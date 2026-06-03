from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
import torchvision.models.video as video_models
import torchvision.transforms as transforms


DEFAULT_VIDEOMAE_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
VIDEOMAE_EMBEDDING_DIM = 768
DEFAULT_VIDEOMAE_BATCH_SIZE = 4


@lru_cache(maxsize=32)
def _hog_descriptor(image_size: int) -> cv2.HOGDescriptor:
    cell_size = max(4, image_size // 8)
    block_size = cell_size * 2
    block_stride = cell_size
    return cv2.HOGDescriptor(
        _winSize=(image_size, image_size),
        _blockSize=(block_size, block_size),
        _blockStride=(block_stride, block_stride),
        _cellSize=(cell_size, cell_size),
        _nbins=9,
    )


def _normalize_histogram(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total == 0.0:
        return values.astype(np.float32)
    return (values / total).astype(np.float32)


def _channel_histograms(image: np.ndarray, bins: int = 16) -> np.ndarray:
    histograms = []
    for channel_index in range(image.shape[2]):
        histogram, _ = np.histogram(image[..., channel_index], bins=bins, range=(0.0, 1.0))
        histograms.append(_normalize_histogram(histogram.astype(np.float32)))
    return np.concatenate(histograms)


def _image_statistics(image: np.ndarray) -> np.ndarray:
    channel_mean = image.mean(axis=(0, 1)).astype(np.float32)
    channel_std = image.std(axis=(0, 1)).astype(np.float32)
    return np.concatenate([channel_mean, channel_std])


def _hog_features(gray_image: np.ndarray, image_size: int) -> np.ndarray:
    hog = _hog_descriptor(image_size)
    descriptor = hog.compute((gray_image * 255.0).astype(np.uint8))
    if descriptor is None:
        return np.zeros((0,), dtype=np.float32)
    return descriptor.reshape(-1).astype(np.float32)


def _edge_features(gray_image: np.ndarray) -> np.ndarray:
    edges = cv2.Canny((gray_image * 255.0).astype(np.uint8), 80, 160)
    edge_density = np.array([edges.mean() / 255.0], dtype=np.float32)
    return edge_density


def _spatial_thumbnail(image: np.ndarray, size: int = 16) -> np.ndarray:
    thumbnail = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return thumbnail.reshape(-1).astype(np.float32)


@lru_cache(maxsize=1)
def _get_resnet50_model():
    """Load pretrained ResNet-50 and remove the classification head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
    model.to(device)
    model.eval()
    return model, device


@lru_cache(maxsize=1)
def _get_vgg16_model():
    """Load pretrained VGG16 and keep it as a feature extractor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
    model.to(device)
    model.eval()
    return model, device


@lru_cache(maxsize=1)
def _get_r3d_18_model():
    """Load pretrained R3D-18 on Kinetics-400 and remove the classification head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = video_models.r3d_18(weights=video_models.R3D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model, device


@lru_cache(maxsize=1)
def _get_r2plus1d_18_model():
    """Load pretrained R(2+1)D-18 on Kinetics-400 and remove the classification head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = video_models.r2plus1d_18(weights=video_models.R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model, device


@lru_cache(maxsize=2)
def _get_videomae_model(model_name: str = DEFAULT_VIDEOMAE_MODEL_NAME):
    try:
        from transformers import AutoImageProcessor, VideoMAEModel
    except ImportError as error:
        raise ImportError(
            "The 'transformers' package is required for feature_extractor='videomae'. "
            "Install project dependencies again before rebuilding the pipeline."
        ) from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model, device


def _pretrained_embedding_dim(feature_extractor: str) -> int:
    if feature_extractor == "resnet50":
        return 2048
    if feature_extractor == "vgg16":
        return 4096
    if feature_extractor in {"r3d_18", "r2plus1d_18"}:
        return 512
    if feature_extractor == "videomae":
        return VIDEOMAE_EMBEDDING_DIM
    raise ValueError(f"Unknown pretrained feature extractor: {feature_extractor}")


def _is_video_feature_extractor(feature_extractor: str) -> bool:
    return feature_extractor in {"r3d_18", "r2plus1d_18", "videomae"}


def load_image_vector_pretrained(image_path: str | Path, feature_extractor: str = "resnet50") -> np.ndarray:
    """Extract pretrained CNN embeddings from an image."""
    if feature_extractor == "resnet50":
        model, device = _get_resnet50_model()
    elif feature_extractor == "vgg16":
        model, device = _get_vgg16_model()
    else:
        raise ValueError(f"Unknown pretrained feature extractor: {feature_extractor}")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(tensor).squeeze().cpu().numpy()

    return embedding.astype(np.float32)


def _frame_identity(image_path: str | Path) -> tuple[str, int] | None:
    path = Path(image_path)
    match = re.match(r"(?P<video>.+)_(?P<frame>\d+)$", path.stem)
    if match is None:
        return None
    return match.group("video"), int(match.group("frame"))


def _clip_frame_paths(anchor_path: str | Path, clip_len: int = 16) -> list[Path]:
    path = Path(anchor_path)
    identity = _frame_identity(path)
    if identity is None:
        return [path] * clip_len

    video_id, anchor_frame = identity
    candidates = []
    for frame_path in path.parent.glob(f"{video_id}_*.png"):
        frame_identity = _frame_identity(frame_path)
        if frame_identity is None:
            continue
        _, frame_number = frame_identity
        candidates.append((abs(frame_number - anchor_frame), frame_number, frame_path))

    if not candidates:
        return [path] * clip_len

    selected = [frame_path for _, _, frame_path in sorted(candidates)[:clip_len]]
    selected = sorted(selected, key=lambda frame_path: _frame_identity(frame_path)[1] if _frame_identity(frame_path) else 0)
    if len(selected) < clip_len:
        selected.extend([selected[-1]] * (clip_len - len(selected)))
    return selected[:clip_len]


def _load_video_clip_tensor(anchor_path: str | Path, clip_len: int = 16) -> torch.Tensor:
    frame_paths = _clip_frame_paths(anchor_path, clip_len=clip_len)
    preprocess = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    ])

    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(preprocess(image.convert("RGB")))

    return torch.stack(frames, dim=1).unsqueeze(0)


def _load_video_clip_frames(anchor_path: str | Path, clip_len: int = 16) -> list[Image.Image]:
    frames = []
    for frame_path in _clip_frame_paths(anchor_path, clip_len=clip_len):
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB").copy())
    return frames


def load_video_vector_pretrained(anchor_path: str | Path, feature_extractor: str = "r2plus1d_18") -> np.ndarray:
    """Extract Kinetics-400 video embeddings from a frame-centered clip."""
    if feature_extractor == "r3d_18":
        model, device = _get_r3d_18_model()
    elif feature_extractor == "r2plus1d_18":
        model, device = _get_r2plus1d_18_model()
    else:
        raise ValueError(f"Unknown video feature extractor: {feature_extractor}")

    tensor = _load_video_clip_tensor(anchor_path).to(device)
    with torch.no_grad():
        embedding = model(tensor).squeeze().cpu().numpy()

    return embedding.astype(np.float32)


def load_video_vector_videomae(
    anchor_path: str | Path,
    model_name: str = DEFAULT_VIDEOMAE_MODEL_NAME,
    clip_len: int = 16,
) -> np.ndarray:
    """Extract a VideoMAE transformer embedding from a frame-centered clip."""
    return load_video_vectors_videomae([anchor_path], model_name=model_name, clip_len=clip_len)[0]


def load_video_vectors_videomae(
    anchor_paths: Sequence[str | Path],
    model_name: str = DEFAULT_VIDEOMAE_MODEL_NAME,
    clip_len: int = 16,
) -> list[np.ndarray]:
    """Extract VideoMAE transformer embeddings for several frame-centered clips."""
    if not anchor_paths:
        return []

    processor, model, device = _get_videomae_model(model_name)
    videos = [_load_video_clip_frames(anchor_path, clip_len=clip_len) for anchor_path in anchor_paths]

    try:
        inputs = processor(videos, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
    except Exception:
        processed_videos = [
            processor(frames, return_tensors="pt")["pixel_values"].squeeze(0)
            for frames in videos
        ]
        pixel_values = torch.stack(processed_videos, dim=0)

    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    return [embedding.astype(np.float32) for embedding in embeddings]


def _feature_cache_namespace(feature_extractor: str, video_model_name: str | None = None) -> str:
    if feature_extractor != "videomae" or not video_model_name:
        return feature_extractor
    digest = hashlib.sha256(video_model_name.encode("utf-8")).hexdigest()[:12]
    return f"{feature_extractor}-{digest}"


def _embedding_cache_path(
    image_path: str | Path,
    feature_extractor: str,
    cache_dir: str | Path,
    video_model_name: str | None = None,
) -> Path:
    path = Path(image_path)
    try:
        stat = path.stat()
        fingerprint = f"{path.resolve()}|{feature_extractor}|{video_model_name or ''}|{stat.st_mtime_ns}|{stat.st_size}"
    except OSError:
        fingerprint = f"{path}|{feature_extractor}|{video_model_name or ''}"
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    return Path(cache_dir) / _feature_cache_namespace(feature_extractor, video_model_name) / f"{digest}.npy"


def load_image_vector_pretrained_cached(
    image_path: str | Path,
    feature_extractor: str = "resnet50",
    cache_dir: str | Path | None = None,
    video_model_name: str = DEFAULT_VIDEOMAE_MODEL_NAME,
) -> np.ndarray:
    if cache_dir is None:
        if feature_extractor == "videomae":
            return load_video_vector_videomae(image_path, model_name=video_model_name)
        if _is_video_feature_extractor(feature_extractor):
            return load_video_vector_pretrained(image_path, feature_extractor=feature_extractor)
        return load_image_vector_pretrained(image_path, feature_extractor=feature_extractor)

    cache_path = _embedding_cache_path(
        image_path,
        feature_extractor,
        cache_dir,
        video_model_name=video_model_name if feature_extractor == "videomae" else None,
    )
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    if feature_extractor == "videomae":
        embedding = load_video_vector_videomae(image_path, model_name=video_model_name)
    elif _is_video_feature_extractor(feature_extractor):
        embedding = load_video_vector_pretrained(image_path, feature_extractor=feature_extractor)
    else:
        embedding = load_image_vector_pretrained(image_path, feature_extractor=feature_extractor)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embedding)
    return embedding


def load_video_vectors_videomae_cached_batch(
    anchor_paths: Sequence[str | Path],
    cache_dir: str | Path | None = None,
    video_model_name: str = DEFAULT_VIDEOMAE_MODEL_NAME,
    batch_size: int = DEFAULT_VIDEOMAE_BATCH_SIZE,
) -> list[np.ndarray]:
    paths = list(anchor_paths)
    if not paths:
        return []

    batch_size = max(1, int(batch_size))
    if cache_dir is None:
        embeddings = []
        for start in range(0, len(paths), batch_size):
            embeddings.extend(
                load_video_vectors_videomae(
                    paths[start : start + batch_size],
                    model_name=video_model_name,
                )
            )
        return embeddings

    cached_embeddings: list[np.ndarray | None] = [None] * len(paths)
    missing_by_cache_path: dict[Path, tuple[Path, list[int]]] = {}
    for index, path in enumerate(paths):
        cache_path = _embedding_cache_path(
            path,
            "videomae",
            cache_dir,
            video_model_name=video_model_name,
        )
        if cache_path.exists():
            cached_embeddings[index] = np.load(cache_path).astype(np.float32)
        else:
            if cache_path not in missing_by_cache_path:
                missing_by_cache_path[cache_path] = (Path(path), [])
            missing_by_cache_path[cache_path][1].append(index)

    missing = list(missing_by_cache_path.items())
    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        chunk_embeddings = load_video_vectors_videomae(
            [path for _, (path, _) in chunk],
            model_name=video_model_name,
        )
        for (cache_path, (_, indexes)), embedding in zip(chunk, chunk_embeddings):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embedding)
            for index in indexes:
                cached_embeddings[index] = embedding

    return [embedding if embedding is not None else np.zeros(VIDEOMAE_EMBEDDING_DIM, dtype=np.float32) for embedding in cached_embeddings]


def load_image_vector(image_path: str | Path, image_size: int, color_mode: str) -> np.ndarray:
    mode = "RGB" if color_mode == "rgb" else "L"

    with Image.open(image_path) as image:
        image = image.convert(mode)
        image = image.resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0

    if color_mode == "rgb":
        rgb_image = array
        gray_image = cv2.cvtColor((rgb_image * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        features = [
            _hog_features(gray_image, image_size),
            _channel_histograms(rgb_image),
            _image_statistics(rgb_image),
            _edge_features(gray_image),
            _spatial_thumbnail(rgb_image, size=max(8, image_size // 4)),
        ]
        return np.concatenate(features).astype(np.float32)

    gray_image = array if array.ndim == 2 else array[..., 0]
    gray_image = gray_image.astype(np.float32)
    gray_image = gray_image if gray_image.ndim == 2 else gray_image.reshape(image_size, image_size)
    gray_image = np.clip(gray_image, 0.0, 1.0)
    features = [
        _hog_features(gray_image, image_size),
        _image_statistics(gray_image[..., None]),
        _edge_features(gray_image),
        _spatial_thumbnail(gray_image, size=max(8, image_size // 4)),
    ]
    return np.concatenate(features).astype(np.float32)


def build_feature_matrix(
    manifest: pd.DataFrame,
    image_size: int,
    color_mode: str,
    use_pretrained: bool = True,
    feature_extractor: str = "resnet50",
    cache_dir: str | Path | None = None,
    video_model_name: str = DEFAULT_VIDEOMAE_MODEL_NAME,
    videomae_batch_size: int = DEFAULT_VIDEOMAE_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix from images.
    
    Args:
        manifest: DataFrame with 'path' and 'label' columns
        image_size: Size for traditional HOG features (ignored if use_pretrained=True)
        color_mode: 'rgb' or 'grayscale' (ignored if use_pretrained=True)
        use_pretrained: If True, use CNN embeddings; else use HOG+color+edge
        feature_extractor: backbone to use for embeddings ('resnet50', 'vgg16', 'r3d_18', 'r2plus1d_18' or 'videomae')
        cache_dir: Optional folder to cache embeddings and speed up repeated runs
        video_model_name: Hugging Face model id used when feature_extractor='videomae'
        videomae_batch_size: Number of clips processed together when using VideoMAE
    
    Returns:
        Tuple of (features array, labels array)
    """
    if use_pretrained:
        embedding_dim = _pretrained_embedding_dim(feature_extractor)
        print(f"Loading {feature_extractor} pretrained model...")
        features = []
        if feature_extractor == "videomae":
            paths = manifest["path"].tolist()
            batch_size = max(1, int(videomae_batch_size))
            for start in range(0, len(paths), batch_size):
                chunk_paths = paths[start : start + batch_size]
                try:
                    features.extend(
                        load_video_vectors_videomae_cached_batch(
                            chunk_paths,
                            cache_dir=cache_dir,
                            video_model_name=video_model_name,
                            batch_size=batch_size,
                        )
                    )
                except Exception as batch_error:
                    print(f"  Warning: batch failed for rows {start + 1}-{start + len(chunk_paths)}: {batch_error}")
                    for path in chunk_paths:
                        try:
                            features.append(
                                load_image_vector_pretrained_cached(
                                    path,
                                    feature_extractor=feature_extractor,
                                    cache_dir=cache_dir,
                                    video_model_name=video_model_name,
                                )
                            )
                        except Exception as error:
                            print(f"  Warning: failed to process {path}: {error}")
                            features.append(np.zeros(embedding_dim, dtype=np.float32))

                processed = min(start + len(chunk_paths), len(paths))
                if processed % 50 == 0 or processed == len(paths):
                    print(f"  Processed {processed}/{len(manifest)} videos")

            labels = manifest["label"].to_numpy()
            return np.vstack(features), labels

        for idx, row in enumerate(manifest.itertuples(index=False)):
            try:
                embedding = load_image_vector_pretrained_cached(
                    row.path,
                    feature_extractor=feature_extractor,
                    cache_dir=cache_dir,
                    video_model_name=video_model_name,
                )
                features.append(embedding)
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(manifest)} images")
            except Exception as e:
                print(f"  Warning: failed to process {row.path}: {e}")
                features.append(np.zeros(embedding_dim, dtype=np.float32))
        labels = manifest["label"].to_numpy()
        return np.vstack(features), labels
    else:
        features = [
            load_image_vector(row.path, image_size=image_size, color_mode=color_mode)
            for row in manifest.itertuples(index=False)
        ]
        labels = manifest["label"].to_numpy()
        return np.vstack(features), labels
