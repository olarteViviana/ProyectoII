from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms


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


def load_image_vector_pretrained(image_path: str | Path, embedding_dim: int = 2048) -> np.ndarray:
    """Extract ResNet-50 embeddings from an image (2048-dim feature vector)."""
    model, device = _get_resnet50_model()
    
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix from images.
    
    Args:
        manifest: DataFrame with 'path' and 'label' columns
        image_size: Size for traditional HOG features (ignored if use_pretrained=True)
        color_mode: 'rgb' or 'grayscale' (ignored if use_pretrained=True)
        use_pretrained: If True, use ResNet-50 embeddings (2048-dim); else use HOG+color+edge
    
    Returns:
        Tuple of (features array, labels array)
    """
    if use_pretrained:
        print("Loading ResNet-50 pretrained model...")
        features = []
        for idx, row in enumerate(manifest.itertuples(index=False)):
            try:
                embedding = load_image_vector_pretrained(row.path)
                features.append(embedding)
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(manifest)} images")
            except Exception as e:
                print(f"  Warning: failed to process {row.path}: {e}")
                features.append(np.zeros(2048, dtype=np.float32))
        labels = manifest["label"].to_numpy()
        return np.vstack(features), labels
    else:
        features = [
            load_image_vector(row.path, image_size=image_size, color_mode=color_mode)
            for row in manifest.itertuples(index=False)
        ]
        labels = manifest["label"].to_numpy()
        return np.vstack(features), labels
