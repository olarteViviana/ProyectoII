from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ucf_crime_recognition.predict import predict_image_details


HIGH_RISK_LABELS = {"Abuse", "Arson", "Assault", "Explosion", "Fighting", "Robbery", "Shooting"}
MEDIUM_RISK_LABELS = {"Burglary", "RoadAccidents", "Shoplifting", "Stealing", "Vandalism"}
NORMAL_LABEL = "NormalVideos"
CLASS_RISK_WEIGHTS = {
    NORMAL_LABEL: 0.7,
    "Fighting": 2.7,
    "Assault": 2.5,
    "Abuse": 2.35,
    "Shooting": 2.3,
    "Explosion": 2.25,
    "Arson": 2.15,
    "Robbery": 1.9,
    "Vandalism": 1.6,
}
MOTION_AFFINITY = {
    "Fighting": 0.42,
    "Assault": 0.35,
    "Abuse": 0.28,
    "Vandalism": 0.18,
    "Robbery": 0.08,
}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
FRAME_SAMPLES = 16
VIDEO_CLIP_LEN = 16
CLIP_WINDOW_SECONDS = 2.0
MOTION_PRIORITY = True
MOTION_STRIDE_SECONDS = 1.0
MOTION_PRIORITY_RATIO = 0.65
MOTION_THUMBNAIL_SIZE = (96, 54)
MOTION_EVENT_QUANTILE = 0.7
SEGMENT_VIDEO_CODECS = (
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("H264", ".mp4"),
    ("MJPG", ".avi"),
)

LABEL_ACTIONS = {
    "Abuse": "Escalar al equipo de seguridad y revisar el clip completo.",
    "Arson": "Bloquear zona, alertar a emergencias y guardar evidencia.",
    "Assault": "Enviar alerta inmediata y marcar la cámara para revisión humana.",
    "Explosion": "Evacuar el área virtualmente y notificar supervisión.",
    "Fighting": "Priorizar revisión y generar ticket de incidente.",
    "Robbery": "Restringir acceso, revisar rutas de salida y compartir evidencia.",
    "Shooting": "Escalar como incidente crítico de forma inmediata.",
    "Burglary": "Abrir verificación manual y revisar perímetro.",
    "RoadAccidents": "Notificar al operador y registrar ubicación.",
    "Shoplifting": "Guardar clips clave y activar seguimiento interno.",
    "Stealing": "Registrar el evento y alertar al guardia de turno.",
    "Vandalism": "Marcar para revisión y comparar con cámaras cercanas.",
    "NormalVideos": "Sin alerta crítica. Mantener seguimiento pasivo.",
}


def risk_profile(label: str, confidence: float) -> tuple[str, str, str]:
    if label in HIGH_RISK_LABELS:
        tier = "Crítico"
        score = "Alto"
    elif label in MEDIUM_RISK_LABELS:
        tier = "Vigilancia"
        score = "Medio"
    else:
        tier = "Normal"
        score = "Bajo"

    if label == NORMAL_LABEL and confidence < 0.5:
        tier = "Revisión"
        score = "Medio"

    if confidence < 0.45 and tier != "Normal":
        score = "Medio"
        tier = "Revisión"

    recommendation = LABEL_ACTIONS.get(label, "Sin recomendación específica disponible.")
    return tier, score, recommendation


def risk_signal_from_scores(class_scores: dict) -> tuple[str, float]:
    risk_scores = {
        str(class_name): float(probability)
        for class_name, probability in class_scores.items()
        if str(class_name) != NORMAL_LABEL
    }
    if not risk_scores:
        return "N/D", 0.0

    class_name = max(risk_scores, key=risk_scores.get)
    return class_name, risk_scores[class_name]


def normalize_motion_scores(frames: list[dict]) -> list[dict]:
    if not frames:
        return frames

    max_motion = max((float(frame.get("motion_score") or 0.0) for frame in frames), default=0.0)
    for frame in frames:
        motion_score = float(frame.get("motion_score") or 0.0)
        frame["motion_intensity"] = motion_score / max_motion if max_motion > 0.0 else 0.0
    return frames


def adjust_class_scores_for_motion(class_scores: dict, motion_intensity: float) -> dict[str, float]:
    adjusted = {str(label): float(score) for label, score in (class_scores or {}).items()}
    if not adjusted:
        return adjusted

    motion = max(0.0, min(float(motion_intensity or 0.0), 1.0))
    if motion < 0.25 or NORMAL_LABEL not in adjusted:
        return adjusted

    risk_scores = {label: score for label, score in adjusted.items() if label != NORMAL_LABEL}
    if not risk_scores or max(risk_scores.values()) < 0.04:
        return adjusted

    normal_dampening = min(0.6, 0.18 + 0.42 * motion)
    adjusted[NORMAL_LABEL] = max(0.0, adjusted[NORMAL_LABEL] * (1.0 - normal_dampening))

    for label, score in list(adjusted.items()):
        if label == NORMAL_LABEL or score < 0.03:
            continue
        affinity = MOTION_AFFINITY.get(label, 0.1 if label in HIGH_RISK_LABELS else 0.04)
        adjusted[label] = min(1.0, score + affinity * motion * (1.0 - score))

    return adjusted


def motion_adjusted_prediction_label(
    original_label: str,
    adjusted_scores: dict[str, float],
    motion_intensity: float,
) -> tuple[str, float]:
    if not adjusted_scores:
        return original_label, 0.0

    normal_score = float(adjusted_scores.get(NORMAL_LABEL, 0.0))
    risk_scores = {label: score for label, score in adjusted_scores.items() if label != NORMAL_LABEL}
    if not risk_scores:
        return original_label, normal_score

    top_risk_label = max(risk_scores, key=risk_scores.get)
    top_risk_score = float(risk_scores[top_risk_label])
    original_score = float(adjusted_scores.get(original_label, top_risk_score))

    if (
        original_label == NORMAL_LABEL
        and motion_intensity >= 0.45
        and top_risk_score >= 0.08
        and top_risk_score >= normal_score * 0.45
    ):
        return top_risk_label, top_risk_score

    return original_label, original_score


def format_active_labels(prediction: dict) -> str:
    labels = prediction.get("predictions")
    if labels:
        return " | ".join(str(label) for label in labels)
    return str(prediction.get("prediction", ""))


def is_video_filename(filename: str | Path) -> bool:
    return Path(str(filename)).suffix.lower().lstrip(".") in VIDEO_EXTENSIONS


def is_video_file(uploaded_file) -> bool:
    return is_video_filename(uploaded_file.name)


def _sample_anchor_indexes(total_frames: int, frame_samples: int) -> list[int]:
    sample_count = min(frame_samples, total_frames)
    return sorted({int(round(index)) for index in np.linspace(0, total_frames - 1, sample_count)})


def _motion_frame_scores(
    video_path: str | Path,
    total_frames: int,
    fps: float,
    stride_seconds: float = MOTION_STRIDE_SECONDS,
) -> dict[int, float]:
    if total_frames <= 1:
        return {}

    stride = max(1, int(round(float(fps) * float(stride_seconds))))
    sample_indexes = list(range(0, total_frames, stride))
    if sample_indexes[-1] != total_frames - 1:
        sample_indexes.append(total_frames - 1)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {}

    previous_gray: np.ndarray | None = None
    scores: dict[int, float] = {}
    try:
        for frame_index in sample_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, MOTION_THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
            if previous_gray is None:
                scores[frame_index] = 0.0
            else:
                difference = cv2.absdiff(gray, previous_gray)
                scores[frame_index] = float(difference.mean() / 255.0)
            previous_gray = gray
    finally:
        capture.release()

    return scores


def _select_motion_anchor_indexes(
    motion_scores: dict[int, float],
    anchor_count: int,
    fps: float,
    clip_window_seconds: float = CLIP_WINDOW_SECONDS,
) -> list[int]:
    if anchor_count <= 0 or not motion_scores:
        return []

    minimum_gap = max(1, int(round(float(fps) * max(1.0, float(clip_window_seconds) * 0.85))))
    selected: list[int] = []
    for frame_index, score in sorted(motion_scores.items(), key=lambda item: item[1], reverse=True):
        if score <= 0.0:
            break
        if all(abs(frame_index - existing) >= minimum_gap for existing in selected):
            selected.append(frame_index)
        if len(selected) >= anchor_count:
            break

    return sorted(selected)


def _append_anchor_with_gap(selected: list[int], frame_index: int, anchor_count: int, minimum_gap: int) -> None:
    if len(selected) >= anchor_count:
        return
    if frame_index in selected:
        return
    if all(abs(frame_index - existing) >= minimum_gap for existing in selected):
        selected.append(frame_index)


def _motion_event_ranges(motion_scores: dict[int, float], fps: float) -> list[dict]:
    positive_items = [(index, score) for index, score in sorted(motion_scores.items()) if score > 0.0]
    if not positive_items:
        return []

    values = np.asarray([score for _, score in positive_items], dtype=np.float32)
    threshold = max(float(np.quantile(values, MOTION_EVENT_QUANTILE)), float(values.mean()))
    active_items = [(index, score) for index, score in positive_items if score >= threshold]
    if not active_items:
        return []

    sample_gaps = np.diff([index for index, _ in sorted(motion_scores.items())])
    median_gap = int(np.median(sample_gaps)) if len(sample_gaps) else max(1, int(round(float(fps))))
    max_event_gap = max(1, int(round(median_gap * 2.5)))

    groups: list[list[tuple[int, float]]] = []
    current_group: list[tuple[int, float]] = []
    for frame_index, score in active_items:
        if not current_group or frame_index - current_group[-1][0] <= max_event_gap:
            current_group.append((frame_index, score))
        else:
            groups.append(current_group)
            current_group = [(frame_index, score)]
    if current_group:
        groups.append(current_group)

    events = []
    for group in groups:
        start_frame = group[0][0]
        end_frame = group[-1][0]
        peak_frame, peak_score = max(group, key=lambda item: item[1])
        total_score = float(sum(score for _, score in group))
        duration_seconds = max((end_frame - start_frame + 1) / float(fps), median_gap / float(fps))
        events.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "peak_frame": peak_frame,
                "peak_score": float(peak_score),
                "total_score": total_score,
                "duration_seconds": float(duration_seconds),
                "priority": total_score + float(peak_score) * 2.0 + duration_seconds * 0.05,
            }
        )

    return sorted(events, key=lambda event: event["priority"], reverse=True)


def _event_anchor_candidates(event: dict, fps: float, clip_window_seconds: float) -> list[int]:
    start_frame = int(event["start_frame"])
    end_frame = int(event["end_frame"])
    peak_frame = int(event["peak_frame"])
    duration_frames = max(1, end_frame - start_frame + 1)
    spacing_frames = max(1, int(round(float(fps) * max(0.75, float(clip_window_seconds) / 2.0))))
    candidate_count = max(1, int(np.ceil(duration_frames / spacing_frames)))

    anchors = [peak_frame]
    if candidate_count > 1:
        anchors.extend(int(round(index)) for index in np.linspace(start_frame, end_frame, candidate_count))
    else:
        anchors.extend([start_frame, end_frame])

    return sorted({max(start_frame, min(anchor, end_frame)) for anchor in anchors})


def _select_motion_event_anchor_indexes(
    motion_scores: dict[int, float],
    anchor_count: int,
    fps: float,
    clip_window_seconds: float = CLIP_WINDOW_SECONDS,
) -> list[int]:
    if anchor_count <= 0 or not motion_scores:
        return []

    events = _motion_event_ranges(motion_scores, fps)
    if not events:
        return _select_motion_anchor_indexes(
            motion_scores,
            anchor_count,
            fps,
            clip_window_seconds=clip_window_seconds,
        )

    minimum_gap = max(1, int(round(float(fps) * max(1.0, float(clip_window_seconds) * 0.85))))
    selected: list[int] = []

    for event in events:
        _append_anchor_with_gap(selected, int(event["peak_frame"]), anchor_count, minimum_gap)

    candidate_queues = [
        [
            candidate
            for candidate in _event_anchor_candidates(event, fps, clip_window_seconds)
            if candidate != int(event["peak_frame"])
        ]
        for event in events
    ]
    while len(selected) < anchor_count and any(candidate_queues):
        for candidates in candidate_queues:
            if not candidates:
                continue
            _append_anchor_with_gap(selected, candidates.pop(0), anchor_count, minimum_gap)
            if len(selected) >= anchor_count:
                break

    if len(selected) < anchor_count:
        for anchor in _select_motion_anchor_indexes(
            motion_scores,
            anchor_count,
            fps,
            clip_window_seconds=clip_window_seconds,
        ):
            _append_anchor_with_gap(selected, anchor, anchor_count, minimum_gap)
            if len(selected) >= anchor_count:
                break

    return sorted(selected)


def _motion_score_for_anchor(anchor_index: int, motion_scores: dict[int, float]) -> float:
    if not motion_scores:
        return 0.0
    nearest_index = min(motion_scores, key=lambda frame_index: abs(frame_index - anchor_index))
    return float(motion_scores[nearest_index])


def _motion_prioritized_anchor_indexes(
    video_path: str | Path,
    total_frames: int,
    fps: float,
    frame_samples: int,
    clip_window_seconds: float = CLIP_WINDOW_SECONDS,
    motion_stride_seconds: float = MOTION_STRIDE_SECONDS,
) -> tuple[list[int], dict[int, float], set[int], set[int]]:
    if frame_samples <= 3:
        anchors = _sample_anchor_indexes(total_frames, frame_samples)
        return anchors, {}, set(anchors), set()

    sample_count = min(frame_samples, total_frames)
    temporal_count = max(3, int(round(sample_count * (1.0 - MOTION_PRIORITY_RATIO))))
    temporal_count = min(temporal_count, sample_count)
    motion_count = max(0, sample_count - temporal_count)

    temporal_anchors = _sample_anchor_indexes(total_frames, temporal_count)
    motion_scores = _motion_frame_scores(
        video_path,
        total_frames,
        fps,
        stride_seconds=motion_stride_seconds,
    )
    motion_anchors = _select_motion_event_anchor_indexes(
        motion_scores,
        motion_count,
        fps,
        clip_window_seconds=clip_window_seconds,
    )

    anchors = sorted({*temporal_anchors, *motion_anchors})
    if len(anchors) < sample_count:
        for anchor_index in _sample_anchor_indexes(total_frames, sample_count):
            if anchor_index not in anchors:
                anchors.append(anchor_index)
            if len(anchors) >= sample_count:
                break

    anchors = sorted(anchors[:sample_count])
    temporal_anchor_set = set(temporal_anchors).intersection(anchors)
    motion_anchor_set = set(motion_anchors).intersection(anchors)
    return anchors, motion_scores, temporal_anchor_set, motion_anchor_set


def _sampling_reason(anchor_index: int, temporal_anchors: set[int], motion_anchors: set[int]) -> str:
    reasons = []
    if anchor_index in motion_anchors:
        reasons.append("movimiento")
    if anchor_index in temporal_anchors:
        reasons.append("cobertura")
    if not reasons:
        reasons.append("relleno")
    return "+".join(reasons)


def _clip_window_indexes(
    anchor_frame: int,
    total_frames: int,
    fps: float,
    clip_len: int = VIDEO_CLIP_LEN,
    clip_window_seconds: float = CLIP_WINDOW_SECONDS,
) -> list[int]:
    if total_frames <= clip_len:
        return list(range(total_frames))

    window_frames = max(clip_len, int(round(float(clip_window_seconds) * float(fps))))
    window_frames = min(window_frames, total_frames)
    start = anchor_frame - window_frames // 2
    start = max(0, min(start, total_frames - window_frames))
    end = start + window_frames - 1

    candidates = list(range(start, end + 1))
    targets = np.linspace(start, end, min(clip_len, len(candidates)))
    selected: list[int] = []
    for target in targets:
        frame_index = min(candidates, key=lambda index: (abs(index - target), abs(index - anchor_frame)))
        if frame_index not in selected:
            selected.append(frame_index)

    if anchor_frame not in selected:
        selected.append(anchor_frame)

    while len(selected) > clip_len:
        removable = [index for index in selected if index != anchor_frame]
        selected.remove(max(removable, key=lambda index: abs(index - anchor_frame)))

    remaining = [index for index in candidates if index not in selected]
    while len(selected) < min(clip_len, len(candidates)) and remaining:
        frame_index = min(remaining, key=lambda index: abs(index - anchor_frame))
        selected.append(frame_index)
        remaining.remove(frame_index)

    return sorted(selected)


def _safe_video_id(video_path: str | Path) -> str:
    return Path(video_path).stem.replace(" ", "_") or "video"


def _open_segment_writer(
    output_stem: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, Path]:
    for codec, suffix in SEGMENT_VIDEO_CODECS:
        output_path = output_stem.with_suffix(suffix)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer, output_path
        writer.release()

    raise ValueError("No se pudo crear un clip reproducible con OpenCV.")


def export_video_segment(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    output_dir: str | Path | None = None,
) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("No se pudo abrir el video.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("El video no contiene fotogramas legibles.")

    safe_start = max(0, min(int(start_frame), total_frames - 1))
    safe_end = max(safe_start, min(int(end_frame), total_frames - 1))
    output_root = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="ucf-crime-segments-"))
    output_root.mkdir(parents=True, exist_ok=True)
    output_stem = output_root / f"{_safe_video_id(video_path)}_{safe_start:06d}_{safe_end:06d}"

    writer, output_path = _open_segment_writer(output_stem, fps, (width, height))
    frames_written = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, safe_start)
        for _ in range(safe_start, safe_end + 1):
            success, frame = capture.read()
            if not success:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            frames_written += 1
    finally:
        writer.release()
        capture.release()

    if frames_written == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("No se pudo exportar ningún frame del segmento.")

    return output_path


def extract_sampled_frames(
    video_path: Path,
    frame_samples: int = FRAME_SAMPLES,
    clip_window_seconds: float = CLIP_WINDOW_SECONDS,
    motion_priority: bool = MOTION_PRIORITY,
    motion_stride_seconds: float = MOTION_STRIDE_SECONDS,
    output_dir: str | Path | None = None,
) -> list[dict]:
    if frame_samples < 1:
        raise ValueError("frame_samples must be at least 1.")
    if clip_window_seconds <= 0:
        raise ValueError("clip_window_seconds must be greater than 0.")
    if motion_stride_seconds <= 0:
        raise ValueError("motion_stride_seconds must be greater than 0.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("No se pudo abrir el video.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    if total_frames <= 0:
        capture.release()
        raise ValueError("El video no contiene fotogramas legibles.")

    temporal_anchors = set(_sample_anchor_indexes(total_frames, min(frame_samples, total_frames)))
    motion_scores: dict[int, float] = {}
    motion_anchors: set[int] = set()
    if motion_priority:
        anchor_indexes, motion_scores, temporal_anchors, motion_anchors = _motion_prioritized_anchor_indexes(
            video_path,
            total_frames,
            fps,
            frame_samples,
            clip_window_seconds=clip_window_seconds,
            motion_stride_seconds=motion_stride_seconds,
        )
    else:
        anchor_indexes = sorted(temporal_anchors)
    temp_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="ucf-crime-frames-"))
    temp_dir.mkdir(parents=True, exist_ok=True)

    video_id = _safe_video_id(video_path)

    extracted_frames: list[dict] = []
    for clip_position, anchor_index in enumerate(anchor_indexes):
        clip_indexes = _clip_window_indexes(
            anchor_index,
            total_frames,
            fps,
            clip_window_seconds=clip_window_seconds,
        )
        clip_dir = temp_dir / f"clip_{clip_position:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: dict[int, Path] = {}

        for frame_index in clip_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue

            image_path = clip_dir / f"{video_id}_{frame_index:06d}.png"
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(image_rgb).save(image_path)
            saved_paths[frame_index] = image_path

        image_path = saved_paths.get(anchor_index)
        if image_path is None:
            continue
        timestamp_seconds = anchor_index / fps
        clip_start_frame = min(clip_indexes)
        clip_end_frame = max(clip_indexes)
        clip_duration_seconds = (clip_end_frame - clip_start_frame + 1) / fps
        extracted_frames.append(
            {
                "frame_index": anchor_index,
                "timestamp_seconds": timestamp_seconds,
                "image_path": image_path,
                "clip_start_frame": clip_start_frame,
                "clip_end_frame": clip_end_frame,
                "clip_duration_seconds": clip_duration_seconds,
                "clip_window_seconds": clip_window_seconds,
                "video_fps": fps,
                "motion_score": _motion_score_for_anchor(anchor_index, motion_scores),
                "sampling_reason": _sampling_reason(anchor_index, temporal_anchors, motion_anchors),
            }
        )

    capture.release()
    return extracted_frames


def aggregate_video_predictions(frame_results: list[dict]) -> dict:
    if not frame_results:
        raise ValueError("No fue posible extraer ningún frame del video.")

    result_frame = pd.DataFrame(frame_results)
    score_rows: list[dict] = []
    for frame in frame_results:
        for class_name, probability in frame.get("class_scores", {}).items():
            score_rows.append(
                {
                    "class_name": class_name,
                    "frame_index": frame["frame_index"],
                    "timestamp_s": frame["timestamp_s"],
                    "probability": float(probability),
                }
            )

    score_frame = pd.DataFrame(score_rows)
    if score_frame.empty:
        dominant_label = (
            result_frame.groupby("prediction")
            .agg(count=("prediction", "size"), mean_confidence=("confidence", "mean"), max_confidence=("confidence", "max"))
            .sort_values(["count", "mean_confidence"], ascending=False)
            .reset_index()
            .iloc[0]
        )
        label = str(dominant_label["prediction"])
        confidence = float(dominant_label["max_confidence"])
        class_summary = pd.DataFrame()
        top_alternative = None
        probability_leader_label = label
        probability_leader_value = confidence
        decision_reason = "prediccion_frame"
        decision_note = "La decisión se calculó por mayoría de predicciones en los frames muestreados."
    else:
        class_summary = (
            score_frame.groupby("class_name")
            .agg(
                mean_probability=("probability", "mean"),
                max_probability=("probability", "max"),
                frame_hits=("probability", lambda values: int((values >= 0.2).sum())),
            )
            .reset_index()
        )
        class_summary["risk_weight"] = class_summary["class_name"].map(
            lambda class_name: CLASS_RISK_WEIGHTS.get(
                class_name,
                2.0 if class_name in HIGH_RISK_LABELS else 1.3 if class_name in MEDIUM_RISK_LABELS else 1.0,
            )
        )
        class_summary["decision_score"] = (
            class_summary["mean_probability"] * class_summary["risk_weight"]
            + class_summary["max_probability"] * 0.15
            + class_summary["frame_hits"] * 0.01
        )
        class_summary = class_summary.sort_values(
            ["decision_score", "max_probability", "mean_probability"], ascending=False
        ).reset_index(drop=True)

        best_row = class_summary.iloc[0]
        label = str(best_row["class_name"])
        confidence = float(best_row["max_probability"])
        top_alternative = class_summary.iloc[1] if len(class_summary) > 1 else None
        probability_leader = class_summary.sort_values("max_probability", ascending=False).iloc[0]
        probability_leader_label = str(probability_leader["class_name"])
        probability_leader_value = float(probability_leader["max_probability"])
        decision_reason = "score_operativo"
        decision_note = "La decisión usa score operativo: promedio, pico, persistencia y peso de riesgo por clase."

    if not class_summary.empty:
        normal_row = class_summary[class_summary["class_name"] == NORMAL_LABEL]
        risk_rows = class_summary[class_summary["class_name"] != NORMAL_LABEL]
        if not normal_row.empty and not risk_rows.empty:
            normal_score = float(normal_row.iloc[0]["decision_score"])
            risk_row = risk_rows.sort_values(["decision_score", "max_probability", "mean_probability"], ascending=False).iloc[0]
            risk_score = float(risk_row["decision_score"])
            risk_hits = int(risk_row["frame_hits"])
            risk_peak = float(risk_row["max_probability"])

            if risk_peak >= 0.20 and risk_hits >= 2 and risk_score >= normal_score * 0.55:
                label = str(risk_row["class_name"])
                confidence = risk_peak
                decision_reason = "prioridad_riesgo"
                decision_note = (
                    f"{label} no necesariamente tiene la mayor probabilidad bruta; se priorizó porque aparece "
                    f"en {risk_hits} frames con señal de riesgo y su score operativo es cercano al de {NORMAL_LABEL}."
                )
            elif label == NORMAL_LABEL and top_alternative is not None:
                alternative_label = str(top_alternative["class_name"])
                alternative_score = float(top_alternative["decision_score"])
                if alternative_label != NORMAL_LABEL and alternative_score >= normal_score * 0.9:
                    label = alternative_label
                    confidence = float(top_alternative["max_probability"])
                    decision_reason = "alternativa_cercana"
                    decision_note = (
                        f"{alternative_label} quedó muy cerca de {NORMAL_LABEL} en score operativo; "
                        "se muestra como evento para revisión humana."
                    )

    tier, score, recommendation = risk_profile(label, confidence)
    if not class_summary.empty:
        class_summary = class_summary.copy()
        class_summary["selected"] = class_summary["class_name"].eq(label)

    return {
        "prediction": label,
        "confidence": confidence,
        "decision_reason": decision_reason,
        "decision_note": decision_note,
        "probability_leader": probability_leader_label,
        "probability_leader_confidence": probability_leader_value,
        "tier": tier,
        "score": score,
        "recommendation": recommendation,
        "summary_frame": result_frame,
        "class_summary": class_summary,
    }


def predict_video_details(
    video_path: str | Path,
    frame_samples: int = FRAME_SAMPLES,
    clip_window_seconds: float = CLIP_WINDOW_SECONDS,
    motion_priority: bool = MOTION_PRIORITY,
    motion_stride_seconds: float = MOTION_STRIDE_SECONDS,
    frame_output_dir: str | Path | None = None,
    predictor: Callable[[str | Path], dict] = predict_image_details,
) -> dict:
    frames = extract_sampled_frames(
        Path(video_path),
        frame_samples=frame_samples,
        clip_window_seconds=clip_window_seconds,
        motion_priority=motion_priority,
        motion_stride_seconds=motion_stride_seconds,
        output_dir=frame_output_dir,
    )
    frames = normalize_motion_scores(frames)
    frame_results: list[dict] = []

    for frame in frames:
        prediction = predictor(frame["image_path"])
        original_label = prediction["prediction"]
        raw_confidence = float(prediction["confidence"])
        raw_class_scores = prediction.get("class_scores", {})
        motion_intensity = float(frame.get("motion_intensity") or 0.0)
        class_scores = adjust_class_scores_for_motion(raw_class_scores, motion_intensity)
        label, confidence = motion_adjusted_prediction_label(original_label, class_scores, motion_intensity)
        if confidence == 0.0:
            confidence = raw_confidence
        risk_class, risk_probability = risk_signal_from_scores(class_scores)
        tier, score, recommendation = risk_profile(label, confidence)
        frame_results.append(
            {
                "timestamp_s": round(frame["timestamp_seconds"], 2),
                "frame_index": frame["frame_index"],
                "clip_start_frame": frame.get("clip_start_frame"),
                "clip_end_frame": frame.get("clip_end_frame"),
                "clip_duration_seconds": frame.get("clip_duration_seconds"),
                "clip_window_seconds": frame.get("clip_window_seconds"),
                "video_fps": frame.get("video_fps"),
                "motion_score": frame.get("motion_score"),
                "motion_intensity": motion_intensity,
                "sampling_reason": frame.get("sampling_reason"),
                "prediction": label,
                "raw_prediction": original_label,
                "active_labels": label,
                "confidence": confidence,
                "normal_probability": float(class_scores.get(NORMAL_LABEL, 0.0)),
                "raw_normal_probability": float(raw_class_scores.get(NORMAL_LABEL, 0.0)),
                "top_risk_class": risk_class,
                "top_risk_probability": risk_probability,
                "class_scores": class_scores,
                "raw_class_scores": raw_class_scores,
                "tier": tier,
                "score": score,
                "recommendation": recommendation,
            }
        )

    aggregate = aggregate_video_predictions(frame_results)
    return {"aggregate": aggregate, "frames": frame_results}
