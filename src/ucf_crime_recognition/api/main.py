from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from ucf_crime_recognition.config import IMAGE_EXTENSIONS
from ucf_crime_recognition.predict import predict_image_details
from ucf_crime_recognition.services.triage import (
    CLIP_WINDOW_SECONDS,
    FRAME_SAMPLES,
    LABEL_ACTIONS,
    MOTION_PRIORITY,
    MOTION_STRIDE_SECONDS,
    VIDEO_EXTENSIONS,
    predict_video_details,
)


API_TITLE = "UCF Crime Recognition API"
API_VERSION = "0.1.0"
IMAGE_UPLOAD_EXTENSIONS = IMAGE_EXTENSIONS
VIDEO_UPLOAD_EXTENSIONS = {f".{extension}" for extension in VIDEO_EXTENSIONS}


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class MetadataResponse(BaseModel):
    labels: list[str]
    image_extensions: list[str]
    video_extensions: list[str]


class ImagePredictionResponse(BaseModel):
    prediction: str
    predictions: list[str] = Field(default_factory=list)
    confidence: float
    class_scores: dict[str, float] = Field(default_factory=dict)
    class_thresholds: dict[str, float] = Field(default_factory=dict)
    model_path: str | None = None


class FramePredictionResponse(BaseModel):
    timestamp_s: float
    frame_index: int
    clip_start_frame: int | None = None
    clip_end_frame: int | None = None
    clip_duration_seconds: float | None = None
    clip_window_seconds: float | None = None
    video_fps: float | None = None
    motion_score: float | None = None
    motion_intensity: float | None = None
    sampling_reason: str | None = None
    prediction: str
    raw_prediction: str | None = None
    active_labels: str
    confidence: float
    normal_probability: float
    raw_normal_probability: float | None = None
    top_risk_class: str
    top_risk_probability: float
    class_scores: dict[str, float] = Field(default_factory=dict)
    tier: str
    score: str
    recommendation: str


class ClassSummaryResponse(BaseModel):
    selected: bool = False
    class_name: str
    mean_probability: float
    max_probability: float
    frame_hits: int
    risk_weight: float | None = None
    decision_score: float | None = None


class VideoPredictionResponse(BaseModel):
    prediction: str
    confidence: float
    decision_reason: str
    decision_note: str
    probability_leader: str
    probability_leader_confidence: float
    tier: str
    score: str
    recommendation: str
    frames: list[FramePredictionResponse] = Field(default_factory=list)
    class_summary: list[ClassSummaryResponse] = Field(default_factory=list)


def _file_extension(filename: str | None) -> str:
    return Path(filename or "").suffix.lower()


def _validate_upload(file: UploadFile, allowed_extensions: set[str], media_name: str) -> str:
    extension = _file_extension(file.filename)
    if extension not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported {media_name} extension '{extension or '<none>'}'. Allowed: {allowed}",
        )
    return extension


async def _save_upload_to(uploaded_file: UploadFile, destination: Path) -> None:
    contents = await uploaded_file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    destination.write_bytes(contents)


def _float(value: Any) -> float:
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _int(value: Any) -> int:
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def _score_dict(values: dict | None) -> dict[str, float]:
    return {str(key): _float(value) for key, value in (values or {}).items()}


def _normalize_image_prediction(details: dict) -> ImagePredictionResponse:
    return ImagePredictionResponse(
        prediction=str(details.get("prediction", "")),
        predictions=[str(label) for label in details.get("predictions", [])],
        confidence=_float(details.get("confidence", 0.0)),
        class_scores=_score_dict(details.get("class_scores", {})),
        class_thresholds=_score_dict(details.get("class_thresholds", {})),
        model_path=str(details["model_path"]) if details.get("model_path") else None,
    )


def _dataframe_records(frame: pd.DataFrame | None) -> list[dict]:
    if frame is None or frame.empty:
        return []
    cleaned = frame.replace({np.nan: None})
    return cleaned.to_dict(orient="records")


def _normalize_frame_prediction(frame: dict) -> FramePredictionResponse:
    return FramePredictionResponse(
        timestamp_s=_float(frame.get("timestamp_s", 0.0)),
        frame_index=_int(frame.get("frame_index", 0)),
        clip_start_frame=_int(frame["clip_start_frame"]) if frame.get("clip_start_frame") is not None else None,
        clip_end_frame=_int(frame["clip_end_frame"]) if frame.get("clip_end_frame") is not None else None,
        clip_duration_seconds=_float(frame["clip_duration_seconds"]) if frame.get("clip_duration_seconds") is not None else None,
        clip_window_seconds=_float(frame["clip_window_seconds"]) if frame.get("clip_window_seconds") is not None else None,
        video_fps=_float(frame["video_fps"]) if frame.get("video_fps") is not None else None,
        motion_score=_float(frame["motion_score"]) if frame.get("motion_score") is not None else None,
        motion_intensity=_float(frame["motion_intensity"]) if frame.get("motion_intensity") is not None else None,
        sampling_reason=str(frame["sampling_reason"]) if frame.get("sampling_reason") is not None else None,
        prediction=str(frame.get("prediction", "")),
        raw_prediction=str(frame["raw_prediction"]) if frame.get("raw_prediction") is not None else None,
        active_labels=str(frame.get("active_labels", "")),
        confidence=_float(frame.get("confidence", 0.0)),
        normal_probability=_float(frame.get("normal_probability", 0.0)),
        raw_normal_probability=_float(frame["raw_normal_probability"]) if frame.get("raw_normal_probability") is not None else None,
        top_risk_class=str(frame.get("top_risk_class", "")),
        top_risk_probability=_float(frame.get("top_risk_probability", 0.0)),
        class_scores=_score_dict(frame.get("class_scores", {})),
        tier=str(frame.get("tier", "")),
        score=str(frame.get("score", "")),
        recommendation=str(frame.get("recommendation", "")),
    )


def _normalize_class_summary(row: dict) -> ClassSummaryResponse:
    return ClassSummaryResponse(
        selected=bool(row.get("selected", False)),
        class_name=str(row.get("class_name", "")),
        mean_probability=_float(row.get("mean_probability", 0.0)),
        max_probability=_float(row.get("max_probability", 0.0)),
        frame_hits=_int(row.get("frame_hits", 0)),
        risk_weight=_float(row["risk_weight"]) if row.get("risk_weight") is not None else None,
        decision_score=_float(row["decision_score"]) if row.get("decision_score") is not None else None,
    )


def _normalize_video_prediction(details: dict) -> VideoPredictionResponse:
    aggregate = details["aggregate"]
    summary_rows = _dataframe_records(aggregate.get("class_summary"))
    return VideoPredictionResponse(
        prediction=str(aggregate.get("prediction", "")),
        confidence=_float(aggregate.get("confidence", 0.0)),
        decision_reason=str(aggregate.get("decision_reason", "")),
        decision_note=str(aggregate.get("decision_note", "")),
        probability_leader=str(aggregate.get("probability_leader", "")),
        probability_leader_confidence=_float(aggregate.get("probability_leader_confidence", 0.0)),
        tier=str(aggregate.get("tier", "")),
        score=str(aggregate.get("score", "")),
        recommendation=str(aggregate.get("recommendation", "")),
        frames=[_normalize_frame_prediction(frame) for frame in details.get("frames", [])],
        class_summary=[_normalize_class_summary(row) for row in summary_rows],
    )


def _prediction_error(error: Exception) -> HTTPException:
    if isinstance(error, FileNotFoundError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model file not available: {error}",
        )
    if isinstance(error, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Inference failed: {error}",
    )


def create_app() -> FastAPI:
    api = FastAPI(title=API_TITLE, version=API_VERSION)

    @api.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": API_TITLE, "docs": "/docs", "health": "/api/v1/health"}

    @api.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service=API_TITLE, version=API_VERSION)

    @api.get("/api/v1/metadata", response_model=MetadataResponse)
    def metadata() -> MetadataResponse:
        return MetadataResponse(
            labels=sorted(LABEL_ACTIONS),
            image_extensions=sorted(IMAGE_UPLOAD_EXTENSIONS),
            video_extensions=sorted(VIDEO_UPLOAD_EXTENSIONS),
        )

    @api.post("/api/v1/predict/image", response_model=ImagePredictionResponse)
    async def predict_image_endpoint(file: UploadFile = File(...)) -> ImagePredictionResponse:
        extension = _validate_upload(file, IMAGE_UPLOAD_EXTENSIONS, "image")
        with tempfile.TemporaryDirectory(prefix="ucf-crime-api-image-") as temp_dir:
            image_path = Path(temp_dir) / f"upload{extension}"
            await _save_upload_to(file, image_path)
            try:
                details = predict_image_details(image_path)
            except Exception as error:
                raise _prediction_error(error) from error
        return _normalize_image_prediction(details)

    @api.post("/api/v1/predict/video", response_model=VideoPredictionResponse)
    async def predict_video_endpoint(
        file: UploadFile = File(...),
        frame_samples: int = Query(
            default=FRAME_SAMPLES,
            ge=1,
            le=128,
            description="Cantidad de anclas temporales a evaluar. Cada ancla reconstruye un clip técnico de 16 frames.",
        ),
        clip_window_seconds: float = Query(
            default=CLIP_WINDOW_SECONDS,
            ge=0.25,
            le=10.0,
            description="Duración temporal que cubren los 16 frames del clip técnico.",
        ),
        motion_priority: bool = Query(
            default=MOTION_PRIORITY,
            description="Si está activo, combina cobertura temporal con anclas de alto movimiento.",
        ),
        motion_stride_seconds: float = Query(
            default=MOTION_STRIDE_SECONDS,
            ge=0.1,
            le=2.0,
            description="Separación temporal usada para estimar movimiento entre frames.",
        ),
    ) -> VideoPredictionResponse:
        extension = _validate_upload(file, VIDEO_UPLOAD_EXTENSIONS, "video")
        with tempfile.TemporaryDirectory(prefix="ucf-crime-api-video-") as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / f"upload{extension}"
            frame_dir = temp_path / "frames"
            await _save_upload_to(file, video_path)
            try:
                details = predict_video_details(
                    video_path,
                    frame_samples=frame_samples,
                    clip_window_seconds=clip_window_seconds,
                    motion_priority=motion_priority,
                    motion_stride_seconds=motion_stride_seconds,
                    frame_output_dir=frame_dir,
                )
            except Exception as error:
                raise _prediction_error(error) from error
        return _normalize_video_prediction(details)

    return api


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("ucf_crime_recognition.api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
