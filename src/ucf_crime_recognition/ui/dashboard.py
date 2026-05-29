from __future__ import annotations

from io import BytesIO
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from ucf_crime_recognition.config import load_config, project_path
from ucf_crime_recognition.features.engineering import _clip_frame_paths, _frame_identity
from ucf_crime_recognition.predict import predict_image_details
from ucf_crime_recognition.services import triage as triage_service


APP_NAME = "Sentinel Review"
FRAME_SAMPLES = triage_service.FRAME_SAMPLES
TEMPORAL_COVERAGE_OPTIONS = {
    "Rápida": 8,
    "Balanceada": FRAME_SAMPLES,
    "Alta": 64,
    "Máxima": 128,
}
CLIP_WINDOW_OPTIONS = {
    "Corta": 1.0,
    "Media": triage_service.CLIP_WINDOW_SECONDS,
    "Larga": 4.0,
    "Amplia": 6.0,
    "Extendida": 10.0,
}
REVIEW_WINDOW_SECONDS = 2.0
GALLERY_COLUMNS = 4


def _load_summary() -> pd.DataFrame | None:
    config = load_config()
    summary_path = project_path(config["reports"]["experiment_summary"])
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return None


def _load_report_text() -> str | None:
    config = load_config()
    report_path = project_path(config["reports"]["classification_report"])
    if report_path.exists():
        return report_path.read_text()
    return None


def _render_score_histogram(
    scores: pd.Series | list[float],
    labels: pd.Series | list[str],
    title: str,
) -> None:
    chart_data = pd.DataFrame(
        {
            "label": pd.Series(labels, dtype="string"),
            "score": pd.Series(scores, dtype="float64"),
        }
    ).dropna()
    if chart_data.empty:
        return

    bin_edges = np.linspace(0.0, 1.0, 6)
    chart_data["score"] = chart_data["score"].clip(0.0, 1.0)
    chart_data["bin"] = pd.cut(
        chart_data["score"],
        bins=bin_edges,
        include_lowest=True,
        right=True,
    )
    grouped = chart_data.groupby("bin", observed=False)
    counts = grouped.size().to_numpy()
    bin_labels = [f"{left:.1f}-{right:.1f}" for left, right in zip(bin_edges[:-1], bin_edges[1:])]
    class_labels = []
    for _, group in grouped:
        names = group.sort_values("score", ascending=False)["label"].astype(str).tolist()
        if len(names) > 4:
            names = [*names[:4], f"+{len(names) - 4} más"]
        class_labels.append("\n".join(names))

    fig_height = 4.2 + max((label.count("\n") for label in class_labels), default=0) * 0.25
    fig, ax = plt.subplots(figsize=(9, fig_height))
    bars = ax.bar(
        bin_labels,
        counts,
        color="#60a5fa",
        edgecolor="#172033",
        linewidth=1.1,
        alpha=0.9,
    )
    max_count = max(counts.max(), 1)
    for bar, count, label_text in zip(bars, counts, class_labels):
        if count == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + max_count * 0.04,
            label_text,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#111827",
        )

    ax.set_title(title, fontsize=12, color="#111827", pad=12)
    ax.set_xlabel("Rango de score")
    ax.set_ylabel("Cantidad de clases")
    ax.set_ylim(0, max_count * 1.55)
    ax.grid(axis="y", color="#d7deea", alpha=0.8)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#d7deea")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _risk_profile(label: str, confidence: float) -> tuple[str, str, str]:
    return triage_service.risk_profile(label, confidence)


def _incident_summary(label: str, confidence: float) -> str:
    tier, score, recommendation = _risk_profile(label, confidence)
    return (
        f"**Nivel operativo:** {tier}\n\n"
        f"**Severidad comercial:** {score}\n\n"
        f"**Acción sugerida:** {recommendation}"
    )


def _format_percent(value: float) -> str:
    return f"{float(value):.1%}"


def _risk_signal_from_scores(class_scores: dict) -> tuple[str, float]:
    return triage_service.risk_signal_from_scores(class_scores)


def _format_active_labels(prediction: dict) -> str:
    return triage_service.format_active_labels(prediction)


def _save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    temp_dir = Path(tempfile.mkdtemp(prefix="ucf-crime-ui-"))
    temp_path = temp_dir / f"uploaded{suffix}"
    temp_path.write_bytes(uploaded_file.getbuffer())
    return temp_path


def _is_video_file(uploaded_file) -> bool:
    return triage_service.is_video_file(uploaded_file)


def _extract_sampled_frames(
    video_path: Path,
    frame_samples: int = FRAME_SAMPLES,
    clip_window_seconds: float = triage_service.CLIP_WINDOW_SECONDS,
    motion_priority: bool = triage_service.MOTION_PRIORITY,
) -> list[dict]:
    return triage_service.extract_sampled_frames(
        video_path,
        frame_samples=frame_samples,
        clip_window_seconds=clip_window_seconds,
        motion_priority=motion_priority,
    )


def _aggregate_video_predictions(frame_results: list[dict]) -> dict:
    return triage_service.aggregate_video_predictions(frame_results)


def _build_clip_contact_sheet(
    anchor_path: str | Path,
    clip_len: int = triage_service.VIDEO_CLIP_LEN,
    columns: int = 4,
    thumbnail_size: tuple[int, int] = (150, 95),
) -> Image.Image:
    anchor_path = Path(anchor_path)
    clip_paths = _clip_frame_paths(anchor_path, clip_len=clip_len)
    rows = int(np.ceil(len(clip_paths) / columns))
    cell_width = thumbnail_size[0] + 14
    cell_height = thumbnail_size[1] + 28
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#f8fafc")

    for index, frame_path in enumerate(clip_paths):
        with Image.open(frame_path) as frame:
            thumbnail = ImageOps.fit(frame.convert("RGB"), thumbnail_size, method=Image.Resampling.BILINEAR)

        border_color = "#2563eb" if Path(frame_path) == anchor_path else "#d7deea"
        thumbnail = ImageOps.expand(thumbnail, border=3, fill=border_color)
        x = (index % columns) * cell_width + 4
        y = (index // columns) * cell_height + 4
        sheet.paste(thumbnail, (x, y))

        identity = _frame_identity(frame_path)
        label = f"f{identity[1]}" if identity else frame_path.stem
        ImageDraw.Draw(sheet).text((x + 4, y + thumbnail.height + 4), label, fill="#111827")

    return sheet


def _build_clip_animation(
    anchor_path: str | Path,
    fps: float | None = None,
    clip_len: int = triage_service.VIDEO_CLIP_LEN,
    frame_size: tuple[int, int] = (480, 300),
) -> bytes:
    clip_paths = _clip_frame_paths(anchor_path, clip_len=clip_len)
    frames = []
    for frame_path in clip_paths:
        with Image.open(frame_path) as frame:
            frames.append(
                ImageOps.pad(
                    frame.convert("RGB"),
                    frame_size,
                    method=Image.Resampling.BILINEAR,
                    color="#000000",
                )
            )

    if not frames:
        return b""

    safe_fps = max(float(fps or 8.0), 1.0)
    duration_ms = int(round(1000.0 / min(safe_fps, 12.0)))
    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return output.getvalue()


def _build_clip_thumbnail(
    anchor_path: str | Path,
    frame_size: tuple[int, int] = (240, 150),
) -> Image.Image:
    clip_paths = _clip_frame_paths(anchor_path, clip_len=triage_service.VIDEO_CLIP_LEN)
    if len(clip_paths) >= 4:
        selected_paths = [
            clip_paths[0],
            clip_paths[len(clip_paths) // 3],
            clip_paths[(len(clip_paths) * 2) // 3],
            clip_paths[-1],
        ]
    else:
        selected_paths = clip_paths or [Path(anchor_path)]

    tile_width = max(1, frame_size[0] // len(selected_paths))
    thumbnail = Image.new("RGB", frame_size, "#000000")
    for index, frame_path in enumerate(selected_paths):
        with Image.open(frame_path) as frame:
            tile = ImageOps.fit(
                frame.convert("RGB"),
                (tile_width, frame_size[1]),
                method=Image.Resampling.BILINEAR,
            )
        thumbnail.paste(tile, (index * tile_width, 0))

    return thumbnail


def _video_format(video_path: Path) -> str:
    if video_path.suffix.lower() == ".mp4":
        return "video/mp4"
    return "video/x-msvideo"


def _numeric_value(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _rank_event_clips(event_table: pd.DataFrame, order: str) -> pd.DataFrame:
    if order == "Tiempo":
        return event_table.sort_values("timestamp_s").reset_index(drop=True)
    if order == "Movimiento" and "motion_score" in event_table:
        return event_table.sort_values(
            ["motion_score", "top_risk_probability"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return event_table.sort_values(
        ["top_risk_probability", "normal_probability"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _clip_effective_fps(selected_clip: pd.Series) -> float | None:
    clip_duration = float(selected_clip.get("clip_duration_seconds") or 0.0)
    if clip_duration <= 0:
        return None
    return triage_service.VIDEO_CLIP_LEN / clip_duration


def _render_original_video_window(selected_clip: pd.Series) -> None:
    video_path = selected_clip.get("source_video_path")
    if not video_path:
        return

    path = Path(str(video_path))
    if not path.exists():
        return

    clip_start_frame = _numeric_value(selected_clip.get("clip_start_frame"))
    clip_end_frame = _numeric_value(selected_clip.get("clip_end_frame"))
    video_fps = _numeric_value(selected_clip.get("video_fps"))
    if clip_start_frame is not None and clip_end_frame is not None:
        try:
            segment_path = triage_service.export_video_segment(
                path,
                start_frame=int(clip_start_frame),
                end_frame=int(clip_end_frame),
            )
            st.video(
                segment_path.read_bytes(),
                format=_video_format(segment_path),
                width="stretch",
            )
            return
        except Exception:
            pass

    timestamp_s = float(selected_clip.get("timestamp_s", 0.0))
    clip_duration = float(selected_clip.get("clip_duration_seconds") or 0.0)
    context_seconds = max(REVIEW_WINDOW_SECONDS, clip_duration / 2.0)
    if video_fps and clip_start_frame is not None and clip_end_frame is not None:
        start_time = max(0.0, clip_start_frame / video_fps)
        end_time = max(start_time + 0.1, (clip_end_frame + 1) / video_fps)
    else:
        start_time = max(0.0, timestamp_s - context_seconds)
        end_time = timestamp_s + context_seconds
    st.video(path.read_bytes(), start_time=start_time, end_time=end_time, width="stretch")


def _render_clip_gallery(ranked_clips: pd.DataFrame, gallery_count: int) -> None:
    if gallery_count < 1:
        return

    preview_clips = ranked_clips.head(gallery_count)
    for start in range(0, len(preview_clips), GALLERY_COLUMNS):
        cols = st.columns(GALLERY_COLUMNS)
        for column, (_, clip) in zip(cols, preview_clips.iloc[start : start + GALLERY_COLUMNS].iterrows()):
            with column:
                st.image(_build_clip_thumbnail(clip["image_path"]), width="stretch")
                st.caption(
                    f"{clip['timestamp_s']:.2f}s | "
                    f"{clip['top_risk_class']} {_format_percent(clip['top_risk_probability'])} | "
                    f"mov {_format_percent(clip.get('motion_score', 0.0))}"
                )


def _render_clip_inspector(event_table: pd.DataFrame) -> None:
    if event_table.empty or "image_path" not in event_table:
        return

    with st.expander("Ver clips evaluados"):
        order = st.segmented_control(
            "Orden",
            options=["Mayor riesgo", "Movimiento", "Tiempo"],
            default="Mayor riesgo",
            required=True,
        )
        ranked_clips = _rank_event_clips(event_table, str(order))
        gallery_count = st.slider(
            "Clips en galería",
            min_value=1,
            max_value=len(ranked_clips),
            value=len(ranked_clips),
        )
        _render_clip_gallery(ranked_clips, gallery_count)

        clip_index = st.selectbox(
            "Reproducir clip",
            options=list(ranked_clips.index),
            format_func=lambda index: (
                f"{ranked_clips.loc[index, 'timestamp_s']:.2f}s | "
                f"{ranked_clips.loc[index, 'active_labels']} | "
                f"{ranked_clips.loc[index, 'top_risk_class']} "
                f"{_format_percent(ranked_clips.loc[index, 'top_risk_probability'])} | "
                f"mov {_format_percent(ranked_clips.loc[index].get('motion_score', 0.0))}"
            ),
        )
        if clip_index is None:
            return

        selected_clip = ranked_clips.loc[int(clip_index)]
        st.caption("Segmento reproducible del video original")
        _render_original_video_window(selected_clip)

        st.caption("Clip técnico de 16 frames usado por el modelo")
        clip_duration = float(selected_clip.get("clip_duration_seconds") or 0.0)
        animation = _build_clip_animation(
            selected_clip["image_path"],
            fps=_clip_effective_fps(selected_clip),
        )
        if animation:
            st.image(animation, width="stretch")

        with st.expander("Ver frames del clip"):
            sheet = _build_clip_contact_sheet(selected_clip["image_path"])
            st.image(sheet, width="stretch")

        cols = st.columns(6)
        cols[0].metric("Predicción clip", selected_clip["prediction"])
        cols[1].metric("Normal ajustado", _format_percent(selected_clip["normal_probability"]))
        cols[2].metric(str(selected_clip["top_risk_class"]), _format_percent(selected_clip["top_risk_probability"]))
        cols[3].metric("Duración clip", f"{clip_duration:.2f}s")
        cols[4].metric("Movimiento", _format_percent(selected_clip.get("motion_score", 0.0)))
        cols[5].metric("Normal bruto", _format_percent(selected_clip.get("raw_normal_probability", selected_clip["normal_probability"])))


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f5f7fb;
            color: #0f172a;
        }
        [data-testid="stHeader"] {
            background: rgba(245, 247, 251, 0.92);
        }
        .hero {
            padding: 1.35rem 1.5rem;
            border-radius: 8px;
            background: #101827;
            color: #f8fafc;
            border: 1px solid rgba(15, 23, 42, 0.12);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            letter-spacing: 0;
        }
        .hero p {
            margin: 0.45rem 0 0 0;
            max-width: 860px;
            color: #cbd5e1;
            font-size: 0.98rem;
            line-height: 1.5;
        }
        .soft-card {
            padding: 1rem;
            border-radius: 8px;
            background: #ffffff;
            border: 1px solid #d8dee9;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            color: #0f172a;
        }
        .small-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 700;
            color: #475569;
        }
        .business-note {
            padding: 0.9rem 1rem;
            border-left: 4px solid #2563eb;
            background: #eff6ff;
            border-radius: 0 8px 8px 0;
            color: #172554;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d8dee9;
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
        }
        div[data-testid="stMetric"] label {
            color: #475569;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a;
        }
        .stDataFrame {
            border: 1px solid #d8dee9;
            border-radius: 8px;
        }
        .decision-note {
            padding: 0.9rem 1rem;
            border-radius: 8px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #7c2d12;
            margin: 0.6rem 0 0.8rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_kpis(summary: pd.DataFrame | None) -> None:
    cols = st.columns(4)
    cols[0].metric("Modelo activo", "Optuna + MLflow", help="Entrena y selecciona el mejor candidato automáticamente.")
    cols[1].metric("Supervisión", "Incidentes", help="Pensado para triage operativo y revisión rápida.")
    if summary is not None and not summary.empty:
        best_row = summary.sort_values("validation_f1_macro", ascending=False).iloc[0]
        cols[2].metric("Mejor validación", f"{best_row['validation_f1_macro']:.3f}")
        cols[3].metric("Candidato líder", best_row["model_name"])
    else:
        cols[2].metric("Mejor validación", "N/D")
        cols[3].metric("Candidato líder", "N/D")


def _render_experiment_panel(summary: pd.DataFrame | None, report_text: str | None) -> None:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-label">Triage de incidente</div>', unsafe_allow_html=True)
        st.subheader("Sube un video o una imagen y recibe una decisión operativa")
        uploaded = st.file_uploader("Archivo del incidente", type=["png", "jpg", "jpeg", "webp", "bmp", "mp4", "mov", "avi", "mkv", "webm"])
        if uploaded is not None:
            temp_path = _save_upload(uploaded)
            if _is_video_file(uploaded):
                coverage_label = st.segmented_control(
                    "Cobertura temporal",
                    options=list(TEMPORAL_COVERAGE_OPTIONS),
                    default="Balanceada",
                    required=True,
                )
                frame_samples = TEMPORAL_COVERAGE_OPTIONS.get(str(coverage_label or "Balanceada"), FRAME_SAMPLES)
                window_label = st.segmented_control(
                    "Duración del clip",
                    options=list(CLIP_WINDOW_OPTIONS),
                    default="Media",
                    required=True,
                )
                clip_window_seconds = CLIP_WINDOW_OPTIONS.get(
                    str(window_label or "Media"),
                    triage_service.CLIP_WINDOW_SECONDS,
                )
                motion_priority = st.checkbox("Priorizar movimiento", value=triage_service.MOTION_PRIORITY)
                st.video(temp_path.read_bytes())
                try:
                    with st.spinner("Detectando movimiento y extrayendo clips..."):
                        frames = triage_service.extract_sampled_frames(
                            temp_path,
                            frame_samples=frame_samples,
                            clip_window_seconds=clip_window_seconds,
                            motion_priority=motion_priority,
                        )
                        frames = triage_service.normalize_motion_scores(frames)
                    frame_results: list[dict] = []

                    prediction_progress = st.progress(0)
                    for position, frame in enumerate(frames, start=1):
                        prediction = predict_image_details(frame["image_path"])
                        raw_label = prediction["prediction"]
                        raw_confidence = float(prediction["confidence"])
                        raw_class_scores = prediction.get("class_scores", {})
                        motion_intensity = float(frame.get("motion_intensity") or 0.0)
                        class_scores = triage_service.adjust_class_scores_for_motion(
                            raw_class_scores,
                            motion_intensity,
                        )
                        label, confidence = triage_service.motion_adjusted_prediction_label(
                            raw_label,
                            class_scores,
                            motion_intensity,
                        )
                        if confidence == 0.0:
                            confidence = raw_confidence
                        risk_class, risk_probability = _risk_signal_from_scores(class_scores)
                        tier, score, recommendation = _risk_profile(label, confidence)
                        frame_results.append(
                            {
                                "timestamp_s": round(frame["timestamp_seconds"], 2),
                                "frame_index": frame["frame_index"],
                                "clip_start_frame": frame.get("clip_start_frame"),
                                "clip_end_frame": frame.get("clip_end_frame"),
                                "clip_duration_seconds": frame.get("clip_duration_seconds"),
                                "clip_window_seconds": frame.get("clip_window_seconds"),
                                "video_fps": frame.get("video_fps"),
                                "motion_score": frame.get("motion_score", 0.0),
                                "motion_intensity": motion_intensity,
                                "sampling_reason": frame.get("sampling_reason", "cobertura"),
                                "image_path": str(frame["image_path"]),
                                "source_video_path": str(temp_path),
                                "prediction": label,
                                "raw_prediction": raw_label,
                                "active_labels": label,
                                "confidence": confidence,
                                "normal_probability": float(class_scores.get(triage_service.NORMAL_LABEL, 0.0)),
                                "raw_normal_probability": float(raw_class_scores.get(triage_service.NORMAL_LABEL, 0.0)),
                                "top_risk_class": risk_class,
                                "top_risk_probability": risk_probability,
                                "class_scores": class_scores,
                                "raw_class_scores": raw_class_scores,
                                "tier": tier,
                                "score": score,
                                "recommendation": recommendation,
                            }
                        )
                        prediction_progress.progress(position / max(len(frames), 1))
                    prediction_progress.empty()

                    aggregate = _aggregate_video_predictions(frame_results)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Decisión operativa", aggregate["prediction"])
                    m2.metric("Evidencia del evento", _format_percent(aggregate["confidence"]))
                    m3.metric("Clips evaluados", len(frame_results))
                    m4.metric("Pico mayor", _format_percent(aggregate["probability_leader_confidence"]))

                    st.markdown(_incident_summary(aggregate["prediction"], aggregate["confidence"]))
                    st.markdown(
                        f"""
                        <div class="decision-note">
                        <strong>Por qué la decisión puede diferir del máximo bruto:</strong><br>
                        {aggregate["decision_note"]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.success(f"Recomendación operativa: {aggregate['recommendation']}")

                    if not aggregate["class_summary"].empty:
                        st.caption("Evidencia agregada por clase con ajuste operativo")
                        display_summary = aggregate["class_summary"][
                            [
                                "selected",
                                "class_name",
                                "mean_probability",
                                "max_probability",
                                "frame_hits",
                                "decision_score",
                            ]
                        ].rename(
                            columns={
                                "selected": "decisión",
                                "class_name": "clase",
                                "mean_probability": "prob. media",
                                "max_probability": "prob. máxima",
                                "frame_hits": "clips con señal",
                                "decision_score": "score operativo",
                            }
                        )
                        display_summary["decisión"] = display_summary["decisión"].map(
                            {True: "Seleccionada", False: ""}
                        )
                        display_summary = display_summary.sort_values(
                            ["decisión", "score operativo", "prob. máxima"],
                            ascending=[False, False, False],
                        )
                        st.dataframe(
                            display_summary,
                            width="stretch",
                            hide_index=True,
                        )

                    event_table = pd.DataFrame(frame_results)
                    st.caption("Timeline multi-etiqueta por clip")
                    st.dataframe(
                        event_table[
                            [
                                "timestamp_s",
                                "clip_start_frame",
                                "clip_end_frame",
                                "clip_duration_seconds",
                                "motion_score",
                                "motion_intensity",
                                "sampling_reason",
                                "raw_prediction",
                                "active_labels",
                                "raw_normal_probability",
                                "normal_probability",
                                "top_risk_class",
                                "top_risk_probability",
                                "tier",
                                "score",
                            ]
                        ]
                        .rename(
                            columns={
                                "timestamp_s": "tiempo (s)",
                                "clip_start_frame": "inicio clip",
                                "clip_end_frame": "fin clip",
                                "clip_duration_seconds": "duración clip (s)",
                                "motion_score": "movimiento",
                                "motion_intensity": "mov. relativa",
                                "sampling_reason": "muestreo",
                                "raw_prediction": "pred. bruta",
                                "active_labels": "etiquetas activas",
                                "raw_normal_probability": "Normal bruto",
                                "normal_probability": "Normal ajustado",
                                "top_risk_class": "mayor riesgo",
                                "top_risk_probability": "score mayor riesgo",
                                "tier": "nivel",
                                "score": "severidad",
                            }
                        )
                        .sort_values("tiempo (s)"),
                        width="stretch",
                        hide_index=True,
                    )

                    key_moments = event_table.sort_values("top_risk_probability", ascending=False).head(3)
                    st.caption("Momentos clave por señal de riesgo")
                    st.dataframe(
                        key_moments[
                            [
                                "timestamp_s",
                                "top_risk_class",
                                "top_risk_probability",
                                "normal_probability",
                                "active_labels",
                                "recommendation",
                            ]
                        ].rename(
                            columns={
                                "timestamp_s": "tiempo (s)",
                                "top_risk_class": "mayor riesgo",
                                "top_risk_probability": "score riesgo",
                                "normal_probability": "score NormalVideos",
                                "active_labels": "etiquetas activas",
                                "recommendation": "recomendación",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    _render_clip_inspector(event_table)

                    if not aggregate["class_summary"].empty:
                        st.caption("Histograma de score operativo")
                        _render_score_histogram(
                            aggregate["class_summary"]["decision_score"],
                            aggregate["class_summary"]["class_name"],
                            "Distribución de evidencia operativa por clase",
                        )
                except Exception as error:
                    st.error(f"No se pudo evaluar el video: {error}")
            else:
                image = Image.open(temp_path)
                st.image(image, width="stretch")
                try:
                    prediction = predict_image_details(temp_path)
                    label = prediction["prediction"]
                    confidence = float(prediction["confidence"])
                    tier, score, recommendation = _risk_profile(label, confidence)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Clase detectada", label)
                    m2.metric("Confianza", f"{confidence:.1%}")
                    m3.metric("Nivel", tier)

                    st.markdown(_incident_summary(label, confidence))
                    st.info(f"**Recomendación operativa:** {recommendation}")

                    scores = prediction.get("class_scores", {})
                    if scores:
                        score_frame = pd.DataFrame(
                            [{"Clase": class_name, "Score": probability} for class_name, probability in scores.items()]
                        ).sort_values("Score", ascending=False)
                        st.caption("Histograma de scores del modelo")
                        _render_score_histogram(
                            score_frame["Score"],
                            score_frame["Clase"],
                            "Distribución de confianza interna por clase",
                        )
                        st.dataframe(score_frame, width="stretch", hide_index=True)
                except Exception as error:
                    st.error(f"No se pudo evaluar la imagen: {error}")
        else:
            st.write("Carga una imagen o un video del incidente para obtener el triage.")

        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-label">Operación</div>', unsafe_allow_html=True)
        st.subheader("Señales del negocio")
        st.markdown(
            """
            <div class="business-note">
            <strong>Qué vende esta interfaz:</strong><br>
            ahorro de tiempo para equipos de seguridad, menos revisión manual de video, y reportes listos para auditoría.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("Casos de uso: tiendas, colegios, estacionamientos, bodegas y conjuntos residenciales.")
        st.write("Modelo comercial sugerido: suscripción por cámara o por sitio con alertas y reportes premium.")

        if summary is not None and not summary.empty:
            st.caption("Resumen del último entrenamiento")
            latest = summary.sort_values("validation_f1_macro", ascending=False)
            st.dataframe(
                latest[["model_name", "validation_f1_macro", "validation_accuracy", "best_params"]],
                width="stretch",
                hide_index=True,
            )

        if report_text:
            with st.expander("Ver reporte de clasificación"):
                st.text(report_text)
        st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide")
    _apply_styles()

    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_NAME}</h1>
            <p>
                Una interfaz de triage de incidentes para cámaras de seguridad. Sube una imagen,
                clasifica el evento, prioriza la revisión y genera un resumen claro para el equipo operativo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = _load_summary()
    report_text = _load_report_text()
    _render_kpis(summary)
    st.divider()
    _render_experiment_panel(summary, report_text)


if __name__ == "__main__":
    main()
