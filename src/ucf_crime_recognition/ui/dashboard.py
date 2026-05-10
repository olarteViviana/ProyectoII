from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from ucf_crime_recognition.config import load_config, project_path
from ucf_crime_recognition.predict import predict_image_details


APP_NAME = "Sentinel Review"
HIGH_RISK_LABELS = {"Abuse", "Arson", "Assault", "Explosion", "Fighting", "Robbery", "Shooting"}
MEDIUM_RISK_LABELS = {"Burglary", "RoadAccidents", "Shoplifting", "Stealing", "Vandalism"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
FRAME_SAMPLES = 16

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


def _risk_profile(label: str, confidence: float) -> tuple[str, str, str]:
    if label in HIGH_RISK_LABELS:
        tier = "Crítico"
        score = "Alto"
    elif label in MEDIUM_RISK_LABELS:
        tier = "Vigilancia"
        score = "Medio"
    else:
        tier = "Normal"
        score = "Bajo"

    if label == "NormalVideos" and confidence < 0.5:
        tier = "Revisión"
        score = "Medio"

    if confidence < 0.45 and tier != "Normal":
        score = "Medio"
        tier = "Revisión"

    recommendation = LABEL_ACTIONS.get(label, "Sin recomendación específica disponible.")
    return tier, score, recommendation


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
    risk_scores = {
        str(class_name): float(probability)
        for class_name, probability in class_scores.items()
        if str(class_name) != "NormalVideos"
    }
    if not risk_scores:
        return "N/D", 0.0

    class_name = max(risk_scores, key=risk_scores.get)
    return class_name, risk_scores[class_name]


def _format_active_labels(prediction: dict) -> str:
    labels = prediction.get("predictions")
    if labels:
        return " | ".join(str(label) for label in labels)
    return str(prediction.get("prediction", ""))


def _save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    temp_dir = Path(tempfile.mkdtemp(prefix="ucf-crime-ui-"))
    temp_path = temp_dir / f"uploaded{suffix}"
    temp_path.write_bytes(uploaded_file.getbuffer())
    return temp_path


def _is_video_file(uploaded_file) -> bool:
    return Path(uploaded_file.name).suffix.lower().lstrip(".") in VIDEO_EXTENSIONS


def _extract_sampled_frames(video_path: Path, frame_samples: int = FRAME_SAMPLES) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("No se pudo abrir el video.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    if total_frames <= 0:
        capture.release()
        raise ValueError("El video no contiene fotogramas legibles.")

    sample_count = min(frame_samples, total_frames)
    frame_indexes = sorted({int(round(index)) for index in np.linspace(0, total_frames - 1, sample_count)})

    extracted_frames: list[dict] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="ucf-crime-frames-"))

    for position, frame_index in enumerate(frame_indexes):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            continue

        timestamp_seconds = frame_index / fps
        image_path = temp_dir / f"frame_{position:03d}_{frame_index:06d}.png"
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(image_rgb).save(image_path)
        extracted_frames.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "image_path": image_path,
            }
        )

    capture.release()
    return extracted_frames


def _aggregate_video_predictions(frame_results: list[dict]) -> dict:
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
            lambda class_name: 2.0 if class_name in HIGH_RISK_LABELS else 1.3 if class_name in MEDIUM_RISK_LABELS else 1.0
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
        normal_row = class_summary[class_summary["class_name"] == "NormalVideos"]
        risk_rows = class_summary[class_summary["class_name"] != "NormalVideos"]
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
                    f"en {risk_hits} frames con señal de riesgo y su score operativo es cercano al de NormalVideos."
                )
            elif label == "NormalVideos" and top_alternative is not None:
                alternative_label = str(top_alternative["class_name"])
                alternative_score = float(top_alternative["decision_score"])
                if alternative_label != "NormalVideos" and alternative_score >= normal_score * 0.9:
                    label = alternative_label
                    confidence = float(top_alternative["max_probability"])
                    decision_reason = "alternativa_cercana"
                    decision_note = (
                        f"{alternative_label} quedó muy cerca de NormalVideos en score operativo; "
                        "se muestra como evento para revisión humana."
                    )

    tier, score, recommendation = _risk_profile(label, confidence)
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
                st.video(temp_path.read_bytes())
                try:
                    frames = _extract_sampled_frames(temp_path)
                    frame_results: list[dict] = []

                    for frame in frames:
                        prediction = predict_image_details(frame["image_path"])
                        label = prediction["prediction"]
                        confidence = float(prediction["confidence"])
                        class_scores = prediction.get("class_scores", {})
                        risk_class, risk_probability = _risk_signal_from_scores(class_scores)
                        tier, score, recommendation = _risk_profile(label, confidence)
                        frame_results.append(
                            {
                                "timestamp_s": round(frame["timestamp_seconds"], 2),
                                "frame_index": frame["frame_index"],
                                "prediction": label,
                                "active_labels": _format_active_labels(prediction),
                                "confidence": confidence,
                                "normal_probability": float(class_scores.get("NormalVideos", 0.0)),
                                "top_risk_class": risk_class,
                                "top_risk_probability": risk_probability,
                                "class_scores": class_scores,
                                "tier": tier,
                                "score": score,
                                "recommendation": recommendation,
                            }
                        )

                    aggregate = _aggregate_video_predictions(frame_results)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Decisión operativa", aggregate["prediction"])
                    m2.metric("Evidencia del evento", _format_percent(aggregate["confidence"]))
                    m3.metric("Mayor probabilidad", aggregate["probability_leader"])
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
                        st.caption("Evidencia agregada por clase")
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
                                "frame_hits": "frames con señal",
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
                    st.caption("Timeline multi-etiqueta por frame")
                    st.dataframe(
                        event_table[
                            [
                                "timestamp_s",
                                "active_labels",
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
                                "active_labels": "etiquetas activas",
                                "normal_probability": "score NormalVideos",
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

                    if not aggregate["class_summary"].empty:
                        st.caption("Score operativo por clase")
                        st.bar_chart(aggregate["class_summary"].set_index("class_name")["decision_score"])
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
                        st.caption("Distribución interna del modelo")
                        st.bar_chart(score_frame.set_index("Clase"))
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
