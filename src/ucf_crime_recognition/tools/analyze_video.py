from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from ucf_crime_recognition.ui.dashboard import _extract_sampled_frames
from ucf_crime_recognition.predict import predict_image_details


def _collect_frame_predictions(video_path: Path, frame_samples: int) -> pd.DataFrame:
    frames = _extract_sampled_frames(video_path, frame_samples)
    results = []
    for f in frames:
        details = predict_image_details(f["image_path"])
        results.append(
            {
                "timestamp_s": f["timestamp_seconds"],
                "frame_index": f["frame_index"],
                "prediction": details["prediction"],
                "confidence": details["confidence"],
                "class_scores": details.get("class_scores", {}),
            }
        )

    return pd.DataFrame(results)


def analyze_video(
    video_path: Path,
    frame_samples: int = 32,
    peak_threshold: float = 0.40,
    seq_threshold: float = 0.30,
    seq_len: int = 2,
    policy: str = "risk_override",
    risk_max: float = 0.2,
    risk_hits: int = 2,
) -> dict:
    df = _collect_frame_predictions(video_path, frame_samples)
    if df.empty:
        raise ValueError("No fue posible extraer predicciones del video.")

    # Peak rule: any frame where a non-NormalVideos class has prob >= peak_threshold
    peaks = []
    for _, row in df.iterrows():
        for cls, prob in row["class_scores"].items():
            if cls != "NormalVideos" and prob >= peak_threshold:
                peaks.append({"timestamp_s": row["timestamp_s"], "frame_index": row["frame_index"], "class": cls, "prob": prob})

    if policy == "peak" and peaks:
        # choose the highest-prob peak
        peaks_df = pd.DataFrame(peaks)
        top = peaks_df.sort_values("prob", ascending=False).iloc[0]
        return {"decision": str(top["class"]), "reason": "peak", "timestamp_s": float(top["timestamp_s"]), "detail": peaks_df}

    # Consecutive rule: same class appears with prob >= seq_threshold for seq_len consecutive frames
    seq_hits = []
    classes = set(cls for row in df["class_scores"] for cls in row)
    for cls in classes:
        if cls == "NormalVideos":
            continue
        counts = 0
        for _, row in df.iterrows():
            prob = float(row["class_scores"].get(cls, 0.0))
            if prob >= seq_threshold:
                counts += 1
                if counts >= seq_len:
                    seq_hits.append({"class": cls, "end_frame": row["frame_index"], "count": counts})
                    break
            else:
                counts = 0

    if policy == "consecutive" and seq_hits:
        seq_df = pd.DataFrame(seq_hits)
        top = seq_df.sort_values("count", ascending=False).iloc[0]
        return {"decision": str(top["class"]), "reason": "consecutive", "detail": seq_df}

    # Fallback: aggregate average non-normal probability
    agg = []
    for cls in classes:
        probs = [float(r.get(cls, 0.0)) for r in df["class_scores"]]
        agg.append({"class": cls, "mean_prob": sum(probs) / len(probs) if probs else 0.0})
    agg_df = pd.DataFrame(agg).sort_values("mean_prob", ascending=False)
    top_risk = agg_df[agg_df["class"] != "NormalVideos"].head(1)
    if policy == "aggregate" and not top_risk.empty and float(top_risk["mean_prob"].iloc[0]) > 0.15:
        return {"decision": str(top_risk["class"].iloc[0]), "reason": "aggregate", "detail": agg_df}

    # Risk override: repeated robbery-like cues should not be flattened into NormalVideos.
    if policy == "risk_override" and not df.empty:
        score_rows = []
        for row in df.itertuples(index=False):
            for cls, prob in row.class_scores.items():
                if cls == "NormalVideos":
                    continue
                score_rows.append({"class": cls, "prob": float(prob)})

        if score_rows:
            score_df = pd.DataFrame(score_rows)
            risk_df = (
                score_df.groupby("class")
                .agg(mean_prob=("prob", "mean"), max_prob=("prob", "max"), hits=("prob", lambda values: int((values >= float(risk_max)).sum())))
                .reset_index()
                .sort_values(["mean_prob", "max_prob", "hits"], ascending=False)
            )
            top_risk = risk_df.iloc[0]
            if float(top_risk["max_prob"]) >= float(risk_max) and int(top_risk["hits"]) >= int(risk_hits):
                return {"decision": str(top_risk["class"]), "reason": "risk_override", "detail": risk_df}

    if policy not in {"peak", "consecutive", "aggregate", "risk_override"}:
        raise ValueError(f"Unknown policy: {policy}")

    return {"decision": "NormalVideos", "reason": "none", "detail": df}


def main():
    parser = argparse.ArgumentParser(description="Analyze a video with frame-level predictions and temporal rules")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--samples", type=int, default=32, help="Number of frames to sample")
    parser.add_argument("--peak", type=float, default=0.40, help="Peak probability threshold")
    parser.add_argument("--seq", type=float, default=0.30, help="Consecutive probability threshold")
    parser.add_argument("--seq_len", type=int, default=2, help="Consecutive frames required")
    parser.add_argument(
        "--policy",
        choices=["risk_override", "peak", "consecutive", "aggregate"],
        default="risk_override",
        help="Temporal aggregation policy to use.",
    )
    parser.add_argument("--risk_max", type=float, default=0.2, help="Max-prob threshold for risk_override (used to count hits)")
    parser.add_argument("--risk_hits", type=int, default=2, help="Minimum hits required for risk_override to trigger")

    args = parser.parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    result = analyze_video(
        video_path,
        frame_samples=args.samples,
        peak_threshold=args.peak,
        seq_threshold=args.seq,
        seq_len=args.seq_len,
        policy=args.policy,
        risk_max=args.risk_max,
        risk_hits=args.risk_hits,
    )
    print("Decision:", result["decision"], "(reason:", result["reason"], ")")
    if "timestamp_s" in result:
        print("Timestamp (s):", result["timestamp_s"])

    # Print detail briefly
    detail = result.get("detail")
    if isinstance(detail, pd.DataFrame):
        print(detail.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
