from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ucf_crime_recognition.predict import predict_image_details
from ucf_crime_recognition.tools.analyze_video import analyze_video

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _iter_videos(dataset_root: Path):
    for path in sorted(dataset_root.rglob("*")):
        if path.suffix.lower() in VIDEO_EXTS:
            yield path


def _iter_image_sequences(dataset_root: Path):
    groups = defaultdict(list)
    pattern = re.compile(r"(.+?)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)

    for path in sorted(dataset_root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        match = pattern.match(path.name)
        prefix = match.group(1) if match else path.stem
        groups[str(path.parent / prefix)].append(path)

    for key, files in groups.items():
        yield Path(key), sorted(files)


def _infer_label(video_path: Path, dataset_root: Path) -> str:
    relative = video_path.relative_to(dataset_root)
    return relative.parts[0]


def _collect_frame_predictions_from_images(image_paths: list[Path], frame_samples: int) -> pd.DataFrame:
    if not image_paths:
        return pd.DataFrame()

    if len(image_paths) <= frame_samples:
        indices = list(range(len(image_paths)))
    else:
        step = len(image_paths) / float(frame_samples)
        indices = [int(i * step) for i in range(frame_samples)]

    rows = []
    for sampled_index, image_index in enumerate(indices):
        image_path = image_paths[image_index]
        details = predict_image_details(image_path)
        rows.append(
            {
                "timestamp_s": float(sampled_index),
                "frame_index": int(image_index),
                "prediction": details["prediction"],
                "confidence": details["confidence"],
                "class_scores": details.get("class_scores", {}),
            }
        )

    return pd.DataFrame(rows)


def _apply_policies_on_df(df: pd.DataFrame, peak: float, seq: float, seq_len: int, policy: str, risk_max: float = 0.2, risk_hits: int = 2) -> dict:
    if df.empty:
        return {"decision": "NormalVideos", "reason": "empty"}

    peaks = []
    for _, row in df.iterrows():
        for cls, prob in row["class_scores"].items():
            if cls != "NormalVideos" and prob >= peak:
                peaks.append(
                    {
                        "timestamp_s": row["timestamp_s"],
                        "frame_index": row["frame_index"],
                        "class": cls,
                        "prob": prob,
                    }
                )

    if policy == "peak" and peaks:
        peaks_df = pd.DataFrame(peaks)
        top = peaks_df.sort_values("prob", ascending=False).iloc[0]
        return {"decision": str(top["class"]), "reason": "peak", "timestamp_s": float(top["timestamp_s"]), "detail": peaks_df}

    seq_hits = []
    classes = set(cls for row in df["class_scores"] for cls in row)
    for cls in classes:
        if cls == "NormalVideos":
            continue

        count = 0
        for _, row in df.iterrows():
            prob = float(row["class_scores"].get(cls, 0.0))
            if prob >= seq:
                count += 1
                if count >= seq_len:
                    seq_hits.append({"class": cls, "end_frame": row["frame_index"], "count": count})
                    break
            else:
                count = 0

    if policy == "consecutive" and seq_hits:
        seq_df = pd.DataFrame(seq_hits)
        top = seq_df.sort_values("count", ascending=False).iloc[0]
        return {"decision": str(top["class"]), "reason": "consecutive", "detail": seq_df}

    agg = []
    for cls in classes:
        probs = [float(r.get(cls, 0.0)) for r in df["class_scores"]]
        agg.append({"class": cls, "mean_prob": sum(probs) / len(probs) if probs else 0.0})

    agg_df = pd.DataFrame(agg).sort_values("mean_prob", ascending=False)
    top_risk = agg_df[agg_df["class"] != "NormalVideos"].head(1)
    if policy == "aggregate" and not top_risk.empty and float(top_risk["mean_prob"].iloc[0]) > 0.15:
        return {"decision": str(top_risk["class"].iloc[0]), "reason": "aggregate", "detail": agg_df}

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


def validate(dataset_root: Path, frame_samples: int, peak: float, seq: float, seq_len: int, risk_max: float = 0.2, risk_hits: int = 2) -> pd.DataFrame:
    rows = []
    any_videos = False
    for video_path in _iter_videos(dataset_root):
        any_videos = True
        true_label = _infer_label(video_path, dataset_root)
        for policy in ["peak", "consecutive", "aggregate", "risk_override"]:
            result = analyze_video(
                video_path,
                frame_samples=frame_samples,
                peak_threshold=peak,
                seq_threshold=seq,
                seq_len=seq_len,
                policy=policy,
                risk_max=risk_max,
                risk_hits=risk_hits,
            )
            rows.append(
                {
                    "video": str(video_path),
                    "true_label": true_label,
                    "policy": policy,
                    "prediction": result["decision"],
                    "reason": result["reason"],
                }
            )

    if not any_videos:
        for sequence_id_path, image_paths in _iter_image_sequences(dataset_root):
            true_label = image_paths[0].parent.name
            df = _collect_frame_predictions_from_images(image_paths, frame_samples=frame_samples)
            for policy in ["peak", "consecutive", "aggregate", "risk_override"]:
                result = _apply_policies_on_df(df, peak=peak, seq=seq, seq_len=seq_len, policy=policy, risk_max=risk_max, risk_hits=risk_hits)
                rows.append(
                    {
                        "video": str(sequence_id_path),
                        "true_label": true_label,
                        "policy": policy,
                        "prediction": result["decision"],
                        "reason": result["reason"],
                    }
                )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "temporal_policy_predictions.csv", index=False)

    summary_rows = []
    for policy, group in results.groupby("policy"):
        y_true = group["true_label"]
        y_pred = group["prediction"]
        summary_rows.append(
            {
                "policy": policy,
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "n_samples": len(group),
            }
        )

        report = classification_report(y_true, y_pred, zero_division=0)
        (output_dir / f"classification_report_{policy}.txt").write_text(report)

        labels = sorted(set(y_true) | set(y_pred))
        matrix = pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=labels),
            index=labels,
            columns=labels,
        )
        matrix.to_csv(output_dir / f"confusion_matrix_{policy}.csv")

    summary = pd.DataFrame(summary_rows).sort_values("f1_macro", ascending=False)
    summary.to_csv(output_dir / "temporal_policy_summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate temporal video policies on labeled videos grouped by folder.")
    parser.add_argument("dataset_root", help="Root folder where each subfolder is a class label and contains videos.")
    parser.add_argument("--output", default="reports/temporal_policy_validation", help="Folder for validation outputs.")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--peak", type=float, default=0.40)
    parser.add_argument("--seq", type=float, default=0.30)
    parser.add_argument("--seq_len", type=int, default=2)
    parser.add_argument("--risk_max", type=float, default=0.2, help="Max-prob threshold for risk_override (used to count hits)")
    parser.add_argument("--risk_hits", type=int, default=2, help="Minimum hits required for risk_override to trigger")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    results = validate(
        dataset_root,
        frame_samples=args.samples,
        peak=args.peak,
        seq=args.seq,
        seq_len=args.seq_len,
        risk_max=args.risk_max,
        risk_hits=args.risk_hits,
    )
    summarize(results, Path(args.output))


if __name__ == "__main__":
    main()