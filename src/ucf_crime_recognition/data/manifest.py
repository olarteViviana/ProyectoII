from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ucf_crime_recognition.config import IMAGE_EXTENSIONS, UCF_CRIME_LABELS, load_config, project_path


def _infer_label(image_path: Path) -> str:
    for part in image_path.parts:
        normalized = part.replace(" ", "").replace("_", "").lower()
        if normalized in UCF_CRIME_LABELS:
            return part

    return image_path.parent.name


def _infer_existing_split(image_path: Path) -> str | None:
    for part in image_path.parts:
        normalized = part.lower()
        if normalized in {"train", "training"}:
            return "train"
        if normalized in {"test", "testing", "val", "valid", "validation"}:
            return "test"

    return None


def _iter_image_files(raw_dir: Path):
    for root, _, files in os.walk(raw_dir, followlinks=True):
        root_path = Path(root)
        for filename in files:
            image_path = root_path / filename
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield image_path


def build_manifest(config_path: str | Path | None = None) -> pd.DataFrame:
    config = load_config(config_path) if config_path else load_config()
    raw_dir = project_path(config["dataset"]["raw_dir"])
    manifest_path = project_path(config["dataset"]["manifest_path"])
    test_size = config["preprocessing"]["test_size"]
    random_state = config["preprocessing"]["random_state"]

    records = []

    for image_path in sorted(_iter_image_files(raw_dir)):
        records.append(
            {
                "path": str(image_path),
                "label": _infer_label(image_path),
                "source_split": _infer_existing_split(image_path),
            }
        )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError(f"No images found in {raw_dir}")

    if manifest["source_split"].notna().all():
        manifest["split"] = manifest["source_split"]
    else:
        stratify = manifest["label"] if manifest["label"].nunique() > 1 else None
        train_df, test_df = train_test_split(
            manifest,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        train_df = train_df.assign(split="train")
        test_df = test_df.assign(split="test")
        manifest = pd.concat([train_df, test_df], ignore_index=True)

    manifest = manifest.drop(columns=["source_split"])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"Manifest saved to: {manifest_path}")
    print(manifest["split"].value_counts().to_string())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an image manifest for UCF Crime.")
    parser.add_argument("--config", default=None, help="Path to a TOML config file.")
    args = parser.parse_args()

    build_manifest(args.config)


if __name__ == "__main__":
    main()
