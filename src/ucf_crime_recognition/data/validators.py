from __future__ import annotations

import pandas as pd


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"path", "label", "split"}
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Manifest is missing required columns: {missing}")

    if manifest.empty:
        raise ValueError("Manifest is empty")

    if manifest["label"].nunique() < 2:
        raise ValueError("Manifest must contain at least two classes")

    return manifest
