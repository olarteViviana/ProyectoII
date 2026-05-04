from __future__ import annotations

from pathlib import Path

import pandas as pd

from ucf_crime_recognition.config import load_config, project_path


def load_manifest(config: dict | None = None, config_path: str | Path | None = None) -> pd.DataFrame:
    config = config or (load_config(config_path) if config_path else load_config())
    manifest_path = project_path(config["dataset"]["manifest_path"])
    return pd.read_csv(manifest_path)


def split_manifest(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_manifest = manifest[manifest["split"] == "train"]
    test_manifest = manifest[manifest["split"] == "test"]
    return train_manifest, test_manifest


def sample_manifest(manifest: pd.DataFrame, max_samples: int | None, random_state: int) -> pd.DataFrame:
    if not max_samples or len(manifest) <= max_samples:
        return manifest

    sampled_groups = []
    for _, group in manifest.groupby("label"):
        group_size = max(1, round(max_samples * len(group) / len(manifest)))
        sampled_groups.append(
            group.sample(
                n=min(group_size, len(group)),
                random_state=random_state,
            )
        )

    return pd.concat(sampled_groups).sample(frac=1, random_state=random_state).head(max_samples)


def rebalance_manifest(
    manifest: pd.DataFrame,
    target_min_per_class: int | None,
    target_max_per_class: int | None,
    random_state: int,
) -> pd.DataFrame:
    if target_min_per_class is None and target_max_per_class is None:
        return manifest

    balanced_groups = []
    for _, group in manifest.groupby("label"):
        desired_size = len(group)
        if target_min_per_class is not None:
            desired_size = max(desired_size, target_min_per_class)
        if target_max_per_class is not None:
            desired_size = min(desired_size, target_max_per_class)

        if desired_size <= len(group):
            sampled_group = group.sample(n=desired_size, random_state=random_state)
        else:
            sampled_group = group.sample(n=desired_size, replace=True, random_state=random_state)

        balanced_groups.append(sampled_group)

    return pd.concat(balanced_groups).sample(frac=1, random_state=random_state).reset_index(drop=True)
