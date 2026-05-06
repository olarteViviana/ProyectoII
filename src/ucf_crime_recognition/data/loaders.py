from __future__ import annotations

from collections.abc import Callable
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
    label_parser: Callable[[object], tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    if target_min_per_class is None and target_max_per_class is None:
        return manifest

    if label_parser is not None:
        return _rebalance_multilabel_manifest(
            manifest,
            target_min_per_class=target_min_per_class,
            target_max_per_class=target_max_per_class,
            random_state=random_state,
            label_parser=label_parser,
        )

    class_counts = manifest["label"].value_counts()
    median_class_size = int(class_counts.median())

    if target_min_per_class is not None and target_max_per_class is not None:
        target_size = max(target_min_per_class, min(median_class_size, target_max_per_class))
    elif target_min_per_class is not None:
        target_size = target_min_per_class
    else:
        target_size = target_max_per_class

    balanced_groups = []
    for _, group in manifest.groupby("label"):
        if target_size <= len(group):
            sampled_group = group.sample(n=target_size, random_state=random_state)
        else:
            sampled_group = group.sample(n=target_size, replace=True, random_state=random_state)

        balanced_groups.append(sampled_group)

    return pd.concat(balanced_groups).sample(frac=1, random_state=random_state).reset_index(drop=True)


def _rebalance_multilabel_manifest(
    manifest: pd.DataFrame,
    target_min_per_class: int | None,
    target_max_per_class: int | None,
    random_state: int,
    label_parser: Callable[[object], tuple[str, ...]],
) -> pd.DataFrame:
    label_sets = manifest["label"].map(label_parser)
    class_counts = label_sets.explode().value_counts()
    median_class_size = int(class_counts.median())

    if target_min_per_class is not None and target_max_per_class is not None:
        target_size = max(target_min_per_class, min(median_class_size, target_max_per_class))
    elif target_min_per_class is not None:
        target_size = target_min_per_class
    else:
        target_size = target_max_per_class

    balanced_manifest = manifest.copy()
    singleton_labels = label_sets.map(lambda labels: len(labels) == 1)

    if target_max_per_class is not None:
        overrepresented_labels = class_counts[class_counts > target_size].index
        drop_indices = []
        for label in overrepresented_labels:
            label_only_rows = manifest[singleton_labels & label_sets.map(lambda labels: label in labels)]
            if len(label_only_rows) <= target_size:
                continue
            keep_indices = set(label_only_rows.sample(n=target_size, random_state=random_state).index)
            drop_indices.extend(index for index in label_only_rows.index if index not in keep_indices)
        if drop_indices:
            balanced_manifest = balanced_manifest.drop(index=drop_indices)

    balanced_label_sets = balanced_manifest["label"].map(label_parser)
    balanced_counts = balanced_label_sets.explode().value_counts()
    sampled_groups = [balanced_manifest]

    for label in class_counts.index:
        current_count = int(balanced_counts.get(label, 0))
        if current_count >= target_size:
            continue

        label_rows = manifest[label_sets.map(lambda labels: label in labels)]
        if label_rows.empty:
            continue

        needed = target_size - current_count
        sampled_groups.append(label_rows.sample(n=needed, replace=True, random_state=random_state))

    return pd.concat(sampled_groups).sample(frac=1, random_state=random_state).reset_index(drop=True)
