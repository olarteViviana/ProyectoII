from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ucf_crime_recognition.config import load_config, project_path
from ucf_crime_recognition.features import build_feature_matrix


def _sample_manifest(manifest: pd.DataFrame, max_samples: int | None, random_state: int) -> pd.DataFrame:
    if not max_samples or len(manifest) <= max_samples:
        return manifest

    return (
        manifest.groupby("label", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=max(1, round(max_samples * len(group) / len(manifest))),
                random_state=random_state,
            )
        )
        .sample(frac=1, random_state=random_state)
        .head(max_samples)
    )


def train(config_path: str | Path | None = None) -> Pipeline:
    config = load_config(config_path) if config_path else load_config()
    manifest_path = project_path(config["dataset"]["manifest_path"])
    model_path = project_path(config["model"]["output_path"])
    report_path = project_path(config["reports"]["classification_report"])
    matrix_path = project_path(config["reports"]["confusion_matrix"])

    manifest = pd.read_csv(manifest_path)
    random_state = config["preprocessing"]["random_state"]
    train_manifest = manifest[manifest["split"] == "train"]
    test_manifest = manifest[manifest["split"] == "test"]
    train_manifest = _sample_manifest(
        train_manifest,
        config["model"].get("max_train_samples"),
        random_state,
    )
    test_manifest = _sample_manifest(
        test_manifest,
        config["model"].get("max_test_samples"),
        random_state,
    )

    image_size = config["preprocessing"]["image_size"]
    color_mode = config["preprocessing"]["color_mode"]

    x_train, y_train = build_feature_matrix(train_manifest, image_size, color_mode)
    x_test, y_test = build_feature_matrix(test_manifest, image_size, color_mode)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=config["model"]["max_iter"],
                    class_weight=config["model"]["class_weight"],
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    report = classification_report(y_test, predictions)
    labels = sorted(pd.Series(y_test).unique())
    matrix = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=labels,
        columns=labels,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    report_path.write_text(report)
    matrix.to_csv(matrix_path)

    print(f"Model saved to: {model_path}")
    print(f"Report saved to: {report_path}")
    print(report)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline UCF Crime image classifier.")
    parser.add_argument("--config", default=None, help="Path to a TOML config file.")
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
