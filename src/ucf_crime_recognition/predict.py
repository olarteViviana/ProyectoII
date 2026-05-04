from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from ucf_crime_recognition.config import load_config, project_path
from ucf_crime_recognition.features import load_image_vector


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def _prediction_details(model, features: np.ndarray) -> dict:
    predicted_label = model.predict(features)[0]

    class_scores: dict[str, float] = {}
    confidence: float

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        class_scores = {str(label): float(probability) for label, probability in zip(model.classes_, probabilities)}
        confidence = float(class_scores[str(predicted_label)])
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features))
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        probabilities = _softmax(scores[0])
        class_scores = {str(label): float(probability) for label, probability in zip(model.classes_, probabilities)}
        confidence = float(class_scores[str(predicted_label)])
    else:
        confidence = 1.0
        class_scores = {str(predicted_label): 1.0}

    return {
        "prediction": str(predicted_label),
        "confidence": confidence,
        "class_scores": class_scores,
    }


def predict_image_details(image_path: str | Path, config_path: str | Path | None = None) -> dict:
    config = load_config(config_path) if config_path else load_config()
    model_path = project_path(config["model"]["output_path"])
    image_size = config["preprocessing"]["image_size"]
    color_mode = config["preprocessing"]["color_mode"]

    model = joblib.load(model_path)
    features = load_image_vector(image_path, image_size, color_mode).reshape(1, -1)

    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and features.shape[1] != expected_features:
        raise ValueError(
            "The saved model was trained with "
            f"{expected_features} features, but the current extractor produces {features.shape[1]}. "
            "Run `uv run ucf-flow --rebuild-manifest` again so the saved model matches the active feature extractor."
        )

    details = _prediction_details(model, features)
    details["model_path"] = str(model_path)
    return details


def predict_image(image_path: str | Path, config_path: str | Path | None = None) -> str:
    return predict_image_details(image_path, config_path)["prediction"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the class of one UCF Crime image.")
    parser.add_argument("image_path", help="Path to an image.")
    parser.add_argument("--config", default=None, help="Path to a TOML config file.")
    args = parser.parse_args()

    prediction = predict_image(args.image_path, args.config)
    print(prediction)


if __name__ == "__main__":
    main()
