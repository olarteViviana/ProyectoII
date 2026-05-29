from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from ucf_crime_recognition.config import load_config, project_path
from ucf_crime_recognition.features import (
    load_image_vector,
    load_image_vector_pretrained,
    load_video_vector_pretrained,
    load_video_vector_videomae,
)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _transform_pipeline_features(model, features: np.ndarray) -> np.ndarray:
    if not hasattr(model, "steps"):
        return features

    transformed = features
    for _, step in model.steps[:-1]:
        if hasattr(step, "transform"):
            transformed = step.transform(transformed)
    return transformed


def _positive_probability(probabilities: np.ndarray, classes) -> float:
    if classes is None:
        return float(probabilities[0, -1])
    classes = list(classes)
    if 1 not in classes:
        return 0.0
    return float(probabilities[0, classes.index(1)])


def _multioutput_class_scores(model, features: np.ndarray, label_classes: list[str]) -> dict[str, float]:
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else model

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if isinstance(probabilities, list):
            scores = {}
            output_classes = getattr(classifier, "classes_", [None] * len(probabilities))
            for label, output_probabilities, classes in zip(label_classes, probabilities, output_classes):
                output_probabilities = np.asarray(output_probabilities)
                scores[str(label)] = _positive_probability(output_probabilities, classes)
            return scores

    transformed = _transform_pipeline_features(model, features)
    estimators = getattr(classifier, "estimators_", None)
    if estimators is None:
        return {}

    scores = {}
    for label, estimator in zip(label_classes, estimators):
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(transformed)
            scores[str(label)] = _positive_probability(probabilities, estimator.classes_)
        elif hasattr(estimator, "decision_function"):
            decision = np.asarray(estimator.decision_function(transformed)).reshape(-1)[0]
            scores[str(label)] = _sigmoid(float(decision))
    return scores


def _prediction_details(model, features: np.ndarray) -> dict:
    predicted_label = model.predict(features)[0]
    label_classes = getattr(model, "label_classes_", None)

    if label_classes is not None:
        label_classes = [str(label) for label in label_classes]
        class_scores = _multioutput_class_scores(model, features, label_classes)
        thresholds = getattr(model, "label_thresholds_", None)
        if thresholds is not None and class_scores:
            predicted_outputs = np.asarray(
                [
                    int(class_scores.get(label, 0.0) >= float(threshold))
                    for label, threshold in zip(label_classes, thresholds)
                ]
            )
        else:
            predicted_outputs = np.asarray(predicted_label).astype(int)
        active_labels = [
            label
            for label, is_active in zip(label_classes, predicted_outputs)
            if int(is_active) == 1
        ]

        if not active_labels and class_scores:
            active_labels = [max(class_scores, key=class_scores.get)]
        elif not active_labels:
            active_labels = ["NormalVideos"] if "NormalVideos" in label_classes else []

        confidence = max((class_scores.get(label, 1.0) for label in active_labels), default=0.0)
        return {
            "prediction": "|".join(active_labels),
            "predictions": active_labels,
            "confidence": float(confidence),
            "class_scores": class_scores,
            "class_thresholds": {
                label: float(threshold)
                for label, threshold in zip(label_classes, thresholds)
            }
            if thresholds is not None
            else {},
        }

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


def _config_cache_key(config_path: str | Path | None) -> str:
    if config_path is None:
        return ""
    return str(Path(config_path).expanduser().resolve())


@lru_cache(maxsize=16)
def _load_prediction_config(config_key: str) -> dict:
    if config_key:
        return load_config(Path(config_key))
    return load_config()


def _model_fingerprint(model_path: Path) -> tuple[int, int]:
    try:
        stat = model_path.stat()
    except OSError:
        return 0, 0
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _load_prediction_model(model_path: str, mtime_ns: int, size: int):
    return joblib.load(Path(model_path))


def predict_image_details(image_path: str | Path, config_path: str | Path | None = None) -> dict:
    config = _load_prediction_config(_config_cache_key(config_path))
    model_path = project_path(config["model"]["output_path"])
    image_size = config["preprocessing"]["image_size"]
    color_mode = config["preprocessing"]["color_mode"]

    model_mtime_ns, model_size = _model_fingerprint(model_path)
    model = _load_prediction_model(str(model_path), model_mtime_ns, model_size)
    
    feature_extractor = getattr(model, "feature_extractor_", config["preprocessing"].get("feature_extractor", None))
    expected_features = getattr(model, "n_features_in_", None)

    if feature_extractor in {"resnet50", "vgg16"}:
        features = load_image_vector_pretrained(image_path, feature_extractor=feature_extractor).reshape(1, -1)
    elif feature_extractor == "videomae":
        video_model_name = getattr(
            model,
            "video_model_name_",
            config["preprocessing"].get("video_model_name", "MCG-NJU/videomae-base-finetuned-kinetics"),
        )
        features = load_video_vector_videomae(image_path, model_name=video_model_name).reshape(1, -1)
    elif feature_extractor in {"r3d_18", "r2plus1d_18"}:
        features = load_video_vector_pretrained(image_path, feature_extractor=feature_extractor).reshape(1, -1)
    elif expected_features == 2048:
        features = load_image_vector_pretrained(image_path, feature_extractor="resnet50").reshape(1, -1)
    elif expected_features == 4096:
        features = load_image_vector_pretrained(image_path, feature_extractor="vgg16").reshape(1, -1)
    elif expected_features == 512:
        fallback_extractor = config["preprocessing"].get("feature_extractor", "r2plus1d_18")
        if fallback_extractor not in {"r3d_18", "r2plus1d_18"}:
            fallback_extractor = "r2plus1d_18"
        features = load_video_vector_pretrained(image_path, feature_extractor=fallback_extractor).reshape(1, -1)
    elif expected_features == 768 and config["preprocessing"].get("feature_extractor") == "videomae":
        video_model_name = config["preprocessing"].get("video_model_name", "MCG-NJU/videomae-base-finetuned-kinetics")
        features = load_video_vector_videomae(image_path, model_name=video_model_name).reshape(1, -1)
    else:
        features = load_image_vector(image_path, image_size, color_mode).reshape(1, -1)

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
