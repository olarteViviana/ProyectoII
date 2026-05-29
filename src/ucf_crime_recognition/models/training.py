from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, hamming_loss, jaccard_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.exceptions import ConvergenceWarning
import warnings

from ucf_crime_recognition.config import load_config, project_path, setup_mlflow
from ucf_crime_recognition.data import rebalance_manifest, load_manifest, sample_manifest, split_manifest, validate_manifest
from ucf_crime_recognition.features import build_feature_matrix
from ucf_crime_recognition.models.candidates import build_model, suggest_model_params
from ucf_crime_recognition.models.registry import register_best_model


def _is_multi_output_config(config: dict) -> bool:
    return bool(config["model"].get("multi_output", False))


def _parse_label_cell(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        labels = [str(label).strip() for label in value]
    else:
        text = str(value)
        for separator in ("|", ";", ",", "+"):
            text = text.replace(separator, "|")
        labels = [label.strip() for label in text.split("|")]

    labels = [label for label in labels if label]
    if len(labels) > 1 and "NormalVideos" in labels:
        labels = [label for label in labels if label != "NormalVideos"]
    return tuple(dict.fromkeys(labels))


def _fit_multilabel_binarizer(manifest: pd.DataFrame) -> MultiLabelBinarizer:
    label_sets = manifest["label"].map(_parse_label_cell)
    return MultiLabelBinarizer().fit(label_sets)


def _transform_multilabel_targets(labels, binarizer: MultiLabelBinarizer) -> np.ndarray:
    return binarizer.transform([_parse_label_cell(label) for label in labels])


def _is_multilabel_target(y_true) -> bool:
    return np.asarray(y_true).ndim == 2


def _transform_pipeline_features(model, features: np.ndarray) -> np.ndarray:
    if not hasattr(model, "steps"):
        return features

    transformed = features
    for _, step in model.steps[:-1]:
        if hasattr(step, "transform"):
            transformed = step.transform(transformed)
    return transformed


def _positive_probability(probabilities: np.ndarray, classes) -> np.ndarray:
    if classes is None:
        return probabilities[:, -1]
    classes = list(classes)
    if 1 not in classes:
        return np.zeros(probabilities.shape[0], dtype=float)
    return probabilities[:, classes.index(1)]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _multilabel_scores(model, features: np.ndarray, n_outputs: int) -> np.ndarray:
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else model

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if isinstance(probabilities, list):
            output_classes = getattr(classifier, "classes_", [None] * len(probabilities))
            scores = [
                _positive_probability(np.asarray(output_probabilities), classes)
                for output_probabilities, classes in zip(probabilities, output_classes)
            ]
            return np.vstack(scores).T

    transformed_features = _transform_pipeline_features(model, features)
    estimators = getattr(classifier, "estimators_", None)
    if estimators is None:
        return np.asarray(model.predict(features), dtype=float)

    scores = []
    for estimator in estimators:
        if hasattr(estimator, "predict_proba"):
            probabilities = np.asarray(estimator.predict_proba(transformed_features))
            scores.append(_positive_probability(probabilities, estimator.classes_))
        elif hasattr(estimator, "decision_function"):
            decision = np.asarray(estimator.decision_function(transformed_features), dtype=float)
            scores.append(_sigmoid(decision.reshape(-1)))
        else:
            scores.append(np.asarray(estimator.predict(transformed_features), dtype=float))

    if not scores:
        return np.zeros((features.shape[0], n_outputs), dtype=float)
    return np.vstack(scores).T


def _apply_multilabel_thresholds(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (scores >= thresholds.reshape(1, -1)).astype(int)


def _optimize_multilabel_thresholds(y_true: np.ndarray, scores: np.ndarray, config: dict) -> np.ndarray:
    threshold_config = config["model"].get("threshold_search", {})
    start = float(threshold_config.get("min", 0.05))
    stop = float(threshold_config.get("max", 0.95))
    steps = int(threshold_config.get("steps", 19))
    candidates = np.linspace(start, stop, steps)
    thresholds = np.full(y_true.shape[1], 0.5, dtype=float)

    for label_index in range(y_true.shape[1]):
        if int(y_true[:, label_index].sum()) == 0:
            continue

        best_score = -1.0
        best_threshold = 0.5
        for threshold in candidates:
            predictions = (scores[:, label_index] >= threshold).astype(int)
            score = f1_score(y_true[:, label_index], predictions, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        thresholds[label_index] = best_threshold

    return thresholds


def _risk_label_indices(label_classes: list[str] | None) -> list[int]:
    if not label_classes:
        return []
    return [index for index, label in enumerate(label_classes) if str(label) != "NormalVideos"]


def _compute_metrics(y_true, predictions, label_classes: list[str] | None = None) -> dict:
    if _is_multilabel_target(y_true):
        metrics = {
            "accuracy": accuracy_score(y_true, predictions),
            "f1_macro": f1_score(y_true, predictions, average="macro", zero_division=0),
            "f1_micro": f1_score(y_true, predictions, average="micro", zero_division=0),
            "f1_weighted": f1_score(y_true, predictions, average="weighted", zero_division=0),
            "f1_samples": f1_score(y_true, predictions, average="samples", zero_division=0),
            "jaccard_samples": jaccard_score(y_true, predictions, average="samples", zero_division=0),
            "hamming_loss": hamming_loss(y_true, predictions),
        }
        risk_indices = _risk_label_indices(label_classes)
        if risk_indices:
            risk_true = y_true[:, risk_indices]
            risk_predictions = predictions[:, risk_indices]
            metrics.update(
                {
                    "risk_f1_macro": f1_score(risk_true, risk_predictions, average="macro", zero_division=0),
                    "risk_recall_macro": recall_score(
                        risk_true,
                        risk_predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "risk_f1_micro": f1_score(risk_true, risk_predictions, average="micro", zero_division=0),
                }
            )
        return metrics

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "f1_macro": f1_score(y_true, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, predictions, average="weighted", zero_division=0),
    }


def _log_metrics(prefix: str, metrics: dict) -> None:
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", float(metric_value))


def _log_params(params: dict) -> None:
    if params:
        mlflow.log_params({key: str(value) for key, value in params.items()})


def _flatten_params(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True)


def _log_label_distribution(title: str, manifest: pd.DataFrame) -> None:
    exploded_labels = manifest["label"].map(_parse_label_cell).explode()
    counts = exploded_labels.value_counts().sort_index()
    print(f"{title} ({len(manifest)} samples)")
    print(counts.to_string())


def _safe_validation_split(
    train_manifest: pd.DataFrame,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_labels = train_manifest["label"] if train_manifest["label"].nunique() > 1 else None
    try:
        return train_test_split(
            train_manifest,
            test_size=validation_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
    except ValueError:
        return train_test_split(
            train_manifest,
            test_size=validation_size,
            random_state=random_state,
            stratify=None,
        )


def _run_optuna_search(
    model_name: str,
    x_train,
    y_train,
    x_validation,
    y_validation,
    config: dict,
    label_classes: list[str] | None = None,
) -> dict:
    selection_metric = config["model"]["selection_metric"]
    random_state = config["preprocessing"]["random_state"]
    n_trials = config["optuna"]["n_trials"]
    timeout_seconds = config["optuna"].get("timeout_seconds")
    sampler = optuna.samplers.TPESampler(seed=random_state)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_study",
        sampler=sampler,
    )

    with mlflow.start_run(run_name=model_name) as search_run:
        mlflow.set_tag("model_candidate", model_name)
        mlflow.set_tag("search_strategy", "optuna")
        mlflow.log_param("optuna_n_trials", n_trials)
        if timeout_seconds is not None:
            mlflow.log_param("optuna_timeout_seconds", timeout_seconds)

        def objective(trial: optuna.Trial) -> float:
            params = suggest_model_params(trial, model_name, config)
            model = build_model(model_name, config, params=params)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(x_train, y_train)
            validation_thresholds = None
            if _is_multilabel_target(y_validation):
                validation_scores = _multilabel_scores(model, x_validation, y_validation.shape[1])
                validation_thresholds = _optimize_multilabel_thresholds(y_validation, validation_scores, config)
                predictions = _apply_multilabel_thresholds(validation_scores, validation_thresholds)
            else:
                predictions = model.predict(x_validation)
            metrics = _compute_metrics(y_validation, predictions, label_classes)

            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.set_tag("model_candidate", model_name)
                mlflow.set_tag("search_strategy", "optuna")
                mlflow.set_tag("optuna_trial_number", trial.number)
                _log_params(params)
                _log_metrics("validation", metrics)

            trial.set_user_attr("params", params)
            trial.set_user_attr("validation_metrics", metrics)
            if validation_thresholds is not None:
                trial.set_user_attr("validation_thresholds", validation_thresholds.tolist())
            return metrics[selection_metric]

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            catch=(ValueError, RuntimeError),
        )

        if not study.best_trials:
            raise RuntimeError(f"No valid Optuna trials were completed for model '{model_name}'.")

        best_trial = study.best_trial
        best_params = dict(best_trial.user_attrs.get("params", best_trial.params))
        validation_metrics = dict(best_trial.user_attrs.get("validation_metrics", {}))
        validation_thresholds = best_trial.user_attrs.get("validation_thresholds")

        mlflow.log_param("best_trial_number", best_trial.number)
        mlflow.log_param("best_params_json", _flatten_params(best_params))
        if validation_thresholds is not None:
            mlflow.log_param("best_validation_thresholds_json", _flatten_params(validation_thresholds))
        _log_params(best_params)
        _log_metrics("best_validation", validation_metrics)

        return {
            "model_name": model_name,
            "search_run_id": search_run.info.run_id,
            "best_params": best_params,
            "best_trial_number": best_trial.number,
            "validation_metrics": validation_metrics,
            "validation_thresholds": validation_thresholds,
            "selection_score": validation_metrics[selection_metric],
        }


def _train_final_model(
    model_name: str,
    best_params: dict,
    search_run_id: str,
    best_trial_number: int,
        validation_metrics: dict,
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
    config: dict,
    label_classes: list[str] | None = None,
    label_thresholds: list[float] | None = None,
) -> dict:
    x_full_train = np.vstack([x_train, x_validation])
    y_full_train = np.concatenate([y_train, y_validation])
    final_model = build_model(model_name, config, params=best_params)

    with mlflow.start_run(run_name=f"{model_name}_final") as final_run:
        mlflow.set_tag("model_candidate", model_name)
        mlflow.set_tag("search_strategy", "optuna")
        mlflow.set_tag("optuna_search_run_id", search_run_id)
        mlflow.log_param("best_trial_number", best_trial_number)
        mlflow.log_param("best_params_json", _flatten_params(best_params))
        if label_thresholds is not None:
            mlflow.log_param("label_thresholds_json", _flatten_params(label_thresholds))
        _log_params(best_params)

        final_model.fit(x_full_train, y_full_train)
        final_model.feature_extractor_ = config["preprocessing"].get("feature_extractor", "resnet50")
        if final_model.feature_extractor_ == "videomae":
            final_model.video_model_name_ = config["preprocessing"].get(
                "video_model_name",
                "MCG-NJU/videomae-base-finetuned-kinetics",
            )
        if label_classes is not None:
            final_model.label_classes_ = list(label_classes)
            final_model.multi_output_ = True
            if label_thresholds is not None:
                final_model.label_thresholds_ = [float(threshold) for threshold in label_thresholds]

        if label_classes is not None and label_thresholds is not None:
            test_scores = _multilabel_scores(final_model, x_test, y_test.shape[1])
            predictions = _apply_multilabel_thresholds(test_scores, np.asarray(label_thresholds, dtype=float))
        else:
            predictions = final_model.predict(x_test)
        test_metrics = _compute_metrics(y_test, predictions, label_classes)

        _log_metrics("validation", validation_metrics)
        _log_metrics("test", test_metrics)
        mlflow.sklearn.log_model(final_model, name="model")

        return {
            "model_name": model_name,
            "run_id": final_run.info.run_id,
            "model": final_model,
            "predictions": predictions,
            "best_params": best_params,
            "best_trial_number": best_trial_number,
            "validation_accuracy": validation_metrics.get("accuracy"),
            "validation_f1_macro": validation_metrics.get("f1_macro"),
            "validation_f1_micro": validation_metrics.get("f1_micro"),
            "validation_f1_weighted": validation_metrics.get("f1_weighted"),
            "validation_f1_samples": validation_metrics.get("f1_samples"),
            "validation_risk_f1_macro": validation_metrics.get("risk_f1_macro"),
            "validation_risk_recall_macro": validation_metrics.get("risk_recall_macro"),
            "accuracy": test_metrics["accuracy"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_micro": test_metrics.get("f1_micro"),
            "f1_weighted": test_metrics["f1_weighted"],
            "f1_samples": test_metrics.get("f1_samples"),
            "risk_f1_macro": test_metrics.get("risk_f1_macro"),
            "risk_recall_macro": test_metrics.get("risk_recall_macro"),
            "jaccard_samples": test_metrics.get("jaccard_samples"),
            "hamming_loss": test_metrics.get("hamming_loss"),
        }


def train(config_path: str | Path | None = None) -> dict:
    config = load_config(config_path) if config_path else load_config()
    setup_mlflow(config)

    model_path = project_path(config["model"]["output_path"])
    report_path = project_path(config["reports"]["classification_report"])
    matrix_path = project_path(config["reports"]["confusion_matrix"])
    summary_path = project_path(config["reports"]["experiment_summary"])

    manifest = validate_manifest(load_manifest(config=config))
    _log_label_distribution("Full manifest", manifest)
    random_state = config["preprocessing"]["random_state"]
    train_manifest, test_manifest = split_manifest(manifest)
    _log_label_distribution("Train split before sampling", train_manifest)
    _log_label_distribution("Test split before sampling", test_manifest)
    train_manifest = sample_manifest(
        train_manifest,
        config["model"].get("max_train_samples"),
        random_state,
    )
    _log_label_distribution("Train split after sampling", train_manifest)

    validation_size = config["preprocessing"].get("validation_size", 0.25)
    train_manifest, validation_manifest = _safe_validation_split(train_manifest, validation_size, random_state)
    validation_manifest = sample_manifest(
        validation_manifest,
        config["model"].get("max_validation_samples", config["model"].get("max_test_samples")),
        random_state,
    )
    _log_label_distribution("Train split after validation split", train_manifest)
    _log_label_distribution("Validation split after sampling", validation_manifest)
    train_manifest = rebalance_manifest(
        train_manifest,
        target_min_per_class=config["model"].get("min_train_samples_per_class"),
        target_max_per_class=config["model"].get("max_train_samples_per_class"),
        random_state=random_state,
        label_parser=_parse_label_cell if _is_multi_output_config(config) else None,
    )
    _log_label_distribution("Train split after rebalance", train_manifest)
    test_manifest = sample_manifest(
        test_manifest,
        config["model"].get("max_test_samples"),
        random_state,
    )
    _log_label_distribution("Test split after sampling", test_manifest)

    image_size = config["preprocessing"]["image_size"]
    color_mode = config["preprocessing"]["color_mode"]
    feature_extractor = config["preprocessing"].get("feature_extractor", "resnet50")
    video_model_name = config["preprocessing"].get("video_model_name", "MCG-NJU/videomae-base-finetuned-kinetics")
    embedding_cache_dir = config["preprocessing"].get("embedding_cache_dir")
    embedding_cache_path = project_path(embedding_cache_dir) if embedding_cache_dir else None

    x_train, y_train = build_feature_matrix(
        train_manifest,
        image_size,
        color_mode,
        feature_extractor=feature_extractor,
        cache_dir=embedding_cache_path,
        video_model_name=video_model_name,
    )
    x_validation, y_validation = build_feature_matrix(
        validation_manifest,
        image_size,
        color_mode,
        feature_extractor=feature_extractor,
        cache_dir=embedding_cache_path,
        video_model_name=video_model_name,
    )
    x_test, y_test = build_feature_matrix(
        test_manifest,
        image_size,
        color_mode,
        feature_extractor=feature_extractor,
        cache_dir=embedding_cache_path,
        video_model_name=video_model_name,
    )

    label_classes = None
    if _is_multi_output_config(config):
        label_binarizer = _fit_multilabel_binarizer(manifest)
        label_classes = [str(label) for label in label_binarizer.classes_]
        print(f"Multi-output labels: {', '.join(label_classes)}")
        y_train = _transform_multilabel_targets(y_train, label_binarizer)
        y_validation = _transform_multilabel_targets(y_validation, label_binarizer)
        y_test = _transform_multilabel_targets(y_test, label_binarizer)

    candidate_results = []
    for model_name in config["model"]["candidate_models"]:
        search_result = _run_optuna_search(
            model_name,
            x_train,
            y_train,
            x_validation,
            y_validation,
            config,
            label_classes,
        )
        candidate_results.append(search_result)

    selection_metric = config["model"]["selection_metric"]
    best_search_result = max(candidate_results, key=lambda result: result["selection_score"])
    best_result = _train_final_model(
        best_search_result["model_name"],
        best_search_result["best_params"],
        best_search_result["search_run_id"],
        best_search_result["best_trial_number"],
        best_search_result["validation_metrics"],
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        config,
        label_classes,
        best_search_result.get("validation_thresholds"),
    )

    if _is_multi_output_config(config):
        report = classification_report(
            y_test,
            best_result["predictions"],
            target_names=label_classes,
            zero_division=0,
        )
        matrix_rows = []
        label_thresholds = best_search_result.get("validation_thresholds")
        for label_index, label in enumerate(label_classes):
            tn, fp, fn, tp = confusion_matrix(
                y_test[:, label_index],
                best_result["predictions"][:, label_index],
                labels=[0, 1],
            ).ravel()
            matrix_rows.append(
                {
                    "label": label,
                    "threshold": label_thresholds[label_index] if label_thresholds is not None else None,
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                }
            )
        matrix = pd.DataFrame(matrix_rows)
    else:
        report = classification_report(y_test, best_result["predictions"], zero_division=0)
        labels = sorted(pd.Series(y_test).unique())
        matrix = pd.DataFrame(
            confusion_matrix(y_test, best_result["predictions"], labels=labels),
            index=labels,
            columns=labels,
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_result["model"], model_path)
    report_path.write_text(report)
    matrix.to_csv(matrix_path)

    summary_rows = []
    for result in candidate_results:
        row = {
            "model_name": result["model_name"],
            "search_run_id": result["search_run_id"],
            "best_trial_number": result["best_trial_number"],
            "best_params": _flatten_params(result["best_params"]),
            "validation_thresholds": _flatten_params(result["validation_thresholds"])
            if result.get("validation_thresholds") is not None
            else None,
            "validation_accuracy": result["validation_metrics"].get("accuracy"),
            "validation_f1_macro": result["validation_metrics"].get("f1_macro"),
            "validation_f1_micro": result["validation_metrics"].get("f1_micro"),
            "validation_f1_weighted": result["validation_metrics"].get("f1_weighted"),
            "validation_f1_samples": result["validation_metrics"].get("f1_samples"),
            "validation_risk_f1_macro": result["validation_metrics"].get("risk_f1_macro"),
            "validation_risk_recall_macro": result["validation_metrics"].get("risk_recall_macro"),
            "validation_risk_f1_micro": result["validation_metrics"].get("risk_f1_micro"),
            "validation_jaccard_samples": result["validation_metrics"].get("jaccard_samples"),
            "validation_hamming_loss": result["validation_metrics"].get("hamming_loss"),
            "final_run_id": None,
            "accuracy": None,
            "f1_macro": None,
            "f1_micro": None,
            "f1_weighted": None,
            "f1_samples": None,
            "risk_f1_macro": None,
            "risk_recall_macro": None,
            "jaccard_samples": None,
            "hamming_loss": None,
            "is_best": False,
        }
        if result["model_name"] == best_result["model_name"]:
            row.update(
                {
                    "final_run_id": best_result["run_id"],
                    "accuracy": best_result["accuracy"],
                    "f1_macro": best_result["f1_macro"],
                    "f1_micro": best_result.get("f1_micro"),
                    "f1_weighted": best_result["f1_weighted"],
                    "f1_samples": best_result.get("f1_samples"),
                    "risk_f1_macro": best_result.get("risk_f1_macro"),
                    "risk_recall_macro": best_result.get("risk_recall_macro"),
                    "jaccard_samples": best_result.get("jaccard_samples"),
                    "hamming_loss": best_result.get("hamming_loss"),
                    "is_best": True,
                }
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("validation_" + selection_metric, ascending=False)
    summary.to_csv(summary_path, index=False)

    model_version = register_best_model(config, best_result["run_id"], best_result[selection_metric])

    print(f"Model saved to: {model_path}")
    print(f"Report saved to: {report_path}")
    print(f"Experiment summary saved to: {summary_path}")
    print(
        f"Best model: {best_result['model_name']} "
        f"(validation {selection_metric}={best_search_result['selection_score']:.4f}, "
        f"test {selection_metric}={best_result[selection_metric]:.4f})"
    )
    print(f"Best params: {_flatten_params(best_result['best_params'])}")
    if model_version:
        print(f"Registered model version: {model_version}")
    print(report)

    return {
        "best_model_name": best_result["model_name"],
        "best_run_id": best_result["run_id"],
        "best_score": best_result[selection_metric],
        "validation_score": best_search_result["selection_score"],
        "best_params": best_result["best_params"],
        "model_version": model_version,
        "summary_path": str(summary_path),
        "model_path": str(model_path),
    }
