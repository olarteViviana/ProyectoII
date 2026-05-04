from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import warnings

from ucf_crime_recognition.config import load_config, project_path, setup_mlflow
from ucf_crime_recognition.data import rebalance_manifest, load_manifest, sample_manifest, split_manifest, validate_manifest
from ucf_crime_recognition.features import build_feature_matrix
from ucf_crime_recognition.models.candidates import build_model, suggest_model_params
from ucf_crime_recognition.models.registry import register_best_model


def _compute_metrics(y_true, predictions) -> dict:
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "f1_macro": f1_score(y_true, predictions, average="macro"),
        "f1_weighted": f1_score(y_true, predictions, average="weighted"),
    }


def _log_metrics(prefix: str, metrics: dict) -> None:
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"{prefix}_{metric_name}", float(metric_value))


def _log_params(params: dict) -> None:
    if params:
        mlflow.log_params({key: str(value) for key, value in params.items()})


def _flatten_params(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True)


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
            predictions = model.predict(x_validation)
            metrics = _compute_metrics(y_validation, predictions)

            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.set_tag("model_candidate", model_name)
                mlflow.set_tag("search_strategy", "optuna")
                mlflow.set_tag("optuna_trial_number", trial.number)
                _log_params(params)
                _log_metrics("validation", metrics)

            trial.set_user_attr("params", params)
            trial.set_user_attr("validation_metrics", metrics)
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

        mlflow.log_param("best_trial_number", best_trial.number)
        mlflow.log_param("best_params_json", _flatten_params(best_params))
        _log_params(best_params)
        _log_metrics("best_validation", validation_metrics)

        return {
            "model_name": model_name,
            "search_run_id": search_run.info.run_id,
            "best_params": best_params,
            "best_trial_number": best_trial.number,
            "validation_metrics": validation_metrics,
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
        _log_params(best_params)

        final_model.fit(x_full_train, y_full_train)
        predictions = final_model.predict(x_test)
        test_metrics = _compute_metrics(y_test, predictions)

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
            "validation_f1_weighted": validation_metrics.get("f1_weighted"),
            "accuracy": test_metrics["accuracy"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_weighted": test_metrics["f1_weighted"],
        }


def train(config_path: str | Path | None = None) -> dict:
    config = load_config(config_path) if config_path else load_config()
    setup_mlflow(config)

    model_path = project_path(config["model"]["output_path"])
    report_path = project_path(config["reports"]["classification_report"])
    matrix_path = project_path(config["reports"]["confusion_matrix"])
    summary_path = project_path(config["reports"]["experiment_summary"])

    manifest = validate_manifest(load_manifest(config=config))
    random_state = config["preprocessing"]["random_state"]
    train_manifest, test_manifest = split_manifest(manifest)
    train_manifest = sample_manifest(
        train_manifest,
        config["model"].get("max_train_samples"),
        random_state,
    )

    validation_size = config["preprocessing"].get("validation_size", 0.25)
    train_manifest, validation_manifest = _safe_validation_split(train_manifest, validation_size, random_state)
    validation_manifest = sample_manifest(
        validation_manifest,
        config["model"].get("max_validation_samples", config["model"].get("max_test_samples")),
        random_state,
    )
    train_manifest = rebalance_manifest(
        train_manifest,
        target_min_per_class=config["model"].get("min_train_samples_per_class"),
        target_max_per_class=config["model"].get("max_train_samples_per_class"),
        random_state=random_state,
    )
    test_manifest = sample_manifest(
        test_manifest,
        config["model"].get("max_test_samples"),
        random_state,
    )

    image_size = config["preprocessing"]["image_size"]
    color_mode = config["preprocessing"]["color_mode"]

    x_train, y_train = build_feature_matrix(train_manifest, image_size, color_mode)
    x_validation, y_validation = build_feature_matrix(validation_manifest, image_size, color_mode)
    x_test, y_test = build_feature_matrix(test_manifest, image_size, color_mode)

    candidate_results = []
    for model_name in config["model"]["candidate_models"]:
        search_result = _run_optuna_search(model_name, x_train, y_train, x_validation, y_validation, config)
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
    )

    report = classification_report(y_test, best_result["predictions"])
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
            "validation_accuracy": result["validation_metrics"].get("accuracy"),
            "validation_f1_macro": result["validation_metrics"].get("f1_macro"),
            "validation_f1_weighted": result["validation_metrics"].get("f1_weighted"),
            "final_run_id": None,
            "accuracy": None,
            "f1_macro": None,
            "f1_weighted": None,
            "is_best": False,
        }
        if result["model_name"] == best_result["model_name"]:
            row.update(
                {
                    "final_run_id": best_result["run_id"],
                    "accuracy": best_result["accuracy"],
                    "f1_macro": best_result["f1_macro"],
                    "f1_weighted": best_result["f1_weighted"],
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
