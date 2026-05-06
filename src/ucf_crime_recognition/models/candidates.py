from __future__ import annotations

from typing import Any

import optuna
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class RiskAwareClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, normal_label: str = "NormalVideos", normal_switch_ratio: float = 0.9):
        self.estimator = estimator
        self.normal_label = normal_label
        self.normal_switch_ratio = normal_switch_ratio

    def fit(self, x, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x, y)
        self.classes_ = getattr(self.estimator_, "classes_", None)
        self.n_features_in_ = getattr(self.estimator_, "n_features_in_", None)
        return self

    def _predict_scores(self, x) -> np.ndarray:
        if hasattr(self.estimator_, "predict_proba"):
            scores = np.asarray(self.estimator_.predict_proba(x), dtype=float)
        elif hasattr(self.estimator_, "decision_function"):
            scores = np.asarray(self.estimator_.decision_function(x), dtype=float)
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            shifted = scores - np.max(scores, axis=1, keepdims=True)
            exponentiated = np.exp(shifted)
            scores = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        else:
            scores = np.ones((x.shape[0], len(self.classes_)), dtype=float)
            scores /= scores.sum(axis=1, keepdims=True)

        return scores

    def predict(self, x):
        scores = self._predict_scores(x)
        best_indices = np.argmax(scores, axis=1)
        normal_index = None
        if self.classes_ is not None:
            try:
                normal_index = list(self.classes_).index(self.normal_label)
            except ValueError:
                normal_index = None

        if normal_index is None:
            return self.classes_[best_indices]

        predictions = []
        for row_index, best_index in enumerate(best_indices):
            if best_index == normal_index:
                row = scores[row_index]
                risk_scores = row.copy()
                risk_scores[normal_index] = -1.0
                risk_index = int(np.argmax(risk_scores))
                if row[risk_index] >= row[normal_index] * self.normal_switch_ratio:
                    best_index = risk_index
            predictions.append(self.classes_[best_index])

        return np.asarray(predictions)

    def predict_proba(self, x):
        return self._predict_scores(x)

    def decision_function(self, x):
        return self._predict_scores(x)


def suggest_model_params(trial: optuna.Trial, model_name: str, config: dict) -> dict[str, Any]:
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-4, 200.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": trial.suggest_int("max_iter", 300, 2000, step=100),
            "normal_switch_ratio": trial.suggest_float("normal_switch_ratio", 0.75, 0.98),
        }

    if model_name == "linear_svm":
        return {
            "C": trial.suggest_float("C", 1e-4, 200.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 300, 3000, step=100),
            "normal_switch_ratio": trial.suggest_float("normal_switch_ratio", 0.75, 0.98),
        }

    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=25),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30, 40]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "normal_switch_ratio": trial.suggest_float("normal_switch_ratio", 0.75, 0.98),
        }

    if model_name == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 20, 30, 40, 50]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "normal_switch_ratio": trial.suggest_float("normal_switch_ratio", 0.75, 0.98),
        }

    raise ValueError(f"Unknown model candidate: {model_name}")


def build_model(model_name: str, config: dict, params: dict[str, Any] | None = None) -> Pipeline:
    params = params or {}
    max_iter = config["model"]["max_iter"]
    class_weight = config["model"]["class_weight"]
    random_state = config["preprocessing"]["random_state"]
    multi_output = bool(config["model"].get("multi_output", False))

    def wrap_classifier(classifier):
        if multi_output:
            return MultiOutputClassifier(classifier, n_jobs=-1)
        return RiskAwareClassifier(
            classifier,
            normal_switch_ratio=params.get("normal_switch_ratio", 0.9),
        )

    if model_name == "logistic_regression":
        classifier = wrap_classifier(
            LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", max_iter),
                solver=params.get("solver", "lbfgs"),
                class_weight=class_weight,
                random_state=random_state,
            )
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])

    if model_name == "linear_svm":
        classifier = wrap_classifier(
            LinearSVC(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", max_iter),
                class_weight=class_weight,
                random_state=random_state,
            )
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])

    if model_name == "random_forest":
        classifier = wrap_classifier(
            RandomForestClassifier(
                n_estimators=params.get("n_estimators", 150),
                max_depth=params.get("max_depth"),
                min_samples_split=params.get("min_samples_split", 2),
                min_samples_leaf=params.get("min_samples_leaf", 1),
                max_features=params.get("max_features", "sqrt"),
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
            )
        )
        return Pipeline([("classifier", classifier)])

    if model_name == "extra_trees":
        classifier = wrap_classifier(
            ExtraTreesClassifier(
                n_estimators=params.get("n_estimators", 250),
                max_depth=params.get("max_depth"),
                min_samples_split=params.get("min_samples_split", 2),
                min_samples_leaf=params.get("min_samples_leaf", 1),
                max_features=params.get("max_features", "sqrt"),
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
            )
        )
        return Pipeline([("classifier", classifier)])

    raise ValueError(f"Unknown model candidate: {model_name}")
