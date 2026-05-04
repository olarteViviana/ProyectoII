from __future__ import annotations

import os
from pathlib import Path

import mlflow

from ucf_crime_recognition.config.settings import load_config, project_path


def _resolve_tracking_uri(uri: str) -> str:
    if uri.startswith("sqlite:///"):
        db_path = uri.replace("sqlite:///", "", 1)
        if not Path(db_path).is_absolute():
            return f"sqlite:///{project_path(db_path)}"

    return uri


def setup_mlflow(config: dict | None = None) -> str:
    config = config or load_config()
    mlflow_config = config["mlflow"]
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", mlflow_config["tracking_uri"])
    tracking_uri = _resolve_tracking_uri(tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    return tracking_uri
