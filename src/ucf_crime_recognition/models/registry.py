from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient


def register_best_model(config: dict, run_id: str, score: float) -> str | None:
    if not config["registry"].get("enabled", False):
        return None

    model_name = config["registry"]["model_name"]
    model_uri = f"runs:/{run_id}/model"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient()
    client.set_model_version_tag(model_name, model_details.version, "selection_metric", config["model"]["selection_metric"])
    client.set_model_version_tag(model_name, model_details.version, "score", f"{score:.6f}")
    client.set_model_version_tag(model_name, model_details.version, "framework", "prefect+mlflow")

    return str(model_details.version)
