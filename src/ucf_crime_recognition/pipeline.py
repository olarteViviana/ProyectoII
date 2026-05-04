from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact

from ucf_crime_recognition.config import load_config, project_path, setup_mlflow
from ucf_crime_recognition.data import build_manifest, download_dataset
from ucf_crime_recognition.models import train


@task(name="download_dataset", retries=1)
def download_dataset_task(config_path: str | None) -> str:
    return str(download_dataset(config_path))


@task(name="build_manifest", retries=1)
def build_manifest_task(config_path: str | None) -> int:
    manifest = build_manifest(config_path)
    return len(manifest)


@task(name="train_and_select_best_model", retries=1)
def train_task(config_path: str | None) -> dict:
    return train(config_path)


@flow(
    name="UCF Crime MLflow Prefect Pipeline",
    description="Download data, build manifest, train candidate models, and register the best model.",
    log_prints=True,
)
def ucf_crime_training_flow(
    config_path: str | None = None,
    download: bool = False,
    rebuild_manifest: bool = False,
) -> str:
    logger = get_run_logger()
    config = load_config(config_path) if config_path else load_config()
    tracking_uri = setup_mlflow(config)

    manifest_path = project_path(config["dataset"]["manifest_path"])
    if download:
        logger.info("Downloading dataset from Kaggle...")
        download_dataset_task(config_path)

    if rebuild_manifest or not manifest_path.exists():
        logger.info("Building image manifest...")
        manifest_rows = build_manifest_task(config_path)
        logger.info("Manifest rows: %s", manifest_rows)

    result = train_task(config_path)
    best_run_id = result["best_run_id"]
    best_model_name = result["best_model_name"]
    best_score = result["best_score"]
    selection_metric = config["model"]["selection_metric"]

    table = [
        ["Best model", best_model_name],
        [selection_metric, f"{best_score:.4f}"],
        ["MLflow run ID", best_run_id],
        ["MLflow tracking URI", tracking_uri],
    ]

    create_table_artifact(
        key="ucf-best-model",
        table=table,
        description="Best UCF Crime model selected by the pipeline.",
    )

    mlflow_ui_hint = tracking_uri.replace("sqlite:///", "mlflow ui --backend-store-uri sqlite:///")
    summary = f"""
    # UCF Crime Pipeline Summary

    ## Best Model
    - **Model**: {best_model_name}
    - **Selection metric**: {selection_metric}
    - **Score**: {best_score:.4f}
    - **MLflow run ID**: `{best_run_id}`

    ## Tracking
    - **MLflow tracking URI**: `{tracking_uri}`
    - To inspect locally, run:

    ```bash
    {mlflow_ui_hint}
    ```
    """

    create_markdown_artifact(
        key="ucf-pipeline-summary",
        markdown=summary,
        description="UCF Crime model selection summary.",
    )

    Path("prefect_run_id.txt").write_text(best_run_id)
    return best_run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Prefect + MLflow UCF Crime pipeline.")
    parser.add_argument("--config", default=None, help="Path to a TOML config file.")
    parser.add_argument("--download", action="store_true", help="Download the Kaggle dataset before training.")
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Rebuild the image manifest before training.",
    )
    args = parser.parse_args()

    run_id = ucf_crime_training_flow(
        config_path=args.config,
        download=args.download,
        rebuild_manifest=args.rebuild_manifest,
    )
    print("Pipeline completed successfully.")
    print(f"Best MLflow run_id: {run_id}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
