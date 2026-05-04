from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub

from ucf_crime_recognition.config import load_config, project_path


def download_dataset(config_path: str | Path | None = None) -> Path:
    config = load_config(config_path) if config_path else load_config()
    slug = config["dataset"]["kaggle_slug"]
    raw_dir = project_path(config["dataset"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    kaggle_path = Path(kagglehub.dataset_download(slug))
    destination = raw_dir / slug.split("/")[-1]

    if destination.exists():
        print(f"Dataset already available at: {destination}")
        return destination

    try:
        destination.symlink_to(kaggle_path, target_is_directory=True)
        print(f"Dataset linked to: {destination}")
    except OSError:
        shutil.copytree(kaggle_path, destination)
        print(f"Dataset copied to: {destination}")

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the UCF Crime dataset from Kaggle.")
    parser.add_argument("--config", default=None, help="Path to a TOML config file.")
    args = parser.parse_args()

    path = download_dataset(args.config)
    print("Path to dataset files:", path)


if __name__ == "__main__":
    main()
