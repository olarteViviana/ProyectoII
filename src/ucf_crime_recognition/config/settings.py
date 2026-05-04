from __future__ import annotations

from pathlib import Path
import tomllib

from ucf_crime_recognition.config.constants import DEFAULT_CONFIG_PATH, PROJECT_ROOT


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
