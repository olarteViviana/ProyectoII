from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.toml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UCF_CRIME_LABELS = {
    "abuse",
    "arrest",
    "arson",
    "assault",
    "burglary",
    "explosion",
    "fighting",
    "normalvideos",
    "roadaccidents",
    "robbery",
    "shooting",
    "shoplifting",
    "stealing",
    "vandalism",
}
