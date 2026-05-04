from __future__ import annotations

import argparse
from pathlib import Path

from ucf_crime_recognition.models.training import train

__all__ = ["train", "main"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UCF Crime image classifiers.")
    parser.add_argument("--config", default=None, help="Path to a TOML config file.")
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
