from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from ucf_crime_recognition.config import load_config, project_path
from ucf_crime_recognition.features import load_image_vector


def predict_image(image_path: str | Path, config_path: str | Path | None = None) -> str:
    config = load_config(config_path) if config_path else load_config()
    model_path = project_path(config["model"]["output_path"])
    image_size = config["preprocessing"]["image_size"]
    color_mode = config["preprocessing"]["color_mode"]

    model = joblib.load(model_path)
    features = load_image_vector(image_path, image_size, color_mode).reshape(1, -1)
    prediction = model.predict(features)[0]
    return str(prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the class of one UCF Crime image.")
    parser.add_argument("image_path", help="Path to an image.")
    parser.add_argument("--config", default=None, help="Path to a TOML config file.")
    args = parser.parse_args()

    prediction = predict_image(args.image_path, args.config)
    print(prediction)


if __name__ == "__main__":
    main()
