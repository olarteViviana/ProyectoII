from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_image_vector(image_path: str | Path, image_size: int, color_mode: str) -> np.ndarray:
    mode = "RGB" if color_mode == "rgb" else "L"

    with Image.open(image_path) as image:
        image = image.convert(mode)
        image = image.resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0

    return array.reshape(-1)


def build_feature_matrix(
    manifest: pd.DataFrame,
    image_size: int,
    color_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    features = [
        load_image_vector(row.path, image_size=image_size, color_mode=color_mode)
        for row in manifest.itertuples(index=False)
    ]
    labels = manifest["label"].to_numpy()
    return np.vstack(features), labels
