import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from ucf_crime_recognition.features.engineering import (
    _is_video_feature_extractor,
    _pretrained_embedding_dim,
    build_feature_matrix,
)
from ucf_crime_recognition.predict import predict_image_details


class FakeVideoMaeClassifier:
    classes_ = np.asarray(["Fighting", "NormalVideos"])
    feature_extractor_ = "videomae"
    video_model_name_ = "test/videomae"
    n_features_in_ = 768

    def predict(self, features):
        return np.asarray(["Fighting"])

    def predict_proba(self, features):
        return np.asarray([[0.81, 0.19]])


class VideoMaeExtractorTests(unittest.TestCase):
    def test_videomae_is_registered_as_video_feature_extractor(self):
        self.assertTrue(_is_video_feature_extractor("videomae"))
        self.assertEqual(_pretrained_embedding_dim("videomae"), 768)

    def test_build_feature_matrix_passes_videomae_batch_options(self):
        manifest = pd.DataFrame(
            [
                {"path": "clip_001.png", "label": "Fighting"},
                {"path": "clip_002.png", "label": "NormalVideos"},
            ]
        )

        with patch("ucf_crime_recognition.features.engineering.load_video_vectors_videomae_cached_batch") as loader:
            loader.return_value = [
                np.ones(768, dtype=np.float32),
                np.full(768, 2.0, dtype=np.float32),
            ]
            features, labels = build_feature_matrix(
                manifest,
                image_size=96,
                color_mode="rgb",
                feature_extractor="videomae",
                video_model_name="test/videomae",
                videomae_batch_size=2,
            )

        self.assertEqual(features.shape, (2, 768))
        self.assertEqual(labels.tolist(), ["Fighting", "NormalVideos"])
        self.assertEqual(loader.call_args.args[0], ["clip_001.png", "clip_002.png"])
        self.assertEqual(loader.call_args.kwargs["video_model_name"], "test/videomae")
        self.assertEqual(loader.call_args.kwargs["batch_size"], 2)

    def test_predict_image_details_uses_videomae_vector_for_videomae_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "clip_000001.png"
            model_path = root / "model.joblib"
            config_path = root / "pipeline.toml"
            Image.new("RGB", (12, 12), color=(20, 80, 220)).save(image_path)
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        f'output_path = "{model_path}"',
                        "",
                        "[preprocessing]",
                        "image_size = 96",
                        'color_mode = "rgb"',
                        'feature_extractor = "videomae"',
                        'video_model_name = "test/videomae"',
                    ]
                ),
                encoding="utf-8",
            )

            with (
                patch("ucf_crime_recognition.predict.joblib.load", return_value=FakeVideoMaeClassifier()),
                patch("ucf_crime_recognition.predict.load_video_vector_videomae", return_value=np.ones(768, dtype=np.float32)) as loader,
            ):
                details = predict_image_details(image_path, config_path=config_path)

        self.assertEqual(details["prediction"], "Fighting")
        self.assertEqual(details["class_scores"], {"Fighting": 0.81, "NormalVideos": 0.19})
        self.assertEqual(loader.call_args.kwargs["model_name"], "test/videomae")


if __name__ == "__main__":
    unittest.main()
