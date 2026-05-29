import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ucf_crime_recognition.predict import predict_image_details
from ucf_crime_recognition.services.triage import (
    adjust_class_scores_for_motion,
    motion_adjusted_prediction_label,
)
from ucf_crime_recognition.ui.dashboard import (
    _aggregate_video_predictions,
    _format_active_labels,
    _is_video_file,
    _risk_signal_from_scores,
)


class FakeUpload:
    def __init__(self, name: str):
        self.name = name


class FakeImageClassifier:
    classes_ = np.asarray(["Abuse", "NormalVideos"])

    def predict(self, features):
        return np.asarray(["Abuse"])

    def predict_proba(self, features):
        return np.asarray([[0.72, 0.28]])


class FrontendBackendContractTests(unittest.TestCase):
    def test_predict_image_details_returns_fields_consumed_by_streamlit_image_flow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "incident.png"
            model_path = root / "model.joblib"
            config_path = root / "pipeline.toml"

            Image.new("RGB", (12, 12), color=(220, 10, 30)).save(image_path)
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        f'output_path = "{model_path}"',
                        "",
                        "[preprocessing]",
                        "image_size = 8",
                        'color_mode = "rgb"',
                        'feature_extractor = "traditional"',
                    ]
                ),
                encoding="utf-8",
            )

            with patch("ucf_crime_recognition.predict.joblib.load", return_value=FakeImageClassifier()):
                details = predict_image_details(image_path, config_path=config_path)

            self.assertEqual(details["prediction"], "Abuse")
            self.assertAlmostEqual(details["confidence"], 0.72)
            self.assertEqual(details["class_scores"], {"Abuse": 0.72, "NormalVideos": 0.28})
            self.assertEqual(details["model_path"], str(model_path))

    def test_frontend_helpers_understand_prediction_payload_shape(self):
        prediction = {
            "prediction": "Abuse|Fighting",
            "predictions": ["Abuse", "Fighting"],
            "class_scores": {"NormalVideos": 0.84, "Abuse": 0.66, "Fighting": 0.52},
        }

        self.assertEqual(_format_active_labels(prediction), "Abuse | Fighting")
        self.assertEqual(_risk_signal_from_scores(prediction["class_scores"]), ("Abuse", 0.66))

    def test_video_upload_detection_matches_streamlit_allowed_extensions(self):
        self.assertTrue(_is_video_file(FakeUpload("clip.MP4")))
        self.assertTrue(_is_video_file(FakeUpload("incident.webm")))
        self.assertFalse(_is_video_file(FakeUpload("frame.png")))

    def test_aggregate_video_predictions_prioritizes_persistent_risk_signal(self):
        frame_results = [
            {
                "frame_index": 1,
                "timestamp_s": 0.0,
                "prediction": "NormalVideos",
                "confidence": 0.95,
                "class_scores": {"NormalVideos": 0.95, "Abuse": 0.45},
            },
            {
                "frame_index": 2,
                "timestamp_s": 0.5,
                "prediction": "NormalVideos",
                "confidence": 0.93,
                "class_scores": {"NormalVideos": 0.93, "Abuse": 0.45},
            },
            {
                "frame_index": 3,
                "timestamp_s": 1.0,
                "prediction": "NormalVideos",
                "confidence": 0.91,
                "class_scores": {"NormalVideos": 0.91, "Abuse": 0.45},
            },
        ]

        aggregate = _aggregate_video_predictions(frame_results)

        self.assertEqual(aggregate["prediction"], "Abuse")
        self.assertEqual(aggregate["decision_reason"], "prioridad_riesgo")
        self.assertEqual(aggregate["tier"], "Crítico")
        self.assertIn("class_summary", aggregate)
        selected_rows = aggregate["class_summary"][aggregate["class_summary"]["selected"]]
        self.assertEqual(selected_rows["class_name"].tolist(), ["Abuse"])

    def test_motion_adjustment_reduces_normal_and_prefers_fighting_when_close(self):
        raw_scores = {"NormalVideos": 0.92, "Robbery": 0.22, "Fighting": 0.2}

        adjusted = adjust_class_scores_for_motion(raw_scores, motion_intensity=0.95)
        label, confidence = motion_adjusted_prediction_label("NormalVideos", adjusted, motion_intensity=0.95)

        self.assertLess(adjusted["NormalVideos"], raw_scores["NormalVideos"])
        self.assertGreater(adjusted["Fighting"], adjusted["Robbery"])
        self.assertEqual(label, "Fighting")
        self.assertEqual(confidence, adjusted["Fighting"])


if __name__ == "__main__":
    unittest.main()
