import unittest
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from ucf_crime_recognition.api.main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_metadata_endpoint_exposes_supported_media_and_labels(self):
        response = self.client.get("/api/v1/metadata")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(".png", payload["image_extensions"])
        self.assertIn(".mp4", payload["video_extensions"])
        self.assertIn("NormalVideos", payload["labels"])

    def test_predict_image_endpoint_returns_backend_prediction_contract(self):
        fake_prediction = {
            "prediction": "Abuse",
            "confidence": 0.72,
            "class_scores": {"Abuse": 0.72, "NormalVideos": 0.28},
            "model_path": "models/fake.joblib",
        }

        with patch("ucf_crime_recognition.api.main.predict_image_details", return_value=fake_prediction):
            response = self.client.post(
                "/api/v1/predict/image",
                files={"file": ("incident.png", b"fake-image-bytes", "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prediction"], "Abuse")
        self.assertEqual(payload["confidence"], 0.72)
        self.assertEqual(payload["class_scores"], {"Abuse": 0.72, "NormalVideos": 0.28})
        self.assertEqual(payload["model_path"], "models/fake.joblib")

    def test_predict_image_endpoint_rejects_unsupported_extension(self):
        response = self.client.post(
            "/api/v1/predict/image",
            files={"file": ("incident.txt", b"not-an-image", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported image extension", response.json()["detail"])

    def test_predict_video_endpoint_returns_aggregate_and_frame_contract(self):
        fake_video_prediction = {
            "aggregate": {
                "prediction": "Abuse",
                "confidence": 0.45,
                "decision_reason": "prioridad_riesgo",
                "decision_note": "Se priorizo la senal de riesgo persistente.",
                "probability_leader": "NormalVideos",
                "probability_leader_confidence": 0.95,
                "tier": "Crítico",
                "score": "Alto",
                "recommendation": "Escalar al equipo de seguridad.",
                "class_summary": pd.DataFrame(
                    [
                        {
                            "selected": True,
                            "class_name": "Abuse",
                            "mean_probability": 0.45,
                            "max_probability": 0.45,
                            "frame_hits": 3,
                            "risk_weight": 2.0,
                            "decision_score": 1.02,
                        }
                    ]
                ),
            },
            "frames": [
                {
                    "timestamp_s": 0.0,
                    "frame_index": 1,
                    "clip_start_frame": 0,
                    "clip_end_frame": 15,
                    "clip_duration_seconds": 0.53,
                    "clip_window_seconds": 2.0,
                    "video_fps": 30.0,
                    "motion_score": 0.31,
                    "motion_intensity": 0.8,
                    "sampling_reason": "movimiento",
                    "prediction": "Abuse",
                    "raw_prediction": "NormalVideos",
                    "active_labels": "Abuse",
                    "confidence": 0.45,
                    "normal_probability": 0.95,
                    "raw_normal_probability": 0.99,
                    "top_risk_class": "Abuse",
                    "top_risk_probability": 0.45,
                    "class_scores": {"Abuse": 0.45, "NormalVideos": 0.95},
                    "tier": "Crítico",
                    "score": "Alto",
                    "recommendation": "Escalar al equipo de seguridad.",
                }
            ],
        }

        with patch("ucf_crime_recognition.api.main.predict_video_details", return_value=fake_video_prediction):
            response = self.client.post(
                "/api/v1/predict/video?frame_samples=3&clip_window_seconds=2.0&motion_priority=true",
                files={"file": ("incident.mp4", b"fake-video-bytes", "video/mp4")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prediction"], "Abuse")
        self.assertEqual(payload["decision_reason"], "prioridad_riesgo")
        self.assertEqual(payload["frames"][0]["top_risk_class"], "Abuse")
        self.assertEqual(payload["frames"][0]["clip_start_frame"], 0)
        self.assertEqual(payload["frames"][0]["clip_end_frame"], 15)
        self.assertEqual(payload["frames"][0]["clip_window_seconds"], 2.0)
        self.assertEqual(payload["frames"][0]["video_fps"], 30.0)
        self.assertEqual(payload["frames"][0]["motion_score"], 0.31)
        self.assertEqual(payload["frames"][0]["motion_intensity"], 0.8)
        self.assertEqual(payload["frames"][0]["sampling_reason"], "movimiento")
        self.assertEqual(payload["frames"][0]["raw_prediction"], "NormalVideos")
        self.assertEqual(payload["frames"][0]["raw_normal_probability"], 0.99)
        self.assertEqual(payload["class_summary"][0]["class_name"], "Abuse")


if __name__ == "__main__":
    unittest.main()
