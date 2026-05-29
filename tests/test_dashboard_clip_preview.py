import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from ucf_crime_recognition.services.triage import VIDEO_CLIP_LEN
from ucf_crime_recognition.ui.dashboard import (
    _build_clip_animation,
    _build_clip_contact_sheet,
    _build_clip_thumbnail,
    _clip_effective_fps,
    _rank_event_clips,
)


class DashboardClipPreviewTests(unittest.TestCase):
    def test_build_clip_contact_sheet_uses_video_clip_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for frame_index in range(VIDEO_CLIP_LEN):
                image = Image.new(
                    "RGB",
                    (24, 24),
                    color=(frame_index * 10, frame_index * 5, frame_index * 3),
                )
                image.save(root / f"fight_clip_{frame_index:06d}.png")

            sheet = _build_clip_contact_sheet(root / "fight_clip_000008.png")

            self.assertGreater(sheet.width, 0)
            self.assertGreater(sheet.height, 0)

    def test_build_clip_animation_returns_gif_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for frame_index in range(VIDEO_CLIP_LEN):
                image = Image.new(
                    "RGB",
                    (24, 24),
                    color=(frame_index * 10, frame_index * 5, frame_index * 3),
                )
                image.save(root / f"fight_clip_{frame_index:06d}.png")

            animation = _build_clip_animation(root / "fight_clip_000008.png", fps=8.0)

            self.assertTrue(animation.startswith(b"GIF"))

    def test_build_clip_thumbnail_builds_strip_from_clip_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for frame_index in range(VIDEO_CLIP_LEN):
                image = Image.new(
                    "RGB",
                    (24, 24),
                    color=(frame_index * 10, frame_index * 5, frame_index * 3),
                )
                image.save(root / f"fight_clip_{frame_index:06d}.png")

            thumbnail = _build_clip_thumbnail(root / "fight_clip_000008.png")

            self.assertEqual(thumbnail.size, (240, 150))

    def test_rank_event_clips_can_sort_by_time_or_risk(self):
        event_table = pd.DataFrame(
            [
                {"timestamp_s": 12.0, "top_risk_probability": 0.1, "normal_probability": 0.8},
                {"timestamp_s": 3.0, "top_risk_probability": 0.6, "normal_probability": 0.2, "motion_score": 0.3},
                {"timestamp_s": 8.0, "top_risk_probability": 0.4, "normal_probability": 0.3, "motion_score": 0.7},
            ]
        )

        by_time = _rank_event_clips(event_table, "Tiempo")
        by_risk = _rank_event_clips(event_table, "Mayor riesgo")
        by_motion = _rank_event_clips(event_table, "Movimiento")

        self.assertEqual(by_time["timestamp_s"].tolist(), [3.0, 8.0, 12.0])
        self.assertEqual(by_risk["top_risk_probability"].tolist(), [0.6, 0.4, 0.1])
        self.assertEqual(by_motion["timestamp_s"].tolist()[0], 8.0)

    def test_clip_effective_fps_uses_technical_frame_count_over_clip_duration(self):
        selected_clip = pd.Series({"clip_duration_seconds": 2.0})

        self.assertEqual(_clip_effective_fps(selected_clip), VIDEO_CLIP_LEN / 2.0)


if __name__ == "__main__":
    unittest.main()
