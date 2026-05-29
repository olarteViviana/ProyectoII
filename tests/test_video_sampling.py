import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ucf_crime_recognition.features.engineering import _clip_frame_paths, _frame_identity
from ucf_crime_recognition.services.triage import (
    CLIP_WINDOW_SECONDS,
    VIDEO_CLIP_LEN,
    _clip_window_indexes,
    _motion_event_ranges,
    _motion_frame_scores,
    _select_motion_event_anchor_indexes,
    export_video_segment,
    extract_sampled_frames,
)


class VideoSamplingTests(unittest.TestCase):
    def test_clip_window_indexes_cover_context_around_anchor(self):
        indexes = _clip_window_indexes(
            anchor_frame=20,
            total_frames=80,
            fps=10.0,
            clip_len=VIDEO_CLIP_LEN,
            clip_window_seconds=CLIP_WINDOW_SECONDS,
        )

        self.assertEqual(len(indexes), VIDEO_CLIP_LEN)
        self.assertIn(20, indexes)
        self.assertGreater(max(indexes) - min(indexes) + 1, VIDEO_CLIP_LEN)

    def test_extracted_frames_share_prefix_used_by_clip_reconstruction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "fight_clip.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (24, 24),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV VideoWriter is not available in this environment")

            for index in range(40):
                frame = np.full((24, 24, 3), index % 255, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            frames = extract_sampled_frames(video_path, frame_samples=3, output_dir=root / "frames")

            self.assertGreaterEqual(len(frames), 2)
            self.assertEqual(frames[0]["frame_index"], 0)
            self.assertEqual(frames[-1]["frame_index"], 39)
            identities = [_frame_identity(frame["image_path"]) for frame in frames]
            self.assertTrue(all(identity is not None for identity in identities))
            self.assertEqual({identity[0] for identity in identities}, {"fight_clip"})
            self.assertTrue(all(frame["clip_window_seconds"] == CLIP_WINDOW_SECONDS for frame in frames))
            self.assertTrue(all(frame["clip_duration_seconds"] > 0 for frame in frames))

            middle_frame = frames[len(frames) // 2]
            clip_paths = _clip_frame_paths(middle_frame["image_path"], clip_len=VIDEO_CLIP_LEN)
            clip_numbers = [_frame_identity(path)[1] for path in clip_paths]

            self.assertEqual(len({path.name for path in clip_paths}), VIDEO_CLIP_LEN)
            self.assertIn(middle_frame["frame_index"], clip_numbers)

    def test_motion_priority_samples_high_motion_regions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "motion_clip.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (32, 32),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV VideoWriter is not available in this environment")

            for index in range(60):
                frame = np.zeros((32, 32, 3), dtype=np.uint8)
                if 30 <= index < 45:
                    x = (index - 30) % 16
                    frame[8:24, x : x + 8] = 255
                writer.write(frame)
            writer.release()

            scores = _motion_frame_scores(video_path, total_frames=60, fps=10.0)
            frames = extract_sampled_frames(
                video_path,
                frame_samples=8,
                motion_priority=True,
                output_dir=root / "frames",
            )

            self.assertGreater(max(scores.values()), 0.0)
            self.assertTrue(any("movimiento" in frame["sampling_reason"] for frame in frames))
            self.assertTrue(any(25 <= frame["frame_index"] <= 45 for frame in frames))
            self.assertTrue(all("motion_score" in frame for frame in frames))

    def test_motion_events_can_yield_multiple_anchors_in_one_active_region(self):
        scores = {
            0: 0.0,
            10: 0.0,
            20: 0.05,
            30: 0.4,
            40: 0.5,
            50: 0.45,
            60: 0.47,
            70: 0.44,
            80: 0.43,
            90: 0.04,
            100: 0.0,
        }

        events = _motion_event_ranges(scores, fps=10.0)
        anchors = _select_motion_event_anchor_indexes(
            scores,
            anchor_count=3,
            fps=10.0,
            clip_window_seconds=2.0,
        )

        self.assertEqual(len(events), 1)
        self.assertGreaterEqual(len(anchors), 2)
        self.assertTrue(all(30 <= anchor <= 80 for anchor in anchors[:2]))
        self.assertGreaterEqual(min(abs(a - b) for a, b in zip(anchors, anchors[1:])), 17)

    def test_export_video_segment_writes_original_frames_between_clip_bounds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "fight_clip.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (32, 24),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV VideoWriter is not available in this environment")

            for index in range(30):
                frame = np.full((24, 32, 3), index % 255, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            try:
                segment_path = export_video_segment(video_path, 5, 15, output_dir=root / "segments")
            except ValueError as error:
                self.skipTest(str(error))

            self.assertTrue(segment_path.exists())
            capture = cv2.VideoCapture(str(segment_path))
            self.assertTrue(capture.isOpened())
            self.assertGreaterEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            self.assertGreater(float(capture.get(cv2.CAP_PROP_FPS) or 0), 0)
            capture.release()


if __name__ == "__main__":
    unittest.main()
