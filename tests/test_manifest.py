import tempfile
import unittest
from pathlib import Path

from ucf_crime_recognition.data.manifest import build_manifest


class BuildManifestTests(unittest.TestCase):
    def test_build_manifest_uses_existing_dataset_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            manifest_path = root / "processed" / "manifest.csv"

            train_image = raw_dir / "Training" / "Abuse" / "frame_001.jpg"
            test_image = raw_dir / "Testing" / "NormalVideos" / "frame_002.png"
            train_image.parent.mkdir(parents=True)
            test_image.parent.mkdir(parents=True)
            train_image.touch()
            test_image.touch()

            config_path = root / "pipeline.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[dataset]",
                        f'raw_dir = "{raw_dir}"',
                        f'manifest_path = "{manifest_path}"',
                        "",
                        "[preprocessing]",
                        "test_size = 0.5",
                        "random_state = 42",
                    ]
                ),
                encoding="utf-8",
            )

            manifest = build_manifest(config_path)

            self.assertTrue(manifest_path.exists())
            self.assertEqual(set(manifest["label"]), {"Abuse", "NormalVideos"})
            self.assertEqual(set(manifest["split"]), {"train", "test"})
            split_by_label = dict(zip(manifest["label"], manifest["split"]))
            self.assertEqual(split_by_label["Abuse"], "train")
            self.assertEqual(split_by_label["NormalVideos"], "test")

    def test_build_manifest_rejects_empty_image_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()

            config_path = root / "pipeline.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[dataset]",
                        f'raw_dir = "{raw_dir}"',
                        f'manifest_path = "{root / "manifest.csv"}"',
                        "",
                        "[preprocessing]",
                        "test_size = 0.5",
                        "random_state = 42",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "No images found"):
                build_manifest(config_path)


if __name__ == "__main__":
    unittest.main()
