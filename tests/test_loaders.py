import unittest

import pandas as pd

from ucf_crime_recognition.data.loaders import rebalance_manifest, sample_manifest, split_manifest


class LoaderTests(unittest.TestCase):
    def test_split_manifest_separates_train_and_test_rows(self):
        manifest = pd.DataFrame(
            {
                "path": ["train_a.jpg", "test_a.jpg", "train_b.jpg"],
                "label": ["Abuse", "Abuse", "NormalVideos"],
                "split": ["train", "test", "train"],
            }
        )

        train_manifest, test_manifest = split_manifest(manifest)

        self.assertEqual(train_manifest["path"].tolist(), ["train_a.jpg", "train_b.jpg"])
        self.assertEqual(test_manifest["path"].tolist(), ["test_a.jpg"])

    def test_sample_manifest_caps_total_rows_and_keeps_known_labels(self):
        manifest = pd.DataFrame(
            {
                "path": [f"image_{index}.jpg" for index in range(8)],
                "label": ["Abuse"] * 5 + ["NormalVideos"] * 3,
                "split": ["train"] * 8,
            }
        )

        sampled = sample_manifest(manifest, max_samples=4, random_state=42)

        self.assertLessEqual(len(sampled), 4)
        self.assertEqual(set(sampled["label"]), {"Abuse", "NormalVideos"})

    def test_rebalance_manifest_upsamples_underrepresented_classes(self):
        manifest = pd.DataFrame(
            {
                "path": ["abuse_1.jpg", "normal_1.jpg", "normal_2.jpg", "normal_3.jpg"],
                "label": ["Abuse", "NormalVideos", "NormalVideos", "NormalVideos"],
                "split": ["train"] * 4,
            }
        )

        balanced = rebalance_manifest(
            manifest,
            target_min_per_class=2,
            target_max_per_class=2,
            random_state=42,
        )

        self.assertEqual(balanced["label"].value_counts().to_dict(), {"Abuse": 2, "NormalVideos": 2})


if __name__ == "__main__":
    unittest.main()
