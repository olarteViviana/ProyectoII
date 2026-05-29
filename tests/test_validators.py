import unittest

import pandas as pd

from ucf_crime_recognition.data.validators import validate_manifest


class ValidateManifestTests(unittest.TestCase):
    def test_accepts_manifest_with_required_columns_and_two_classes(self):
        manifest = pd.DataFrame(
            {
                "path": ["a.jpg", "b.jpg"],
                "label": ["Abuse", "NormalVideos"],
                "split": ["train", "test"],
            }
        )

        validated = validate_manifest(manifest)

        self.assertIs(validated, manifest)

    def test_rejects_manifest_missing_required_columns(self):
        manifest = pd.DataFrame({"path": ["a.jpg"], "label": ["Abuse"]})

        with self.assertRaisesRegex(ValueError, "split"):
            validate_manifest(manifest)

    def test_rejects_empty_manifest(self):
        manifest = pd.DataFrame(columns=["path", "label", "split"])

        with self.assertRaisesRegex(ValueError, "empty"):
            validate_manifest(manifest)

    def test_rejects_manifest_with_single_class(self):
        manifest = pd.DataFrame(
            {
                "path": ["a.jpg", "b.jpg"],
                "label": ["Abuse", "Abuse"],
                "split": ["train", "test"],
            }
        )

        with self.assertRaisesRegex(ValueError, "at least two classes"):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
