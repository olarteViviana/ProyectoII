import unittest

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ucf_crime_recognition.models.candidates import RiskAwareClassifier, build_model


class FixedProbabilityEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, probabilities=None):
        self.probabilities = probabilities

    def fit(self, x, y):
        self.classes_ = np.asarray(["Abuse", "NormalVideos"])
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        return np.asarray(self.probabilities, dtype=float)[: x.shape[0]]


class CandidateModelTests(unittest.TestCase):
    def test_risk_aware_classifier_switches_from_normal_to_close_risk_score(self):
        x = np.zeros((1, 2))
        y = np.asarray(["NormalVideos"])
        estimator = FixedProbabilityEstimator(probabilities=[[0.86, 0.90]])
        classifier = RiskAwareClassifier(estimator, normal_switch_ratio=0.9)

        classifier.fit(x, y)

        self.assertEqual(classifier.predict(x).tolist(), ["Abuse"])

    def test_build_model_rejects_unknown_candidate_name(self):
        config = {
            "model": {"max_iter": 100, "class_weight": None},
            "preprocessing": {"random_state": 42},
        }

        with self.assertRaisesRegex(ValueError, "Unknown model candidate"):
            build_model("not_a_model", config)


if __name__ == "__main__":
    unittest.main()
