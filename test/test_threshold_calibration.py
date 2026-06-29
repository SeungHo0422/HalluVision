"""
Unit tests for analyzer.threshold_calibration (R2#7).
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from analyzer.threshold_calibration import (  # noqa: E402
    calibrate,
    cross_validate,
    fit_threshold,
)


class TestFitThreshold(unittest.TestCase):
    def test_perfectly_separable_lower_means_halluc(self):
        # cosine_sim-style: low scores → hallucinated
        scores = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95]
        labels = [1,   1,   1,   1,   0,   0,   0,   0]
        fit = fit_threshold(scores, labels, "auto")
        self.assertEqual(fit.direction, "lower")
        self.assertAlmostEqual(fit.auc, 1.0)
        self.assertAlmostEqual(fit.f1, 1.0)
        self.assertAlmostEqual(fit.youden_j, 1.0)

    def test_perfectly_separable_higher_means_halluc(self):
        # freq_distance-style: high scores → hallucinated
        scores = [0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95]
        labels = [0,    0,   0,    0,   1,   1,    1,   1]
        fit = fit_threshold(scores, labels, "auto")
        self.assertEqual(fit.direction, "higher")
        self.assertAlmostEqual(fit.f1, 1.0)

    def test_explicit_direction_respected(self):
        scores = [0.1, 0.2, 0.3, 0.9]
        labels = [1, 0, 1, 0]
        fit = fit_threshold(scores, labels, "higher")
        self.assertEqual(fit.direction, "higher")


class TestCrossValidate(unittest.TestCase):
    def test_cv_returns_finite_on_balanced_set(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        p, r, f1 = cross_validate(scores, labels, "lower", k=5, seed=42)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        self.assertGreaterEqual(f1, 0.0)


class TestCalibrate(unittest.TestCase):
    def test_calibrate_skips_metric_with_one_class(self):
        rows = [
            {"label_a": "faithful", "label_b": "faithful", "cosine_sim": "0.9"},
            {"label_a": "faithful", "label_b": "faithful", "cosine_sim": "0.95"},
        ]
        out = calibrate(rows, ["cosine_sim"], {}, folds=2, seed=42)
        self.assertEqual(len(out), 1)
        self.assertIn("note", out[0])

    def test_calibrate_picks_direction_and_threshold(self):
        rows = [
            {"label_a": "hallucinated", "label_b": "hallucinated", "cosine_sim": "0.1"},
            {"label_a": "hallucinated", "label_b": "partial",      "cosine_sim": "0.2"},
            {"label_a": "partial",      "label_b": "hallucinated", "cosine_sim": "0.3"},
            {"label_a": "faithful",     "label_b": "faithful",     "cosine_sim": "0.8"},
            {"label_a": "faithful",     "label_b": "faithful",     "cosine_sim": "0.9"},
            {"label_a": "faithful",     "label_b": "faithful",     "cosine_sim": "0.95"},
        ]
        out = calibrate(rows, ["cosine_sim"], {"cosine_sim": "lower"}, folds=3, seed=42)
        self.assertEqual(out[0]["direction"], "lower")
        self.assertGreater(out[0]["fit_f1"], 0.7)


if __name__ == "__main__":
    unittest.main()
