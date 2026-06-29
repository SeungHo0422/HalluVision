"""
Unit tests for analyzer.combined_scorer (R2#1).
"""

from __future__ import annotations

import os
import random
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from analyzer.combined_scorer import (  # noqa: E402
    _predict_proba,
    cross_validate,
    train_model,
)


def _gen_synthetic(n: int, seed: int = 42):
    """Two features with opposite directionality."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n // 2):
        # Hallucinated: cosine low (~0.2), freq high (~0.8)
        X.append([rng.gauss(0.2, 0.05), rng.gauss(0.8, 0.05)])
        y.append(1)
    for _ in range(n - n // 2):
        X.append([rng.gauss(0.9, 0.05), rng.gauss(0.1, 0.05)])
        y.append(0)
    return X, y


class TestTrainModel(unittest.TestCase):
    def test_separable_two_features(self):
        X, y = _gen_synthetic(60)
        signs = [-1.0, 1.0]  # cosine lower=halluc, freq higher=halluc
        model = train_model(X, y, ["cosine_sim", "freq_distance"], signs)
        # both signed coefficients should be positive (drive halluc up)
        self.assertGreater(model.coefficients[0], 0.0)
        self.assertGreater(model.coefficients[1], 0.0)
        # predict_proba sanity check: hallucinated example → high prob
        p_halluc = _predict_proba(model, [0.2, 0.8])
        p_faithful = _predict_proba(model, [0.9, 0.1])
        self.assertGreater(p_halluc, p_faithful)
        self.assertGreater(p_halluc, 0.5)
        self.assertLess(p_faithful, 0.5)

    def test_cv_high_auc_on_separable(self):
        X, y = _gen_synthetic(80)
        signs = [-1.0, 1.0]
        cv = cross_validate(X, y, ["a", "b"], signs, folds=5, seed=42)
        self.assertGreater(cv["auc_mean"], 0.9)
        self.assertGreater(cv["f1_mean"], 0.8)
        self.assertEqual(cv["n_folds"], 5)


class TestPredict(unittest.TestCase):
    def test_predict_proba_in_range(self):
        X, y = _gen_synthetic(40)
        model = train_model(X, y, ["a", "b"], [-1.0, 1.0])
        for x in X:
            p = _predict_proba(model, x)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


if __name__ == "__main__":
    unittest.main()
