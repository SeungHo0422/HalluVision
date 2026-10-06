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
    _leave_one_out_auc,
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


class TestEqualWeightScorer(unittest.TestCase):
    def test_equal_weights_are_parameter_free(self):
        X, y = _gen_synthetic(60)
        signs = [-1.0, 1.0]  # cosine lower=halluc, freq higher=halluc
        model = train_model(X, y, ["cosine_sim", "freq_distance"], signs)
        self.assertEqual(model.method, "equal")
        # weights are fixed at 1/d, not learned
        self.assertEqual(model.weights, [0.5, 0.5])
        self.assertEqual(model.intercept, 0.0)

    def test_separates_halluc_from_faithful(self):
        X, y = _gen_synthetic(60)
        signs = [-1.0, 1.0]
        model = train_model(X, y, ["cosine_sim", "freq_distance"], signs)
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

    def test_logistic_comparison_does_not_beat_equal_much(self):
        X, y = _gen_synthetic(80)
        signs = [-1.0, 1.0]
        cv_eq = cross_validate(X, y, ["a", "b"], signs, folds=5, seed=42, method="equal")
        cv_lg = cross_validate(X, y, ["a", "b"], signs, folds=5, seed=42, method="logistic")
        # both discriminate well; learned weights give no large AUC advantage
        self.assertGreater(cv_eq["auc_mean"], 0.9)
        self.assertGreater(cv_lg["auc_mean"], 0.9)
        self.assertLess(cv_lg["auc_mean"] - cv_eq["auc_mean"], 0.05)


class TestPredict(unittest.TestCase):
    def test_predict_proba_in_range(self):
        X, y = _gen_synthetic(40)
        model = train_model(X, y, ["a", "b"], [-1.0, 1.0])
        for x in X:
            p = _predict_proba(model, x)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


class TestLeaveOneOut(unittest.TestCase):
    def test_loo_reports_each_signal(self):
        X, y = _gen_synthetic(60)
        loo = _leave_one_out_auc(X, y, ["a", "b"], [-1.0, 1.0])
        self.assertIn("a", loo)
        self.assertIn("b", loo)
        for auc in loo.values():
            self.assertGreaterEqual(auc, 0.0)
            self.assertLessEqual(auc, 1.0)


if __name__ == "__main__":
    unittest.main()
