"""
Unit tests for analyzer.agreement (Reviewer 2 #3).

Exercises Cohen's kappa, confusion matrix, Spearman ρ, AUC, and the
markdown renderer using a tiny hand-crafted dataset where the
expected outcomes can be checked by inspection.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from analyzer.agreement import (  # noqa: E402
    _cohen_kappa,
    _roc_auc,
    _spearman,
    compute_report,
    render_markdown,
)


class TestKappaAndStats(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        labels = ["faithful", "partial", "hallucinated"]
        a = ["faithful", "partial", "hallucinated", "faithful"]
        b = list(a)
        self.assertAlmostEqual(_cohen_kappa(a, b, labels), 1.0)

    def test_random_agreement_is_near_zero(self):
        labels = ["yes", "no"]
        a = ["yes", "no", "yes", "no", "yes", "no", "yes", "no"]
        b = ["no", "yes", "no", "yes", "no", "yes", "no", "yes"]
        self.assertLess(_cohen_kappa(a, b, labels), 0.0)

    def test_spearman_monotone(self):
        rho = _spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(rho, 1.0)

    def test_auc_perfect_separation(self):
        # higher score should be the positive class
        auc = _roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
        self.assertAlmostEqual(auc, 1.0)


class TestReport(unittest.TestCase):
    def _make_csv(self, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        f.close()
        return f.name

    def test_end_to_end_report(self):
        rows = [
            {"doc_id": "1", "label_a": "faithful",     "label_b": "faithful",     "cosine_sim": "0.95"},
            {"doc_id": "2", "label_a": "partial",      "label_b": "partial",      "cosine_sim": "0.7"},
            {"doc_id": "3", "label_a": "hallucinated", "label_b": "hallucinated", "cosine_sim": "0.3"},
            {"doc_id": "4", "label_a": "hallucinated", "label_b": "partial",      "cosine_sim": "0.4"},
            {"doc_id": "5", "label_a": "faithful",     "label_b": "partial",      "cosine_sim": "0.85"},
        ]
        report = compute_report(rows, metric_cols=["cosine_sim"])
        self.assertEqual(report["n_rows"], 5)
        self.assertEqual(report["n_dual_labeled"], 5)
        self.assertGreater(report["overall_kappa"], 0.0)
        # cosine_sim should be inversely correlated with hallucination
        cs = report["metric_correlation"]["cosine_sim"]
        self.assertLess(cs["spearman_vs_majority"], 0.0)
        self.assertLess(cs["auc_halluc_vs_faithful"], 0.5)

    def test_render_markdown_smoke(self):
        rows = [
            {"doc_id": "1", "label_a": "faithful", "label_b": "faithful"},
            {"doc_id": "2", "label_a": "partial",  "label_b": "hallucinated"},
        ]
        report = compute_report(rows, metric_cols=[])
        md = render_markdown(report, source_path="dummy.csv")
        self.assertIn("Cohen's κ", md)
        self.assertIn("Confusion matrix", md)


if __name__ == "__main__":
    unittest.main()
