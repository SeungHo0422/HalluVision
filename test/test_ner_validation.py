"""
Unit tests for the §2a NER-validation pipeline (Reviewer 2 #2a).

Covers (1) the stratified sampler's precision/recall split and
determinism, and (2) reuse of ``analyzer.agreement`` for the NER
correctness levels via the parameterized ``compute_report``.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from analyzer.agreement import compute_report  # noqa: E402
from analyzer.ner_sampler import LABELS, stratified_sample  # noqa: E402

NER_LEVELS = ["correct", "partial", "incorrect"]


def _fake_original_ner(n_docs: int = 40) -> dict:
    """Each label is present in ~half the docs so both with/without pools
    are large enough for the 70/30 precision/recall split."""
    out = {}
    for i in range(n_docs):
        out[str(i)] = {
            lbl: ([f"{lbl}_ent_{i}"] if (i % 2 == 0) else [])
            for lbl in LABELS
        }
    return out


class TestStratifiedSample(unittest.TestCase):
    def test_counts_and_split(self):
        rows = stratified_sample(_fake_original_ner(), n_per_label=20, seed=42)
        self.assertEqual(len(rows), 20 * len(LABELS))
        for lbl in LABELS:
            cells = [r for r in rows if r["label"] == lbl]
            self.assertEqual(len(cells), 20)
            prec = [r for r in cells if r["kind"] == "precision_check"]
            rec = [r for r in cells if r["kind"] == "recall_check"]
            self.assertEqual(len(prec), 14)  # round(20 * 0.7)
            self.assertEqual(len(rec), 6)
            # precision cells carry the predicted entities; recall cells do not
            self.assertTrue(all(r["predicted_entities"] for r in prec))
            self.assertTrue(all(r["predicted_entities"] == "" for r in rec))

    def test_deterministic_with_seed(self):
        original = _fake_original_ner()
        a = stratified_sample(original, n_per_label=20, seed=42)
        b = stratified_sample(original, n_per_label=20, seed=42)
        self.assertEqual([r["doc_id"] for r in a], [r["doc_id"] for r in b])


class TestAgreementReuseForNER(unittest.TestCase):
    def test_kappa_over_correctness_levels(self):
        rows = [
            {"doc_id": "1", "annotator_a_correct": "correct",   "annotator_b_correct": "correct"},
            {"doc_id": "2", "annotator_a_correct": "incorrect", "annotator_b_correct": "incorrect"},
            {"doc_id": "3", "annotator_a_correct": "partial",   "annotator_b_correct": "partial"},
            {"doc_id": "4", "annotator_a_correct": "correct",   "annotator_b_correct": "partial"},
            {"doc_id": "5", "annotator_a_correct": "incorrect", "annotator_b_correct": "correct"},
        ]
        report = compute_report(
            rows,
            metric_cols=[],
            col_a="annotator_a_correct",
            col_b="annotator_b_correct",
            levels=NER_LEVELS,
        )
        self.assertEqual(report["levels"], NER_LEVELS)
        self.assertEqual(report["n_dual_labeled"], 5)
        # κ must be a real number (the old default columns would give NaN here)
        self.assertEqual(report["overall_kappa"], report["overall_kappa"])
        self.assertIn("correct", report["per_class_kappa"])
        self.assertEqual(report["confusion_matrix"][0][0], 1)  # correct/correct cell

    def test_default_hallucination_behavior_unchanged(self):
        rows = [
            {"doc_id": "1", "label_a": "faithful",     "label_b": "faithful"},
            {"doc_id": "2", "label_a": "hallucinated", "label_b": "hallucinated"},
        ]
        report = compute_report(rows, metric_cols=[])
        self.assertEqual(report["levels"], ["faithful", "partial", "hallucinated"])
        self.assertEqual(report["n_dual_labeled"], 2)


if __name__ == "__main__":
    unittest.main()
