"""
Unit tests for analyzer.cosine_similarity + the generic-metric path in
analyzer.stats (R2 #8, summary-task original-vs-summary cosine).

The torch/jina forward pass is not exercised here; these cover the data
plumbing and the maths around it, plus the stats wiring that consumes the
generated doc-stats file.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from analyzer.cosine_similarity import (  # noqa: E402
    METRIC_KEY,
    _cosine_rows,
    _read_checkpoint,
    _write_checkpoint,
    load_pairs,
    summarize,
    to_doc_stats,
)
from analyzer.stats import (  # noqa: E402
    _normalize_doc_stats,
    bootstrap_metric_table,
    paired_comparison_table,
)


class TestCosineRows(unittest.TestCase):
    def test_identical_vectors_are_one(self):
        a = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
        sims = _cosine_rows(a, a.copy())
        for s in sims:
            self.assertAlmostEqual(s, 1.0, places=10)

    def test_orthogonal_vectors_are_zero(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        self.assertAlmostEqual(_cosine_rows(a, b)[0], 0.0, places=10)

    def test_opposite_vectors_are_minus_one(self):
        a = np.array([[1.0, 2.0]])
        self.assertAlmostEqual(_cosine_rows(a, -a)[0], -1.0, places=10)

    def test_scale_invariant(self):
        a = np.array([[1.0, 2.0, 3.0]])
        self.assertAlmostEqual(_cosine_rows(a, a * 7.5)[0], 1.0, places=10)

    def test_zero_vector_yields_nan_not_zerodivision(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 1.0]])
        self.assertTrue(math.isnan(_cosine_rows(a, b)[0]))


class TestLoadPairs(unittest.TestCase):
    def _write(self, payload) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return Path(tmp.name)

    def test_reads_id_original_summary(self):
        path = self._write(
            [{"id": 7, "original_text": "orig", "summary_text": "summ", "qa_text": "x"}]
        )
        self.assertEqual(load_pairs(path), [("7", "orig", "summ")])

    def test_skips_missing_or_blank_summary(self):
        path = self._write(
            [
                {"id": 1, "original_text": "a", "summary_text": "b"},
                {"id": 2, "original_text": "a", "summary_text": "   "},
                {"id": 3, "original_text": "a", "summary_text": None},
                {"id": 4, "original_text": "", "summary_text": "b"},
            ]
        )
        self.assertEqual([doc for doc, _, _ in load_pairs(path)], ["1"])

    def test_doc_ids_are_strings(self):
        path = self._write([{"id": 11, "original_text": "a", "summary_text": "b"}])
        self.assertIsInstance(load_pairs(path)[0][0], str)


class TestCheckpoint(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ckpt.json"
            _write_checkpoint(p, {"1": 0.9, "2": 0.8})
            self.assertEqual(_read_checkpoint(p), {"1": 0.9, "2": 0.8})

    def test_missing_file_is_empty(self):
        self.assertEqual(_read_checkpoint(Path("/nonexistent/ckpt.json")), {})

    def test_corrupt_file_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ckpt.json"
            p.write_text("{not json", encoding="utf-8")
            self.assertEqual(_read_checkpoint(p), {})


class TestSummarize(unittest.TestCase):
    def test_counts_low_similarity_tail(self):
        cosines = {"1": 0.95, "2": 0.9, "3": 0.75, "4": 0.5}
        out = summarize(cosines, low_threshold=0.8)
        self.assertEqual(out["n"], 4)
        self.assertEqual(out["n_below_0.8"], 2)
        self.assertAlmostEqual(out["pct_below_0.8"], 50.0)

    def test_empty_input(self):
        self.assertEqual(summarize({}), {"n": 0})


class TestStatsIntegration(unittest.TestCase):
    """The generated doc-stats file must flow through stats.py unchanged."""

    def _write(self, payload) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, tmp)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return Path(tmp.name)

    def test_to_doc_stats_shape_normalizes(self):
        norm = _normalize_doc_stats(to_doc_stats({"1": 0.9}), [METRIC_KEY])
        self.assertEqual(norm, {"1": {"_overall_": {"cosine_sim": 0.9}}})

    def test_flat_cosine_dict_also_normalizes(self):
        norm = _normalize_doc_stats({"1": {"cosine_sim": 0.9}}, [METRIC_KEY])
        self.assertEqual(norm, {"1": {"_overall_": {"cosine_sim": 0.9}}})

    def test_bootstrap_on_cosine_metric(self):
        path = self._write(to_doc_stats({str(i): 0.9 for i in range(50)}))
        table = bootstrap_metric_table(
            path, ["_overall_"], n_boot=200, metrics=[METRIC_KEY]
        )
        res = table["_overall_"][METRIC_KEY]
        self.assertEqual(res.n, 50)
        self.assertAlmostEqual(res.mean, 0.9, places=6)

    def test_paired_cosine_detects_a_real_shift(self):
        a = self._write(to_doc_stats({str(i): 0.70 for i in range(200)}))
        b = self._write(to_doc_stats({str(i): 0.80 for i in range(200)}))
        table = paired_comparison_table(
            a, b, ["_overall_"], n_boot=500, metrics=[METRIC_KEY]
        )
        res = table["_overall_"][METRIC_KEY]
        self.assertEqual(res.n, 200)
        self.assertAlmostEqual(res.delta_mean, 0.10, places=6)
        self.assertLess(res.p_value, 0.05)

    def test_paired_cosine_aligns_on_shared_docs_only(self):
        a = self._write(to_doc_stats({"1": 0.9, "2": 0.8, "3": 0.7}))
        b = self._write(to_doc_stats({"2": 0.85, "3": 0.75, "9": 0.5}))
        table = paired_comparison_table(
            a, b, ["_overall_"], n_boot=200, metrics=[METRIC_KEY]
        )
        self.assertEqual(table["_overall_"][METRIC_KEY].n, 2)

    def test_default_metrics_still_precision_recall_f1(self):
        """Backward compatibility: existing doc_stats files are unaffected."""
        path = self._write(
            {"1": {"_overall_": {"precision": 0.5, "recall": 0.6, "f1": 0.55}}}
        )
        table = bootstrap_metric_table(path, ["_overall_"], n_boot=100)
        self.assertEqual(set(table["_overall_"]), {"precision", "recall", "f1"})
        self.assertAlmostEqual(table["_overall_"]["precision"].mean, 0.5)


if __name__ == "__main__":
    unittest.main()
