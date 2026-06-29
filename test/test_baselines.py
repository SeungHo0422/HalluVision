"""
Unit tests for analyzer.baselines.* (Reviewer 2 #6).

Uses a stub LLM client so no API key / GPU is needed. Tests cover
FactScore decomposition + verification flow, QAGS QG → QA → agreement
flow, and HalluVision-vs-baseline correlation/contingency math.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from analyzer.baselines import factscore_runner, qags_runner  # noqa: E402
from analyzer.baselines.baseline_compare import compare, _spearman  # noqa: E402
from analyzer.baselines.subset import pick_subset  # noqa: E402


class StubClient:
    """Tiny BaseLLMClient stand-in that returns scripted responses."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[str] = []

    def chat_completion(self, messages, **kwargs):
        prompt = messages[-1]["content"] if messages else ""
        self.calls.append(prompt)
        return self.responses.pop(0) if self.responses else ""

    def simple_prompt(self, prompt: str, **kwargs):
        self.calls.append(prompt)
        return self.responses.pop(0) if self.responses else ""


class TestFactScoreRunner(unittest.TestCase):
    def test_score_doc_basic(self):
        # 2 claims, one supported, one unsupported -> 0.5
        client = StubClient([
            "ScanBox is a JavaScript framework.\nScanBox targets banks.",
            "SUPPORTED",
            "UNSUPPORTED",
        ])
        doc = {
            "id": 0,
            "summary_text": "ScanBox is a JS framework. It targets banks.",
            "original_text": "ScanBox is a JavaScript reconnaissance framework.",
        }
        out = factscore_runner.score_doc(doc, client, judge_model="stub")
        self.assertAlmostEqual(out["factscore"], 0.5)
        self.assertEqual(out["n_claims"], 2)
        self.assertEqual(out["verdicts"], ["supported", "unsupported"])

    def test_score_doc_empty_summary_safe(self):
        client = StubClient([])
        out = factscore_runner.score_doc(
            {"id": 7, "summary_text": "", "original_text": "abc"},
            client, judge_model="stub")
        self.assertIsNone(out["factscore"])
        self.assertIn("missing", out.get("note", ""))


class TestQAGSRunner(unittest.TestCase):
    def test_score_doc_agreement(self):
        # 2 questions. (a) answers match exactly, (b) answers diverge.
        client = StubClient([
            "What is ScanBox?\nWho is the threat actor?",
            "ScanBox is a keylogger",  # answer from summary q1
            "ScanBox is a keylogger",  # answer from source q1
            "TA423",                    # summary q2
            "Lazarus Group",            # source q2 (disagrees)
        ])
        doc = {
            "id": 1,
            "summary_text": "ScanBox keylogger by TA423.",
            "original_text": "ScanBox keylogger attributed to TA423 / Red Ladon.",
        }
        out = qags_runner.score_doc(doc, n_questions=2, judge=client, judge_model="stub")
        self.assertEqual(out["n_questions"], 2)
        self.assertIsNotNone(out["qags"])
        # first pair agrees fully (token-F1 = 1.0), second pair disagrees
        self.assertGreaterEqual(out["qags"], 0.4)
        self.assertLessEqual(out["qags"], 0.6)


class TestBaselineCompare(unittest.TestCase):
    def test_compare_lower_means_halluc(self):
        # cosine_sim lower → halluc, factscore lower → halluc → expect positive ρ
        hv = {str(i): {"cosine_sim": v} for i, v in enumerate([0.9, 0.8, 0.5, 0.3, 0.1])}
        factscore = {str(i): v for i, v in enumerate([0.95, 0.9, 0.6, 0.4, 0.05])}
        corr, cont = compare(hv, factscore, qags={}, hv_directions={"cosine_sim": "lower"})
        self.assertEqual(len(corr), 1)
        self.assertAlmostEqual(corr[0]["spearman"], 1.0)
        self.assertEqual(corr[0]["n"], 5)
        c = cont[0]["contingency"]
        # both metrics median-split: top half of each → halluc
        self.assertEqual(c["agree_halluc"] + c["agree_faithful"],
                         5 - c["halluvision_only"] - c["baseline_only"])


class TestPickSubset(unittest.TestCase):
    def test_subset_prioritizes_human_overlap(self):
        docs = [
            {"id": i, "summary_text": "s", "qa_text": "q", "original_text": "o"}
            for i in range(50)
        ]
        human = {"5", "10", "15"}
        out = pick_subset(docs, human, total=10, seed=42)
        self.assertEqual(len(out), 10)
        ids = [str(d["id"]) for d in out]
        self.assertTrue(human.issubset(set(ids)))


if __name__ == "__main__":
    unittest.main()
