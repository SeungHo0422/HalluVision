"""
Unit tests for §10 minor-experiment scaffolds (R2 minor).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from analyzer.qa_error_taxonomy import _classify, _parse_gold, classify_rows  # noqa: E402
from analyzer.role_ablation import _doc_prf, _label_prf, aggregate, run_ablation  # noqa: E402
from analyzer.case_study_selector import select  # noqa: E402


class TestQAErrorTaxonomy(unittest.TestCase):
    def test_format_error(self):
        self.assertEqual(_classify("", ["abc"], None), "format_error")
        self.assertEqual(_classify("no answer", ["abc"], None), "format_error")

    def test_surface_variance(self):
        self.assertEqual(
            _classify("LockBit ransomware", ["lockbit", "evil corp"], None),
            "surface_variance",
        )

    def test_ner_recall_miss(self):
        text = "the malware Zeus was deployed against Windows hosts in 2022."
        self.assertEqual(_classify("Zeus", ["lockbit"], text), "ner_recall_miss")

    def test_ioc_token_form(self):
        # gold has dotted CVE, pred has dashless version -> punct mismatch
        self.assertEqual(
            _classify("CVE20241234", ["CVE-2024-1234"], None),
            "ioc_token_form",
        )

    def test_llm_hallucination(self):
        self.assertEqual(
            _classify("AcmeBot 9000", ["lockbit"], "totally unrelated text"),
            "llm_hallucination",
        )

    def test_parse_gold_handles_list_repr(self):
        self.assertEqual(_parse_gold("['a', 'b']"), ["a", "b"])
        self.assertEqual(_parse_gold(""), [])


class StubClient:
    def __init__(self, response: str):
        self.response = response
        self.prompts = []

    def chat_completion(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return self.response

    def simple_prompt(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.response


class TestRoleAblation(unittest.TestCase):
    def test_label_prf_subset_hit(self):
        p, r, f = _label_prf("LockBit ransomware exfiltrates data", ["lockbit"])
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(f, 1.0)

    def test_doc_prf_overall(self):
        gold = {"Malware": ["lockbit"], "Organization": ["evil corp"]}
        out = _doc_prf("LockBit ransomware run by Evil Corp", gold)
        self.assertGreater(out["_overall_"]["f1"], 0.9)

    def test_run_ablation_invokes_three_variants(self):
        client = StubClient("LockBit and Evil Corp summary")
        docs = [{"id": 1, "original_text": "LockBit by Evil Corp"}]
        gold = {"1": {"Malware": ["lockbit"], "Organization": ["evil corp"]}}
        rows = run_ablation(docs, gold, client, model="stub", min_w=10, max_w=20)
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]["variants"].keys()), {"v0", "v1", "v2"})
        # 3 prompts were sent
        self.assertEqual(len(client.prompts), 3)
        agg = aggregate(rows)
        for v in ("v0", "v1", "v2"):
            self.assertGreater(agg[v]["f1"], 0.5)


class TestCaseStudySelector(unittest.TestCase):
    def test_select_stratifies_by_error_type(self):
        classified = [
            {"doc_id": "1", "label": "Malware", "error_type": "format_error",
             "prediction": "", "gold_entities": ["a"]},
            {"doc_id": "2", "label": "Indicator", "error_type": "format_error",
             "prediction": "", "gold_entities": ["b"]},
            {"doc_id": "3", "label": "Malware", "error_type": "llm_hallucination",
             "prediction": "x", "gold_entities": ["c"]},
            {"doc_id": "4", "label": "Indicator", "error_type": "llm_hallucination",
             "prediction": "y", "gold_entities": ["d"]},
            {"doc_id": "5", "label": "Malware", "error_type": "surface_variance",
             "prediction": "z", "gold_entities": ["e"]},
        ]
        picks = select(classified, per_bucket=1, seed=42)
        self.assertEqual(len(picks), 3)
        self.assertEqual(
            {p["error_type"] for p in picks},
            {"format_error", "llm_hallucination", "surface_variance"},
        )


if __name__ == "__main__":
    unittest.main()
