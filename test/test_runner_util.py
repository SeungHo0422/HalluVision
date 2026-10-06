"""
Unit tests for analyzer.baselines.runner_util (R2 #6 execution layer).

The full 200-doc baseline run is ~4.4K judge calls, so resume-after-kill
and retry-on-transient-error are what keep it from being a coin flip.
No API key needed — the "judge" here is a counter.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from analyzer.baselines import runner_util as ru  # noqa: E402


class TestRetry(unittest.TestCase):
    def setUp(self):
        self._delay = ru.BASE_DELAY
        ru.BASE_DELAY = 0.0  # keep the suite fast

    def tearDown(self):
        ru.BASE_DELAY = self._delay

    def test_succeeds_after_transient_failures(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("429 rate limit")
            return "ok"

        self.assertEqual(ru.retry(flaky), "ok")
        self.assertEqual(calls["n"], 3)

    def test_raises_after_max_attempts(self):
        def always_fails():
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            ru.retry(always_fails)


class TestCheckpointResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "scores.json")
        self.docs = [{"id": str(i)} for i in range(6)]

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_skips_cached_docs(self):
        seen_first = []
        ru.run_docs(self.docs[:3], self.out,
                    lambda d: (seen_first.append(d["id"]),
                               {"doc_id": d["id"], "score": 1.0})[1],
                    workers=2)
        self.assertEqual(sorted(seen_first), ["0", "1", "2"])

        seen_second = []
        results = ru.run_docs(self.docs, self.out,
                              lambda d: (seen_second.append(d["id"]),
                                         {"doc_id": d["id"], "score": 1.0})[1],
                              workers=2)
        # only the three new docs get re-scored
        self.assertEqual(sorted(seen_second), ["3", "4", "5"])
        # output holds all six, in the input order the caller passed
        self.assertEqual([r["doc_id"] for r in results],
                         ["0", "1", "2", "3", "4", "5"])

    def test_output_order_is_input_order(self):
        def slow_for_early_docs(d):
            # finish out of order on purpose
            return {"doc_id": d["id"], "score": float(d["id"])}

        results = ru.run_docs(self.docs, self.out, slow_for_early_docs, workers=4)
        self.assertEqual([r["doc_id"] for r in results],
                         [d["id"] for d in self.docs])
        written = json.loads(open(self.out, encoding="utf-8").read())
        self.assertEqual(len(written), 6)

    def test_failing_doc_does_not_kill_the_run(self):
        def sometimes_raises(d):
            if d["id"] == "2":
                raise ValueError("bad doc")
            return {"doc_id": d["id"], "score": 1.0}

        results = ru.run_docs(self.docs, self.out, sometimes_raises, workers=2)
        self.assertEqual(len(results), 6)
        bad = next(r for r in results if r["doc_id"] == "2")
        self.assertIn("failed", bad.get("note", ""))
        self.assertNotIn("score", bad)

    def test_truncated_checkpoint_line_is_ignored(self):
        with open(ru.ckpt_path(self.out), "w", encoding="utf-8") as f:
            f.write(json.dumps({"doc_id": "0", "score": 1.0}) + "\n")
            f.write('{"doc_id": "1", "sco')  # killed mid-write
        done = ru.load_done(self.out)
        self.assertEqual(list(done), ["0"])


if __name__ == "__main__":
    unittest.main()
