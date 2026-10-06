"""Tests for the QA output normaliser (Reviewer 2 #4)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyzer.qa_format import (  # noqa: E402
    compliance_report,
    normalize_qa_text,
    parse_answers,
)


class TestQAFormat(unittest.TestCase):
    def test_canonical_is_untouched(self):
        text = "{LockBit, no answer, no answer, FakeUpdates, Evil Corp}"
        self.assertEqual(normalize_qa_text(text), text)

    def test_answer_n_lines(self):
        text = (
            "Here are the answers to the questions:\n\n"
            "Answer 1: Ducktail\n\nAnswer 2: No answer\n\nAnswer 3: No answer\n\n"
            "Answer 4: Telegram channel\n\nAnswer 5: Intel 471"
        )
        self.assertEqual(
            parse_answers(text),
            ["Ducktail", "No answer", "No answer", "Telegram channel", "Intel 471"],
        )

    def test_question_echo_with_prefix(self):
        text = "\n".join(
            f"Question {i}: What is the main X covered in this article?\nA{i}"
            for i in range(1, 6)
        )
        self.assertEqual(parse_answers(text), [f"A{i}" for i in range(1, 6)])

    def test_question_echo_without_prefix_keeps_prompt_order(self):
        # Prompt order is MALWARE, SYSTEM, VULNERABILITY, INDICATOR, ORGANIZATION;
        # the answers must come back in that order, not in question-text order.
        text = (
            "What is the main MALWARE covered in this article?\nFlubot\n\n"
            "What is the main SYSTEM covered in this article?\nAndroid\n\n"
            "What is the main VULNERABILITY covered in this article?\nNo answer\n\n"
            "What is the main INDICATOR covered in this article?\nRussian domain\n\n"
            "What is the main ORGANIZATION covered in this article?\nEuropol"
        )
        self.assertEqual(
            parse_answers(text),
            ["Flubot", "Android", "No answer", "Russian domain", "Europol"],
        )

    def test_commas_inside_answers_do_not_break_the_split(self):
        text = "Answer 1: Emotet, Trickbot\nAnswer 2: b\nAnswer 3: c\nAnswer 4: d\nAnswer 5: e"
        out = normalize_qa_text(text)
        self.assertEqual(len([p for p in out.strip("{}").split(",") if p.strip()]), 5)

    def test_truncated_output_is_unrecoverable(self):
        text = "Answer 1: Flubot\n\nAnswer 2: Android\n\nAnswer 3: No ans"
        self.assertIsNone(parse_answers(text))
        self.assertEqual(normalize_qa_text(text), "")

    def test_compliance_report_counts(self):
        records = [
            {"qa_text": "{a, b, c, d, e}"},
            {"qa_text": "Answer 1: a\nAnswer 2: b\nAnswer 3: c\nAnswer 4: d\nAnswer 5: e"},
            {"qa_text": "sorry, I cannot answer"},
        ]
        self.assertEqual(
            compliance_report(records),
            {"n": 3, "canonical": 1, "recovered": 1, "unrecoverable": 1},
        )


if __name__ == "__main__":
    unittest.main()
