"""
Smoke tests for LlamaClient (Reviewer 2 #4, Reviewer 3) — vLLM HTTP
and Ollama HTTP backends are exercised with a mocked ``requests.post``;
the HF in-process backend is exercised with a stub pipeline so the test
does not require torch/transformers installs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

# Make `config.settings` importable without going through final_pipeline/__init__.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from llm_proxy.llama_client import LlamaClient, _build_llama_prompt  # noqa: E402


class TestLlamaPromptTemplate(unittest.TestCase):
    def test_system_then_user_template(self):
        messages = [
            {"role": "system", "content": "You are a CTI expert."},
            {"role": "user", "content": "Summarize this."},
        ]
        prompt = _build_llama_prompt(messages)
        self.assertIn("<|begin_of_text|>", prompt)
        self.assertIn("<|start_header_id|>system<|end_header_id|>", prompt)
        self.assertIn("You are a CTI expert.", prompt)
        self.assertIn("Summarize this.", prompt)
        self.assertTrue(prompt.rstrip().endswith(
            "<|start_header_id|>assistant<|end_header_id|>"
        ) or "<|start_header_id|>assistant<|end_header_id|>" in prompt)


class TestLlamaClientVLLM(unittest.TestCase):
    def setUp(self):
        self.manifest = tempfile.NamedTemporaryFile(
            "w+", suffix=".jsonl", delete=False
        ).name
        self.client = LlamaClient(
            vllm_host="http://localhost:8000",
            manifest_path=self.manifest,
        )

    def tearDown(self):
        if os.path.exists(self.manifest):
            os.remove(self.manifest)

    def test_vllm_call_and_manifest(self):
        fake_resp = mock.MagicMock()
        fake_resp.raise_for_status = mock.MagicMock()
        fake_resp.json.return_value = {
            "id": "cmpl-xyz",
            "choices": [{"message": {"content": "ok"}}],
        }
        with mock.patch("requests.post", return_value=fake_resp) as p:
            text = self.client.chat_completion(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "u"},
                ],
                temperature=0.0,
                top_p=1.0,
                seed=42,
                max_tokens=8,
            )
        self.assertEqual(text, "ok")
        called_url = p.call_args[0][0]
        self.assertTrue(called_url.endswith("/v1/chat/completions"))
        # manifest record was appended with sha hashes + backend tag
        with open(self.manifest) as f:
            rec = json.loads(f.readline())
        self.assertEqual(rec["backend"], "vllm")
        self.assertEqual(len(rec["system_msg_hash"]), 64)
        self.assertEqual(rec["response_id"], "cmpl-xyz")


class TestLlamaClientOllama(unittest.TestCase):
    def test_ollama_call_shape(self):
        fake_resp = mock.MagicMock()
        fake_resp.raise_for_status = mock.MagicMock()
        fake_resp.json.return_value = {
            "message": {"content": "hello"},
            "created_at": "2025-01-01T00:00:00Z",
        }
        client = LlamaClient(
            ollama_host="http://localhost:11434",
            manifest_path=tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name,
        )
        with mock.patch("requests.post", return_value=fake_resp) as p:
            out = client.chat_completion(
                [{"role": "user", "content": "hi"}], temperature=0.7, top_p=0.9
            )
        self.assertEqual(out, "hello")
        payload = p.call_args.kwargs["json"]
        self.assertIn("options", payload)
        self.assertAlmostEqual(payload["options"]["temperature"], 0.7)


class TestLlamaClientHFStub(unittest.TestCase):
    def test_hf_backend_uses_pipeline_output(self):
        client = LlamaClient(
            manifest_path=tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name,
        )

        def fake_pipe(prompt, **kw):
            return [{"generated_text": "stubbed response"}]

        client._hf_pipe = fake_pipe  # bypass torch / transformers entirely
        out = client.chat_completion(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
            ],
            temperature=0.0,
        )
        self.assertEqual(out, "stubbed response")


if __name__ == "__main__":
    unittest.main()
