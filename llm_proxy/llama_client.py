"""
Llama-3-8B-Instruct backend (Reviewer 2 #4, Reviewer 3).

Three backends in priority order — chosen via env vars / kwargs:

1. **vLLM HTTP** (``VLLM_HOST`` env or ``vllm_host`` kwarg). OpenAI-
   compatible ``/v1/chat/completions`` endpoint.
2. **Ollama HTTP** (``OLLAMA_HOST`` env or ``ollama_host`` kwarg).
3. **HuggingFace ``transformers``** in-process. Lazy import so the
   module loads without a GPU. ``max_new_tokens`` is bounded by the
   Llama-3 8k context budget.

Per-call manifest is appended to the same JSONL the GPTClient uses
(``pipeline_output/run_manifest.jsonl`` by default), so the
reproducibility audit covers all backends uniformly.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from config.settings import Settings
except ImportError:  # pragma: no cover
    from final_pipeline.config.settings import Settings  # type: ignore


LLAMA_CHAT_TEMPLATE_SYSTEM = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|>"
)
LLAMA_CHAT_TEMPLATE_USER = (
    "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


def _build_llama_prompt(messages: List[Dict[str, str]]) -> str:
    sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    prompt = ""
    if sys_msg:
        prompt += LLAMA_CHAT_TEMPLATE_SYSTEM.format(system=sys_msg)
    else:
        prompt += "<|begin_of_text|>"
    for u in user_msgs:
        prompt += LLAMA_CHAT_TEMPLATE_USER.format(user=u)
    return prompt


class LlamaClient:
    """Thin client implementing :class:`BaseLLMClient` for Llama-3-8B."""

    model: str

    def __init__(
        self,
        model: str = "llama-3-8b-instruct",
        vllm_host: Optional[str] = None,
        ollama_host: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ):
        self.model = model
        self.snapshot_model = Settings.resolve_model_snapshot(model)
        self.vllm_host = vllm_host or os.getenv("VLLM_HOST")
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST")
        self.manifest_path = manifest_path or os.getenv(
            "LLM_RUN_MANIFEST", "pipeline_output/run_manifest.jsonl"
        )
        self._hf_pipe = None  # set lazily

    # ─── public API ───────────────────────────────────────────────────────

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = 42,
        max_tokens: Optional[int] = 512,
        record_manifest: bool = True,
        **kwargs: Any,
    ) -> str:
        logical = model or self.model
        snapshot = Settings.resolve_model_snapshot(logical)

        if self.vllm_host:
            text, response_id = self._call_vllm(
                messages, snapshot, temperature, top_p, seed, max_tokens
            )
        elif self.ollama_host:
            text, response_id = self._call_ollama(
                messages, snapshot, temperature, top_p, seed, max_tokens
            )
        else:
            text, response_id = self._call_hf(
                messages, snapshot, temperature, top_p, seed, max_tokens
            )

        if record_manifest:
            self._append_manifest(
                messages=messages,
                model_logical=logical,
                model_snapshot=snapshot,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                response_id=response_id,
            )
        return text

    def simple_prompt(self, prompt: str, **kwargs: Any) -> str:
        return self.chat_completion([{"role": "user", "content": prompt}], **kwargs)

    # ─── backends ─────────────────────────────────────────────────────────

    def _call_vllm(self, messages, snapshot, temperature, top_p, seed, max_tokens):
        import requests
        payload = {
            "model": snapshot,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        r = requests.post(f"{self.vllm_host.rstrip('/')}/v1/chat/completions",
                          json=payload, timeout=300)
        r.raise_for_status()
        body = r.json()
        text = body["choices"][0]["message"]["content"]
        return text, body.get("id")

    def _call_ollama(self, messages, snapshot, temperature, top_p, seed, max_tokens):
        import requests
        payload = {
            "model": snapshot,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens or -1,
                **({"seed": seed} if seed is not None else {}),
            },
        }
        r = requests.post(f"{self.ollama_host.rstrip('/')}/api/chat",
                          json=payload, timeout=600)
        r.raise_for_status()
        body = r.json()
        text = (body.get("message") or {}).get("content", "")
        return text, body.get("created_at")

    def _call_hf(self, messages, snapshot, temperature, top_p, seed, max_tokens):
        if self._hf_pipe is None:
            self._hf_pipe = self._load_hf_pipeline(snapshot)
        prompt = _build_llama_prompt(messages)
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_tokens or 512,
            "do_sample": do_sample,
            "return_full_text": False,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        if seed is not None:
            try:
                import torch  # type: ignore
                torch.manual_seed(seed)
            except ImportError:
                pass
        out = self._hf_pipe(prompt, **gen_kwargs)
        text = out[0]["generated_text"] if out else ""
        return text.strip(), None

    def _load_hf_pipeline(self, snapshot: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch  # type: ignore

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        tok = AutoTokenizer.from_pretrained(snapshot)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, torch_dtype=dtype, device_map=device_map
        )
        return pipeline("text-generation", model=model, tokenizer=tok)

    # ─── manifest (mirrors GPTClient) ─────────────────────────────────────

    def _append_manifest(
        self,
        *,
        messages: List[Dict[str, str]],
        model_logical: str,
        model_snapshot: str,
        temperature: float,
        top_p: float,
        seed: Optional[int],
        response_id: Optional[str],
    ) -> None:
        try:
            Path(self.manifest_path).parent.mkdir(parents=True, exist_ok=True)
            sys_msg = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            user_msg = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_logical": model_logical,
                "model_snapshot": model_snapshot,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "system_msg_hash": hashlib.sha256(sys_msg.encode("utf-8")).hexdigest(),
                "user_msg_hash": hashlib.sha256(user_msg.encode("utf-8")).hexdigest(),
                "response_id": response_id,
                "system_fingerprint": None,
                "usage": None,
                "backend": "vllm" if self.vllm_host else ("ollama" if self.ollama_host else "hf"),
            }
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] llama manifest write failed: {e}")
