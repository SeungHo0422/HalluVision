"""
Llama-3-8B-Instruct local inference driver (Reviewer 2 #4).

Runs the *exact* main-pipeline prompts (AbstractiveSummarizer v2 summary
prompt, QuestionAnsweringHandler 5-question QA prompt) against a locally
loaded HuggingFace checkpoint, batched so a 4,940-doc corpus finishes in
hours rather than days.

Why not `LlamaClient._call_hf`: that path is one prompt per forward pass,
which is correct but ~10x slower than a left-padded batch. Prompts,
system messages, and decoding parameters here are pulled from the same
`Settings` / prompt builders the OpenAI runs used, so the only variable
that changes between the GPT and Llama runs is the model.

Usage:
    python final_pipeline/run_llama_inference.py --limit 8 --batch-size 4
    python final_pipeline/run_llama_inference.py --tasks summarization qa

Outputs (JSONL, appended incrementally so the run is resumable):
    pipeline_output/llama/llama_results_<model>.jsonl
plus per-call records in the shared pipeline_output/run_manifest.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402
from llm_proxy.summarizer import AbstractiveSummarizer  # noqa: E402
from llm_proxy.qa_handler import QuestionAnsweringHandler  # noqa: E402


DEFAULT_SOURCE = ROOT / "datasets" / "llm_results_gpt4o_mini_2503.json"
DEFAULT_OUT_DIR = ROOT.parent / "pipeline_output" / "llama"
MANIFEST_PATH = ROOT.parent / "pipeline_output" / "run_manifest.jsonl"


class _StubClient:
    """Satisfies the Summarizer/QAHandler constructors without an API key.

    Only the prompt builders are used here; generation happens in-process.
    """

    model = "llama-3-8b-instruct"

    def chat_completion(self, *a, **kw):  # pragma: no cover - never called
        raise RuntimeError("stub client: generation runs through the batch loop")

    def simple_prompt(self, *a, **kw):  # pragma: no cover - never called
        raise RuntimeError("stub client: generation runs through the batch loop")


# ─── corpus ───────────────────────────────────────────────────────────────

def load_corpus(source: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load [{id, original_text}, ...] from a pipeline results JSON/JSONL."""
    raw = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = list(raw.values())

    docs = []
    for rec in raw:
        text = rec.get("original_text") or rec.get("source_text") or rec.get("content")
        if not text:
            continue
        docs.append({"id": rec["id"], "original_text": text})
    docs.sort(key=lambda d: d["id"])
    return docs[:limit] if limit else docs


def load_done_ids(out_path: Path) -> set:
    """Ids already present in the output JSONL, for resume."""
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


# ─── generation ───────────────────────────────────────────────────────────

class BatchLlama:
    def __init__(
        self,
        model_path: str,
        max_input_tokens: int = 6144,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_input_tokens = max_input_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Llama-3 ships no pad token; left padding keeps generated tokens
        # contiguous at the end of each row.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        self.model.eval()

        # Llama-3-Instruct ends turns with <|eot_id|>, which is not the
        # tokenizer's eos_token — without this the model runs to max_new_tokens.
        eot = self.tokenizer.get_vocab().get("<|eot_id|>")
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        if eot is not None and eot not in self.eos_token_ids:
            self.eos_token_ids.append(eot)

        self.truncated = 0

    def build_prompt(self, system: str, user: str) -> str:
        # An empty system string means "no system turn at all" — the GPT runs
        # went through GPTClient.simple_prompt, which sends the user message
        # only (their manifest rows carry the SHA-256 of the empty string).
        # Keeping that shape makes the model the only variable.
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _truncate_user(self, user: str, system: str, reserve: int) -> str:
        """Clip the article body so system+user fits the context budget."""
        budget = self.max_input_tokens - reserve
        ids = self.tokenizer(user, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            return user
        self.truncated += 1
        return self.tokenizer.decode(ids[:budget], skip_special_tokens=True)

    @staticmethod
    def _seed(seed: Optional[int]):
        if seed is None:
            return
        import torch
        torch.manual_seed(seed)

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = 42,
    ) -> List[str]:
        torch = self.torch
        self._seed(seed)

        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_ids,
        )
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kwargs)

        new_tokens = out[:, enc["input_ids"].shape[1]:]
        return [
            self.tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in new_tokens
        ]


# ─── manifest (mirrors LlamaClient._append_manifest) ──────────────────────

def append_manifest(
    records: List[Dict[str, Any]],
    *,
    model_logical: str,
    model_snapshot: str,
    temperature: float,
    top_p: float,
    seed: Optional[int],
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps({
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "model_logical": model_logical,
                    "model_snapshot": model_snapshot,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "system_msg_hash": hashlib.sha256(
                        rec["system"].encode("utf-8")).hexdigest(),
                    "user_msg_hash": hashlib.sha256(
                        rec["user"].encode("utf-8")).hexdigest(),
                    "response_id": None,
                    "system_fingerprint": None,
                    "usage": None,
                    "backend": "hf",
                    "task": rec["task"],
                    "doc_id": rec["id"],
                }, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] manifest write failed: {e}")


# ─── driver ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Local Llama-3 inference for HalluVision")
    ap.add_argument("--model", default="llama-3-8b-instruct",
                    help="logical model name (resolved via Settings.MODEL_VERSIONS)")
    ap.add_argument("--model-path", default=None,
                    help="override the HF repo id / local path")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--tasks", nargs="+", default=["summarization", "qa"],
                    choices=["summarization", "qa"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-input-tokens", type=int, default=6144)
    ap.add_argument("--max-new-summary", type=int, default=256)
    ap.add_argument("--max-new-qa", type=int, default=128)
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in the output JSONL")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--no-system", action="store_true",
                    help="drop the system turn, matching the GPT runs "
                         "(GPTClient.simple_prompt sends the user message only)")
    args = ap.parse_args()

    model_path = args.model_path or Settings.resolve_model_snapshot(args.model)

    docs = load_corpus(args.source, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"llama_results_{args.model}.jsonl"

    if args.resume:
        done = load_done_ids(out_path)
        before = len(docs)
        docs = [d for d in docs if d["id"] not in done]
        print(f"[resume] {before - len(docs)} docs already done, {len(docs)} remaining")

    if not docs:
        print("[done] nothing to do")
        return 0

    print(f"[load] {model_path}")
    t0 = time.time()
    engine = BatchLlama(model_path, max_input_tokens=args.max_input_tokens)
    print(f"[load] ready in {time.time() - t0:.1f}s")

    # Prompt builders straight from the pipeline modules.
    summarizer = AbstractiveSummarizer(model=args.model, client=_StubClient())
    qa = QuestionAnsweringHandler(model=args.model, client=_StubClient())

    sum_sys = "" if args.no_system else Settings.SUMMARY_SYSTEM_MESSAGE
    qa_sys = "" if args.no_system else Settings.QA_SYSTEM_MESSAGE
    seed = Settings.LLM_SEED

    do_sum = "summarization" in args.tasks
    do_qa = "qa" in args.tasks

    t_start = time.time()
    n_done = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for i in range(0, len(docs), args.batch_size):
            batch = docs[i:i + args.batch_size]
            results = {d["id"]: {"id": d["id"], "model": args.model,
                                 "model_snapshot": model_path}
                       for d in batch}
            manifest_records = []

            if do_sum:
                users, prompts = [], []
                for d in batch:
                    body = engine._truncate_user(
                        d["original_text"], sum_sys, reserve=256)
                    user = summarizer._prompt_ver2(body)
                    users.append(user)
                    prompts.append(engine.build_prompt(sum_sys, user))
                outs = engine.generate(
                    prompts, args.max_new_summary,
                    temperature=Settings.SUMMARY_TEMPERATURE,
                    top_p=Settings.LLM_TOP_P, seed=seed)
                for d, u, o in zip(batch, users, outs):
                    results[d["id"]]["summary_text"] = o
                    manifest_records.append({"id": d["id"], "task": "summarization",
                                             "system": sum_sys, "user": u})

            if do_qa:
                users, prompts = [], []
                for d in batch:
                    body = engine._truncate_user(
                        d["original_text"], qa_sys, reserve=512)
                    user = qa.create_qa_prompt(body)
                    users.append(user)
                    prompts.append(engine.build_prompt(qa_sys, user))
                outs = engine.generate(
                    prompts, args.max_new_qa,
                    temperature=Settings.QA_TEMPERATURE,
                    top_p=Settings.LLM_TOP_P, seed=seed)
                for d, u, o in zip(batch, users, outs):
                    results[d["id"]]["qa_text"] = o
                    manifest_records.append({"id": d["id"], "task": "qa",
                                             "system": qa_sys, "user": u})

            for d in batch:
                fout.write(json.dumps(results[d["id"]], ensure_ascii=False) + "\n")
            fout.flush()

            if not args.no_manifest:
                append_manifest(
                    manifest_records,
                    model_logical=args.model,
                    model_snapshot=model_path,
                    temperature=Settings.LLM_TEMPERATURE,
                    top_p=Settings.LLM_TOP_P,
                    seed=seed,
                )

            n_done += len(batch)
            elapsed = time.time() - t_start
            rate = n_done / elapsed
            eta = (len(docs) - n_done) / rate if rate else 0
            print(f"[{n_done}/{len(docs)}] {rate:.2f} docs/s "
                  f"elapsed {elapsed/60:.1f}m eta {eta/60:.1f}m "
                  f"truncated {engine.truncated}", flush=True)

    print(f"[done] {n_done} docs -> {out_path}")
    print(f"[done] truncated inputs: {engine.truncated}")

    # Convenience: also emit the aggregated JSON the analyzer modules read.
    json_path = out_path.with_suffix(".json")
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["id"])
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] aggregated -> {json_path} ({len(rows)} docs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
