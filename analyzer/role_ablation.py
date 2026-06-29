"""
Role-prompt ablation runner (Reviewer 2 minor #1).

Compares three summarisation prompts on the same N-doc subset:

* ``v1``   — *"You're a famous NER Expert in CTI"* (current)
* ``v2``   — *"You're a famous Expert in CTI"* (current variant)
* ``v0``   — no role prompt, plain "Summarize the article below" baseline

Per doc we score the produced summary against the original NER gold
labels using the existing summary-vs-NER overlap metric, and we
report mean P/R/F1 plus paired Δ(v1 − v0) and Δ(v2 − v0).

The LLM client is injected so unit tests can run with a stub. In
real runs the default ``GPTClient`` (or any ``BaseLLMClient``) is
used; OpenAI cost is N docs × 3 prompt variants.

Usage::

    python -m final_pipeline.analyzer.role_ablation \
        --gpt-results gpt_results_2503.json \
        --original-ner original_ner.json \
        --n 30 --seed 42 \
        --model gpt-4o-mini \
        --output notes/role_ablation.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from llm_proxy.base_client import BaseLLMClient, get_client_for_model
    from config.settings import Settings
except ImportError:  # pragma: no cover
    from final_pipeline.llm_proxy.base_client import BaseLLMClient, get_client_for_model  # type: ignore
    from final_pipeline.config.settings import Settings  # type: ignore


LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]

PROMPT_V0 = "Summarize the article below in {min}–{max} words.\n\n{text}"
PROMPT_V1 = """# Main Role
- You're a famous NER Expert in CTI (Cyber Threat Intelligence) domain. Your task is to generate a short summary of a malware information obtained from popular malware blog posts, following the instructions below.

# Main Instructions
- Summarize the content below in at least {min} words, and at most {max} words.
- You should answer using only the words and contents in the summary as much as possible.

## Input blog text data
{text}
"""
PROMPT_V2 = """# Main Role
- You're a famous Expert in CTI (Cyber Threat Intelligence) domain. Your task is to generate a short summary of a malware information obtained from popular malware blog posts, following the instructions below.

# Main Instructions
- Summarize the content below in at least {min} words, and at most {max} words.

## Input blog text data
{text}
"""

PROMPTS = {"v0": PROMPT_V0, "v1": PROMPT_V1, "v2": PROMPT_V2}


def _token_set(text: str) -> set:
    return set(re.findall(r"[A-Za-z0-9]+(?:[.\-_][A-Za-z0-9]+)*", text.lower()))


def _label_prf(summary: str, golds: List[str]) -> Tuple[float, float, float]:
    if not golds:
        return 0.0, 0.0, 0.0
    summary_tokens = _token_set(summary)
    hit = 0
    for g in golds:
        g_tokens = _token_set(g)
        if g_tokens and g_tokens.issubset(summary_tokens):
            hit += 1
    recall = hit / len(golds)
    # precision is approximated as recall here (per-label gold is a list of
    # entities and we lack a "prediction set" — same approximation as the
    # original summary analyzer)
    precision = recall
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _doc_prf(summary: str, gold_by_label: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    out = {}
    overall_p, overall_r, overall_f = [], [], []
    for lbl in LABELS:
        p, r, f = _label_prf(summary, gold_by_label.get(lbl, []))
        out[lbl] = {"precision": p, "recall": r, "f1": f}
        if gold_by_label.get(lbl):
            overall_p.append(p)
            overall_r.append(r)
            overall_f.append(f)
    out["_overall_"] = {
        "precision": statistics.mean(overall_p) if overall_p else 0.0,
        "recall": statistics.mean(overall_r) if overall_r else 0.0,
        "f1": statistics.mean(overall_f) if overall_f else 0.0,
    }
    return out


def _summarise(client: BaseLLMClient, model: str, variant: str, text: str,
               min_w: int, max_w: int) -> str:
    prompt = PROMPTS[variant].format(text=text[:8000], min=min_w, max=max_w)
    return client.simple_prompt(prompt, model=model, temperature=0.0)


def run_ablation(docs: List[dict], gold_ner: Dict[str, dict],
                 client: BaseLLMClient, model: str,
                 min_w: int, max_w: int) -> List[dict]:
    rows = []
    for d in docs:
        doc_id = str(d.get("id"))
        text = (d.get("original_text") or "").strip()
        if not text:
            continue
        golds = gold_ner.get(doc_id, {})
        per_variant = {}
        for variant in ("v0", "v1", "v2"):
            summary = _summarise(client, model, variant, text, min_w, max_w)
            per_variant[variant] = {
                "summary": summary,
                "prf": _doc_prf(summary, golds),
            }
        rows.append({"doc_id": doc_id, "variants": per_variant})
    return rows


def aggregate(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, List[float]]] = {
        v: {"precision": [], "recall": [], "f1": []} for v in ("v0", "v1", "v2")
    }
    for r in rows:
        for v in ("v0", "v1", "v2"):
            overall = r["variants"][v]["prf"]["_overall_"]
            agg[v]["precision"].append(overall["precision"])
            agg[v]["recall"].append(overall["recall"])
            agg[v]["f1"].append(overall["f1"])
    out: Dict[str, Dict[str, float]] = {}
    for v, m in agg.items():
        out[v] = {k: statistics.mean(vals) if vals else 0.0 for k, vals in m.items()}
    return out


def render_markdown(rows: List[dict], summary_means: Dict[str, Dict[str, float]],
                    n_docs: int, model: str) -> str:
    lines = [
        "# Role-Prompt Ablation (R2 minor)",
        "",
        f"Subset size: {n_docs}",
        f"Model: `{model}`",
        "",
        "## Summary-vs-NER overall metrics (mean over subset)",
        "",
        "| variant | description | P | R | F1 |",
        "|---|---|---:|---:|---:|",
    ]
    descriptions = {
        "v0": "no role prompt",
        "v1": "NER expert role",
        "v2": "CTI expert role",
    }
    for v in ("v0", "v1", "v2"):
        m = summary_means[v]
        lines.append(
            f"| {v} | {descriptions[v]} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['f1']:.3f} |"
        )
    lines += [
        "",
        "## Paired deltas vs v0 (no-role baseline)",
        "",
        "| variant | ΔP | ΔR | ΔF1 |",
        "|---|---:|---:|---:|",
    ]
    base = summary_means["v0"]
    for v in ("v1", "v2"):
        m = summary_means[v]
        lines.append(
            f"| {v} − v0 | {m['precision'] - base['precision']:+.3f} | "
            f"{m['recall'] - base['recall']:+.3f} | "
            f"{m['f1'] - base['f1']:+.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Role-prompt ablation runner")
    p.add_argument("--gpt-results", required=True)
    p.add_argument("--original-ner", required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--min-words", type=int, default=getattr(Settings, "SUMMARY_MIN_WORDS", 80))
    p.add_argument("--max-words", type=int, default=getattr(Settings, "SUMMARY_MAX_WORDS", 150))
    p.add_argument("--output", required=True)
    p.add_argument("--dump-rows", default=None)
    args = p.parse_args()

    docs = json.loads(Path(args.gpt_results).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    subset = docs[: args.n]
    gold = json.loads(Path(args.original_ner).read_text(encoding="utf-8"))

    client = get_client_for_model(args.model)
    rows = run_ablation(subset, gold, client, args.model, args.min_words, args.max_words)
    summary_means = aggregate(rows)

    md = render_markdown(rows, summary_means, len(rows), args.model)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    if args.dump_rows:
        Path(args.dump_rows).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"[role_ablation] {len(rows)} docs × 3 variants → {args.output}")


if __name__ == "__main__":
    main()
