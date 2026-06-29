"""
FactScore-style baseline runner (Reviewer 2 #6).

Re-implements the *atomic fact decomposition + closed-domain support
check* idea from FActScore (Min et al., 2023, EMNLP) using an LLM judge
against the original Threatpost / THN article as the knowledge source.

This is the in-paper baseline — we are *not* shipping the upstream
FActScore package (it pins old transformer versions and assumes a Wiki
knowledge source). Instead we expose the same metric shape (per-doc
support ratio in [0, 1]) so HalluVision and FactScore-style numbers
are directly comparable.

Pipeline per doc:
    1. Ask the judge to decompose the summary into atomic claims
    2. Ask the judge to verify each claim against the original text
       (supported / unsupported / inconclusive)
    3. Score = supported / total_atomic_claims

The judge defaults to the HalluVision GPTClient with gpt-4o-mini at
``temperature=0.0`` — the same hyperparameters used everywhere else,
so reproducibility carries over. A different ``BaseLLMClient`` can be
injected for mocked tests (see ``test/test_baselines.py``).

Usage::

    python -m final_pipeline.analyzer.baselines.factscore_runner \
        --subset notes/baseline_subset.json \
        --output notes/factscore_scores.json \
        --judge-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from llm_proxy.base_client import BaseLLMClient, get_client_for_model
except ImportError:  # pragma: no cover
    from final_pipeline.llm_proxy.base_client import BaseLLMClient, get_client_for_model  # type: ignore


ATOMIC_DECOMP_PROMPT = """Break the SUMMARY below into atomic claims. \
An atomic claim is a single, self-contained factual statement (subject + \
predicate). Output one claim per line. No numbering, no preamble.

SUMMARY:
{summary}
"""

VERIFY_PROMPT = """Read the SOURCE ARTICLE and decide whether the CLAIM is \
directly supported by it. Reply with exactly one of: SUPPORTED, UNSUPPORTED, \
INCONCLUSIVE. No other text.

SOURCE ARTICLE:
{source}

CLAIM: {claim}
"""


def _decompose_claims(summary: str, judge: BaseLLMClient,
                      judge_model: Optional[str]) -> List[str]:
    text = judge.simple_prompt(
        ATOMIC_DECOMP_PROMPT.format(summary=summary.strip()),
        model=judge_model,
        temperature=0.0,
    )
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip() for l in text.splitlines()]
    return [l for l in lines if l]


def _verify_claim(source: str, claim: str, judge: BaseLLMClient,
                  judge_model: Optional[str]) -> str:
    text = judge.simple_prompt(
        VERIFY_PROMPT.format(source=source.strip()[:8000], claim=claim.strip()),
        model=judge_model,
        temperature=0.0,
    ).strip().upper()
    if "UNSUPPORTED" in text:
        return "unsupported"
    if "SUPPORTED" in text:
        return "supported"
    return "inconclusive"


def score_doc(doc: dict, judge: BaseLLMClient,
              judge_model: Optional[str]) -> Dict[str, Any]:
    summary = doc.get("summary_text") or ""
    source = doc.get("original_text") or ""
    if not summary.strip() or not source.strip():
        return {"doc_id": str(doc.get("id")),
                "factscore": None, "n_claims": 0, "note": "missing summary/source"}
    claims = _decompose_claims(summary, judge, judge_model)
    verdicts = [_verify_claim(source, c, judge, judge_model) for c in claims]
    supported = sum(1 for v in verdicts if v == "supported")
    n = len(verdicts)
    return {
        "doc_id": str(doc.get("id")),
        "factscore": (supported / n) if n else None,
        "n_claims": n,
        "supported": supported,
        "verdicts": verdicts,
        "claims": claims,
    }


def run(subset_path: str, output_path: str, judge_model: str,
        client: Optional[BaseLLMClient] = None,
        api_key: Optional[str] = None,
        limit: Optional[int] = None) -> List[dict]:
    payload = json.loads(Path(subset_path).read_text(encoding="utf-8"))
    docs = payload.get("docs", [])
    if limit is not None:
        docs = docs[:limit]
    judge = client if client is not None else get_client_for_model(judge_model, api_key=api_key)
    results = []
    for d in docs:
        results.append(score_doc(d, judge, judge_model))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="FactScore-style baseline runner")
    p.add_argument("--subset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap (smoke runs)")
    args = p.parse_args()

    results = run(args.subset, args.output, args.judge_model, limit=args.limit)
    n_scored = sum(1 for r in results if r.get("factscore") is not None)
    print(f"[factscore] scored {n_scored}/{len(results)} → {args.output}")


if __name__ == "__main__":
    main()
