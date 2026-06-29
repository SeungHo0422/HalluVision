"""
QAGS-style baseline runner (Reviewer 2 #6).

Re-implements the QAGS protocol (Wang et al., 2020 ACL) using an LLM
judge: generate questions from the SUMMARY, then answer them against
the SUMMARY and the SOURCE ARTICLE; agreement between the two answer
sets is the QAGS score in [0, 1].

Per-doc score = mean(exact-token overlap between summary-answer and
source-answer across all generated questions).

The runner deliberately stays light: a pure LLM judge with the same
``BaseLLMClient`` abstraction used by FactScore. Real upstream QAGS
plugs in BART-QA / SQuAD models; for our paper we want comparable
metric shape, not architectural fidelity to a 2020 model.

Usage::

    python -m final_pipeline.analyzer.baselines.qags_runner \
        --subset notes/baseline_subset.json \
        --output notes/qags_scores.json \
        --judge-model gpt-4o-mini --n-questions 5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from llm_proxy.base_client import BaseLLMClient, get_client_for_model
    from analyzer.span_match import token_f1
except ImportError:  # pragma: no cover
    from final_pipeline.llm_proxy.base_client import BaseLLMClient, get_client_for_model  # type: ignore
    from final_pipeline.analyzer.span_match import token_f1  # type: ignore


QUESTION_GEN_PROMPT = """Generate {n} factual questions about the SUMMARY below. \
Each question must target a specific entity, number, or claim that appears in \
the summary. Output one question per line. No numbering, no preamble.

SUMMARY:
{summary}
"""

ANSWER_PROMPT = """Answer the QUESTION using only the TEXT below. \
Reply with a short answer (≤10 words). If the TEXT does not contain the answer, \
reply with the literal token UNANSWERABLE.

TEXT:
{text}

QUESTION: {question}
"""


def _gen_questions(summary: str, n: int, judge: BaseLLMClient,
                   judge_model: Optional[str]) -> List[str]:
    text = judge.simple_prompt(
        QUESTION_GEN_PROMPT.format(n=n, summary=summary.strip()),
        model=judge_model,
        temperature=0.0,
    )
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip("?").strip() for l in text.splitlines()]
    questions = [l + "?" for l in lines if l]
    return questions[:n]


def _answer(text: str, question: str, judge: BaseLLMClient,
            judge_model: Optional[str]) -> str:
    return judge.simple_prompt(
        ANSWER_PROMPT.format(text=text.strip()[:8000], question=question),
        model=judge_model,
        temperature=0.0,
    ).strip()


def _answers_agree(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if "UNANSWERABLE" in a.upper() and "UNANSWERABLE" in b.upper():
        return 1.0
    if "UNANSWERABLE" in a.upper() or "UNANSWERABLE" in b.upper():
        return 0.0
    return token_f1(a, b)


def score_doc(doc: dict, n_questions: int, judge: BaseLLMClient,
              judge_model: Optional[str]) -> Dict[str, Any]:
    summary = doc.get("summary_text") or ""
    source = doc.get("original_text") or ""
    if not summary.strip() or not source.strip():
        return {"doc_id": str(doc.get("id")),
                "qags": None, "n_questions": 0, "note": "missing summary/source"}
    questions = _gen_questions(summary, n_questions, judge, judge_model)
    pairs = []
    for q in questions:
        sa = _answer(summary, q, judge, judge_model)
        ta = _answer(source, q, judge, judge_model)
        pairs.append({"question": q, "summary_answer": sa, "source_answer": ta,
                      "agreement": _answers_agree(sa, ta)})
    score = sum(p["agreement"] for p in pairs) / len(pairs) if pairs else None
    return {
        "doc_id": str(doc.get("id")),
        "qags": score,
        "n_questions": len(pairs),
        "pairs": pairs,
    }


def run(subset_path: str, output_path: str, judge_model: str, n_questions: int,
        client: Optional[BaseLLMClient] = None,
        api_key: Optional[str] = None,
        limit: Optional[int] = None) -> List[dict]:
    payload = json.loads(Path(subset_path).read_text(encoding="utf-8"))
    docs = payload.get("docs", [])
    if limit is not None:
        docs = docs[:limit]
    judge = client if client is not None else get_client_for_model(judge_model, api_key=api_key)
    results = [score_doc(d, n_questions, judge, judge_model) for d in docs]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="QAGS-style baseline runner")
    p.add_argument("--subset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--n-questions", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    results = run(args.subset, args.output, args.judge_model, args.n_questions,
                  limit=args.limit)
    n_scored = sum(1 for r in results if r.get("qags") is not None)
    print(f"[qags] scored {n_scored}/{len(results)} → {args.output}")


if __name__ == "__main__":
    main()
