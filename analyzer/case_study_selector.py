"""
Pick 8–10 case studies per error category (Reviewer 2 minor #3).

Consumes the classified rows from ``qa_error_taxonomy`` and the
underlying ``gpt_results`` JSON, then samples ``--per-bucket`` rows
from each error type so the appendix case-study section in the
revised paper covers the full taxonomy uniformly. Sampling is
stratified by label too, so each error_type × label cell contributes
when populated.

Output: a markdown file with one ``### Case <i>: <error_type>``
section per pick, embedding the original text excerpt, prediction,
gold list, and a short pre-written analysis stub the writer can edit.

Usage::

    python -m final_pipeline.analyzer.case_study_selector \
        --classified notes/qa_error_taxonomy_rows.json \
        --gpt-results gpt_results_2503.json \
        --per-bucket 2 --seed 42 \
        --output notes/case_studies_expanded.md
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _text_lookup(path: str) -> Dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(d.get("id")): d for d in data}


def select(classified: List[dict], per_bucket: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for r in classified:
        by_type[r["error_type"]].append(r)
    picks: List[dict] = []
    for et in sorted(by_type.keys()):
        rows = by_type[et]
        rng.shuffle(rows)
        picks.extend(rows[:per_bucket])
    return picks


def render_markdown(picks: List[dict], text_lookup: Dict[str, dict],
                    excerpt_chars: int) -> str:
    lines = [
        "# Expanded Case Studies (R2 minor)",
        "",
        f"Total cases: {len(picks)}",
        "",
    ]
    for i, p in enumerate(picks, start=1):
        doc = text_lookup.get(p["doc_id"], {})
        excerpt = (doc.get("original_text") or "")[:excerpt_chars]
        summary = doc.get("summary_text", "")
        qa = doc.get("qa_text", "")
        gold = p.get("gold_entities") or []
        lines += [
            f"## Case {i}: error_type = `{p['error_type']}` · label = `{p['label']}` · doc {p['doc_id']}",
            "",
            "**Prediction:** `" + (p.get("prediction") or "") + "`",
            "",
            "**Gold entities:** " + (", ".join(f"`{g}`" for g in gold) if gold else "*(none)*"),
            "",
            "**Source excerpt (truncated):**",
            "",
            "```",
            excerpt,
            "```",
            "",
            "**Generated summary:**",
            "",
            summary or "*(empty)*",
            "",
            "**Full QA answer string:** `" + qa + "`",
            "",
            "**Analyst notes (TODO):** _explain what went wrong, which side (NER vs LLM) is responsible, and whether §3 span-level matching reclaims it._",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Pick stratified case studies")
    p.add_argument("--classified", required=True,
                   help="JSON dump from qa_error_taxonomy --dump-rows")
    p.add_argument("--gpt-results", required=True)
    p.add_argument("--per-bucket", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--excerpt-chars", type=int, default=1200)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    classified = json.loads(Path(args.classified).read_text(encoding="utf-8"))
    text_lookup = _text_lookup(args.gpt_results)
    picks = select(classified, args.per_bucket, args.seed)
    md = render_markdown(picks, text_lookup, args.excerpt_chars)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    print(f"[case_study_selector] picked {len(picks)} cases → {args.output}")


if __name__ == "__main__":
    main()
