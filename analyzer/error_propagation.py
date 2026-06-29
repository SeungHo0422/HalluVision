"""
NER -> downstream metric error propagation (Reviewer 2 #2b).

Given (a) the reported NER (``original_ner.json``) and (b) a partially
hand-corrected NER for the same doc ids, recompute the summary or QA
metrics restricted to the corrected subset and report Δ. This isolates
how much of the reported metric movement is caused by NER errors vs.
LLM behaviour.

The corrected NER file follows the same schema as ``original_ner.json``::

    {"<doc_id>": {"Malware": [...], "System": [...], ...}, ...}

Doc ids not present in the corrected file are skipped. Entity strings
are compared after lowercasing and stripping.

Usage::

    python -m final_pipeline.analyzer.error_propagation \
        --reported original_ner.json \
        --corrected notes/ner_corrected_subset.json \
        --predictions gpt_results_250427.json \
        --task qa \
        --output notes/error_propagation_qa.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .extract_doc_stats import _qa_doc_stats, _summary_doc_stats
except ImportError:
    from extract_doc_stats import _qa_doc_stats, _summary_doc_stats

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _aggregate(doc_stats: Dict[str, dict], label: str, metric: str) -> float:
    vals = []
    for doc_id, label_map in doc_stats.items():
        cell = label_map.get(label) or label_map.get("_overall_")
        if not cell:
            continue
        v = cell.get(metric)
        if v is None:
            continue
        vals.append(float(v))
    return _mean(vals)


def _restrict(stats: Dict[str, dict], keep_ids: set) -> Dict[str, dict]:
    return {doc_id: v for doc_id, v in stats.items() if doc_id in keep_ids}


def _build_summary_doc_stats(original: dict, summary: dict) -> Dict[str, dict]:
    return _summary_doc_stats(original, summary, split_commas=False)


def main() -> None:
    p = argparse.ArgumentParser(description="NER -> metric error propagation")
    p.add_argument("--reported", required=True, help="original_ner.json (reported NER)")
    p.add_argument("--corrected", required=True, help="hand-corrected NER subset (same schema)")
    p.add_argument(
        "--predictions", required=True,
        help="gpt_results_*.json (qa) or summary NER (summary)",
    )
    p.add_argument("--task", required=True, choices=["qa", "summary"])
    p.add_argument(
        "--match-mode", default="exact",
        choices=["exact", "token_f1", "jaccard", "relaxed"],
        help="(qa only) match mode passed to extract_doc_stats",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()

    reported = json.loads(Path(args.reported).read_text(encoding="utf-8"))
    corrected = json.loads(Path(args.corrected).read_text(encoding="utf-8"))
    with open(args.predictions, encoding="utf-8") as f:
        preds = json.load(f)

    keep_ids = set(corrected.keys()) & set(reported.keys())
    if not keep_ids:
        raise SystemExit("No overlapping doc ids between corrected and reported NER")

    if args.task == "qa":
        if not isinstance(preds, list):
            raise SystemExit("QA predictions must be a JSON list")
        stats_reported = _qa_doc_stats(reported, preds, match_mode=args.match_mode)
        stats_corrected = _qa_doc_stats(corrected, preds, match_mode=args.match_mode)
    else:
        if not isinstance(preds, dict):
            raise SystemExit("Summary predictions must be a JSON dict")
        stats_reported = _build_summary_doc_stats(reported, preds)
        stats_corrected = _build_summary_doc_stats(corrected, preds)

    stats_reported = _restrict(stats_reported, keep_ids)
    stats_corrected = _restrict(stats_corrected, keep_ids)

    lines = [
        f"# NER -> {args.task} metric error propagation",
        "",
        f"- subset: {len(keep_ids)} doc ids (overlap of reported NER and corrected NER)",
        f"- task: {args.task} | match_mode: {args.match_mode if args.task == 'qa' else 'n/a'}",
        "",
        "| Label | metric | reported | corrected | Δ(corrected − reported) |",
        "|---|---|---:|---:|---:|",
    ]
    for label in ["_overall_"] + LABELS:
        for metric in ("precision", "recall", "f1"):
            r = _aggregate(stats_reported, label, metric)
            c = _aggregate(stats_corrected, label, metric)
            delta = c - r if (r == r and c == c) else float("nan")
            lines.append(f"| {label} | {metric} | {r:.4f} | {c:.4f} | {delta:+.4f} |")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[error_propagation] wrote {args.output}")


if __name__ == "__main__":
    main()
