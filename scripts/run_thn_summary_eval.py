#!/usr/bin/env python3
"""Summary-axis answer matching for the THN held-out corpus (Reviewer 2 #5).

Replicates the metric in the root ``eval_summary.py`` -- entity-set matching
between the source-article gold NER and the summary NER, counted per label
over entities (not documents) -- without that script's hardcoded Threatpost
paths and plotting. Also reports the frequency-based L1 distance (the paper's
third metric) so the THN table lines up with Table 6.

Usage::

    python run_thn_summary_eval.py \
        --pair "gpt-4o-mini (THN)" pipeline_output/thn/thn_original_ner.json \
               pipeline_output/thn/thn_summary_ner_gpt-4o-mini.json \
        --pair "gpt-4o-mini (Threatpost)" original_ner.json summary_ner.json \
        --output notes/thn_summary_eval.md \
        --json-output notes/thn_summary_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(gold: dict, summ: dict) -> dict:
    """Per-label TP/FP/FN over entities, plus the L1 frequency distance."""
    counts = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in LABELS}
    l1_distances: List[float] = []
    n_docs = 0
    n_missing = 0

    for doc_id, summ_doc in summ.items():
        orig = gold.get(doc_id)
        if orig is None:
            n_missing += 1
            continue
        n_docs += 1

        for lbl in LABELS:
            true_entities = set(orig.get(lbl, []) or [])
            pred_entities = set(summ_doc.get(lbl, []) or [])
            counts[lbl]["tp"] += len(true_entities & pred_entities)
            counts[lbl]["fn"] += len(true_entities - pred_entities)
            counts[lbl]["fp"] += len(pred_entities - true_entities)

        orig_counts = [len(orig.get(lbl, []) or []) for lbl in LABELS]
        summ_counts = [len(summ_doc.get(lbl, []) or []) for lbl in LABELS]
        orig_total, summ_total = sum(orig_counts), sum(summ_counts)
        orig_ratio = [c / orig_total if orig_total else 0.0 for c in orig_counts]
        summ_ratio = [c / summ_total if summ_total else 0.0 for c in summ_counts]
        l1_distances.append(sum(abs(a - b) for a, b in zip(orig_ratio, summ_ratio)))

    per_label = {}
    for lbl in LABELS:
        c = counts[lbl]
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[lbl] = {"precision": p, "recall": r, "f1": f1, **c}

    tp = sum(counts[l]["tp"] for l in LABELS)
    fp = sum(counts[l]["fp"] for l in LABELS)
    fn = sum(counts[l]["fn"] for l in LABELS)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    micro = {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if (p + r) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

    return {
        "n_docs": n_docs,
        "n_missing_gold": n_missing,
        "per_label": per_label,
        "micro": micro,
        "l1_distance_mean": sum(l1_distances) / len(l1_distances) if l1_distances else 0.0,
    }


def render(results: Dict[str, dict]) -> str:
    lines = [
        "# Summary-axis answer matching — THN held-out vs Threatpost (R2 #5)",
        "",
        "Entity-set matching between source-article gold NER and summary NER,",
        "counted over entities per label (same metric as `eval_summary.py`).",
        "",
    ]
    for name, res in results.items():
        micro = res["micro"]
        lines += [
            f"## {name}",
            "",
            f"- documents: **{res['n_docs']}**"
            + (f" (gold missing for {res['n_missing_gold']})" if res["n_missing_gold"] else ""),
            f"- frequency-based L1 distance (mean): **{res['l1_distance_mean']:.4f}**",
            "",
            "| Label | P | R | F1 | TP | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for lbl in LABELS:
            m = res["per_label"][lbl]
            lines.append(
                f"| {lbl} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                f"{m['tp']} | {m['fp']} | {m['fn']} |"
            )
        lines.append(
            f"| **MICRO** | **{micro['precision']:.4f}** | **{micro['recall']:.4f}** | "
            f"**{micro['f1']:.4f}** | {micro['tp']} | {micro['fp']} | {micro['fn']} |"
        )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("NAME", "GOLD_NER", "SUMMARY_NER"),
        required=True,
    )
    ap.add_argument("--output", required=True, help="markdown report")
    ap.add_argument("--json-output", help="machine-readable results")
    args = ap.parse_args()

    results = {}
    for name, gold_path, summ_path in args.pair:
        results[name] = evaluate(_load(gold_path), _load(summ_path))
        results[name]["gold_path"] = gold_path
        results[name]["summary_path"] = summ_path
        micro = results[name]["micro"]
        print(
            f"{name}: n={results[name]['n_docs']} "
            f"P={micro['precision']:.4f} R={micro['recall']:.4f} F1={micro['f1']:.4f} "
            f"L1={results[name]['l1_distance_mean']:.4f}"
        )

    Path(args.output).write_text(render(results), encoding="utf-8")
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
