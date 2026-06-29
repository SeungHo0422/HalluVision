"""
Stratified human-label sampler (Reviewer 2 #3 — human validation).

Selects ``n`` docs from the full HalluVision output for two-annotator
hallucination labeling. Stratification crosses summary F1 quartile and
QA precision quartile so the sample covers both the metric's hits and
misses uniformly (16 cells, ~6–7 docs per cell at n=100).

Output is a doccano-friendly CSV with the original text excerpt, the
generated summary, the QA answer string, and blank columns for each
annotator's label + free-text reasoning.

Label legend (annotator fills in `label_a` / `label_b`):
    faithful     — summary/QA grounded in source
    partial      — partially grounded, partially hallucinated
    hallucinated — material claim unsupported by source

Usage::

    python -m final_pipeline.analyzer.human_sample \
        --gpt-results gpt_results_2503.json \
        --summary-stats notes/doc_stats_summary_gpt4omini.json \
        --qa-stats notes/doc_stats_qa_gpt4omini.json \
        --n 100 --seed 42 \
        --output notes/human_label_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import quantiles
from typing import Dict, List, Tuple

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]


def _quartile_bucket(value: float, cutoffs: List[float]) -> int:
    for i, c in enumerate(cutoffs):
        if value <= c:
            return i
    return len(cutoffs)


def _compute_cutoffs(values: List[float]) -> List[float]:
    finite = [v for v in values if v is not None]
    if len(finite) < 4:
        return [0.25, 0.5, 0.75]
    return quantiles(finite, n=4)[:3]


def stratified_sample(
    docs: List[dict],
    summary_stats: Dict[str, dict],
    qa_stats: Dict[str, dict],
    n: int,
    seed: int,
) -> List[dict]:
    """Sample ``n`` docs across the 4×4 (summary F1 × QA precision) grid."""
    rng = random.Random(seed)
    eligible: List[Tuple[dict, float, float]] = []
    for d in docs:
        doc_id = str(d.get("id"))
        s = summary_stats.get(doc_id, {}).get("_overall_", {})
        q = qa_stats.get(doc_id, {}).get("_overall_", {})
        if not s or not q:
            continue
        if not (d.get("summary_text") or "").strip():
            continue
        if not (d.get("qa_text") or "").strip():
            continue
        eligible.append((d, s.get("f1", 0.0), q.get("precision", 0.0)))

    sm_cuts = _compute_cutoffs([f for _, f, _ in eligible])
    qa_cuts = _compute_cutoffs([p for _, _, p in eligible])

    grid: Dict[Tuple[int, int], List[dict]] = {(i, j): [] for i in range(4) for j in range(4)}
    for d, sm_f1, qa_p in eligible:
        bucket = (_quartile_bucket(sm_f1, sm_cuts), _quartile_bucket(qa_p, qa_cuts))
        grid[bucket].append(d)
    for cell in grid.values():
        rng.shuffle(cell)

    n_cells = 16
    base = n // n_cells
    remainder = n - base * n_cells

    cells = sorted(grid.keys())
    rng.shuffle(cells)
    sampled: List[dict] = []
    for idx, key in enumerate(cells):
        take = base + (1 if idx < remainder else 0)
        cell_docs = grid[key]
        for d in cell_docs[:take]:
            d_out = dict(d)
            d_out["_sm_f1_bucket_"] = key[0]
            d_out["_qa_p_bucket_"] = key[1]
            sampled.append(d_out)

    if len(sampled) < n:
        deficit = n - len(sampled)
        leftovers: List[dict] = []
        for key in cells:
            cell_docs = grid[key]
            already = sum(1 for s in sampled if str(s.get("id")) in {str(c.get("id")) for c in cell_docs})
            for extra in cell_docs[already:]:
                leftovers.append(extra)
        rng.shuffle(leftovers)
        for d in leftovers[:deficit]:
            d_out = dict(d)
            d_out["_sm_f1_bucket_"] = -1
            d_out["_qa_p_bucket_"] = -1
            sampled.append(d_out)

    return sampled[:n]


def main() -> None:
    p = argparse.ArgumentParser(description="Sample docs for human hallucination labeling")
    p.add_argument("--gpt-results", required=True,
                   help="gpt_results_*.json with id/original_text/summary_text/qa_text")
    p.add_argument("--summary-stats", required=True,
                   help="notes/doc_stats_summary_*.json")
    p.add_argument("--qa-stats", required=True,
                   help="notes/doc_stats_qa_*.json")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--excerpt-chars", type=int, default=1500)
    p.add_argument("--model-tag", default="gpt-4o-mini",
                   help="Label written to the model column")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    docs = json.loads(Path(args.gpt_results).read_text(encoding="utf-8"))
    summary_stats = json.loads(Path(args.summary_stats).read_text(encoding="utf-8"))
    qa_stats = json.loads(Path(args.qa_stats).read_text(encoding="utf-8"))

    sampled = stratified_sample(docs, summary_stats, qa_stats, args.n, args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "doc_id", "model",
            "sm_f1_bucket", "qa_p_bucket",
            "original_text_excerpt", "summary_text", "qa_text",
            "label_a", "reasoning_a",
            "label_b", "reasoning_b",
        ])
        for d in sampled:
            text = (d.get("original_text") or "")[: args.excerpt_chars]
            writer.writerow([
                d.get("id"), args.model_tag,
                d.get("_sm_f1_bucket_"), d.get("_qa_p_bucket_"),
                text,
                d.get("summary_text", ""),
                d.get("qa_text", ""),
                "", "", "", "",
            ])

    print(f"[human_sample] wrote {len(sampled)} rows → {args.output}")


if __name__ == "__main__":
    main()
