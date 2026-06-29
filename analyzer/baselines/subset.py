"""
Pick the 200-doc baseline subset (Reviewer 2 #6).

Strategy: take the §7 human-labeled 100 (so FactScore / QAGS scores
can be compared against gold labels), plus 100 additional random docs
to bring the total to 200. Doc ids and the random seed are written
into the output JSON so the subset is reproducible.

Usage::

    python -m final_pipeline.analyzer.baselines.subset \
        --gpt-results gpt_results_2503.json \
        --human-sample notes/human_label_sample.csv \
        --total 200 --seed 42 \
        --output notes/baseline_subset.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List, Set


def _read_doc_ids(csv_path: str) -> Set[str]:
    out: Set[str] = set()
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            did = (row.get("doc_id") or "").strip()
            if did:
                out.add(did)
    return out


def pick_subset(docs: List[dict], human_ids: Set[str], total: int, seed: int) -> List[dict]:
    by_id = {str(d.get("id")): d for d in docs if d.get("summary_text") and d.get("qa_text")}
    overlap = [by_id[i] for i in human_ids if i in by_id]
    remaining_pool = [d for k, d in by_id.items() if k not in human_ids]
    rng = random.Random(seed)
    rng.shuffle(remaining_pool)
    need = max(0, total - len(overlap))
    return overlap + remaining_pool[:need]


def main() -> None:
    p = argparse.ArgumentParser(description="Pick baseline 200-doc subset")
    p.add_argument("--gpt-results", required=True)
    p.add_argument("--human-sample", required=True)
    p.add_argument("--total", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    docs = json.loads(Path(args.gpt_results).read_text(encoding="utf-8"))
    human_ids = _read_doc_ids(args.human_sample)
    subset = pick_subset(docs, human_ids, args.total, args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "total_requested": args.total,
        "n_human_overlap": sum(1 for d in subset if str(d.get("id")) in human_ids),
        "doc_ids": [str(d.get("id")) for d in subset],
        "docs": subset,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[baseline_subset] {len(subset)} docs → {args.output}")


if __name__ == "__main__":
    main()
