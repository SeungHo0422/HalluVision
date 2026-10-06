#!/usr/bin/env python3
"""Stratified Threatpost subset for the Llama-3 run (Reviewer 2 #4).

Picks N documents spanning the gold entity-density range so the open-source
model is not evaluated on an accidentally easy slice: documents are sorted by
their total gold-NER entity count, split into four equal strata, and sampled
uniformly within each (seed fixed).

The output keeps the `gpt_results_*.json` schema, so it feeds
`final_pipeline/run_llama_inference.py --source` directly and the same ids can
be filtered out of the GPT result files for a paired comparison.

    python make_llama_subset.py --n 500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpt-results", default="gpt_results_2503.json")
    ap.add_argument("--original-ner", default="original_ner.json")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="pipeline_output/llama/tp_subset_500.json")
    ap.add_argument("--ids-output", default="pipeline_output/llama/tp_subset_500_ids.json")
    args = ap.parse_args()

    docs = json.loads(Path(args.gpt_results).read_text(encoding="utf-8"))
    gold = json.loads(Path(args.original_ner).read_text(encoding="utf-8"))

    counted = []
    for d in docs:
        if not (d.get("original_text") or "").strip():
            continue
        g = gold.get(str(d["id"]), {})
        counted.append((sum(len(g.get(l, []) or []) for l in LABELS), d))
    counted.sort(key=lambda t: t[0])

    rng = random.Random(args.seed)
    per = args.n // 4
    picked = []
    for q in range(4):  # interleaved → four equal entity-density quartiles
        stratum = counted[q::4]
        rng.shuffle(stratum)
        picked.extend(stratum[:per])

    picked.sort(key=lambda t: int(t[1]["id"]))
    subset = [d for _, d in picked]
    counts = [c for c, _ in picked]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    Path(args.ids_output).write_text(
        json.dumps([d["id"] for d in subset]), encoding="utf-8"
    )

    counts_sorted = sorted(counts)
    print(f"{len(subset)} docs -> {out}")
    print(f"gold entity count: min {counts_sorted[0]}, "
          f"median {counts_sorted[len(counts_sorted)//2]}, max {counts_sorted[-1]}")
    print(f"corpus-wide median for comparison: "
          f"{sorted(c for c, _ in counted)[len(counted)//2]}")


if __name__ == "__main__":
    main()
