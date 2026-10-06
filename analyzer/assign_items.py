"""
Assign the frozen 100 human-validation items to N annotators (R2 #3).

Two designs, both emitting a unified ``assign`` map (per-annotator doc_id list)
that the web app consumes. Neither re-samples — both reuse the frozen
quartile-stratified order in ``notes/human_label_sample.csv``.

1) DISTRIBUTED redundancy (``--redundancy r``, recommended for many labelers):
   every item is labeled by exactly ``r`` annotators, spread evenly so each
   annotator's load = M*r/N and each annotator's chunk stays bucket-balanced.
   No single-labeled items; agreement is Krippendorff's alpha over all items
   with >=2 raters (i.e. all of them). Example: M=100, N=10, r=3 -> 30 each.

2) ANCHOR + disjoint (``--anchor A``): A items labeled by everyone (Fleiss-kappa
   base) + the rest dealt round-robin (r=1). Per-annotator load = A + (M-A)/N.

Usage::

    # distributed r=3 for 10 labelers -> 30 items each
    python final_pipeline/analyzer/assign_items.py \
        --annotators a,b,c,d,e,f,g,h,i,j --redundancy 3

    # anchor design (N>=3)
    python final_pipeline/analyzer/assign_items.py --annotators a,b,c,d --anchor 52
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import Counter
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_order(sample_path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(sample_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "doc_id": str(row["doc_id"]).strip(),
                    "cell": f"{row.get('sm_f1_bucket','?')}-{row.get('qa_p_bucket','?')}",
                }
            )
    return rows


def even_indices(n: int, k: int) -> List[int]:
    return sorted({(i * n) // k for i in range(k)})


def assign_anchor(order: List[dict], annotators: List[str], anchor: int) -> dict:
    n = len(order)
    if anchor >= n:
        raise SystemExit(f"--anchor ({anchor}) must be < total items ({n})")
    N = len(annotators)
    anchor_pos = set(even_indices(n, anchor))
    anchor_ids = [order[i]["doc_id"] for i in sorted(anchor_pos)]

    assign: Dict[str, List[str]] = {a: list(anchor_ids) for a in annotators}
    j = 0
    for i in range(n):
        if i in anchor_pos:
            continue
        assign[annotators[j % N]].append(order[i]["doc_id"])
        j += 1
    return {
        "version": "assign-v2",
        "mode": "anchor",
        "total_items": n,
        "annotators": annotators,
        "anchor": anchor_ids,
        "anchor_size": len(anchor_ids),
        "assign": assign,
    }


def assign_distributed(order: List[dict], annotators: List[str], r: int) -> dict:
    n = len(order)
    N = len(annotators)
    if not (2 <= r < N):
        raise SystemExit(f"--redundancy must satisfy 2 <= r < N (got r={r}, N={N})")
    # spread the r raters of an item across the annotator ring (offsets 0, s, 2s…)
    s = max(1, N // r)
    assign: Dict[str, List[str]] = {a: [] for a in annotators}
    coincide = Counter()  # unordered pair -> shared item count
    for i, item in enumerate(order):
        raters = [annotators[(i + j * s) % N] for j in range(r)]
        for a in raters:
            assign[a].append(item["doc_id"])
        for pa, pb in itertools.combinations(sorted(raters), 2):
            coincide[(pa, pb)] += 1
    return {
        "version": "assign-v2",
        "mode": "redundancy",
        "redundancy": r,
        "total_items": n,
        "total_labels": n * r,
        "annotators": annotators,
        "assign": assign,
        "_pair_overlap": {f"{a}-{b}": c for (a, b), c in sorted(coincide.items())},
    }


def report(order: List[dict], plan: dict) -> None:
    cell_of = {r["doc_id"]: r["cell"] for r in order}
    N = len(plan["annotators"])

    def cells(ids):
        return len(set(cell_of[d] for d in ids))

    print(f"[assign] mode={plan['mode']} | {plan['total_items']} items | {N} annotators")
    if plan["mode"] == "anchor":
        print(f"[assign] anchor {plan['anchor_size']} (labeled by ALL {N})")
    else:
        print(f"[assign] redundancy r={plan['redundancy']} | {plan['total_labels']} total labels")
    loads = []
    for a in plan["annotators"]:
        ids = plan["assign"][a]
        loads.append(len(ids))
        print(f"[assign]   {a}: {len(ids)} items across {cells(ids)} cells")
    print(f"[assign] per-annotator load: min={min(loads)} max={max(loads)}")
    if plan["mode"] == "redundancy":
        ov = list(plan["_pair_overlap"].values())
        nonzero = [v for v in ov if v]
        print(f"[assign] pair overlap: {sum(1 for v in ov if v)}/{len(ov)} pairs share items "
              f"(shared items min={min(nonzero) if nonzero else 0}, max={max(ov) if ov else 0}) "
              f"-> use Krippendorff alpha")


def main() -> None:
    p = argparse.ArgumentParser(description="Assign frozen items to N annotators (distributed or anchor)")
    p.add_argument("--sample", default=str(REPO_ROOT / "notes/human_label_sample.csv"))
    p.add_argument("--annotators", required=True, help="comma-separated ids, e.g. a,b,c,d,e,f,g,h,i,j")
    p.add_argument("--redundancy", type=int, default=0, help="distributed: labels per item (r>=2). Omit for anchor mode")
    p.add_argument("--anchor", type=int, default=52, help="anchor mode: items labeled by everyone")
    p.add_argument("--out", default=str(REPO_ROOT / "webapp/data/assignments.json"))
    args = p.parse_args()

    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    if len(annotators) < 2:
        raise SystemExit("need at least 2 annotators")

    order = read_order(Path(args.sample))
    if args.redundancy and args.redundancy >= 2:
        plan = assign_distributed(order, annotators, args.redundancy)
    else:
        plan = assign_anchor(order, annotators, args.anchor)

    import json
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    report(order, plan)
    print(f"[assign] wrote {out_path}")


if __name__ == "__main__":
    main()
