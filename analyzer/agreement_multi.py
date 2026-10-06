"""
N-annotator inter-annotator agreement for the human hallucination labels
(R2 #3, anchor+disjoint design with N>=3 raters).

Reads a CSV with one ``label_<id>`` column per annotator (produced by
``merge_human_labels.py``) plus optional metric columns
(``cosine_sim, qa_precision, freq_distance``). Computes:

* Fleiss' kappa on the ANCHOR set (items rated by all N annotators)
* Pairwise Cohen's kappa for every annotator pair (on their common items)
* Overall percent agreement (mean over pairs)
* Majority/consensus label per item (severity tie-break: hallucinated >
  partial > faithful), across whoever labeled that item
* Spearman rho + ROC AUC between each metric and the consensus binary target
  (hallucinated / partial = 1, faithful = 0)

Pure-Python (no sklearn/scipy). For the 2-rater case use ``agreement.py``.

Usage::

    python final_pipeline/analyzer/agreement_multi.py \
        --labels notes/human_labels_2026-07-11.csv \
        --output notes/human_validation.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

LABEL_LEVELS = ["faithful", "partial", "hallucinated"]
HALLUC_POSITIVE = {"partial", "hallucinated"}
SEVERITY = {"faithful": 0, "partial": 1, "hallucinated": 2}


# ---- reused small stats helpers (kept local to avoid package import issues) ----
def _cohen_kappa(a: List[str], b: List[str], labels: List[str]) -> float:
    if not a:
        return float("nan")
    n = len(a)
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in labels)
    return (obs - pe) / (1 - pe) if pe < 1.0 else 1.0


def _fleiss_kappa(items: List[List[str]], labels: List[str]) -> float:
    """Fleiss' kappa. ``items`` = list of items, each a list of category labels
    of identical length n (the number of raters)."""
    if not items:
        return float("nan")
    n = len(items[0])
    if any(len(it) != n for it in items) or n < 2:
        return float("nan")
    k = len(labels)
    idx = {c: i for i, c in enumerate(labels)}
    n_items = len(items)
    total = n_items * n
    col = [0] * k
    p_bar_terms = []
    for it in items:
        counts = [0] * k
        for lbl in it:
            counts[idx[lbl]] += 1
        p_i = (sum(c * c for c in counts) - n) / (n * (n - 1))
        p_bar_terms.append(p_i)
        for j in range(k):
            col[j] += counts[j]
    p_j = [c / total for c in col]
    p_bar = sum(p_bar_terms) / n_items
    p_e = sum(pj * pj for pj in p_j)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1.0 else 1.0


def _krippendorff_alpha(items: List[List[str]], labels: List[str]) -> float:
    """Nominal Krippendorff's alpha over a rotating/incomplete design.

    ``items`` = list of per-item label lists (only items with >=2 raters are
    used; item rater counts may differ). Builds the coincidence matrix and
    returns 1 - D_o/D_e for the nominal metric.
    """
    idx = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    o = [[0.0] * k for _ in range(k)]  # coincidence matrix
    for it in items:
        vals = [v for v in it if v in idx]
        m = len(vals)
        if m < 2:
            continue
        counts = [0] * k
        for v in vals:
            counts[idx[v]] += 1
        for c in range(k):
            for kk in range(k):
                if c == kk:
                    o[c][kk] += counts[c] * (counts[c] - 1) / (m - 1)
                else:
                    o[c][kk] += counts[c] * counts[kk] / (m - 1)
    n_c = [sum(o[c]) for c in range(k)]
    n = sum(n_c)
    if n < 2:
        return float("nan")
    sum_o_diag = sum(o[c][c] for c in range(k))
    d_o = n - sum_o_diag
    d_e = (n * n - sum(nc * nc for nc in n_c)) / (n - 1)
    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e


def _majority(labels_present: List[str]) -> Optional[str]:
    if not labels_present:
        return None
    counts = Counter(labels_present)
    top = max(counts.values())
    tied = [l for l, c in counts.items() if c == top]
    return max(tied, key=lambda x: SEVERITY.get(x, -1))


def _spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")

    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for m in range(i, j + 1):
                r[order[m]] = avg
            i = j + 1
        return r

    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def _roc_auc(scores: List[float], labels: List[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    pairs = sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg)
    return pairs / (len(pos) * len(neg))


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def detect_annotators(fieldnames: List[str]) -> List[str]:
    ids = []
    for f in fieldnames or []:
        m = re.fullmatch(r"label_([A-Za-z0-9]+)", f)
        if m:
            ids.append(m.group(1))
    return ids


def compute_report(rows: List[dict], annotators: List[str], metric_cols: List[str]) -> dict:
    # per-item present labels
    per_item: List[Dict[str, str]] = []
    for r in rows:
        labs = {}
        for a in annotators:
            v = (r.get(f"label_{a}", "") or "").strip().lower()
            if v in LABEL_LEVELS:
                labs[a] = v
        per_item.append(labs)

    # Krippendorff's alpha over every item with >=2 raters (handles the rotating
    # distributed design). This is the primary reliability statistic when there
    # is no full anchor.
    multi_rated = [list(labs.values()) for labs in per_item if len(labs) >= 2]
    alpha = _krippendorff_alpha(multi_rated, LABEL_LEVELS)

    # anchor = items rated by all N annotators (Fleiss only meaningful if these exist)
    anchor_items = [[labs[a] for a in annotators] for labs in per_item if len(labs) == len(annotators)]
    fleiss = _fleiss_kappa(anchor_items, LABEL_LEVELS)

    # pairwise Cohen on common items
    pairwise = {}
    for a, b in itertools.combinations(annotators, 2):
        xa, xb = [], []
        for labs in per_item:
            if a in labs and b in labs:
                xa.append(labs[a])
                xb.append(labs[b])
        pairwise[f"{a}-{b}"] = {
            "kappa": _cohen_kappa(xa, xb, LABEL_LEVELS),
            "n": len(xa),
            "pct_agree": (sum(1 for x, y in zip(xa, xb) if x == y) / len(xa)) if xa else float("nan"),
        }

    mean_pct = (
        sum(p["pct_agree"] for p in pairwise.values() if not math.isnan(p["pct_agree"]))
        / max(1, sum(1 for p in pairwise.values() if not math.isnan(p["pct_agree"])))
    ) if pairwise else float("nan")

    # consensus per item + metric correlation
    consensus = [_majority(list(labs.values())) for labs in per_item]
    consensus_bin = [1 if c in HALLUC_POSITIVE else (0 if c == "faithful" else None) for c in consensus]

    metric_summary = {}
    for col in metric_cols:
        usable = []
        for i, r in enumerate(rows):
            s = _to_float(r.get(col, ""))
            l = consensus_bin[i]
            if s is not None and l is not None:
                usable.append((s, l))
        if not usable:
            continue
        scores = [s for s, _ in usable]
        labels = [l for _, l in usable]
        metric_summary[col] = {
            "n": len(usable),
            "spearman_vs_consensus": _spearman(scores, [float(l) for l in labels]),
            "auc_halluc_vs_faithful": _roc_auc(scores, labels),
        }

    return {
        "annotators": annotators,
        "n_rows": len(rows),
        "n_multi_rated": len(multi_rated),
        "krippendorff_alpha": alpha,
        "n_anchor_complete": len(anchor_items),
        "fleiss_kappa": fleiss,
        "pairwise": pairwise,
        "mean_pairwise_pct_agreement": mean_pct,
        "consensus_distribution": dict(Counter(c for c in consensus if c)),
        "metric_correlation": metric_summary,
        "coverage": {a: sum(1 for labs in per_item if a in labs) for a in annotators},
    }


def render_markdown(rep: dict, source: str) -> str:
    A = rep["annotators"]
    L = [
        "# Human Validation Report (R2#3) — N annotators",
        "",
        f"Source labels: `{source}`",
        f"Annotators ({len(A)}): {', '.join(A)}",
        f"Total items: {rep['n_rows']}  ·  multi-rated (≥2): {rep['n_multi_rated']}  ·  anchor (all-{len(A)}): {rep['n_anchor_complete']}",
        "",
        "## Inter-annotator agreement",
        "",
        f"- **Krippendorff's α (nominal, 3-class): {rep['krippendorff_alpha']:.3f}**  (n={rep['n_multi_rated']} multi-rated items) — primary",
        (
            f"- Fleiss' κ on anchor (3-class): **{rep['fleiss_kappa']:.3f}**  (n={rep['n_anchor_complete']})"
            if rep["n_anchor_complete"] > 0
            else "- Fleiss' κ: n/a (distributed design, no full anchor — use α above)"
        ),
        f"- Mean pairwise percent agreement: **{rep['mean_pairwise_pct_agreement']:.2%}**",
        "",
        "### Pairwise Cohen's κ",
        "",
        "| pair | n | κ | % agree |",
        "|---|---:|---:|---:|",
    ]
    for pair, s in rep["pairwise"].items():
        L.append(f"| {pair} | {s['n']} | {s['kappa']:.3f} | {s['pct_agree']:.2%} |")

    L += ["", "### Coverage per annotator", "", "| annotator | items labeled |", "|---|---:|"]
    for a in A:
        L.append(f"| {a} | {rep['coverage'][a]} |")

    L += ["", "## Consensus (majority-vote) distribution", "", "| label | count |", "|---|---:|"]
    for lbl, c in sorted(rep["consensus_distribution"].items()):
        L.append(f"| {lbl} | {c} |")

    if rep["metric_correlation"]:
        L += [
            "",
            "## Metric vs consensus",
            "",
            "Binary target: `hallucinated` ∪ `partial` = 1, `faithful` = 0.",
            "",
            "| metric | n | Spearman ρ | AUC (halluc vs faithful) |",
            "|---|---:|---:|---:|",
        ]
        for col, s in rep["metric_correlation"].items():
            L.append(
                f"| {col} | {s['n']} | {s['spearman_vs_consensus']:.3f} | {s['auc_halluc_vs_faithful']:.3f} |"
            )
    L.append("")
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser(description="N-annotator agreement (Fleiss + pairwise Cohen)")
    p.add_argument("--labels", required=True)
    p.add_argument("--annotators", default="", help="comma-separated ids; default = auto-detect label_* columns")
    p.add_argument("--metric-cols", nargs="*", default=["cosine_sim", "qa_precision", "freq_distance"])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.labels, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()] or detect_annotators(fieldnames)
    if len(annotators) < 2:
        raise SystemExit(f"need >=2 annotators; detected {annotators}")

    rep = compute_report(rows, annotators, args.metric_cols)
    md = render_markdown(rep, args.labels)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    print(f"[agreement_multi] wrote {args.output}  (fleiss={rep['fleiss_kappa']:.3f}, anchor={rep['n_anchor_complete']}, N={len(annotators)})")


if __name__ == "__main__":
    main()
