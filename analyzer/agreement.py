"""
Inter-annotator agreement for the human hallucination labels (R2#3, R3).

Given a CSV with two annotators' labels (``label_a`` / ``label_b``)
plus optional metric columns (``cosine_sim``, ``qa_precision``,
``freq_distance``), this module computes:

* Cohen's kappa (overall and per-label, treating each rating level as
  a class)
* Percent agreement and a confusion matrix
* Majority-vote label per doc (used downstream by §9 combined scorer)
* Spearman correlation + ROC AUC between each metric and the majority
  vote treated as a binary label (hallucinated/partial = 1, faithful = 0)

The implementation deliberately avoids importing sklearn or scipy at
module load so the CLI still works on a minimal install. When those
libraries are missing the correlation / AUC numbers are skipped with
a clear note rather than the whole report failing.

Usage::

    python -m final_pipeline.analyzer.agreement \
        --labels notes/human_labels_2026-XX.csv \
        --output notes/human_validation.md
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LABEL_LEVELS = ["faithful", "partial", "hallucinated"]
HALLUC_POSITIVE = {"partial", "hallucinated"}


def _cohen_kappa(a: List[str], b: List[str], labels: List[str]) -> float:
    if not a:
        return float("nan")
    idx = {lbl: i for i, lbl in enumerate(labels)}
    n = len(a)
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    a_counts = Counter(a)
    b_counts = Counter(b)
    pe = sum((a_counts.get(lbl, 0) / n) * (b_counts.get(lbl, 0) / n) for lbl in labels)
    return (obs - pe) / (1 - pe) if pe < 1.0 else 1.0


def _confusion(a: List[str], b: List[str], labels: List[str]) -> List[List[int]]:
    idx = {lbl: i for i, lbl in enumerate(labels)}
    mat = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        if x in idx and y in idx:
            mat[idx[x]][idx[y]] += 1
    return mat


def _majority_vote(a: str, b: str) -> Optional[str]:
    if not a or not b:
        return a or b or None
    if a == b:
        return a
    # tie-break: more severe label wins (paper conservative: count as halluc)
    severity = {"faithful": 0, "partial": 1, "hallucinated": 2}
    return max(a, b, key=lambda x: severity.get(x, -1))


def _spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")

    def _rank(values: List[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _roc_auc(scores: List[float], labels: List[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    pairs = 0
    for p in pos:
        for n in neg:
            if p > n:
                pairs += 1
            elif p == n:
                pairs += 0.5
    return pairs / (len(pos) * len(neg))


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_labels(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def compute_report(rows: List[dict], metric_cols: List[str]) -> dict:
    a = [r.get("label_a", "").strip().lower() for r in rows]
    b = [r.get("label_b", "").strip().lower() for r in rows]
    both = [(x, y) for x, y in zip(a, b) if x in LABEL_LEVELS and y in LABEL_LEVELS]
    a_clean = [x for x, _ in both]
    b_clean = [y for _, y in both]

    overall_kappa = _cohen_kappa(a_clean, b_clean, LABEL_LEVELS)
    confusion = _confusion(a_clean, b_clean, LABEL_LEVELS)
    pct_agree = (sum(1 for x, y in both if x == y) / len(both)) if both else float("nan")

    per_class: Dict[str, dict] = {}
    for lbl in LABEL_LEVELS:
        a_bin = ["yes" if x == lbl else "no" for x in a_clean]
        b_bin = ["yes" if y == lbl else "no" for y in b_clean]
        per_class[lbl] = {
            "kappa": _cohen_kappa(a_bin, b_bin, ["yes", "no"]),
            "support_a": sum(1 for x in a_clean if x == lbl),
            "support_b": sum(1 for y in b_clean if y == lbl),
        }

    majority = [_majority_vote(x, y) for x, y in zip(a, b)]
    majority_bin = [
        1 if (m in HALLUC_POSITIVE) else (0 if m == "faithful" else None)
        for m in majority
    ]

    metric_summary: Dict[str, dict] = {}
    for col in metric_cols:
        scored = [
            (_to_float(r.get(col, "")), majority_bin[i])
            for i, r in enumerate(rows)
        ]
        usable = [(s, l) for s, l in scored if s is not None and l is not None]
        if not usable:
            continue
        scores = [s for s, _ in usable]
        labels = [l for _, l in usable]
        metric_summary[col] = {
            "n": len(usable),
            "spearman_vs_majority": _spearman(scores, [float(l) for l in labels]),
            "auc_halluc_vs_faithful": _roc_auc(scores, labels),
        }

    return {
        "n_rows": len(rows),
        "n_dual_labeled": len(both),
        "overall_kappa": overall_kappa,
        "percent_agreement": pct_agree,
        "confusion_matrix": confusion,
        "per_class_kappa": per_class,
        "metric_correlation": metric_summary,
        "majority_distribution": dict(Counter(m for m in majority if m)),
    }


def render_markdown(report: dict, source_path: str) -> str:
    lines = [
        "# Human Validation Report (R2#3, R3)",
        "",
        f"Source labels: `{source_path}`",
        f"Total rows: {report['n_rows']}",
        f"Rows with both annotators: {report['n_dual_labeled']}",
        "",
        "## Inter-annotator agreement",
        "",
        f"- Overall Cohen's κ (3-class faithful/partial/hallucinated): **{report['overall_kappa']:.3f}**",
        f"- Percent agreement: **{report['percent_agreement']:.2%}**",
        "",
        "### Per-class one-vs-rest κ",
        "",
        "| Label | κ | support (A) | support (B) |",
        "|---|---:|---:|---:|",
    ]
    for lbl, stats in report["per_class_kappa"].items():
        lines.append(
            f"| {lbl} | {stats['kappa']:.3f} | {stats['support_a']} | {stats['support_b']} |"
        )

    lines += [
        "",
        "### Confusion matrix (rows = annotator A, cols = annotator B)",
        "",
        "| | " + " | ".join(LABEL_LEVELS) + " |",
        "|" + "---|" * (len(LABEL_LEVELS) + 1),
    ]
    for i, lbl in enumerate(LABEL_LEVELS):
        row = [lbl] + [str(report["confusion_matrix"][i][j]) for j in range(len(LABEL_LEVELS))]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Majority-vote distribution",
        "",
        "| label | count |",
        "|---|---:|",
    ]
    for lbl, count in sorted(report["majority_distribution"].items()):
        lines.append(f"| {lbl} | {count} |")

    if report["metric_correlation"]:
        lines += [
            "",
            "## Metric vs majority-vote",
            "",
            "Binary target: `hallucinated` ∪ `partial` = 1, `faithful` = 0.",
            "",
            "| metric | n | Spearman ρ | AUC (halluc vs faithful) |",
            "|---|---:|---:|---:|",
        ]
        for col, s in report["metric_correlation"].items():
            lines.append(
                f"| {col} | {s['n']} | {s['spearman_vs_majority']:.3f} | {s['auc_halluc_vs_faithful']:.3f} |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Inter-annotator agreement + metric correlation")
    p.add_argument("--labels", required=True, help="CSV with label_a/label_b + optional metric columns")
    p.add_argument("--metric-cols", nargs="*", default=["cosine_sim", "qa_precision", "freq_distance"],
                   help="Numeric columns to correlate against majority vote")
    p.add_argument("--output", required=True, help="Markdown report path")
    args = p.parse_args()

    rows = load_labels(args.labels)
    report = compute_report(rows, args.metric_cols)
    md = render_markdown(report, args.labels)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    print(f"[agreement] wrote {args.output}  (kappa={report['overall_kappa']:.3f}, n={report['n_dual_labeled']})")


if __name__ == "__main__":
    main()
