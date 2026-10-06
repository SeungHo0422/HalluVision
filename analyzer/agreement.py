"""
Inter-annotator agreement for two-annotator labeling sheets.

Serves two tasks with the same machinery:

* §7 human hallucination labeling (R2#3, R3) — default columns
  ``label_a`` / ``label_b`` with levels ``faithful/partial/hallucinated``.
* §2a NER subsample validation (R2#2a) — columns
  ``annotator_a_correct`` / ``annotator_b_correct`` with levels
  ``correct/partial/incorrect`` (pass ``--col-a/--col-b/--levels``).

Given a CSV with two annotators' labels (``col_a`` / ``col_b``)
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

    # §7 hallucination labels (defaults)
    python -m final_pipeline.analyzer.agreement \
        --labels notes/human_labels_2026-XX.csv \
        --output notes/human_validation.md

    # §2a NER validation labels
    python -m final_pipeline.analyzer.agreement \
        --labels notes/ner_validation_sample.csv \
        --levels correct,partial,incorrect \
        --col-a annotator_a_correct --col-b annotator_b_correct \
        --output notes/ner_validation_agreement.md
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default levels/columns preserve the §7 hallucination behavior; the CLI
# (and compute_report) can override them for other labeling tasks (e.g. §2a).
DEFAULT_LEVELS = ["faithful", "partial", "hallucinated"]
DEFAULT_COL_A = "label_a"
DEFAULT_COL_B = "label_b"
# Backward-compatible alias (some callers/tests reference LABEL_LEVELS).
LABEL_LEVELS = DEFAULT_LEVELS


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


def _majority_vote(a: str, b: str, levels: List[str]) -> Optional[str]:
    if not a or not b:
        return a or b or None
    if a == b:
        return a
    # tie-break: more severe label wins (severity = position in `levels`,
    # so the last level is the most severe — paper-conservative choice).
    severity = {lvl: i for i, lvl in enumerate(levels)}
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


def compute_report(
    rows: List[dict],
    metric_cols: List[str],
    col_a: str = DEFAULT_COL_A,
    col_b: str = DEFAULT_COL_B,
    levels: Optional[List[str]] = None,
) -> dict:
    levels = levels or DEFAULT_LEVELS
    # Positive (=1) for the metric-vs-vote binary target: any level above the
    # first (baseline) one. For hallucination this is {partial, hallucinated};
    # for NER validation the metric block is unused (no metric cols).
    baseline = levels[0]
    positive = set(levels[1:])

    a = [r.get(col_a, "").strip().lower() for r in rows]
    b = [r.get(col_b, "").strip().lower() for r in rows]
    both = [(x, y) for x, y in zip(a, b) if x in levels and y in levels]
    a_clean = [x for x, _ in both]
    b_clean = [y for _, y in both]

    overall_kappa = _cohen_kappa(a_clean, b_clean, levels)
    confusion = _confusion(a_clean, b_clean, levels)
    pct_agree = (sum(1 for x, y in both if x == y) / len(both)) if both else float("nan")

    per_class: Dict[str, dict] = {}
    for lbl in levels:
        a_bin = ["yes" if x == lbl else "no" for x in a_clean]
        b_bin = ["yes" if y == lbl else "no" for y in b_clean]
        per_class[lbl] = {
            "kappa": _cohen_kappa(a_bin, b_bin, ["yes", "no"]),
            "support_a": sum(1 for x in a_clean if x == lbl),
            "support_b": sum(1 for y in b_clean if y == lbl),
        }

    majority = [_majority_vote(x, y, levels) for x, y in zip(a, b)]
    majority_bin = [
        1 if (m in positive) else (0 if m == baseline else None)
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
        "levels": levels,
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
    levels = report.get("levels", DEFAULT_LEVELS)
    lines = [
        "# Inter-annotator Validation Report",
        "",
        f"Source labels: `{source_path}`",
        f"Total rows: {report['n_rows']}",
        f"Rows with both annotators: {report['n_dual_labeled']}",
        "",
        "## Inter-annotator agreement",
        "",
        f"- Overall Cohen's κ ({len(levels)}-class {'/'.join(levels)}): **{report['overall_kappa']:.3f}**",
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
        "| | " + " | ".join(levels) + " |",
        "|" + "---|" * (len(levels) + 1),
    ]
    for i, lbl in enumerate(levels):
        row = [lbl] + [str(report["confusion_matrix"][i][j]) for j in range(len(levels))]
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
    p.add_argument("--labels", required=True, help="CSV with two annotator columns + optional metric columns")
    p.add_argument("--metric-cols", nargs="*", default=["cosine_sim", "qa_precision", "freq_distance"],
                   help="Numeric columns to correlate against majority vote")
    p.add_argument("--levels", default=",".join(DEFAULT_LEVELS),
                   help="Comma-separated rating levels, least→most severe (default: %(default)s)")
    p.add_argument("--col-a", default=DEFAULT_COL_A, help="Annotator A column name")
    p.add_argument("--col-b", default=DEFAULT_COL_B, help="Annotator B column name")
    p.add_argument("--output", required=True, help="Markdown report path")
    args = p.parse_args()

    levels = [lvl.strip().lower() for lvl in args.levels.split(",") if lvl.strip()]
    rows = load_labels(args.labels)
    report = compute_report(rows, args.metric_cols, col_a=args.col_a, col_b=args.col_b, levels=levels)
    md = render_markdown(report, args.labels)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    print(f"[agreement] wrote {args.output}  (kappa={report['overall_kappa']:.3f}, n={report['n_dual_labeled']})")


if __name__ == "__main__":
    main()
