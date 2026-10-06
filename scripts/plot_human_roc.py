"""
ROC figure backing the human-validation AUC numbers in the Discussion section.

(a) all three metrics against the human consensus on clear hallucination, over
    the full 100-document sample
(b) cosine similarity on the full sample vs the unanimous subset, i.e. the same
    metric measured against a cleaner target

Every metric is a similarity (higher = closer to the source), so the ROC uses the
negated score: a document is "flagged" when its similarity falls below the sweep
threshold. AUC is the Mann-Whitney statistic with ties counted as 0.5, matching
notes/metric_vs_human.py.

Run with the conda env that has matplotlib::

    python scripts/plot_human_roc.py
"""

from __future__ import annotations

import argparse
import os
import csv
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(os.environ.get("HALLUVISION_WORKSPACE", Path.cwd())).resolve()  # data workspace

_spec = importlib.util.spec_from_file_location(
    "_agreement_multi", Path(__file__).resolve().parents[1] / "analyzer/agreement_multi.py"
)
_am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_am)

# Validated categorical slots 1-3 (blue / orange / aqua). Line style carries the
# same distinction so the figure survives greyscale printing.
SERIES = [
    ("Cosine similarity", "Cosine", "#2a78d6", "-"),
    ("Answer matching similarity", "Answer matching", "#eb6834", "--"),
    ("Frequency-based entity similarity", "Frequency-based", "#1baf7a", ":"),
]
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#d9d8d2"


def load(labels_path: Path) -> list[dict]:
    mvh = importlib.util.spec_from_file_location("_mvh", Path(__file__).resolve().parent / "metric_vs_human.py")
    m = importlib.util.module_from_spec(mvh)
    mvh.loader.exec_module(m)
    metrics = m.load_metrics()

    rows = []
    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        annotators = _am.detect_annotators(reader.fieldnames)
        for r in reader:
            labs = [(r.get(f"label_{a}", "") or "").strip().lower() for a in annotators]
            labs = [v for v in labs if v in _am.LABEL_LEVELS]
            if not labs:
                continue
            cons = _am._majority(labs)
            met = metrics.get(r["doc_id"], {})
            fd = met.get("freq_distance")
            rows.append({
                "y_severe": 1 if cons == "hallucinated" else 0,
                "y_any": 0 if cons == "faithful" else 1,
                "unanimous": len(set(labs)) == 1,
                "Cosine similarity": met.get("cosine_sim"),
                "Answer matching similarity": met.get("qa_precision"),
                "Frequency-based entity similarity": None if fd is None else 1.0 - fd,
            })
    return rows


def roc(scores: list[float], y: list[int]) -> tuple[list[float], list[float], float]:
    """Sweep every distinct score; return (fpr, tpr, auc). `scores` is already
    oriented so that a higher value means more hallucination."""
    pos = [s for s, l in zip(scores, y) if l == 1]
    neg = [s for s, l in zip(scores, y) if l == 0]
    auc = sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg) / (len(pos) * len(neg))
    fpr, tpr = [0.0], [0.0]
    for t in sorted(set(scores), reverse=True):
        tpr.append(sum(1 for p in pos if p >= t) / len(pos))
        fpr.append(sum(1 for n in neg if n >= t) / len(neg))
    fpr.append(1.0)
    tpr.append(1.0)
    return fpr, tpr, auc


def series(rows: list[dict], key: str, target: str = "severe") -> tuple[list[float], list[int]]:
    xs = [(-r[key], r[f"y_{target}"]) for r in rows if r[key] is not None]
    return [a for a, _ in xs], [b for _, b in xs]


def legend(ax) -> None:
    """Lower-right, on an opaque plate so a curve passing underneath cannot make
    the AUC values unreadable in print."""
    lg = ax.legend(fontsize=7, loc="lower right", labelcolor=INK, frameon=True,
                   framealpha=1.0, borderpad=0.5, handlelength=2.2)
    lg.get_frame().set_facecolor("white")
    lg.get_frame().set_edgecolor(GRID)
    lg.get_frame().set_linewidth(0.5)
    lg.set_zorder(5)


def style_axes(ax, title: str | None) -> None:
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate", fontsize=9, color=INK)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, pad=6)
    ax.tick_params(labelsize=8, colors=MUTED, length=3)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def panel_a(ax, rows: list[dict], title: str | None, target: str) -> None:
    """Three metrics, full sample."""
    for name, short, color, ls in SERIES:
        xs, ys = series(rows, name, target)
        fpr, tpr, auc = roc(xs, ys)
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2, zorder=3,
                label=f"{short} (AUC {auc:.3f})")
    style_axes(ax, title)
    ax.set_ylabel("True positive rate", fontsize=9, color=INK)
    legend(ax)


def panel_b(ax, rows: list[dict], unan: list[dict], title: str | None, target: str) -> None:
    """Cosine similarity, full sample vs unanimous subset."""
    for subset, label, ls, color in ((rows, "Full sample ($n$=100)", "-", "#2a78d6"),
                                     (unan, f"Unanimous subset ($n$={len(unan)})", "--", "#eb6834")):
        xs, ys = series(subset, "Cosine similarity", target)
        fpr, tpr, auc = roc(xs, ys)
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2, zorder=3,
                label=f"{label} (AUC {auc:.3f})")
    style_axes(ax, title)
    ax.set_ylabel("True positive rate", fontsize=9, color=INK)
    legend(ax)


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[roc] wrote {out} and {out.with_suffix('.png')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=str(REPO_ROOT / "notes/human_labels_2026-08-01.csv"))
    p.add_argument("--out", default="")
    p.add_argument("--target", choices=["severe", "any"], default="severe",
                   help="severe: hallucinated vs rest; any: hallucinated+partial vs faithful")
    p.add_argument("--combined", action="store_true",
                   help="also write the old single two-panel figure")
    args = p.parse_args()

    rows = load(Path(args.labels))
    unan = [r for r in rows if r["unanimous"]]
    out = Path(args.out) if args.out else \
        REPO_ROOT / ("figures/figure_human_roc.pdf" if args.target == "severe"
                     else "figures/figure_human_roc_any.pdf")

    # One panel per file, each sized for a single \linewidth column so the two can
    # be stacked as \subfloat's in a single-column figure (the panel letter comes
    # from the subfloat caption, hence no in-figure title).
    for suffix, draw in (("_a", lambda ax: panel_a(ax, rows, None, args.target)),
                         ("_b", lambda ax: panel_b(ax, rows, unan, None, args.target))):
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        draw(ax)
        save(fig, out.with_name(f"{out.stem}{suffix}{out.suffix}"))

    if args.combined:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.1))
        panel_a(ax1, rows, "(a) All metrics, full sample ($n$=100)", args.target)
        panel_b(ax2, rows, unan, "(b) Cosine similarity by label agreement", args.target)
        ax2.set_ylabel("")
        save(fig, out)


if __name__ == "__main__":
    main()
