"""
Hallucination rate by metric quartile.

For each metric: sort the annotated documents by score (ascending, so Q1 is the
lowest-scoring quarter), split into four equal-count bins, and plot the share of
each bin whose majority human label is *hallucinated*. The dashed line is the
base rate of the documents plotted, so a bar above the line means that quarter of
the score range is enriched in human-confirmed hallucinations.

Ties matter here: answer matching similarity in particular puts many documents on
the same value, so a bin edge can fall inside a run of equal scores. Ordering is
made deterministic by breaking ties on doc_id, and the per-bin score range is
printed in the console so the arbitrary cuts stay visible.

Run with the conda env that has matplotlib::

    python scripts/plot_quartile_rate.py --task summary
"""

from __future__ import annotations

import argparse
import os
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(os.environ.get("HALLUVISION_WORKSPACE", Path.cwd())).resolve()  # data workspace

_spec = importlib.util.spec_from_file_location(
    "_by_label", Path(__file__).resolve().parent / "plot_metric_by_label.py")
_bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bl)

TASKS = _bl.TASKS
INK, MUTED, GRID = _bl.INK, _bl.MUTED, _bl.GRID
BAR = "#1f5ea8"       # hallucinated
BAR_REST = "#a9c9ec"  # partial + faithful
N_BINS = 4

# Which consensus labels count as a positive. "severe" is the definition the paper
# uses; "any" also counts partially supported outputs.
TARGETS = {
    "severe": ({"hallucinated"}, "Hallucinated", "Partial + Faithful", "Share hallucinated"),
    "any": ({"hallucinated", "partial"}, "Hallucinated + Partial", "Faithful",
            "Share hallucinated or partial"),
}


def bins(rows: list[dict], key: str, positive: set, n_bins: int = N_BINS) -> list[dict]:
    vals = [(r[key], r["doc_id"], r["consensus"]) for r in rows if r.get(key) is not None]
    vals.sort(key=lambda t: (t[0], t[1]))  # deterministic within ties
    n = len(vals)
    out = []
    for b in range(n_bins):
        lo, hi = b * n // n_bins, (b + 1) * n // n_bins
        chunk = vals[lo:hi]
        halluc = sum(1 for _, _, c in chunk if c in positive)
        out.append({"n": len(chunk), "halluc": halluc, "rate": halluc / len(chunk),
                    "lo": chunk[0][0], "hi": chunk[-1][0]})
    return out


def draw(ax, rows, key, title, style="rate", target="severe") -> list[dict]:
    positive, pos_label, neg_label, _ = TARGETS[target]
    bs = bins(rows, key, positive)
    base = sum(1 for r in rows if r.get(key) is not None and r["consensus"] in positive) \
        / sum(1 for r in rows if r.get(key) is not None)

    xs = range(len(bs))
    if style == "stacked":
        # 100% stacked: the two label groups of the severe-only definition, with a
        # surface gap so the segments read as separate marks.
        ax.bar(xs, [b["rate"] for b in bs], width=0.62, color=BAR, zorder=3,
               label=pos_label)
        ax.bar(xs, [1 - b["rate"] for b in bs], width=0.62, color=BAR_REST, zorder=3,
               bottom=[b["rate"] + 0.006 for b in bs], label=neg_label)
        for x, b in zip(xs, bs):
            ax.text(x, b["rate"] / 2, f"{b['halluc']}", ha="center", va="center",
                    fontsize=6.5, color="white", zorder=4)
            ax.text(x, b["rate"] + (1 - b["rate"]) / 2, f"{b['n'] - b['halluc']}",
                    ha="center", va="center", fontsize=6.5, color=INK, zorder=4)
    else:
        ax.bar(xs, [b["rate"] for b in bs], width=0.62, color=BAR, zorder=3)
        for x, b in zip(xs, bs):
            ax.text(x, b["rate"] + 0.015, f"{b['halluc']}/{b['n']}", ha="center", va="bottom",
                    fontsize=6.5, color=INK, zorder=4)
    ax.axhline(base, color=INK if style == "rate" else "#e34948", linestyle="--",
               linewidth=1, zorder=5)
    ax.text(len(bs) - 0.45, base + 0.012, f"base rate {base:.2f}", ha="right", va="bottom",
            fontsize=6.5, color=MUTED if style == "rate" else "#e34948", zorder=5)

    ax.set_title(title, fontsize=8.5, color=INK, pad=5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"Q{i+1}" for i in xs], fontsize=8, color=INK)

    # Score range of each bin, on its own line under the tick label. Narrow-range
    # metrics (cosine) need a third decimal; a bin printed as "1.00-1.00" is a bin
    # whose edges fall inside a run of tied scores.
    spread = max(b["hi"] for b in bs) - min(b["lo"] for b in bs)
    nd = 3 if spread < 0.3 else 2
    for x, b in zip(xs, bs):
        ax.annotate(f"{b['lo']:.{nd}f}\n-{b['hi']:.{nd}f}", xy=(x, 0), xytext=(0, -18),
                    xycoords=("data", "axes fraction"), textcoords="offset points",
                    ha="center", va="top", fontsize=6, color=MUTED, annotation_clip=False)
    ax.set_xlim(-0.6, len(bs) - 0.4)
    ax.set_ylim(0, 1.06 if style == "stacked" else max(0.62, max(b["rate"] for b in bs) + 0.1))
    ax.tick_params(axis="y", labelsize=7, colors=MUTED, length=3)
    ax.tick_params(axis="x", length=0, colors=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    return bs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=str(REPO_ROOT / "notes/human_labels_2026-08-01.csv"))
    p.add_argument("--task", choices=sorted(TASKS), default="summary")
    p.add_argument("--metrics", default="", help="comma-separated metric keys; default = all of the task")
    p.add_argument("--target", choices=sorted(TARGETS), default="severe",
                   help="severe: hallucinated only; any: hallucinated + partial")
    p.add_argument("--style", choices=["rate", "stacked"], default="rate",
                   help="rate: share hallucinated; stacked: both label groups to 100%%")
    p.add_argument("--out", default="")
    args = p.parse_args()

    rows, _ = _bl.load(Path(args.labels))
    _, panels = TASKS[args.task]
    if args.metrics:
        wanted = [m.strip() for m in args.metrics.split(",")]
        panels = [pn for pn in panels if pn[0] in wanted]

    fig, axes = plt.subplots(1, len(panels), figsize=(2.25 * len(panels) + 0.7, 2.7),
                             squeeze=False)
    for ax, (key, title) in zip(axes[0], panels):
        bs = draw(ax, rows, key, title, args.style, args.target)
        print(f"[{key}] " + "  ".join(
            f"Q{i+1}: {b['halluc']}/{b['n']} = {b['rate']:.2f} (score {b['lo']:.3f}-{b['hi']:.3f})"
            for i, b in enumerate(bs)))
    axes[0][0].set_ylabel(TARGETS[args.target][3] if args.style == "rate" else "Share of documents",
                          fontsize=8.5, color=INK)
    if args.style == "stacked":
        axes[0][-1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28),
                           ncol=2, frameon=False, labelcolor=INK)

    fig.tight_layout()
    stem = (args.metrics.replace(",", "-") if args.metrics else args.task) \
        + ("_any" if args.target == "any" else "") \
        + ("_stacked" if args.style == "stacked" else "")
    out = Path(args.out) if args.out else REPO_ROOT / f"figures/figure_quartile_rate_{stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"[quartile] wrote {out}")


if __name__ == "__main__":
    main()
