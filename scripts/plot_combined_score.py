"""
Figure for the combined score S (R2 #1), Discussion section.

Empirical CDF of S over the Threatpost corpus, one curve per model. This is the
output of Algorithm 1 and nothing else: no labels, no calibration, no threshold.
Numbers come from notes/combined_score_corpus.py, which must be run first.

Run with the conda env that has matplotlib::

    python scripts/plot_combined_score.py
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(os.environ.get("HALLUVISION_WORKSPACE", Path.cwd())).resolve()  # data workspace

# Validated categorical slots 1-2 (blue / orange); line style repeats the
# distinction so the figure survives greyscale printing.
SERIES = [
    ("gpt-4o-mini", "#2a78d6", "-"),
    ("gpt-3.5-turbo", "#eb6834", "--"),
]
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#d9d8d2"


def ecdf(xs: list[float]) -> tuple[list[float], list[float]]:
    ys = sorted(xs)
    n = len(ys)
    return ys, [(i + 1) / n for i in range(n)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(REPO_ROOT / "notes/combined_score_corpus.json"))
    p.add_argument("--out", default=str(REPO_ROOT / "figures/figure_combined_score.pdf"))
    args = p.parse_args()

    data = json.loads(Path(args.data).read_text())
    scores = data["scores"]
    summary = data["per_model"]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    for name, color, ls in SERIES:
        xs, ys = ecdf(list(scores[name].values()))
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2, zorder=3,
                label=name)
        # median tick on the curve, so the legend value is locatable
        ax.plot([summary[name]["median"]], [0.5], marker="o", markersize=4.5,
                color=color, markeredgecolor="white", markeredgewidth=1.0, zorder=4)

    ax.axhline(0.5, color=GRID, linewidth=1, zorder=1)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Combined score $S$  (higher = more hallucination)", fontsize=8.5, color=INK)
    ax.set_ylabel("Cumulative fraction of documents", fontsize=8.5, color=INK)
    ax.tick_params(labelsize=8, colors=MUTED, length=3)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    # above the axes: the CDF fills both corners a legend would otherwise sit in
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              labelcolor=INK, frameon=False, handlelength=2.2, columnspacing=1.4,
              borderaxespad=0.4)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"[combined] wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
