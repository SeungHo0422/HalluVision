"""
Per-article scatter/box of each \\ourtool{} metric against the human rating,
split by task.

x is the majority human label (faithful / partial / hallucinated), y is the
metric score, and every point is one of the 100 annotated articles.

  --task summary   cosine similarity, answer matching similarity (summary NER vs
                   source NER), frequency-based entity similarity  [3 panels]
  --task qa        answer matching similarity of the five Q&A answers  [1 panel]

  --style strip    jittered points + median bar
  --style box      quartile box with the points kept, faintly

Points are jittered horizontally (fixed seed) because the metrics take many tied
values -- the Q&A precision in particular is concentrated on 0, 1/3, 1/2, 2/3, 1.

Run with the conda env that has matplotlib::

    python scripts/plot_metric_by_label.py \
        --task summary --style box
"""

from __future__ import annotations

import argparse
import os
import importlib.util
import json
import random
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

REPO_ROOT = Path(os.environ.get("HALLUVISION_WORKSPACE", Path.cwd())).resolve()  # data workspace
SEED = 42

# Grouping of the human labels along x. "binary" is the severe-only definition the
# paper uses: only a majority *hallucinated* counts as a hallucination.
# Ordinal severity -> one hue, light to dark (sequential, not categorical).
GROUPINGS = {
    "three": [("Faithful", {"faithful"}, "#a9c9ec"),
              ("Partial", {"partial"}, "#5b9be0"),
              ("Hallucinated", {"hallucinated"}, "#1f5ea8")],
    "binary": [("Faithful + Partial", {"faithful", "partial"}, "#a9c9ec"),
               ("Hallucinated", {"hallucinated"}, "#1f5ea8")],
}
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#d9d8d2"

# Every metric below is a similarity: higher = closer to the source, so a metric
# that tracks the annotators should slope downward to the right.
TASKS = {
    "summary": ("Abstractive summarization", [
        ("cosine_sim", "Cosine similarity"),
        ("summary_precision", "Answer matching similarity"),
        ("freq_similarity", "Frequency-based entity similarity"),
    ]),
    "qa": ("Question \\& answering", [
        ("qa_precision", "Answer matching similarity"),
    ]),
}


def load(labels_path: Path):
    am_spec = importlib.util.spec_from_file_location(
        "_am", Path(__file__).resolve().parents[1] / "analyzer/agreement_multi.py")
    am = importlib.util.module_from_spec(am_spec)
    am_spec.loader.exec_module(am)

    mvh_spec = importlib.util.spec_from_file_location(
        "_mvh", Path(__file__).resolve().parent / "metric_vs_human.py")
    mvh = importlib.util.module_from_spec(mvh_spec)
    mvh_spec.loader.exec_module(mvh)

    rows, _, _ = mvh.load_rows(labels_path)

    # Summarization-side answer matching (summary NER vs source NER); the QA-side
    # precision already comes in from metric_vs_human.load_metrics().
    summ = json.loads((REPO_ROOT / "notes/doc_stats_summary_gpt4omini.json").read_text())
    for r in rows:
        fd = r.get("freq_distance")
        r["freq_similarity"] = None if fd is None else 1.0 - fd
        rec = summ.get(r["doc_id"], {}).get("_overall_", {})
        r["summary_precision"] = rec.get("precision")
    return rows, am


def draw(ax, rows, key, title, style, am, rng, groups) -> float:
    for i, (_, levels, color) in enumerate(groups):
        vals = [r[key] for r in rows if r["consensus"] in levels and r.get(key) is not None]
        if not vals:
            continue
        if style == "box":
            # Box carries the quartiles; the points stay, faintly, so the reader
            # can still see the ties the box hides.
            bp = ax.boxplot([vals], positions=[i], widths=0.5, showfliers=False,
                            patch_artist=True, zorder=3)
            bp["boxes"][0].set(facecolor=color, alpha=0.45,
                               edgecolor=MUTED, linewidth=0.9)
            for part in ("whiskers", "caps"):
                for ln in bp[part]:
                    ln.set(color=MUTED, linewidth=0.9)
            bp["medians"][0].set(color=INK, linewidth=1.8)
            ax.scatter([i + rng.uniform(-0.16, 0.16) for _ in vals], vals, s=8,
                       facecolor=color, edgecolor="white",
                       linewidth=0.3, alpha=0.75, zorder=4)
        else:
            ax.scatter([i + rng.uniform(-0.22, 0.22) for _ in vals], vals, s=14,
                       facecolor=color, edgecolor="white",
                       linewidth=0.4, alpha=0.9, zorder=3)
            ax.plot([i - 0.34, i + 0.34], [median(vals)] * 2, color=INK,
                    linewidth=1.8, zorder=4)
        ax.text(i + (0.28 if style == "box" else 0.34), median(vals), f" {median(vals):.2f}",
                fontsize=6.5, color=INK, va="center", ha="left", zorder=5,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    vals = [(r[key], r["consensus"]) for r in rows if r.get(key) is not None]
    if len(groups) == 2:  # binary split -> AUC is the natural summary
        stat = am._roc_auc([-v for v, _ in vals],
                           [1 if c == "hallucinated" else 0 for _, c in vals])
        note = f"AUC = {stat:.3f}"
    else:
        stat = am._spearman([v for v, _ in vals],
                            [float(am.SEVERITY[c]) for _, c in vals])
        note = f"$\\rho$ = {stat:+.3f}"
    ax.set_title(title, fontsize=8.5, color=INK, pad=5)
    ax.text(0.03, 0.04, f"{note}  ($n$={len(vals)})", transform=ax.transAxes,
            fontsize=6.5, color=MUTED, ha="left", va="bottom")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5, color=INK)
    ax.set_xlim(-0.65, len(groups) - 0.25)
    ax.tick_params(axis="y", labelsize=7, colors=MUTED, length=3)
    ax.tick_params(axis="x", length=0, colors=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    return stat


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=str(REPO_ROOT / "notes/human_labels_2026-08-01.csv"))
    p.add_argument("--task", choices=sorted(TASKS), default="summary")
    p.add_argument("--style", choices=["strip", "box"], default="strip")
    p.add_argument("--groups", choices=sorted(GROUPINGS), default="three",
                   help="three: faithful/partial/hallucinated; binary: (faithful+partial) vs hallucinated")
    p.add_argument("--out", default="", help="default: figures/figure_metric_by_label_<task>[_box].pdf")
    args = p.parse_args()

    rows, am = load(Path(args.labels))
    rng = random.Random(SEED)
    task_title, panels = TASKS[args.task]
    groups = GROUPINGS[args.groups]

    fig, axes = plt.subplots(1, len(panels), figsize=(2.35 * len(panels) + 0.7, 2.9),
                             squeeze=False)
    rhos = []
    for ax, (key, title) in zip(axes[0], panels):
        rhos.append((key, draw(ax, rows, key, title, args.style, am, rng, groups)))
    axes[0][0].set_ylabel("Metric score", fontsize=8.5, color=INK)

    fig.tight_layout()
    suffix = ("_bin" if args.groups == "binary" else "") + ("_box" if args.style == "box" else "")
    out = Path(args.out) if args.out else \
        REPO_ROOT / f"figures/figure_metric_by_label_{args.task}{suffix}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"[{args.task}/{args.groups}/{args.style}] "
          + "  ".join(f"{k}={v:+.3f}" for k, v in rhos)
          + f"  -> {out.name}")


if __name__ == "__main__":
    main()
