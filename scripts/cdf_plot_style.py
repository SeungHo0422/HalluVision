"""Shared style for the model-comparison CDF figures (paper Figures 3-7).

Reviewer 2 minor concern: the CDF figures look visually identical across
models, making them hard to tell apart. Fix here is presentational only —
a consistent line style per model (gpt-3.5-turbo solid, gpt-4o-mini dashed)
so curves are distinguishable in grayscale/print.

No threshold/cutoff annotation is drawn here on purpose: §4's threshold
framing was removed from the paper body per the R2 #7 response, and a
cutoff line on these figures would reintroduce exactly that.
"""
import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ["gpt-3.5-turbo", "gpt-4o-mini"]

MODEL_STYLE = {
    "gpt-3.5-turbo": dict(color="orange", linestyle="-"),
    "gpt-4o-mini": dict(color="green", linestyle="--"),
}


def plot_model_comparison_cdf(
    data,
    xlabel="Score",
    savepath=None,
    title=None,
    legend_loc="lower right",
    ax=None,
):
    """Single-panel CDF comparing gpt-3.5-turbo vs gpt-4o-mini.

    data: {"gpt-3.5-turbo": array-like, "gpt-4o-mini": array-like} of raw
    (unsorted) per-document scores.
    """
    plt.style.use("default")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    for model in MODEL_ORDER:
        vals = np.sort(np.asarray(data[model], dtype=float))
        vals = vals[~np.isnan(vals)]
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=model, linewidth=1.8, **MODEL_STYLE[model])

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Cumulative Probability", fontsize=9)
    ax.grid(True, linestyle="-", alpha=0.5)
    ax.legend(loc=legend_loc, prop={"size": 9})

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300)
    return fig, ax
