"""
R2 #1 — is the equal-weight combined score S better than its best single component?

The response to R2 #1 no longer claims that combining the three metrics improves
detection; it presents S as a formulation of how the three measurements reduce to
one decision. This script produces the number that backs that framing: the paired
AUC difference between S and each individual metric on the human-labelled sample,
with a paired bootstrap CI, computed on the *same* documents (the 94 for which all
three metrics are defined) so the comparison is apples-to-apples.

It reuses the loaders in ``notes/metric_vs_human.py`` — same metric sources, same
sign convention (higher = more hallucination), same consensus labels.

Usage::

    python scripts/combined_vs_single.py --labels notes/human_labels_2026-08-01.csv
"""

from __future__ import annotations

import argparse
import os
import importlib.util
import json
import random
from pathlib import Path

REPO_ROOT = Path(os.environ.get("HALLUVISION_WORKSPACE", Path.cwd())).resolve()  # data workspace

_spec = importlib.util.spec_from_file_location("_mvh", Path(__file__).resolve().parent / "metric_vs_human.py")
_mvh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mvh)

_roc_auc = _mvh._roc_auc
load_rows = _mvh.load_rows

SIGN = {"cosine_sim": -1.0, "qa_precision": -1.0, "freq_distance": +1.0, "S": +1.0}
SINGLES = ["cosine_sim", "qa_precision", "freq_distance"]
N_BOOT = 2000
SEED = 42


def complete(rows: list[dict]) -> list[dict]:
    keys = SINGLES + ["S"]
    return [r for r in rows if all(r.get(k) is not None for k in keys)]


def paired_delta(rows: list[dict], other: str, target: str, rng: random.Random) -> dict:
    """AUC(S) - AUC(other) on the same docs, with a paired bootstrap CI."""
    s = [SIGN["S"] * r["S"] for r in rows]
    o = [SIGN[other] * r[other] for r in rows]
    y = [r[target] for r in rows]

    auc_s, auc_o = _roc_auc(s, y), _roc_auc(o, y)
    delta = auc_s - auc_o

    n = len(rows)
    vals = []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        by = [y[i] for i in idx]
        if len(set(by)) < 2:
            continue
        d = _roc_auc([s[i] for i in idx], by) - _roc_auc([o[i] for i in idx], by)
        if d == d:
            vals.append(d)
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    p_gt = sum(1 for v in vals if v > 0) / len(vals)
    return {
        "target": target,
        "vs": other,
        "n": n,
        "auc_combined": auc_s,
        "auc_single": auc_o,
        "delta": delta,
        "ci95": [lo, hi],
        "ci_halfwidth": (hi - lo) / 2,
        "p_delta_gt_0": p_gt,
    }


def n_for_halfwidth(half_now: float, n_now: int, target_half: float) -> int:
    """CI half-width shrinks as 1/sqrt(n)."""
    return int(round(n_now * (half_now / target_half) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=str(REPO_ROOT / "notes/human_labels_2026-08-01.csv"))
    ap.add_argument("--output", default=str(REPO_ROOT / "notes/combined_vs_single.md"))
    ap.add_argument("--json", default=str(REPO_ROOT / "notes/combined_vs_single.json"))
    args = ap.parse_args()

    rng = random.Random(SEED)
    rows, annotators, alpha = load_rows(Path(args.labels))
    full = complete(rows)
    unan = [r for r in full if r["unanimous"]]

    results = []
    for label, subset in (("full", full), ("unanimous", unan)):
        for target in ("bin_any", "bin_severe"):
            for other in SINGLES:
                r = paired_delta(subset, other, target, rng)
                r["subset"] = label
                results.append(r)

    ref = next(r for r in results if r["subset"] == "full" and r["target"] == "bin_severe"
               and r["vs"] == "cosine_sim")
    need_005 = n_for_halfwidth(ref["ci_halfwidth"], ref["n"], 0.05)

    L = ["# Combined score vs individual metrics (R2 #1)", ""]
    L.append(f"Labels: `{Path(args.labels).relative_to(REPO_ROOT)}` · "
             f"{len(rows)} docs · {len(annotators)} annotators · Krippendorff's α = {alpha:.3f}")
    L.append("")
    L.append(f"Restricted to the **{len(full)} documents where all three metrics are defined**, so "
             "S and each single metric are scored on exactly the same documents. Every score is "
             "oriented higher = more hallucination. `bin_any` = consensus is partial or "
             "hallucinated; `bin_severe` = consensus is hallucinated.")
    L.append("")
    L.append("S is the parameter-free combination used in the paper: "
             "`S = (1/3) Σ_j sign_j · z(x_j)`.")
    L.append("")

    for label, subset in (("full", full), ("unanimous", unan)):
        L.append(f"## {label} subset (n = {len(subset)})")
        L.append("")
        L.append("| target | vs | AUC(S) | AUC(single) | Δ = S − single | 95% CI (paired boot) | P(Δ>0) |")
        L.append("|---|---|---:|---:|---:|---|---:|")
        for r in results:
            if r["subset"] != label:
                continue
            L.append(f"| `{r['target']}` | `{r['vs']}` | {r['auc_combined']:.3f} | "
                     f"{r['auc_single']:.3f} | {r['delta']:+.3f} | "
                     f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p_delta_gt_0']:.3f} |")
        L.append("")

    L.append("## Sample size required to resolve a difference")
    L.append("")
    L.append(f"On the full subset against `cosine_sim` (`bin_severe`), the paired 95% CI on ΔAUC "
             f"has half-width **{ref['ci_halfwidth']:.3f}** at n = {ref['n']}. Since the half-width "
             f"shrinks as 1/√n, resolving a 0.05 difference needs roughly "
             f"**{need_005} labelled documents** — about "
             f"{need_005 / ref['n']:.1f}× the current sample.")
    L.append("")
    L.append("## Reading")
    L.append("")
    worse = [r for r in results if r["ci95"][1] < 0]
    better = [r for r in results if r["ci95"][0] > 0]
    tie = [r for r in results if r["ci95"][0] <= 0 <= r["ci95"][1]]
    L.append(f"Of the {len(results)} paired comparisons: **{len(worse)} where S is significantly "
             f"worse** (CI entirely below 0), **{len(better)} where S is significantly better**, "
             f"and {len(tie)} indistinguishable (CI contains 0).")
    L.append("")
    for r in worse:
        L.append(f"- worse: {r['subset']} / `{r['target']}` vs `{r['vs']}` — "
                 f"Δ = {r['delta']:+.3f}, CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    for r in better:
        L.append(f"- better: {r['subset']} / `{r['target']}` vs `{r['vs']}` — "
                 f"Δ = {r['delta']:+.3f}, CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    L.append("")
    L.append("The equal-weight combination is *not* an improvement over the best single signal: it "
             "loses to `cosine_sim` by a margin whose CI excludes 0, and only ties or beats the two "
             "entity-level metrics, which are themselves at chance on this sample. That is the "
             "expected behaviour of averaging when two of three components carry no signal — the "
             "informative one gets diluted. Whether a *learned* weighting recovers the difference "
             "cannot be settled here (see the sample-size note above); the paper therefore presents "
             "S as a formulation of the integrated decision, not as a better detector, and leaves "
             "the weighting comparison to future work.")
    L.append("")

    Path(args.output).write_text("\n".join(L) + "\n", encoding="utf-8")
    Path(args.json).write_text(json.dumps(
        {"n_complete": len(full), "n_unanimous": len(unan), "alpha": alpha,
         "n_for_halfwidth_0.05": need_005, "results": results}, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
