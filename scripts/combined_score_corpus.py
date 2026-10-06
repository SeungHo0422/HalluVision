"""
R2 #1 — the combined score S over the whole corpus, with no labels involved.

The Discussion paragraph stops at Algorithm 1: orient, standardize, average. This
script is the corresponding result — S computed for every Threatpost document under
both models, which needs no human labels and no calibration. Human-validated
detection numbers deliberately do NOT appear here; those belong to the threshold
discussion (R2 #3 / #7).

Inputs (repo root unless noted):
  cosine        notes/doc_cosine_{gpt4omini,gpt35}.json      jina-v3, _overall_
  qa precision  notes/doc_stats_qa_{gpt4omini,gpt35}.json    CyNER answer matching, _overall_
  freq distance original_ner.json vs summary_ner.json (4o-mini)
                original_ner.json vs cyner_sum_results_250427.json (3.5)   paper Eq. 5 TV

Standardization is pooled over both models so that the two S distributions are on a
common scale; per-model standardization would centre each model at zero by
construction and make the comparison vacuous.

Usage::

    python scripts/combined_score_corpus.py
"""

from __future__ import annotations

import argparse
import os
import json
import random
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(os.environ.get("HALLUVISION_WORKSPACE", Path.cwd())).resolve()  # data workspace
LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
# The paper's third metric is the frequency-based entity *similarity* F = 1 - TV, so in
# Algorithm 1 all three signs are -1. Here the third feature is the TV distance itself,
# which enters with +1: -F = TV - 1 differs from TV by a constant, and z-scoring removes
# constants, so the two conventions give identical z, S, and every number below.
SIGNS = {"cosine_sim": -1.0, "qa_precision": -1.0, "freq_distance": +1.0}
FEATURES = ["cosine_sim", "qa_precision", "freq_distance"]
MODELS = {
    "gpt-4o-mini": {
        "cosine": "notes/doc_cosine_gpt4omini.json",
        "qa": "notes/doc_stats_qa_gpt4omini.json",
        "summary_ner": "summary_ner.json",
    },
    "gpt-3.5-turbo": {
        "cosine": "notes/doc_cosine_gpt35.json",
        "qa": "notes/doc_stats_qa_gpt35.json",
        "summary_ner": "cyner_sum_results_250427.json",
    },
}
N_BOOT = 2000
SEED = 42


def tv_distance(c_o: list[int], c_g: list[int]) -> float | None:
    so, sg = sum(c_o), sum(c_g)
    if so == 0 or sg == 0:
        return None
    p_o = [x / so for x in c_o]
    p_g = [x / sg for x in c_g]
    return 0.5 * sum(abs(a - b) for a, b in zip(p_o, p_g))


def freq_distances(summary_ner_file: str) -> dict[str, float]:
    orig = json.loads((REPO_ROOT / "original_ner.json").read_text())
    summ = json.loads((REPO_ROOT / summary_ner_file).read_text())
    out = {}
    for did, s in summ.items():
        o = orig.get(did)
        if o is None:
            continue
        tv = tv_distance([len(set(o.get(l, []))) for l in LABELS],
                         [len(set(s.get(l, []))) for l in LABELS])
        if tv is not None:
            out[did] = tv
    return out


def load_model(cfg: dict) -> dict[str, dict]:
    cos = json.loads((REPO_ROOT / cfg["cosine"]).read_text())
    qa = json.loads((REPO_ROOT / cfg["qa"]).read_text())
    freq = freq_distances(cfg["summary_ner"])

    rows = {}
    for did in set(cos) & set(qa) & set(freq):
        c = cos[did].get("_overall_", {}).get("cosine_sim")
        q = qa[did].get("_overall_", {}).get("precision")
        if c is None or q is None:
            continue
        rows[did] = {"cosine_sim": c, "qa_precision": q, "freq_distance": freq[did]}
    return rows


def quantile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    i = q * (len(ys) - 1)
    lo, hi = int(i), min(int(i) + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (i - lo)


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(REPO_ROOT / "notes/combined_score_corpus.md"))
    ap.add_argument("--json", default=str(REPO_ROOT / "notes/combined_score_corpus.json"))
    args = ap.parse_args()

    per_model = {name: load_model(cfg) for name, cfg in MODELS.items()}

    # pooled standardisation over both models
    pooled = {f: [] for f in FEATURES}
    for rows in per_model.values():
        for r in rows.values():
            for f in FEATURES:
                pooled[f].append(SIGNS[f] * r[f])
    stats = {f: (mean(v), pstdev(v)) for f, v in pooled.items()}

    for rows in per_model.values():
        for r in rows.values():
            zs = [(SIGNS[f] * r[f] - stats[f][0]) / stats[f][1] for f in FEATURES]
            r["z"] = dict(zip(FEATURES, zs))
            r["S"] = sum(zs) / len(zs)

    rng = random.Random(SEED)
    S = {m: [r["S"] for r in rows.values()] for m, rows in per_model.items()}
    a, b = S["gpt-3.5-turbo"], S["gpt-4o-mini"]

    # paired over the documents both models cover
    common = sorted(set(per_model["gpt-3.5-turbo"]) & set(per_model["gpt-4o-mini"]))
    diffs = [per_model["gpt-3.5-turbo"][d]["S"] - per_model["gpt-4o-mini"][d]["S"] for d in common]
    boot = []
    for _ in range(N_BOOT):
        boot.append(mean(diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))))
    boot.sort()
    d_lo, d_hi = boot[int(0.025 * N_BOOT)], boot[int(0.975 * N_BOOT)]

    L = ["# Combined score $S$ over the corpus (R2 #1)", ""]
    L.append("No labels, no calibration: this is the output of Algorithm 1 alone. "
             "Features are standardized over the two models pooled, so the two "
             "distributions share a scale.")
    L.append("")
    L.append("| model | n | mean $S$ | sd | p25 | median | p75 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in ("gpt-3.5-turbo", "gpt-4o-mini"):
        v = S[m]
        L.append(f"| {m} | {len(v)} | {mean(v):+.3f} | {pstdev(v):.3f} | "
                 f"{quantile(v, .25):+.3f} | {quantile(v, .5):+.3f} | {quantile(v, .75):+.3f} |")
    L.append("")
    L.append(f"Paired over the {len(common)} documents both models cover, "
             f"$S$(gpt-3.5) $-$ $S$(gpt-4o-mini) = **{mean(diffs):+.3f}** "
             f"(95% bootstrap CI [{d_lo:+.3f}, {d_hi:+.3f}], {N_BOOT} resamples, seed {SEED}). "
             "Higher $S$ means more hallucination, so a positive difference puts gpt-3.5-turbo "
             "on the worse side of the combined score.")
    L.append("")
    L.append("## Where the model gap comes from")
    L.append("")
    L.append("$S$ is the mean of the three standardized components, so the difference in mean "
             "$S$ between the models splits additively over them.")
    L.append("")
    L.append("| component | mean $z$ (3.5) | mean $z$ (4o-mini) | gap | contribution to $\\Delta S$ |")
    L.append("|---|---:|---:|---:|---:|")
    total = 0.0
    for f in FEATURES:
        zs = {m: mean(r["z"][f] for r in rows.values()) for m, rows in per_model.items()}
        gap = zs["gpt-3.5-turbo"] - zs["gpt-4o-mini"]
        total += gap / len(FEATURES)
        L.append(f"| `{f}` | {zs['gpt-3.5-turbo']:+.3f} | {zs['gpt-4o-mini']:+.3f} | "
                 f"{gap:+.3f} | {gap / len(FEATURES):+.3f} |")
    L.append(f"| **total** | | | | **{total:+.3f}** |")
    L.append("")

    L.append("## How much the two populations overlap")
    L.append("")
    med_4o = quantile(S["gpt-4o-mini"], .5)
    worse = sum(1 for d in common
                if per_model["gpt-3.5-turbo"][d]["S"] > per_model["gpt-4o-mini"][d]["S"])
    pool = sorted([(v, "3.5") for v in S["gpt-3.5-turbo"]] +
                  [(v, "4o") for v in S["gpt-4o-mini"]], reverse=True)
    dec = max(1, len(pool) // 10)
    L.append(f"- gpt-3.5-turbo scores worse on **{worse / len(common):.1%}** of the "
             f"{len(common)} documents both models processed (paired, same article).")
    L.append(f"- **{sum(1 for v in S['gpt-3.5-turbo'] if v > med_4o) / len(S['gpt-3.5-turbo']):.1%}** "
             f"of gpt-3.5-turbo documents sit above the gpt-4o-mini median.")
    L.append(f"- **{sum(1 for _, m in pool[:dec] if m == '3.5') / dec:.1%}** of the worst-scoring "
             f"decile of the pooled corpus is gpt-3.5-turbo output; "
             f"**{sum(1 for _, m in pool[-dec:] if m == '4o') / dec:.1%}** of the best-scoring "
             f"decile is gpt-4o-mini.")
    L.append(f"- The gap in means is {mean(S['gpt-3.5-turbo']) - mean(S['gpt-4o-mini']):.3f}, "
             f"about {(mean(S['gpt-3.5-turbo']) - mean(S['gpt-4o-mini'])) / pstdev(S['gpt-3.5-turbo'] + S['gpt-4o-mini']):.2f} "
             f"pooled standard deviations: the distributions separate but overlap over most of "
             f"their range, so $S$ orders populations, not individual documents.")
    L.append("")

    L.append("## Component agreement (Spearman, no labels)")
    L.append("")
    L.append("How much the three oriented components already say the same thing. Weak "
             "correlation is the premise of combining them at all.")
    L.append("")
    L.append("| model | cos–AMS | cos–FES | AMS–FES |")
    L.append("|---|---:|---:|---:|")
    for m, rows in per_model.items():
        z = [r["z"] for r in rows.values()]
        c = [x["cosine_sim"] for x in z]
        q = [x["qa_precision"] for x in z]
        f = [x["freq_distance"] for x in z]
        L.append(f"| {m} | {spearman(c, q):+.3f} | {spearman(c, f):+.3f} | {spearman(q, f):+.3f} |")
    L.append("")

    Path(args.output).write_text("\n".join(L) + "\n", encoding="utf-8")
    Path(args.json).write_text(json.dumps({
        "standardisation": {f: {"mean": stats[f][0], "sd": stats[f][1]} for f in FEATURES},
        "per_model": {m: {"n": len(v), "mean": mean(v), "sd": pstdev(v),
                          "p25": quantile(v, .25), "median": quantile(v, .5),
                          "p75": quantile(v, .75)} for m, v in S.items()},
        "paired_diff_35_minus_4o": {"n": len(common), "mean": mean(diffs),
                                    "ci95": [d_lo, d_hi]},
        "scores": {m: {d: r["S"] for d, r in rows.items()} for m, rows in per_model.items()},
    }, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
