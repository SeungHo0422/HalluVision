"""
R2 #3 — compare each HalluVision metric against the human judgement.

The reviewer asks for the three metrics to be validated *individually* against
human annotators, not just for an agreement number. This script joins the 100
human-labelled docs with the PAPER metrics (not the TF-IDF proxy that
``combined_scorer_demo/labelfree_score.py`` builds for its label-free demo) and
reports, per metric: rank correlation with the human severity, ROC AUC under
both binarisations, the score separation across the three human classes, and
the best achievable operating point.

Metric sources (all gpt-4o-mini — every sampled doc is 4o-mini):
  cosine_sim    notes/doc_cosine_gpt4omini.json          jina-embeddings-v3, full doc
  qa_precision  notes/doc_stats_qa_gpt4omini.json        CyNER answer-matching, _overall_
  freq_distance combined_scorer_demo/entity_freq_gpt4omini.json   paper Eq.5 TV distance
  S             recomputed here with the real combined_scorer z-combination

Human targets (from the 3 raters per doc):
  consensus   majority label, severity tie-break            faithful/partial/hallucinated
  severity    consensus mapped to 0 / 1 / 2                 ordinal
  soft        mean rater severity / 2, in [0, 1]            uses all raters, not just the winner
  bin_any     consensus in {partial, hallucinated}          "did the model hallucinate at all"
  bin_severe  consensus == hallucinated                     "is it a hard hallucination"

Usage::

    python scripts/metric_vs_human.py \
        --labels notes/human_labels_2026-08-01.csv \
        --output notes/metric_vs_human.md
"""

from __future__ import annotations

import argparse
import os
import csv
import importlib.util
import json
import random
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(os.environ.get("HALLUVISION_WORKSPACE", Path.cwd())).resolve()  # data workspace

# agreement_multi is pure-python; import it by path so the analyzer package
# __init__ (numpy/torch) is not pulled in.
_spec = importlib.util.spec_from_file_location(
    "_agreement_multi", Path(__file__).resolve().parents[1] / "analyzer/agreement_multi.py"
)
_am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_am)

LABEL_LEVELS = _am.LABEL_LEVELS
SEVERITY = _am.SEVERITY
_spearman = _am._spearman
_roc_auc = _am._roc_auc
_majority = _am._majority
_krippendorff_alpha = _am._krippendorff_alpha
detect_annotators = _am.detect_annotators

# sign = +1 when a HIGHER value should mean MORE hallucination. The AUC/rho are
# always reported on the hallucination-oriented score, so >0.5 / >0 means the
# metric behaves as the paper claims.
METRICS = [
    ("cosine_sim", -1.0, "jina-v3 cosine(original, summary)"),
    ("qa_precision", -1.0, "CyNER answer-matching precision (overall)"),
    ("freq_distance", +1.0, "entity-label TV distance (Eq. 5)"),
    ("S", +1.0, "combined z-score (equal weights)"),
]

N_BOOT = 2000
SEED = 42

# Read-me-first caveats, appended verbatim to the report so it stands alone.
CAVEATS = [
    "## Caveats that bound every number above",
    "",
    "1. **The human label is a joint judgement.** The labeling screen asks whether *the summary "
    "and all five QA answers* are supported by the article (`webapp/index.html`, guideline block), "
    "and the annotator returns one label for the pair. A summary-level metric (`cosine_sim`) and a "
    "QA-level metric (`qa_precision`) are therefore scored against the same conflated target; "
    "neither gets a clean test. Separate summary/QA labels would be needed to validate them "
    "independently.",
    "2. **α = 0.295.** The target itself is noisy, which attenuates every correlation toward 0. The "
    "unanimous-subset column is the partial control for this — correlations there are the upper "
    "bound of what the current labels can show.",
    "3. **Base rate is 75% (any hallucination).** Accuracy and F1 in the operating-point table are "
    "close to what a constant \"always hallucinated\" predictor gets; only AUC and ρ are "
    "informative at this prevalence.",
    "4. **n = 100 docs, one model (gpt-4o-mini).** CIs are wide; nothing here transfers to the "
    "gpt-3.5-turbo arm without its own labels.",
    "5. Thresholds in the operating-point table are on the *oriented* score, i.e. a `cosine_sim` "
    "threshold of -0.894 means \"flag when cosine < 0.894\".",
]


def load_metrics() -> dict[str, dict]:
    cos = json.loads((REPO_ROOT / "notes/doc_cosine_gpt4omini.json").read_text())
    qa = json.loads((REPO_ROOT / "notes/doc_stats_qa_gpt4omini.json").read_text())
    freq = json.loads((REPO_ROOT / "notes/combined_scorer_demo/entity_freq_gpt4omini.json").read_text())

    out: dict[str, dict] = {}
    for doc_id in set(cos) | set(qa) | set(freq):
        rec: dict[str, float | None] = {}
        c = cos.get(doc_id, {}).get("_overall_", {})
        rec["cosine_sim"] = c.get("cosine_sim")
        q = qa.get(doc_id, {}).get("_overall_", {})
        rec["qa_precision"] = q.get("precision")
        f = freq.get(doc_id, {})
        rec["freq_distance"] = f.get("freq_distance")
        out[doc_id] = rec
    return out


def add_combined(rows: list[dict]) -> None:
    """Recompute the combined score S with the real scorer logic, over the docs
    that have all three metrics."""
    import sys

    spec = importlib.util.spec_from_file_location(
        "_combined_scorer", Path(__file__).resolve().parents[1] / "analyzer/combined_scorer.py"
    )
    cs = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cs  # @dataclass resolves annotations via sys.modules
    spec.loader.exec_module(cs)

    feats = ["cosine_sim", "qa_precision", "freq_distance"]
    signs = [-1.0, -1.0, 1.0]
    complete = [r for r in rows if all(r.get(f) is not None for f in feats)]
    X = [[r[f] for f in feats] for r in complete]
    Z, _, _ = cs._standardise(cs._apply_signs(X, signs))
    for r, z in zip(complete, Z):
        r["S"] = cs._combine_z(z, [1 / 3, 1 / 3, 1 / 3], 0.0)
    for r in rows:
        r.setdefault("S", None)


def load_rows(labels_path: Path) -> tuple[list[dict], list[str], float]:
    metrics = load_metrics()
    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        annotators = detect_annotators(reader.fieldnames)
        rows = []
        per_item = []
        for r in reader:
            labs = {}
            for a in annotators:
                v = (r.get(f"label_{a}", "") or "").strip().lower()
                if v in LABEL_LEVELS:
                    labs[a] = v
            per_item.append(list(labs.values()))
            if not labs:
                continue
            present = list(labs.values())
            cons = _majority(present)
            row = {
                "doc_id": r["doc_id"],
                "n_raters": len(present),
                "consensus": cons,
                "severity": SEVERITY[cons],
                "soft": mean(SEVERITY[v] for v in present) / 2.0,
                "unanimous": len(set(present)) == 1,
                "bin_any": 1 if cons in ("partial", "hallucinated") else 0,
                "bin_severe": 1 if cons == "hallucinated" else 0,
            }
            row.update(metrics.get(r["doc_id"], {}))
            rows.append(row)
    add_combined(rows)
    alpha = _krippendorff_alpha([it for it in per_item if len(it) >= 2], LABEL_LEVELS)
    return rows, annotators, alpha


# ---------------- statistics ----------------
def boot_ci(fn, xs, ys, rng, n=N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap CI over paired (x, y) samples."""
    n_obs = len(xs)
    if n_obs < 5:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n):
        idx = [rng.randrange(n_obs) for _ in range(n_obs)]
        bx = [xs[i] for i in idx]
        by = [ys[i] for i in idx]
        if len(set(by)) < 2:
            continue
        v = fn(bx, by)
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def perm_p(fn, xs, ys, observed, rng, null=0.0, n=N_BOOT) -> float:
    """Two-sided permutation p-value: shuffle the human labels."""
    if observed != observed:
        return float("nan")
    shuf = list(ys)
    hits = 0
    for _ in range(n):
        rng.shuffle(shuf)
        v = fn(xs, shuf)
        if v == v and abs(v - null) >= abs(observed - null):
            hits += 1
    return (hits + 1) / (n + 1)


def operating_point(scores: list[float], y: list[int]) -> dict:
    """Youden-optimal threshold on the hallucination-oriented score."""
    best = None
    for t in sorted(set(scores)):
        pred = [1 if s >= t else 0 for s in scores]
        tp = sum(1 for p, l in zip(pred, y) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(pred, y) if p == 1 and l == 0)
        fn_ = sum(1 for p, l in zip(pred, y) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(pred, y) if p == 0 and l == 0)
        tpr = tp / (tp + fn_) if tp + fn_ else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        j = tpr - fpr
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tpr
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / len(y)
        cand = {"threshold": t, "youden_j": j, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}
        if best is None or j > best["youden_j"]:
            best = cand
    return best or {}


def oriented(rows: list[dict], key: str, sign: float, target: str) -> tuple[list[float], list]:
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is None or r.get(target) is None:
            continue
        xs.append(sign * float(v))
        ys.append(r[target])
    return xs, ys


def metric_block(rows: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for key, sign, desc in METRICS:
        xs_soft, ys_soft = oriented(rows, key, sign, "soft")
        rho = _spearman(xs_soft, ys_soft)
        rho_lo, rho_hi = boot_ci(_spearman, xs_soft, ys_soft, rng)
        rho_p = perm_p(_spearman, xs_soft, ys_soft, rho, rng)

        xs_sev, ys_sev = oriented(rows, key, sign, "severity")
        rho_sev = _spearman(xs_sev, [float(v) for v in ys_sev])

        res = {"key": key, "desc": desc, "n": len(xs_soft), "rho_soft": rho,
               "rho_ci": (rho_lo, rho_hi), "rho_p": rho_p, "rho_sev": rho_sev}

        for tgt, name in (("bin_any", "any"), ("bin_severe", "severe")):
            xs, ys = oriented(rows, key, sign, tgt)
            auc = _roc_auc(xs, ys)
            lo, hi = boot_ci(_roc_auc, xs, ys, rng)
            p = perm_p(_roc_auc, xs, ys, auc, rng, null=0.5)
            res[f"auc_{name}"] = auc
            res[f"auc_{name}_ci"] = (lo, hi)
            res[f"auc_{name}_p"] = p
            res[f"op_{name}"] = operating_point(xs, ys)
            res[f"pos_{name}"] = sum(ys)

        # raw (un-oriented) mean per human class
        by_class = {}
        for lvl in LABEL_LEVELS:
            vals = [float(r[key]) for r in rows if r["consensus"] == lvl and r.get(key) is not None]
            by_class[lvl] = (len(vals), mean(vals) if vals else float("nan"),
                             pstdev(vals) if len(vals) > 1 else float("nan"))
        res["by_class"] = by_class
        out.append(res)
    return out


def fmt_ci(t: tuple[float, float]) -> str:
    lo, hi = t
    return f"[{lo:.3f}, {hi:.3f}]" if lo == lo else "n/a"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=str(REPO_ROOT / "notes/human_labels_2026-08-01.csv"))
    p.add_argument("--output", default=str(REPO_ROOT / "notes/metric_vs_human.md"))
    args = p.parse_args()

    rng = random.Random(SEED)
    rows, annotators, alpha = load_rows(Path(args.labels))
    unan = [r for r in rows if r["unanimous"]]

    L = ["# Metric vs Human Judgement (R2 #3)", ""]
    L.append(f"Labels: `{Path(args.labels).relative_to(REPO_ROOT)}`  ·  "
             f"{len(rows)} docs  ·  {len(annotators)} annotators  ·  3 ratings/doc")
    L.append(f"Inter-annotator reliability: **Krippendorff's α = {alpha:.3f}** (nominal, 3-class). "
             f"Unanimous items: {len(unan)}/{len(rows)}.")
    L.append("")
    L.append("Metrics are the **paper** implementations (jina-v3 cosine, CyNER answer-matching "
             "precision, Eq.5 TV distance) — not the TF-IDF proxy used by the label-free "
             "`combined_scorer_demo`. Every score below is oriented so that **higher = more "
             "hallucination** (cosine and qa_precision are negated), so AUC > 0.5 and ρ > 0 mean "
             "the metric agrees with the annotators.")
    L.append("")

    for title, subset in (("Full sample", rows), (f"Unanimous subset (n={len(unan)})", unan)):
        res = metric_block(subset, rng)
        L.append(f"## {title}")
        L.append("")
        L.append("| metric | n | ρ vs soft human score | 95% CI | p | AUC (any halluc.) | 95% CI | p | AUC (severe only) | 95% CI | p |")
        L.append("|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|")
        for r in res:
            L.append(
                f"| `{r['key']}` | {r['n']} | {r['rho_soft']:.3f} | {fmt_ci(r['rho_ci'])} | {r['rho_p']:.3f} "
                f"| {r['auc_any']:.3f} | {fmt_ci(r['auc_any_ci'])} | {r['auc_any_p']:.3f} "
                f"| {r['auc_severe']:.3f} | {fmt_ci(r['auc_severe_ci'])} | {r['auc_severe_p']:.3f} |"
            )
        L.append("")

        if subset is rows:
            L.append("### Raw metric value by consensus label")
            L.append("")
            L.append("| metric | faithful (n, mean±sd) | partial | hallucinated |")
            L.append("|---|---|---|---|")
            for r in res:
                cells = []
                for lvl in LABEL_LEVELS:
                    n_, m_, s_ = r["by_class"][lvl]
                    cells.append(f"{n_}, {m_:.3f}±{s_:.3f}" if n_ else "—")
                L.append(f"| `{r['key']}` | {cells[0]} | {cells[1]} | {cells[2]} |")
            L.append("")

            L.append("### Best achievable operating point (Youden J, target = any hallucination)")
            L.append("")
            L.append("| metric | threshold (oriented) | precision | recall | F1 | accuracy |")
            L.append("|---|---:|---:|---:|---:|---:|")
            for r in res:
                op = r["op_any"]
                if not op:
                    continue
                L.append(f"| `{r['key']}` | {op['threshold']:.3f} | {op['precision']:.3f} | "
                         f"{op['recall']:.3f} | {op['f1']:.3f} | {op['accuracy']:.3f} |")
            L.append("")
            base = mean(r["bin_any"] for r in subset)
            L.append(f"Positive rate (any hallucination) = {base:.1%}; a constant \"always hallucinated\" "
                     f"predictor scores accuracy {base:.3f}, F1 {2 * base / (1 + base):.3f}.")
            L.append("")

    L.append("## Coverage")
    L.append("")
    L.append("| metric | docs with a value |")
    L.append("|---|---:|")
    for key, _, _ in METRICS:
        L.append(f"| `{key}` | {sum(1 for r in rows if r.get(key) is not None)} / {len(rows)} |")
    L.append("")
    L.append("`freq_distance` is undefined for docs whose entity distribution is empty; "
             "`qa_precision` is missing where the QA stats file has no entry. `S` needs all three.")
    L.append("")
    L.append(f"Bootstrap CIs and permutation p-values: {N_BOOT} resamples, seed {SEED}.")
    L.append("")
    L.extend(CAVEATS)

    out_path = Path(args.output)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[metric_vs_human] wrote {out_path}")


if __name__ == "__main__":
    main()
