"""
Calibrated combined hallucination scorer (Reviewer 2 #1 — a CTI-specific
hallucination scorer that combines the three signals into a single
calibrated decision rule with validated thresholds).

The three HalluVision signals (cosine similarity, answer-matching score /
QA precision, frequency distance) are already CTI-specific measurements.
This module turns them into ONE integrated hallucination score, so a
document is judged holistically rather than by three separate metrics.

Design (deliberately simple but principled):

* **Sign alignment.** ``--directions metric:lower`` flips a feature's
  sign *before* anything else, so "larger = more hallucinated" holds for
  every signal. This is a-priori domain knowledge, not learned.
* **z-score normalisation.** Each signal is standardised so the three
  live on a comparable scale before they are combined.
* **Equal-weight combination.** The combined score is the *mean* of the
  sign-aligned z-scores:

      S = (1/d) * Σ_j z_j

  This has **zero free parameters**, so there is nothing to overfit on
  the small human-label set and no per-signal weight to justify
  subjectively. Equal weight is the least-committal (max-entropy) choice
  absent evidence that one signal deserves more trust.
* **Platt calibration (2 parameters).** A single 1-D logistic map
  ``P = sigmoid(a*S + b)`` turns the combined score into a probability.
  Only two parameters are fit, so calibration barely overfits; crucially
  the class prior (base rate) is absorbed almost entirely by the
  intercept ``b`` — discrimination (AUC, governed by ``a`` and the fixed
  weights) is invariant to the label-set class balance.
* **Validated thresholds.** K-fold cross-validation fits the
  normalisation, the Platt map and the operating threshold on each train
  fold and reports the held-out fold's AUC / F1 / precision / recall.
  Thresholds are therefore validated out-of-sample, never on the data
  they were chosen on.

A learned logistic-regression combiner (per-signal weights fit on the
labels) is reported alongside as a *comparison* only: if learned weights
do not beat equal weights on AUC, the parameter-free rule is justified as
an empirical finding rather than a limitation.

Usage::

    python -m final_pipeline.analyzer.combined_scorer \
        --labels notes/human_labels_filled.csv \
        --features cosine_sim qa_precision freq_distance \
        --directions cosine_sim:lower qa_precision:lower freq_distance:higher \
        --folds 5 --seed 42 \
        --output notes/combined_scorer.md \
        --dump-model notes/combined_scorer.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HALLUC_POSITIVE = {"partial", "hallucinated"}
SEVERITY = {"faithful": 0, "partial": 1, "hallucinated": 2}


@dataclass
class CombinedScorer:
    feature_names: List[str]
    feature_signs: List[float]      # +1 or -1 per feature (a-priori direction)
    feature_means: List[float]
    feature_stds: List[float]
    weights: List[float]            # per-signal combine weights (equal = 1/d)
    intercept: float                # pre-Platt linear intercept (0 for equal)
    platt_a: float                  # calibration slope
    platt_b: float                  # calibration intercept (absorbs base rate)
    threshold_youden: float         # operating point on P, Youden's J
    threshold_f1: float             # operating point on P, F1-max
    method: str                     # "equal" or "logistic"


def _majority_vote(a: str, b: str) -> Optional[str]:
    if not a or not b:
        return a or b or None
    if a == b:
        return a
    return max(a, b, key=lambda x: SEVERITY.get(x, -1))


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _train_logreg(X: List[List[float]], y: List[int],
                  l2: float = 0.01,
                  lr: float = 0.1,
                  epochs: int = 2000) -> Tuple[List[float], float]:
    n_features = len(X[0]) if X else 0
    w = [0.0] * n_features
    b = 0.0
    n = len(X)
    if n == 0 or n_features == 0:
        return w, b
    for _ in range(epochs):
        gw = [0.0] * n_features
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(w[j] * xi[j] for j in range(n_features))
            p = _sigmoid(z)
            err = p - yi
            for j in range(n_features):
                gw[j] += err * xi[j]
            gb += err
        for j in range(n_features):
            gw[j] = gw[j] / n + l2 * w[j]
            w[j] -= lr * gw[j]
        b -= lr * gb / n
    return w, b


def _standardise(X: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    n_features = len(X[0])
    means = [statistics.mean(col) for col in zip(*X)]
    stds = []
    for j in range(n_features):
        col = [x[j] for x in X]
        if len(col) >= 2:
            sd = statistics.pstdev(col)
        else:
            sd = 0.0
        stds.append(sd if sd > 1e-9 else 1.0)
    Z = [[(x[j] - means[j]) / stds[j] for j in range(n_features)] for x in X]
    return Z, means, stds


def _apply_signs(X: List[List[float]], signs: List[float]) -> List[List[float]]:
    return [[x[j] * signs[j] for j in range(len(signs))] for x in X]


def _fit_platt(scores: List[float], labels: List[int]) -> Tuple[float, float]:
    """Fit P = sigmoid(a*S + b) — a 2-parameter 1-D logistic calibration."""
    w, b = _train_logreg([[s] for s in scores], labels)
    a = w[0] if w else 1.0
    return a, b


def _combine_z(z_row: List[float], weights: List[float], intercept: float) -> float:
    return intercept + sum(weights[j] * z_row[j] for j in range(len(weights)))


def _predict_proba(model: CombinedScorer, x_raw: List[float]) -> float:
    if len(x_raw) != len(model.feature_names):
        raise ValueError("feature length mismatch")
    z = [0.0] * len(x_raw)
    for j, v in enumerate(x_raw):
        v_signed = v * model.feature_signs[j]
        z[j] = (v_signed - model.feature_means[j]) / model.feature_stds[j]
    s = _combine_z(z, model.weights, model.intercept)
    return _sigmoid(model.platt_a * s + model.platt_b)


def _auc(scores: List[float], labels: List[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    pairs = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                pairs += 1.0
            elif p == n:
                pairs += 0.5
    return pairs / (len(pos) * len(neg))


def _best_threshold(scores: List[float], labels: List[int]) -> Tuple[float, float, float, float]:
    if not scores:
        return 0.5, 0.5, 0.0, 0.0
    cutoffs = sorted(set(scores))
    best_j = (-1.0, cutoffs[0])
    best_f1 = (-1.0, cutoffs[0])
    for c in cutoffs:
        tp = fp = tn = fn = 0
        for s, l in zip(scores, labels):
            if s >= c and l == 1:
                tp += 1
            elif s >= c and l == 0:
                fp += 1
            elif s < c and l == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if j > best_j[0]:
            best_j = (j, c)
        if f1 > best_f1[0]:
            best_f1 = (f1, c)
    return best_j[1], best_f1[1], best_j[0], best_f1[0]


def train_model(rows: List[List[float]], labels: List[int],
                feature_names: List[str], feature_signs: List[float],
                method: str = "equal") -> CombinedScorer:
    """Build a combined scorer.

    method="equal"    : parameter-free mean of z-scores + Platt calibration
    method="logistic" : learned per-signal weights (reported for comparison)
    """
    d = len(feature_names)
    X_signed = _apply_signs(rows, feature_signs)
    Z, means, stds = _standardise(X_signed)

    if method == "logistic":
        w, b0 = _train_logreg(Z, labels)
        weights, intercept, platt_a, platt_b = w, b0, 1.0, 0.0
    else:  # "equal"
        weights, intercept = [1.0 / d] * d, 0.0
        s_vals = [_combine_z(z, weights, intercept) for z in Z]
        platt_a, platt_b = _fit_platt(s_vals, labels)

    probs = [_sigmoid(platt_a * _combine_z(z, weights, intercept) + platt_b)
             for z in Z]
    thr_y, thr_f1, _, _ = _best_threshold(probs, labels)
    return CombinedScorer(
        feature_names=feature_names,
        feature_signs=feature_signs,
        feature_means=means,
        feature_stds=stds,
        weights=weights,
        intercept=intercept,
        platt_a=platt_a,
        platt_b=platt_b,
        threshold_youden=thr_y,
        threshold_f1=thr_f1,
        method=method,
    )


def cross_validate(X: List[List[float]], y: List[int],
                   feature_names: List[str], feature_signs: List[float],
                   folds: int, seed: int, method: str = "equal") -> Dict[str, float]:
    """Stratified K-fold with normalisation, calibration and threshold all
    fit on the train fold only; metrics reported on the held-out fold."""
    rng = random.Random(seed)
    pos = [i for i in range(len(X)) if y[i] == 1]
    neg = [i for i in range(len(X)) if y[i] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    # stratified assignment: round-robin each class across folds so the
    # positive rate is preserved and folds are not silently dropped.
    fold_assign = [[] for _ in range(folds)]
    for k, i in enumerate(pos):
        fold_assign[k % folds].append(i)
    for k, i in enumerate(neg):
        fold_assign[k % folds].append(i)

    aucs, f1s, precisions, recalls = [], [], [], []
    for fold in fold_assign:
        test = set(fold)
        Xtr = [X[i] for i in range(len(X)) if i not in test]
        ytr = [y[i] for i in range(len(X)) if i not in test]
        Xte = [X[i] for i in fold]
        yte = [y[i] for i in fold]
        if not Xtr or not Xte or len(set(yte)) < 2 or len(set(ytr)) < 2:
            continue
        model = train_model(Xtr, ytr, feature_names, feature_signs, method)
        scores = [_predict_proba(model, x) for x in Xte]
        aucs.append(_auc(scores, yte))
        preds = [1 if s >= model.threshold_f1 else 0 for s in scores]
        tp = sum(1 for p, l in zip(preds, yte) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, yte) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, yte) if p == 0 and l == 1)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return {
        "auc_mean": statistics.mean(aucs) if aucs else float("nan"),
        "f1_mean": statistics.mean(f1s) if f1s else float("nan"),
        "precision_mean": statistics.mean(precisions) if precisions else float("nan"),
        "recall_mean": statistics.mean(recalls) if recalls else float("nan"),
        "n_folds": len(aucs),
    }


def _individual_metric_auc(values: List[float], labels: List[int],
                           sign: float) -> float:
    return _auc([v * sign for v in values], labels)


def _leave_one_out_auc(X: List[List[float]], y: List[int],
                       feature_names: List[str], signs: List[float]) -> Dict[str, float]:
    """Held-in AUC of the equal-weight combiner with each signal removed.
    A large drop means the signal contributes to the combination."""
    d = len(feature_names)
    out = {}
    for drop in range(d):
        keep = [j for j in range(d) if j != drop]
        if not keep:
            continue
        Xk = [[row[j] for j in keep] for row in X]
        names_k = [feature_names[j] for j in keep]
        signs_k = [signs[j] for j in keep]
        model = train_model(Xk, y, names_k, signs_k, method="equal")
        scores = [_predict_proba(model, x) for x in Xk]
        out[feature_names[drop]] = _auc(scores, y)
    return out


def _parse_rows(path: str, feature_names: List[str]) -> List[Tuple[List[float], int]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = (row.get("label_a") or "").strip().lower()
            b = (row.get("label_b") or "").strip().lower()
            m = _majority_vote(a, b)
            if m not in {"faithful", "partial", "hallucinated"}:
                continue
            target = 1 if m in HALLUC_POSITIVE else 0
            feats = [_to_float(row.get(name, "")) for name in feature_names]
            if any(v is None for v in feats):
                continue
            rows.append((feats, target))
    return rows


def _parse_directions(args: List[str], features: List[str]) -> List[float]:
    overrides = {}
    for s in args:
        if ":" in s:
            k, v = s.split(":", 1)
            overrides[k.strip()] = v.strip()
    signs = []
    for name in features:
        d = overrides.get(name, "lower")  # default: low value → halluc
        signs.append(-1.0 if d == "lower" else 1.0)
    return signs


def render_markdown(model: CombinedScorer,
                    cv_equal: Dict[str, float], cv_logit: Dict[str, float],
                    individual_aucs: Dict[str, float],
                    loo_aucs: Dict[str, float], full_auc: float,
                    n_pos: int, n_neg: int, folds: int, source: str) -> str:
    lines = [
        "# Calibrated Combined Scorer (R2#1)",
        "",
        f"Source labels: `{source}`",
        f"Positive class (`partial` ∪ `hallucinated`): {n_pos}",
        f"Negative class (`faithful`): {n_neg}",
        f"CV folds: {folds} (stratified)",
        "",
        "## Cross-validated detection (equal-weight combiner)",
        "",
        f"- AUC: **{cv_equal['auc_mean']:.3f}**",
        f"- F1: **{cv_equal['f1_mean']:.3f}**",
        f"- Precision: {cv_equal['precision_mean']:.3f}",
        f"- Recall: {cv_equal['recall_mean']:.3f}",
        "",
        "## Combination vs individual signals (AUC)",
        "",
        "The combined score integrates the three CTI-specific signals into a "
        "single decision. Learned per-signal weights (logistic) are reported "
        "only for comparison — if they do not beat equal weights, the "
        "parameter-free rule is justified.",
        "",
        "| combiner / signal | AUC |",
        "|---|---:|",
    ]
    for name, auc in individual_aucs.items():
        lines.append(f"| {name} (individual) | {auc:.3f} |")
    lines.append(f"| **equal-weight combined** | **{cv_equal['auc_mean']:.3f}** |")
    lines.append(f"| logistic combined (comparison) | {cv_logit['auc_mean']:.3f} |")
    lines += [
        "",
        "## Signal contribution (leave-one-out, held-in AUC)",
        "",
        f"Full equal-weight combiner AUC: **{full_auc:.3f}**. "
        "Removing a signal and re-combining — a larger AUC drop means the "
        "signal contributes more to the combination.",
        "",
        "| signal removed | remaining AUC | Δ vs full |",
        "|---|---:|---:|",
    ]
    for name, auc in loo_aucs.items():
        lines.append(f"| {name} | {auc:.3f} | {auc - full_auc:+.3f} |")
    lines += [
        "",
        "## Fitted decision rule",
        "",
        "Combined score `S = (1/d) · Σ_j sign_j · z(x_j)` "
        "(equal weights, zero free parameters).",
        f"Calibration `P = sigmoid({model.platt_a:.4f} · S + {model.platt_b:.4f})` "
        "(2-parameter Platt; the intercept absorbs the base rate, so AUC is "
        "invariant to class balance).",
        "",
        "| feature | sign | weight |",
        "|---|:---:|---:|",
    ]
    for name, sign, w in zip(model.feature_names, model.feature_signs, model.weights):
        sign_label = "+" if sign > 0 else "−"
        lines.append(f"| {name} | {sign_label} | {w:.4f} |")
    lines += [
        "",
        f"- Validated threshold (Youden's J, on P): {model.threshold_youden:.4f}",
        f"- Validated threshold (F1-max, on P): {model.threshold_f1:.4f}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Train calibrated combined scorer")
    p.add_argument("--labels", required=True)
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--directions", nargs="*", default=[])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, help="Markdown report")
    p.add_argument("--dump-model", default=None, help="Optional JSON model dump")
    args = p.parse_args()

    rows = _parse_rows(args.labels, args.features)
    if not rows:
        raise SystemExit("[combined_scorer] no usable rows after parsing")
    X = [r[0] for r in rows]
    y = [r[1] for r in rows]
    signs = _parse_directions(args.directions, args.features)

    model = train_model(X, y, args.features, signs, method="equal")
    cv_equal = cross_validate(X, y, args.features, signs, args.folds, args.seed, "equal")
    cv_logit = cross_validate(X, y, args.features, signs, args.folds, args.seed, "logistic")
    individual_aucs = {
        name: _individual_metric_auc([x[j] for x in X], y, signs[j])
        for j, name in enumerate(args.features)
    }
    full_scores = [_predict_proba(model, x) for x in X]
    full_auc = _auc(full_scores, y)
    loo_aucs = _leave_one_out_auc(X, y, args.features, signs)

    n_pos = sum(1 for v in y if v == 1)
    md = render_markdown(model, cv_equal, cv_logit, individual_aucs, loo_aucs,
                         full_auc, n_pos, len(y) - n_pos, args.folds, args.labels)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    if args.dump_model:
        Path(args.dump_model).write_text(json.dumps(asdict(model), indent=2),
                                         encoding="utf-8")
    print(f"[combined_scorer] equal AUC={cv_equal['auc_mean']:.3f} "
          f"F1={cv_equal['f1_mean']:.3f} | logistic AUC={cv_logit['auc_mean']:.3f} "
          f"→ {args.output}")


if __name__ == "__main__":
    main()
