"""
Calibrated combined hallucination scorer (Reviewer 2 #1 — methodology
novelty contribution).

Combines the three HalluVision signals (cosine similarity, answer-
matching score / QA precision, frequency distance) into a single
hallucination probability via logistic regression trained on the §7
human-label majority vote. A coefficient table is dumped to support
the paper's interpretability claim.

The implementation is intentionally minimal:

* Pure-Python gradient descent for logistic regression — no sklearn
  dependency, so the calibration step runs anywhere the rest of the
  analyzer runs.
* z-score feature normalisation so the learned coefficients are
  comparable in magnitude (a proxy for feature importance).
* K-fold cross validation reports AUC and F1 with thresholds picked
  by Youden's J on each train fold. The held-out fold's score is the
  out-of-sample number reported in the paper.
* Direction hints (``--directions metric:lower``) flip the sign of a
  feature *before* training so positive coefficients always mean
  "this feature drives the hallucinated class up".

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
class LogisticModel:
    feature_names: List[str]
    coefficients: List[float]
    intercept: float
    feature_means: List[float]
    feature_stds: List[float]
    feature_signs: List[float]  # +1 or -1 per feature (direction)
    threshold_youden: float
    threshold_f1: float


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


def _predict_proba(model: LogisticModel, x_raw: List[float]) -> float:
    if len(x_raw) != len(model.feature_names):
        raise ValueError("feature length mismatch")
    z = model.intercept
    for j, v in enumerate(x_raw):
        v_signed = v * model.feature_signs[j]
        v_std = (v_signed - model.feature_means[j]) / model.feature_stds[j]
        z += model.coefficients[j] * v_std
    return _sigmoid(z)


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
                feature_names: List[str], feature_signs: List[float]) -> LogisticModel:
    X_signed = _apply_signs(rows, feature_signs)
    Z, means, stds = _standardise(X_signed)
    w, b = _train_logreg(Z, labels)
    scores = [_sigmoid(b + sum(w[j] * z[j] for j in range(len(w)))) for z in Z]
    thr_y, thr_f1, _, _ = _best_threshold(scores, labels)
    return LogisticModel(
        feature_names=feature_names,
        coefficients=w,
        intercept=b,
        feature_means=means,
        feature_stds=stds,
        feature_signs=feature_signs,
        threshold_youden=thr_y,
        threshold_f1=thr_f1,
    )


def cross_validate(X: List[List[float]], y: List[int],
                   feature_names: List[str], feature_signs: List[float],
                   folds: int, seed: int) -> Dict[str, float]:
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    fold_assign = [idx[i::folds] for i in range(folds)]
    aucs, f1s, precisions, recalls = [], [], [], []
    for fold in fold_assign:
        test = set(fold)
        Xtr = [X[i] for i in idx if i not in test]
        ytr = [y[i] for i in idx if i not in test]
        Xte = [X[i] for i in idx if i in test]
        yte = [y[i] for i in idx if i in test]
        if not Xtr or not Xte or len(set(yte)) < 2:
            continue
        model = train_model(Xtr, ytr, feature_names, feature_signs)
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


def render_markdown(model: LogisticModel, cv: Dict[str, float],
                    individual_aucs: Dict[str, float],
                    n_pos: int, n_neg: int, folds: int, source: str) -> str:
    lines = [
        "# Calibrated Combined Scorer (R2#1)",
        "",
        f"Source labels: `{source}`",
        f"Positive class (`partial` ∪ `hallucinated`): {n_pos}",
        f"Negative class (`faithful`): {n_neg}",
        f"CV folds: {folds}",
        "",
        "## Cross-validated detection",
        "",
        f"- AUC: **{cv['auc_mean']:.3f}**",
        f"- F1: **{cv['f1_mean']:.3f}**",
        f"- Precision: {cv['precision_mean']:.3f}",
        f"- Recall: {cv['recall_mean']:.3f}",
        "",
        "## Individual metric AUC (for comparison)",
        "",
        "| metric | AUC | direction-adjusted |",
        "|---|---:|:---:|",
    ]
    for name, auc in individual_aucs.items():
        lines.append(f"| {name} | {auc:.3f} | ✓ |")
    lines += [
        "",
        "## Fitted logistic-regression coefficients",
        "",
        f"- intercept (β₀): {model.intercept:.4f}",
        "",
        "| feature | sign | β (z-scored) | threshold (Youden) | threshold (F1) |",
        "|---|:---:|---:|---:|---:|",
    ]
    for name, coef, sign in zip(model.feature_names, model.coefficients, model.feature_signs):
        sign_label = "+" if sign > 0 else "−"
        lines.append(
            f"| {name} | {sign_label} | {coef:+.4f} | "
            f"{model.threshold_youden:.4f} | {model.threshold_f1:.4f} |"
        )
    lines += [
        "",
        "Positive `β` after the sign flip means: *the metric pushes the score "
        "toward hallucination as it grows in the halluc-aligned direction*. "
        "Magnitudes are comparable because features are z-scored before training.",
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

    model = train_model(X, y, args.features, signs)
    cv = cross_validate(X, y, args.features, signs, args.folds, args.seed)
    individual_aucs = {
        name: _individual_metric_auc([x[j] for x in X], y, signs[j])
        for j, name in enumerate(args.features)
    }

    n_pos = sum(1 for v in y if v == 1)
    md = render_markdown(model, cv, individual_aucs, n_pos, len(y) - n_pos,
                         args.folds, args.labels)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    if args.dump_model:
        Path(args.dump_model).write_text(json.dumps(asdict(model), indent=2),
                                         encoding="utf-8")
    print(f"[combined_scorer] AUC={cv['auc_mean']:.3f} F1={cv['f1_mean']:.3f} → {args.output}")


if __name__ == "__main__":
    main()
