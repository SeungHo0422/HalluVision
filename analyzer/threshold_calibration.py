"""
Threshold calibration for hallucination detection (Reviewer 2 #7).

Existing HalluVision metric cutoffs (cosine 0.85, label-frequency
quartile, etc.) were chosen by inspection. This module fits a single
ROC-optimal cutoff per metric on the human-labeled dev set produced
by ``analyzer.human_sample`` + ``analyzer.agreement`` and reports
detection P/R/F1 plus cross-validated generalization.

* Targets are derived from majority-vote labels: ``hallucinated`` ∪
  ``partial`` = 1, ``faithful`` = 0. Rows lacking dual labels or with
  no metric value are dropped per metric.
* For each metric we sweep all unique values + epsilons as candidate
  cutoffs, compute Youden's J (= TPR − FPR), and select the threshold
  that maximises J. A second pass selects the threshold that maximises
  F1, both are reported.
* ``--direction`` lets callers tell us whether the metric is expected
  to grow with hallucination (e.g. freq_distance) or shrink with it
  (e.g. cosine_sim). Default ``auto`` picks the direction with the
  higher AUC.
* K-fold cross-validation reports out-of-sample P/R/F1 to flag
  overfitting.

The implementation is pure-Python (no sklearn) so the calibration
step does not pull in a heavy dependency just for ROC sweeping.

Usage::

    python -m final_pipeline.analyzer.threshold_calibration \
        --labels notes/human_labels_filled.csv \
        --metrics cosine_sim qa_precision freq_distance \
        --directions cosine_sim:lower qa_precision:lower freq_distance:higher \
        --folds 5 \
        --output notes/threshold_calibration.md
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HALLUC_POSITIVE = {"partial", "hallucinated"}
SEVERITY = {"faithful": 0, "partial": 1, "hallucinated": 2}


@dataclass
class ThresholdResult:
    metric: str
    direction: str  # "higher" or "lower"
    threshold_youden: float
    threshold_f1: float
    youden_j: float
    auc: float
    precision: float
    recall: float
    f1: float
    n_pos: int
    n_neg: int


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


def _confusion_at(scores: List[float], labels: List[int], threshold: float,
                  direction: str) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for s, l in zip(scores, labels):
        if direction == "higher":
            predicted_pos = s >= threshold
        else:
            predicted_pos = s <= threshold
        if predicted_pos and l == 1:
            tp += 1
        elif predicted_pos and l == 0:
            fp += 1
        elif (not predicted_pos) and l == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _auc(scores: List[float], labels: List[int], direction: str) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    pairs = 0.0
    for p in pos:
        for n in neg:
            cmp = (p > n) if direction == "higher" else (p < n)
            if cmp:
                pairs += 1.0
            elif p == n:
                pairs += 0.5
    return pairs / (len(pos) * len(neg))


def _sweep(scores: List[float], labels: List[int], direction: str) -> ThresholdResult:
    if not scores:
        raise ValueError("empty scores")
    cutoffs = sorted(set(scores))
    if not cutoffs:
        raise ValueError("no unique cutoffs")
    best_j = (-1.0, cutoffs[0])
    best_f1 = (-1.0, cutoffs[0], 0.0, 0.0)
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    for c in cutoffs:
        tp, fp, tn, fn = _confusion_at(scores, labels, c, direction)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        p, r, f1 = _prf(tp, fp, fn)
        if j > best_j[0]:
            best_j = (j, c)
        if f1 > best_f1[0]:
            best_f1 = (f1, c, p, r)

    auc = _auc(scores, labels, direction)
    tp, fp, tn, fn = _confusion_at(scores, labels, best_f1[1], direction)
    p, r, f1 = _prf(tp, fp, fn)
    return ThresholdResult(
        metric="?",
        direction=direction,
        threshold_youden=best_j[1],
        threshold_f1=best_f1[1],
        youden_j=best_j[0],
        auc=auc,
        precision=p,
        recall=r,
        f1=f1,
        n_pos=n_pos,
        n_neg=n_neg,
    )


def fit_threshold(
    scores: List[float],
    labels: List[int],
    direction: str = "auto",
) -> ThresholdResult:
    if direction == "auto":
        higher = _auc(scores, labels, "higher")
        lower = _auc(scores, labels, "lower")
        direction = "higher" if (higher >= lower) else "lower"
    return _sweep(scores, labels, direction)


def cross_validate(
    scores: List[float],
    labels: List[int],
    direction: str,
    k: int,
    seed: int,
) -> Tuple[float, float, float]:
    if len(scores) < k:
        return float("nan"), float("nan"), float("nan")
    idx = list(range(len(scores)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    p_sum = r_sum = f1_sum = 0.0
    counted = 0
    for fold in folds:
        test = set(fold)
        train_s = [scores[i] for i in idx if i not in test]
        train_l = [labels[i] for i in idx if i not in test]
        test_s = [scores[i] for i in idx if i in test]
        test_l = [labels[i] for i in idx if i in test]
        if not train_s or not test_s or 1 not in test_l or 0 not in test_l:
            continue
        fit = _sweep(train_s, train_l, direction)
        tp, fp, tn, fn = _confusion_at(test_s, test_l, fit.threshold_f1, direction)
        p, r, f1 = _prf(tp, fp, fn)
        p_sum += p
        r_sum += r
        f1_sum += f1
        counted += 1
    if counted == 0:
        return float("nan"), float("nan"), float("nan")
    return p_sum / counted, r_sum / counted, f1_sum / counted


def _load_rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_directions(args: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for s in args:
        if ":" in s:
            k, v = s.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def calibrate(rows: List[dict], metrics: List[str],
              directions: Dict[str, str], folds: int, seed: int) -> List[dict]:
    results: List[dict] = []
    for metric in metrics:
        scores: List[float] = []
        labels: List[int] = []
        for r in rows:
            a = (r.get("label_a") or "").strip().lower()
            b = (r.get("label_b") or "").strip().lower()
            m = _majority_vote(a, b)
            if m not in {"faithful", "partial", "hallucinated"}:
                continue
            target = 1 if m in HALLUC_POSITIVE else 0
            v = _to_float(r.get(metric, ""))
            if v is None:
                continue
            scores.append(v)
            labels.append(target)
        if not scores or 1 not in labels or 0 not in labels:
            results.append({
                "metric": metric,
                "note": "insufficient labels — skipped",
                "n_pos": labels.count(1),
                "n_neg": labels.count(0),
            })
            continue
        direction = directions.get(metric, "auto")
        fit = fit_threshold(scores, labels, direction)
        fit.metric = metric
        cv_p, cv_r, cv_f1 = cross_validate(scores, labels, fit.direction, folds, seed)
        results.append({
            "metric": metric,
            "direction": fit.direction,
            "threshold_youden": fit.threshold_youden,
            "threshold_f1": fit.threshold_f1,
            "youden_j": fit.youden_j,
            "auc": fit.auc,
            "fit_precision": fit.precision,
            "fit_recall": fit.recall,
            "fit_f1": fit.f1,
            "cv_precision": cv_p,
            "cv_recall": cv_r,
            "cv_f1": cv_f1,
            "n_pos": fit.n_pos,
            "n_neg": fit.n_neg,
        })
    return results


def render_markdown(results: List[dict], source: str, folds: int) -> str:
    lines = [
        "# Threshold Calibration (R2#7)",
        "",
        f"Source labels: `{source}`",
        f"CV folds: {folds}",
        "",
        "| metric | direction | thr (Youden) | thr (F1) | J | AUC | fit P | fit R | fit F1 | CV P | CV R | CV F1 | n+ | n− |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if "note" in r:
            lines.append(
                f"| {r['metric']} | — | — | — | — | — | — | — | — | — | — | — | "
                f"{r['n_pos']} | {r['n_neg']} |  *{r['note']}*"
            )
            continue
        lines.append(
            f"| {r['metric']} | {r['direction']} | "
            f"{r['threshold_youden']:.4f} | {r['threshold_f1']:.4f} | "
            f"{r['youden_j']:.3f} | {r['auc']:.3f} | "
            f"{r['fit_precision']:.3f} | {r['fit_recall']:.3f} | {r['fit_f1']:.3f} | "
            f"{r['cv_precision']:.3f} | {r['cv_recall']:.3f} | {r['cv_f1']:.3f} | "
            f"{r['n_pos']} | {r['n_neg']} |"
        )
    lines += [
        "",
        "**Reading the table.** `thr (Youden)` maximises TPR−FPR on the full dev "
        "set; `thr (F1)` maximises F1. *fit* columns are in-sample detection "
        "metrics at the F1-optimal cutoff; *CV* columns are out-of-sample "
        "K-fold means with the threshold refit per fold — gap between fit and "
        "CV signals overfitting. `direction=lower` means the metric *decreases* "
        "with hallucination; `higher` is the opposite.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate per-metric thresholds")
    p.add_argument("--labels", required=True, help="CSV with label_a/label_b + metric columns")
    p.add_argument("--metrics", nargs="+", required=True)
    p.add_argument("--directions", nargs="*", default=[],
                   help="Per-metric direction overrides, e.g. cosine_sim:lower")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rows = _load_rows(args.labels)
    directions = _parse_directions(args.directions)
    results = calibrate(rows, args.metrics, directions, args.folds, args.seed)
    md = render_markdown(results, args.labels, args.folds)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    print(f"[threshold_calibration] wrote {args.output}")


if __name__ == "__main__":
    main()
