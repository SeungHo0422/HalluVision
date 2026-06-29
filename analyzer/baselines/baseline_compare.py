"""
Compare HalluVision metrics against FactScore / QAGS baselines (R2#6).

Inputs:
    --halluvision   doc-level metric JSON keyed by doc_id
                    {doc_id: {cosine_sim, qa_precision, freq_distance, ...}}
                    Missing keys are skipped per pair.
    --factscore     JSON list from baselines/factscore_runner.py
    --qags          JSON list from baselines/qags_runner.py
    --output        markdown report path

Outputs Spearman ρ and Pearson r matrices, plus an agreement / dis-
agreement contingency table where each metric is binarised at its
median: HalluVision says ``hallucinated`` if the metric value is on
the *worse* side of the median, and similarly for FactScore / QAGS.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")

    def _rank(values: List[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx and dy else float("nan")


def _load_halluvision(path: str) -> Dict[str, Dict[str, float]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): {kk: v for kk, v in vv.items() if isinstance(v, (int, float))}
            for k, vv in data.items()}


def _load_baseline_list(path: Optional[str], key: str) -> Dict[str, float]:
    if not path:
        return {}
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(r.get("doc_id")): r.get(key)
            for r in rows if r.get(key) is not None}


def _pair_values(left: Dict[str, float], right: Dict[str, float]) -> Tuple[List[float], List[float]]:
    shared = set(left.keys()) & set(right.keys())
    xs, ys = [], []
    for k in sorted(shared):
        a, b = left[k], right[k]
        if a is None or b is None:
            continue
        xs.append(float(a))
        ys.append(float(b))
    return xs, ys


def _binarize(values: List[float], direction: str) -> List[int]:
    if not values:
        return []
    median = statistics.median(values)
    if direction == "higher":  # higher = more hallucinated
        return [1 if v >= median else 0 for v in values]
    return [1 if v <= median else 0 for v in values]


def _contingency(a: List[int], b: List[int]) -> Dict[str, int]:
    out = {"agree_halluc": 0, "agree_faithful": 0,
           "halluvision_only": 0, "baseline_only": 0}
    for x, y in zip(a, b):
        if x == 1 and y == 1:
            out["agree_halluc"] += 1
        elif x == 0 and y == 0:
            out["agree_faithful"] += 1
        elif x == 1 and y == 0:
            out["halluvision_only"] += 1
        else:
            out["baseline_only"] += 1
    return out


def render_markdown(corr_rows: List[dict], contingency_rows: List[dict]) -> str:
    lines = [
        "# Baseline Comparison — FactScore / QAGS vs HalluVision (R2#6)",
        "",
        "## Rank correlation (Spearman ρ) & linear correlation (Pearson r)",
        "",
        "| HalluVision metric | direction | Baseline | n | Spearman ρ | Pearson r |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in corr_rows:
        lines.append(
            f"| {r['hv_metric']} | {r['hv_direction']} | {r['baseline']} | "
            f"{r['n']} | {r['spearman']:.3f} | {r['pearson']:.3f} |"
        )
    lines += [
        "",
        "## Median-split agreement contingency",
        "",
        "Both signals binarised at their median; positive class = hallucinated.",
        "",
        "| HV metric | Baseline | both halluc | both faithful | HV only | baseline only |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in contingency_rows:
        lines.append(
            f"| {r['hv_metric']} | {r['baseline']} | "
            f"{r['contingency']['agree_halluc']} | "
            f"{r['contingency']['agree_faithful']} | "
            f"{r['contingency']['halluvision_only']} | "
            f"{r['contingency']['baseline_only']} |"
        )
    lines.append("")
    return "\n".join(lines)


def compare(hv: Dict[str, Dict[str, float]],
            factscore: Dict[str, float],
            qags: Dict[str, float],
            hv_directions: Dict[str, str]) -> Tuple[List[dict], List[dict]]:
    corr_rows: List[dict] = []
    contingency_rows: List[dict] = []
    baselines = {"factscore": factscore, "qags": qags}
    for hv_metric, direction in hv_directions.items():
        hv_values = {k: v.get(hv_metric) for k, v in hv.items() if v.get(hv_metric) is not None}
        for bname, bmap in baselines.items():
            if not bmap:
                continue
            xs, ys = _pair_values(hv_values, bmap)
            if not xs:
                continue
            corr_rows.append({
                "hv_metric": hv_metric,
                "hv_direction": direction,
                "baseline": bname,
                "n": len(xs),
                "spearman": _spearman(xs, ys),
                "pearson": _pearson(xs, ys),
            })
            # baseline scores are *factuality* scores → low value = hallucinated
            hv_bin = _binarize(xs, direction)
            base_bin = _binarize(ys, "lower")
            contingency_rows.append({
                "hv_metric": hv_metric,
                "baseline": bname,
                "contingency": _contingency(hv_bin, base_bin),
            })
    return corr_rows, contingency_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Correlate HalluVision metrics vs FactScore/QAGS")
    p.add_argument("--halluvision", required=True,
                   help="doc_id -> {metric: value} JSON")
    p.add_argument("--factscore", default=None)
    p.add_argument("--qags", default=None)
    p.add_argument("--hv-directions", nargs="*", default=[
        "cosine_sim:lower", "qa_precision:lower", "freq_distance:higher",
    ])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    hv = _load_halluvision(args.halluvision)
    factscore = _load_baseline_list(args.factscore, "factscore")
    qags = _load_baseline_list(args.qags, "qags")
    directions = {s.split(":", 1)[0]: s.split(":", 1)[1] for s in args.hv_directions if ":" in s}

    corr, cont = compare(hv, factscore, qags, directions)
    md = render_markdown(corr, cont)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    print(f"[baseline_compare] wrote {args.output} ({len(corr)} rows)")


if __name__ == "__main__":
    main()
