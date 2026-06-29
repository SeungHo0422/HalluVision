"""
Statistical rigor utilities (Reviewer 2 #8, Reviewer 3).

Provides:
- bootstrap_ci(values, ...)            - mean + bootstrap CI
- paired_bootstrap_diff(a, b, ...)     - Δ(b - a) bootstrap CI + p-value
- bootstrap_metric_table(...)          - per-label summary table for a model
- paired_comparison_table(...)         - per-label paired comparison table

CLI entry point:

    python -m final_pipeline.analyzer.stats \
        --model-a gpt-3.5-turbo --model-b gpt-4o-mini \
        --metrics-a path/to/doc_stats_3.5.json \
        --metrics-b path/to/doc_stats_4o.json \
        --output notes/stats_results.md

Doc-stats JSON expected to be a flat list of {"doc_id", "label", "precision", "recall", "f1"}
or a nested dict {doc_id: {label: {precision, recall, f1}}}. Both shapes accepted.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BootstrapResult:
    mean: float
    ci_lower: float
    ci_upper: float
    n: int


@dataclass
class PairedResult:
    delta_mean: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n: int


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> BootstrapResult:
    """Mean + percentile bootstrap CI for a single sample."""
    values = np.asarray([v for v in values if v is not None and not math.isnan(v)])
    n = len(values)
    if n == 0:
        return BootstrapResult(mean=float("nan"), ci_lower=float("nan"), ci_upper=float("nan"), n=0)
    rng = rng or np.random.default_rng(42)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = values[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return BootstrapResult(
        mean=float(values.mean()),
        ci_lower=float(lo),
        ci_upper=float(hi),
        n=int(n),
    )


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> PairedResult:
    """
    Δ(b - a) mean, percentile CI, and two-sided bootstrap p-value
    (proportion of bootstrap means with opposite sign of observed Δ).

    a, b are paired sequences of equal length (e.g., per-document metric).
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"paired arrays must match: {a_arr.shape} vs {b_arr.shape}")
    diffs = b_arr - a_arr
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return PairedResult(
            delta_mean=float("nan"), ci_lower=float("nan"),
            ci_upper=float("nan"), p_value=float("nan"), n=0,
        )
    rng = rng or np.random.default_rng(42)
    observed = float(diffs.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diffs[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    # Two-sided bootstrap p-value: fraction of bootstrap means at least as
    # extreme (on either side) as 0 under null shifted to observed=0.
    centered = boots - observed
    p = float((np.abs(centered) >= abs(observed)).mean())
    return PairedResult(
        delta_mean=observed,
        ci_lower=float(lo),
        ci_upper=float(hi),
        p_value=p,
        n=int(n),
    )


def _normalize_doc_stats(data) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Accept either:
      - list[{"doc_id", "label", "precision", "recall", "f1"}]
      - dict{doc_id: {label: {precision, recall, f1}}}
      - dict{doc_id: {precision, recall, f1}}  (no per-label key)
    Returns: {doc_id: {label_or_"_overall_": {precision, recall, f1}}}
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    if isinstance(data, list):
        for row in data:
            doc = str(row["doc_id"])
            label = row.get("label", "_overall_")
            out.setdefault(doc, {})[label] = {
                "precision": float(row.get("precision", float("nan"))),
                "recall": float(row.get("recall", float("nan"))),
                "f1": float(row.get("f1", float("nan"))),
            }
    elif isinstance(data, dict):
        first = next(iter(data.values())) if data else {}
        if isinstance(first, dict) and {"precision", "recall", "f1"}.issubset(first.keys()):
            for doc, m in data.items():
                out[str(doc)] = {"_overall_": {k: float(v) for k, v in m.items() if k in {"precision", "recall", "f1"}}}
        else:
            for doc, label_map in data.items():
                out[str(doc)] = {
                    lbl: {k: float(v) for k, v in m.items() if k in {"precision", "recall", "f1"}}
                    for lbl, m in label_map.items()
                }
    else:
        raise ValueError(f"Unsupported doc-stats shape: {type(data)}")
    return out


def _collect_label_values(
    norm: Dict[str, Dict[str, Dict[str, float]]],
    label: str,
    metric: str,
    doc_order: Optional[List[str]] = None,
) -> Tuple[List[str], List[float]]:
    docs = doc_order or sorted(norm.keys())
    vals: List[float] = []
    used: List[str] = []
    for doc in docs:
        cell = norm.get(doc, {}).get(label)
        if cell is None:
            continue
        v = cell.get(metric)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.append(float(v))
        used.append(doc)
    return used, vals


def bootstrap_metric_table(
    metrics_json: Path,
    labels: Iterable[str],
    n_boot: int = 10000,
    ci: float = 0.95,
) -> Dict[str, Dict[str, BootstrapResult]]:
    """Return {label: {metric: BootstrapResult}} for precision/recall/f1."""
    data = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    norm = _normalize_doc_stats(data)
    out: Dict[str, Dict[str, BootstrapResult]] = {}
    for label in labels:
        out[label] = {}
        for metric in ("precision", "recall", "f1"):
            _, vals = _collect_label_values(norm, label, metric)
            out[label][metric] = bootstrap_ci(vals, n_boot=n_boot, ci=ci)
    return out


def paired_comparison_table(
    metrics_a_json: Path,
    metrics_b_json: Path,
    labels: Iterable[str],
    n_boot: int = 10000,
    ci: float = 0.95,
) -> Dict[str, Dict[str, PairedResult]]:
    """Return {label: {metric: PairedResult}} aligned on shared doc_ids."""
    a_norm = _normalize_doc_stats(
        json.loads(Path(metrics_a_json).read_text(encoding="utf-8"))
    )
    b_norm = _normalize_doc_stats(
        json.loads(Path(metrics_b_json).read_text(encoding="utf-8"))
    )
    shared_docs = sorted(set(a_norm) & set(b_norm))
    out: Dict[str, Dict[str, PairedResult]] = {}
    for label in labels:
        out[label] = {}
        for metric in ("precision", "recall", "f1"):
            docs_a, va = _collect_label_values(a_norm, label, metric, shared_docs)
            docs_b, vb = _collect_label_values(b_norm, label, metric, shared_docs)
            common = sorted(set(docs_a) & set(docs_b))
            if not common:
                out[label][metric] = PairedResult(
                    delta_mean=float("nan"), ci_lower=float("nan"),
                    ci_upper=float("nan"), p_value=float("nan"), n=0,
                )
                continue
            a_lookup = dict(zip(docs_a, va))
            b_lookup = dict(zip(docs_b, vb))
            a_aligned = [a_lookup[d] for d in common]
            b_aligned = [b_lookup[d] for d in common]
            out[label][metric] = paired_bootstrap_diff(
                a_aligned, b_aligned, n_boot=n_boot, ci=ci
            )
    return out


def render_markdown_paired(
    table: Dict[str, Dict[str, PairedResult]],
    model_a: str,
    model_b: str,
) -> str:
    lines = [
        f"## Paired comparison: {model_b} − {model_a} (per-document Δ)\n",
        "| Label | Metric | n | Δmean | 95% CI | p (bootstrap) |",
        "|---|---|---:|---:|---|---:|",
    ]
    for label, metrics in table.items():
        for metric, res in metrics.items():
            ci = f"[{res.ci_lower:.4f}, {res.ci_upper:.4f}]"
            lines.append(
                f"| {label} | {metric} | {res.n} | {res.delta_mean:+.4f} | "
                f"{ci} | {res.p_value:.4f} |"
            )
    return "\n".join(lines) + "\n"


def render_markdown_bootstrap(
    table: Dict[str, Dict[str, BootstrapResult]],
    model: str,
) -> str:
    lines = [
        f"## Bootstrap CI: {model}\n",
        "| Label | Metric | n | Mean | 95% CI |",
        "|---|---|---:|---:|---|",
    ]
    for label, metrics in table.items():
        for metric, res in metrics.items():
            ci = f"[{res.ci_lower:.4f}, {res.ci_upper:.4f}]"
            lines.append(f"| {label} | {metric} | {res.n} | {res.mean:.4f} | {ci} |")
    return "\n".join(lines) + "\n"


def _serialize_table(table) -> dict:
    return {
        label: {metric: asdict(res) for metric, res in metrics.items()}
        for label, metrics in table.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stats for HalluVision evaluation")
    parser.add_argument("--metrics-a", required=True, help="JSON file: doc-level metrics for model A")
    parser.add_argument("--metrics-b", required=True, help="JSON file: doc-level metrics for model B")
    parser.add_argument("--model-a", default="model_a")
    parser.add_argument("--model-b", default="model_b")
    parser.add_argument(
        "--labels", nargs="+",
        default=["_overall_", "Malware", "System", "Indicator", "Vulnerability", "Organization"],
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--output", default="notes/stats_results.md")
    args = parser.parse_args()

    boot_a = bootstrap_metric_table(args.metrics_a, args.labels, args.n_boot, args.ci)
    boot_b = bootstrap_metric_table(args.metrics_b, args.labels, args.n_boot, args.ci)
    paired = paired_comparison_table(
        args.metrics_a, args.metrics_b, args.labels, args.n_boot, args.ci
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = (
        f"# Statistical comparison — {args.model_a} vs {args.model_b}\n\n"
        f"- bootstrap samples: {args.n_boot}, confidence: {int(args.ci * 100)}%\n\n"
        + render_markdown_bootstrap(boot_a, args.model_a) + "\n"
        + render_markdown_bootstrap(boot_b, args.model_b) + "\n"
        + render_markdown_paired(paired, args.model_a, args.model_b)
    )
    out_path.write_text(md, encoding="utf-8")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "bootstrap": {args.model_a: _serialize_table(boot_a), args.model_b: _serialize_table(boot_b)},
                "paired": _serialize_table(paired),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[stats] wrote {out_path} and {json_path}")


if __name__ == "__main__":
    main()
