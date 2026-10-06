"""
Assemble the HalluVision metric map for the R2 #6 baseline subset.

``baseline_compare.py`` wants ``{doc_id: {metric: value}}``. The three
paper metrics live in three different artefacts, so this pulls them
together for the 200-doc subset:

* ``cosine_sim``      — jina-v3 CLS cosine, from ``analyzer.cosine_similarity``
                        output (``{doc_id: {_overall_: {cosine_sim}}}``)
* ``qa_precision``    — ``_overall_.precision`` of the QA answer-matching
                        doc stats (``notes/doc_stats_qa_gpt4omini.json``)
* ``freq_distance``   — paper Eq. 5 TV distance of the entity-label
                        distributions (``entity_freq_gpt4omini.json``)

Docs missing a metric simply omit that key; ``baseline_compare`` pairs
per metric, so coverage differences do not silently drop whole rows.

Usage::

    python -m final_pipeline.analyzer.baselines.hv_metrics \
        --subset notes/baseline_subset.json \
        --cosine notes/doc_cosine_baseline_subset.json \
        --qa-stats notes/doc_stats_qa_gpt4omini.json \
        --entity-freq notes/combined_scorer_demo/entity_freq_gpt4omini.json \
        --output notes/hv_metrics_baseline_subset.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build(subset_path: str, cosine_path: str, qa_stats_path: str,
          entity_freq_path: str) -> Dict[str, Dict[str, float]]:
    doc_ids = [str(d) for d in _load(subset_path)["doc_ids"]]
    cosine = _load(cosine_path)
    qa = _load(qa_stats_path)
    freq = _load(entity_freq_path)

    out: Dict[str, Dict[str, float]] = {}
    for did in doc_ids:
        rec: Dict[str, float] = {}

        c = (cosine.get(did) or {}).get("_overall_", {}).get("cosine_sim")
        if c is not None:
            rec["cosine_sim"] = float(c)

        p = (qa.get(did) or {}).get("_overall_", {}).get("precision")
        if p is not None:
            rec["qa_precision"] = float(p)

        f = (freq.get(did) or {}).get("freq_distance")
        if f is not None:
            rec["freq_distance"] = float(f)

        if rec:
            out[did] = rec
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build HV metric map for the baseline subset")
    p.add_argument("--subset", required=True)
    p.add_argument("--cosine", required=True)
    p.add_argument("--qa-stats", required=True)
    p.add_argument("--entity-freq", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    metrics = build(args.subset, args.cosine, args.qa_stats, args.entity_freq)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    cov = {k: sum(1 for v in metrics.values() if k in v)
           for k in ("cosine_sim", "qa_precision", "freq_distance")}
    print(f"[hv_metrics] {len(metrics)} docs → {args.output}; coverage {cov}")


if __name__ == "__main__":
    main()
