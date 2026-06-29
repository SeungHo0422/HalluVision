"""
Extract per-document P/R/F1 from existing pipeline outputs for the
stats module (Reviewer 2 #8). Wraps the legacy top-level scripts.

Outputs JSON per model in the shape consumed by `stats.py`:

  {
    "<doc_id>": {
       "<label>": {"precision": ..., "recall": ..., "f1": ...},
       ...,
       "_overall_": {"precision": ..., "recall": ..., "f1": ...}
    },
    ...
  }

Usage:
    python -m final_pipeline.analyzer.extract_doc_stats \
        --task qa \
        --original original_ner.json \
        --predictions gpt_results_250427.json \
        --output notes/doc_stats_qa_gpt4omini.json

    python -m final_pipeline.analyzer.extract_doc_stats \
        --task summary \
        --original original_ner.json \
        --predictions summary_ner.json \
        --output notes/doc_stats_summary_gpt4omini.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import ujson
from sklearn.metrics import precision_recall_fscore_support

try:
    from .span_match import score_doc_label
except ImportError:
    from span_match import score_doc_label

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]


def _qa_doc_stats(
    original: dict,
    qa_predictions: List[dict],
    match_mode: str = "exact",
    f1_threshold: float = 0.5,
) -> dict:
    """Per-doc stats compatible with analysis_qa.py logic but emitting JSON.

    Mirrors `evaluate_qa_vs_ner` (analysis_qa.py:81-128). ``match_mode``
    routes through ``span_match.score_doc_label`` so the same extractor
    produces exact / token_f1 / jaccard / relaxed metric files.
    """
    out: Dict[str, dict] = {}
    id_to_qa = {
        item["id"]: item["qa_text"]
        for item in qa_predictions
        if "id" in item and "qa_text" in item
    }
    for doc_id, qa_str in id_to_qa.items():
        items = [p.strip() for p in qa_str.strip("{} ").split(",") if p.strip()]
        if len(items) != len(LABELS):
            continue
        ner_entry = original.get(str(doc_id))
        if not ner_entry:
            continue
        out[str(doc_id)] = {}
        overall_t, overall_p = [], []
        for idx, label in enumerate(LABELS):
            # analysis_qa.py swaps idx 2/3 because of label order mismatch.
            if idx == 2:
                pred = items[3]
            elif idx == 3:
                pred = items[2]
            else:
                pred = items[idx]
            gold = list(ner_entry.get(label, []))
            match_int, _ = score_doc_label(pred, gold, match_mode, f1_threshold)
            is_pred = 0 if (not pred or pred.lower().strip() == "no answer") else 1
            y_true = [match_int]
            y_pred = [is_pred]
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            out[str(doc_id)][label] = {"precision": float(p), "recall": float(r), "f1": float(f1)}
            overall_t.append(match_int)
            overall_p.append(is_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            overall_t, overall_p, average="binary", zero_division=0
        )
        out[str(doc_id)]["_overall_"] = {
            "precision": float(p), "recall": float(r), "f1": float(f1),
        }
    return out


def _summary_doc_stats(original: dict, summary: dict, split_commas: bool) -> dict:
    """Per-doc stats matching eval_summary.py's per-label evaluation.

    `split_commas` toggles the gpt-3.5 special branch (eval_summary.py:52-58)
    where comma-joined entities are exploded.
    """
    out: Dict[str, dict] = {}
    for doc_id in summary:
        orig = original.get(doc_id)
        if orig is None:
            continue
        summ = summary[doc_id]
        out[doc_id] = {}
        overall_t, overall_p = [], []
        for label in LABELS:
            true_entities = set(orig.get(label, []))
            if split_commas:
                pre = set(summ.get(label, []))
                pred_entities = set()
                for ent in pre:
                    pred_entities.update(x.strip() for x in ent.split(","))
            else:
                pred_entities = set(summ.get(label, []))
            y_true: List[int] = []
            y_pred: List[int] = []
            for true in true_entities:
                y_true.append(1)
                y_pred.append(1 if true in pred_entities else 0)
            for pred_item in pred_entities:
                if pred_item not in true_entities:
                    y_true.append(0)
                    y_pred.append(1)
            if not y_true:
                continue
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            out[doc_id][label] = {"precision": float(p), "recall": float(r), "f1": float(f1)}
            overall_t.extend(y_true)
            overall_p.extend(y_pred)
        if overall_t:
            p, r, f1, _ = precision_recall_fscore_support(
                overall_t, overall_p, average="binary", zero_division=0
            )
            out[doc_id]["_overall_"] = {
                "precision": float(p), "recall": float(r), "f1": float(f1),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-doc P/R/F1 to JSON")
    parser.add_argument("--task", required=True, choices=["qa", "summary"])
    parser.add_argument("--original", required=True, help="original_ner.json (CyNER ground truth)")
    parser.add_argument("--predictions", required=True, help="Predictions file (qa or summary NER)")
    parser.add_argument(
        "--split-commas", action="store_true",
        help="(summary task) split comma-joined entities like eval_summary.py does for 3.5",
    )
    parser.add_argument(
        "--match-mode", default="exact",
        choices=["exact", "token_f1", "jaccard", "relaxed"],
        help="(qa task) span match mode (Reviewer 2 #10)",
    )
    parser.add_argument(
        "--f1-threshold", type=float, default=0.5,
        help="(qa task) F1 threshold for token_f1/jaccard/relaxed",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.original, encoding="utf-8") as f:
        original = ujson.load(f)
    with open(args.predictions, encoding="utf-8") as f:
        preds = ujson.load(f)

    if args.task == "qa":
        if not isinstance(preds, list):
            raise SystemExit("QA predictions must be a JSON list (gpt_results_*.json shape)")
        stats = _qa_doc_stats(
            original, preds,
            match_mode=args.match_mode,
            f1_threshold=args.f1_threshold,
        )
    else:
        if not isinstance(preds, dict):
            raise SystemExit("Summary predictions must be a JSON dict {doc_id: {label: [...]}}")
        stats = _summary_doc_stats(original, preds, split_commas=args.split_commas)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    print(f"[extract_doc_stats] {args.task}: {len(stats)} docs → {args.output}")


if __name__ == "__main__":
    main()
