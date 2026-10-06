"""
Agreement between two NER entity-set files (R2 #2 / R2 #5).

Compares two files of the shape produced by CyNER runs (``original_ner.json``,
``thn_gold_baseCyNER.json``, ...)::

    {"<doc_id>": {"Malware": [...], "System": [...], "Indicator": [...],
                  "Vulnerability": [...], "Organization": [...]}, ...}

treating ``--gold`` as the reference and ``--pred`` as the candidate, and
reports per-label / micro precision, recall, F1 and mean Jaccard over the
documents present in both files.

Two uses in this project:

1. **Checkpoint identity check** — before running a recovered CyNER checkpoint
   on a new corpus, run it on documents that already have gold and confirm it
   reproduces them (expect F1 ≈ 1.0). A materially lower score means it is not
   the checkpoint that produced the paper's gold, and using it would silently
   break parity.
2. **NER sensitivity analysis (R2 #2)** — quantify how much the measurement
   depends on the NER choice by comparing a substitute NER against the
   fine-tuned gold on the same documents.

Note: divergence shows the two models disagree, i.e. at least one is wrong on
those spans. It does not by itself establish which is correct — that is what
the §2a human validation is for.

Usage::

    python final_pipeline/analyzer/ner_agreement.py \
        --gold original_ner.json \
        --pred pipeline_output/thn/tp_sample_gold_baseCyNER.json \
        --output notes/ner_sensitivity_base_vs_finetuned.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .span_match import token_f1
except ImportError:  # pragma: no cover - direct script execution
    from span_match import token_f1

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]


def _norm(s: str) -> str:
    return s.strip().lower()


def _entity_set(doc: dict, label: str) -> set:
    return {_norm(e) for e in (doc.get(label) or []) if _norm(e)}


def _relaxed_counts(pred: set, gold: set, threshold: float) -> Tuple[int, int, int]:
    """Greedy 1-1 alignment: a pred entity matches an unused gold entity when
    token-F1 >= threshold. Falls back to counting the leftovers as fp/fn."""
    unused = set(gold)
    tp = 0
    for p in pred:
        best, best_score = None, 0.0
        for g in unused:
            score = 1.0 if p == g else token_f1(p, g)
            if score > best_score:
                best, best_score = g, score
        if best is not None and best_score >= threshold:
            unused.discard(best)
            tp += 1
    return tp, len(pred) - tp, len(unused)


def compare(
    gold: Dict[str, dict],
    pred: Dict[str, dict],
    labels: List[str],
    relaxed_threshold: Optional[float] = None,
) -> dict:
    common = [k for k in pred if k in gold]
    per_label: Dict[str, dict] = {}
    micro = [0, 0, 0]

    for label in labels:
        tp = fp = fn = 0
        jaccard: List[float] = []
        for k in common:
            p = _entity_set(pred[k], label)
            g = _entity_set(gold[k], label)
            if relaxed_threshold is None:
                d_tp, d_fp, d_fn = len(p & g), len(p - g), len(g - p)
            else:
                d_tp, d_fp, d_fn = _relaxed_counts(p, g, relaxed_threshold)
            tp += d_tp
            fp += d_fp
            fn += d_fn
            if p or g:
                jaccard.append(len(p & g) / len(p | g))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "mean_jaccard": (sum(jaccard) / len(jaccard)) if jaccard else 0.0,
            "n_pred": tp + fp,
            "n_gold": tp + fn,
        }
        micro[0] += tp
        micro[1] += fp
        micro[2] += fn

    tp, fp, fn = micro
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n_gold_docs": len(gold),
        "n_pred_docs": len(pred),
        "n_common_docs": len(common),
        "match_mode": "exact" if relaxed_threshold is None else f"relaxed@{relaxed_threshold}",
        "per_label": per_label,
        "micro": {
            "precision": precision,
            "recall": recall,
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "n_pred": tp + fp,
            "n_gold": tp + fn,
        },
    }


def render_markdown(report: dict, gold_path: str, pred_path: str) -> str:
    lines = [
        "# NER Agreement Report",
        "",
        f"- Reference (gold): `{gold_path}`",
        f"- Candidate (pred): `{pred_path}`",
        f"- Documents compared: **{report['n_common_docs']}** "
        f"(gold file {report['n_gold_docs']}, pred file {report['n_pred_docs']})",
        f"- Match mode: `{report['match_mode']}`",
        "",
        "| label | P | R | F1 | mean Jaccard | #pred | #gold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, s in report["per_label"].items():
        lines.append(
            f"| {label} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | "
            f"{s['mean_jaccard']:.3f} | {s['n_pred']} | {s['n_gold']} |"
        )
    m = report["micro"]
    lines += [
        f"| **MICRO** | **{m['precision']:.3f}** | **{m['recall']:.3f}** | "
        f"**{m['f1']:.3f}** | | {m['n_pred']} | {m['n_gold']} |",
        "",
        "Divergence means the two models disagree on those spans, i.e. at least one",
        "is wrong; it does not by itself establish which. Correctness is settled by",
        "the §2a human validation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Agreement between two NER entity-set files")
    p.add_argument("--gold", required=True, help="Reference NER JSON (e.g. original_ner.json)")
    p.add_argument("--pred", required=True, help="Candidate NER JSON")
    p.add_argument("--labels", default=",".join(LABELS), help="Comma-separated labels")
    p.add_argument("--relaxed", type=float, default=None,
                   help="Token-F1 threshold for relaxed entity alignment (default: exact match)")
    p.add_argument("--output", default=None, help="Markdown report path (default: stdout only)")
    p.add_argument("--json-output", default=None, help="Optional machine-readable JSON path")
    args = p.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))

    report = compare(gold, pred, labels, relaxed_threshold=args.relaxed)
    md = render_markdown(report, args.gold, args.pred)
    print(md)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"[ner_agreement] wrote {args.output}")
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[ner_agreement] wrote {args.json_output}")


if __name__ == "__main__":
    main()
