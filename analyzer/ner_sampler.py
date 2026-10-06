"""
Stratified NER sample generator (Reviewer 2 #2 — NER validation).

Produces a CSV ready for two-annotator hand validation against the
predictions in ``original_ner.json``. Sampling is stratified by label
so that each label receives roughly equal coverage, and within a label
we draw a mix of "doc has the label predicted" and "doc has the label
absent" cells so reviewers can score both precision and recall errors.

Two annotators independently fill the label columns; agreement (Cohen's κ,
percent agreement) is then produced by ``analyzer.agreement`` with
``--levels correct,partial,incorrect --col-a annotator_a_correct
--col-b annotator_b_correct``.

Output columns (doccano-friendly):
    doc_id, label, kind, predicted_entities, original_text_excerpt,
    annotator_a_correct (blank), annotator_b_correct (blank),
    annotator_missed_entities (blank), annotator_notes (blank)

For ``precision_check`` cells annotators score the listed
``predicted_entities`` as correct / partial / incorrect. For
``recall_check`` cells (``predicted_entities`` empty) annotators read the
excerpt and list any true ``label`` entities the NER missed in
``annotator_missed_entities`` (and mark ``correct`` if none were missed).

Usage::

    python -m final_pipeline.analyzer.ner_sampler \
        --original original_ner.json \
        --predictions gpt_results_250427.json \
        --n-per-label 20 \
        --output notes/ner_validation_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]


def _load_text_lookup(predictions_path: Optional[str]) -> Dict[str, str]:
    """Best-effort doc_id -> original_text lookup from gpt_results_*.json."""
    if not predictions_path or not Path(predictions_path).exists():
        return {}
    with open(predictions_path, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, str] = {}
    if isinstance(data, list):
        for item in data:
            doc_id = str(item.get("id", ""))
            text = item.get("original_text") or item.get("text") or ""
            if doc_id and text:
                out[doc_id] = text
    return out


def stratified_sample(
    original_ner: Dict[str, dict],
    n_per_label: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    rows: List[dict] = []
    for label in LABELS:
        with_label = [doc for doc, ents in original_ner.items() if ents.get(label)]
        without_label = [doc for doc, ents in original_ner.items() if not ents.get(label)]
        rng.shuffle(with_label)
        rng.shuffle(without_label)
        # 70% precision check (label present in prediction), 30% recall check
        n_with = int(round(n_per_label * 0.7))
        n_without = n_per_label - n_with
        for doc in with_label[:n_with]:
            rows.append({
                "doc_id": doc,
                "label": label,
                "predicted_entities": "|".join(original_ner[doc].get(label, [])),
                "kind": "precision_check",
            })
        for doc in without_label[:n_without]:
            rows.append({
                "doc_id": doc,
                "label": label,
                "predicted_entities": "",
                "kind": "recall_check",
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Sample NER cells for hand validation")
    p.add_argument("--original", required=True, help="original_ner.json")
    p.add_argument(
        "--predictions", default=None,
        help="gpt_results_*.json (optional; used to attach text excerpts)",
    )
    p.add_argument("--n-per-label", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--excerpt-chars", type=int, default=800)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    original = json.loads(Path(args.original).read_text(encoding="utf-8"))
    text_lookup = _load_text_lookup(args.predictions)

    rows = stratified_sample(original, args.n_per_label, args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "doc_id", "label", "kind", "predicted_entities",
            "original_text_excerpt",
            "annotator_a_correct",       # to be filled: correct/partial/incorrect
            "annotator_b_correct",       # to be filled: correct/partial/incorrect
            "annotator_missed_entities",  # recall_check: entities NER missed
            "annotator_notes",
        ])
        for row in rows:
            text = text_lookup.get(row["doc_id"], "")
            excerpt = text[: args.excerpt_chars]
            writer.writerow([
                row["doc_id"], row["label"], row["kind"],
                row["predicted_entities"], excerpt, "", "", "", "",
            ])

    print(f"[ner_sampler] wrote {len(rows)} cells → {args.output}")


if __name__ == "__main__":
    main()
