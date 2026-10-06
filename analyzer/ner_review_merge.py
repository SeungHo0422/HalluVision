"""
Apply a review-GUI progress JSON onto the NER validation sheets (R2 #2a).

``review.html``'s "JSON 저장" button exports the annotator's working state::

    {"ent": {"<row_id>": {"verdict": ..., "error_type": ...,
                          "correct_form": ..., "notes": ...}, ...},
     "doc": {"<doc_id>": {"missed": {"<Label>": "A | B"},
                          "doc_done": "y", "doc_notes": ...}, ...},
     "i": 0, "di": 0, "mode": "precision"}

This writes those judgments into the two CSVs so ``analyzer.ner_manual_score``
can read them. Use it when the browser runs on a different machine than the
sheets (VS Code port forwarding, SSH tunnel) and only the JSON made it across —
the CSV export button produces the same result directly.

Unknown row ids / doc ids are reported rather than silently dropped: they mean
the sheets were regenerated after labeling started, and the numbers would be
attached to the wrong entities.

Usage::

    python -m final_pipeline.analyzer.ner_review_merge \
        --progress ner_review_progress.json \
        --entities notes/csv/manual_ner_70/B_entities_precision.csv \
        --docs notes/csv/manual_ner_70/A_docs_recall.csv

With ``--as-annotator-b`` (R2 conditional-accept #3, second rater) only the
``verdict`` values are written, into ``verdict_b``, and the first annotator's
columns and the doc/recall sheet are left untouched; ``analyzer.ner_manual_score``
then reports Cohen's κ. The second rater's error_type/notes stay in their
progress JSON — keep that file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
ENTITY_FIELDS = ["verdict", "error_type", "correct_form", "notes"]


def _read_csv(path: str) -> Tuple[List[str], List[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: str, header: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def merge(progress: dict, ent_rows: List[dict], doc_rows: List[dict],
          as_annotator_b: bool = False) -> dict:
    ent_state: Dict[str, dict] = progress.get("ent") or {}
    doc_state: Dict[str, dict] = progress.get("doc") or {}

    n_ent = 0
    seen_ent = set()
    for row in ent_rows:
        rec = ent_state.get(str(row.get("row_id")))
        if not rec:
            continue
        seen_ent.add(str(row.get("row_id")))
        if as_annotator_b:
            row["verdict_b"] = rec.get("verdict", "") or ""
            if row["verdict_b"]:
                n_ent += 1
            continue
        for field in ENTITY_FIELDS:
            row[field] = rec.get(field, "") or ""
        if row["verdict"]:
            n_ent += 1

    n_doc = 0
    seen_doc = set()
    for row in doc_rows:
        rec = doc_state.get(str(row.get("doc_id")))
        if not rec:
            continue
        seen_doc.add(str(row.get("doc_id")))
        if as_annotator_b:
            # The recall sheet has no *_b columns; κ covers entity verdicts only.
            continue
        missed = rec.get("missed") or {}
        for label in LABELS:
            row[f"missed_{label}"] = missed.get(label, "") or ""
        row["doc_done"] = rec.get("doc_done", "") or ""
        row["doc_notes"] = rec.get("doc_notes", "") or ""
        if row["doc_done"]:
            n_doc += 1

    return {
        "entities_written": n_ent,
        "docs_done": n_doc,
        "orphan_row_ids": sorted(set(ent_state) - seen_ent, key=lambda s: (len(s), s)),
        "orphan_doc_ids": sorted(set(doc_state) - seen_doc, key=lambda s: (len(s), s)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Merge review-GUI progress JSON into the sheets")
    p.add_argument("--progress", required=True, help="ner_review_progress.json")
    p.add_argument("--entities", default="notes/csv/manual_ner_70/B_entities_precision.csv")
    p.add_argument("--docs", default="notes/csv/manual_ner_70/A_docs_recall.csv")
    p.add_argument("--outdir", default=None,
                   help="write the filled sheets here instead of in place")
    p.add_argument("--as-annotator-b", action="store_true",
                   help="write verdicts into verdict_b (second rater, Cohen's κ); "
                        "leaves the first annotator's columns and the recall sheet alone")
    args = p.parse_args()

    progress = json.loads(Path(args.progress).read_text(encoding="utf-8"))
    if "ent" not in progress and "doc" not in progress:
        raise SystemExit("[ner_review_merge] not a review progress file "
                         "(expected top-level 'ent' / 'doc' keys)")

    ent_header, ent_rows = _read_csv(args.entities)
    doc_header, doc_rows = _read_csv(args.docs)
    result = merge(progress, ent_rows, doc_rows, as_annotator_b=args.as_annotator_b)

    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        ent_path = str(out / Path(args.entities).name)
        doc_path = str(out / Path(args.docs).name)
    else:
        ent_path, doc_path = args.entities, args.docs

    _write_csv(ent_path, ent_header, ent_rows)
    _write_csv(doc_path, doc_header, doc_rows)

    if args.as_annotator_b:
        print(f"[ner_review_merge] {result['entities_written']} verdicts → verdict_b "
              f"in {ent_path} (recall sheet untouched)")
    else:
        print(f"[ner_review_merge] {result['entities_written']} verdicts, "
              f"{result['docs_done']} docs marked done → {ent_path}, {doc_path}")
    for kind, ids in (("row_id", result["orphan_row_ids"]), ("doc_id", result["orphan_doc_ids"])):
        if ids:
            print(f"[ner_review_merge] WARNING: {len(ids)} {kind}(s) in the progress "
                  f"file are absent from the sheet — the sheet was likely regenerated "
                  f"after labeling started: {ids[:10]}")


if __name__ == "__main__":
    main()
