"""
Build the blinded labeling payload for the R2#3 human-validation web app.

Joins the frozen 100-row human-validation sheet with the fixed QA questions,
the per-doc auto metrics, and (for post-hoc audit only) the gold NER lists,
emitting ``webapp/data/items.json`` that the static SPA loads.

This script **does not re-sample** — it reads the 100 rows from
``notes/human_label_sample.csv`` in file order, which *is* the seed=42
stratified order produced by ``human_sample.py``.

Blinding contract: the fields that would bias an annotator (gold NER, the
three auto metrics, and the stratification buckets) are carried only under
each item's ``_hidden`` block. The labeling screen never reads ``_hidden``;
it exists solely for a later reveal/audit mode.

Usage::

    python -m final_pipeline.analyzer.build_items
    python -m final_pipeline.analyzer.build_items \
        --sample notes/human_label_sample.csv \
        --metrics notes/combined_scorer_demo/doc_metrics_labelfree.json \
        --gold final_pipeline/datasets/ground_truth_ner.json \
        --output webapp/data/items.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# Repo root = two levels up from this file (final_pipeline/analyzer/build_items.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# The 3-way ordinal scheme shared with agreement.py.
LABEL_SCHEME = ["faithful", "partial", "hallucinated"]

# The 5 fixed QA questions, in **qa_handler order** (Malware, System,
# Vulnerability, Indicator, Organization). This order must match the order of
# the answers packed into the ``qa_text`` braces by the LLM proxy — see
# final_pipeline/llm_proxy/qa_handler.py lines ~63-73. Pairing in any other
# order silently mislabels every QA row.
QA_QUESTIONS = [
    "What is the main MALWARE covered in this article?",
    "What is the main SYSTEM covered in this article?",
    "What is the main VULNERABILITY covered in this article?",
    "What is the main INDICATOR covered in this article?",
    "What is the main ORGANIZATION covered in this article?",
]
QA_LABELS = ["Malware", "System", "Vulnerability", "Indicator", "Organization"]

# Version stamp for items.json. Bumped by hand (Date.now() is intentionally not
# used so runs are reproducible).
PAYLOAD_VERSION = "2026-07-11"


def parse_qa_text(qa_text: str) -> tuple[List[str], bool]:
    """Split ``"{a1, a2, a3, a4, a5}"`` into 5 answer strings.

    Returns ``(answers, warning)``. If the field does not split into exactly
    five tokens (e.g. an answer itself contains a comma), ``answers`` is padded
    to length 5 with the raw string in slot 0 and ``warning`` is True so the
    caller can flag the row for manual review.
    """
    inner = qa_text.strip()

    # Variant B: a few rows stored the raw LLM output rather than the packed
    # brace form, e.g. ``Answer 1 : ```BloodyStealer```\nAnswer 2 : ```Steam```...``.
    # Recover the 5 answers from the triple-backtick spans.
    if "```" in inner:
        spans = [s.strip() for s in re.findall(r"```(.*?)```", inner, flags=re.DOTALL)]
        if len(spans) == 5:
            return spans, False

    # Variant A (canonical): ``{a1, a2, a3, a4, a5}``.
    if inner.startswith("{"):
        inner = inner[1:]
    if inner.endswith("}"):
        inner = inner[:-1]
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != 5:
        padded = [qa_text.strip()] + ["(parse error)"] * 4
        return padded[:5], True
    return parts, False


def load_metrics(path: Path) -> Dict[str, dict]:
    """Index the label-free metrics list by string doc_id."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, dict] = {}
    for row in data:
        out[str(row["doc_id"])] = {
            "cosine_sim": row.get("cosine_sim"),
            "qa_precision": row.get("qa_precision"),
            "freq_distance": row.get("freq_distance"),
        }
    return out


def load_gold(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _to_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_items(
    sample_path: Path,
    metrics_path: Path,
    gold_path: Path,
) -> dict:
    metrics = load_metrics(metrics_path)
    gold = load_gold(gold_path)

    items: List[dict] = []
    warnings: List[str] = []

    with open(sample_path, encoding="utf-8", newline="") as f:
        # csv.DictReader correctly handles the embedded newlines inside the
        # quoted excerpt/summary fields (the file is 100 rows across ~194
        # physical lines).
        for idx, row in enumerate(csv.DictReader(f)):
            doc_id = str(row["doc_id"]).strip()
            answers, warn = parse_qa_text(row.get("qa_text", ""))
            if warn:
                warnings.append(doc_id)

            qa = [
                {"question": QA_QUESTIONS[i], "answer": answers[i]}
                for i in range(5)
            ]

            item = {
                "idx": idx,
                "doc_id": doc_id,
                "article": row.get("original_text_excerpt", ""),
                "summary": row.get("summary_text", ""),
                "qa": qa,
                # Never rendered in the labeling screen — audit/reveal only.
                "_hidden": {
                    "sm_f1_bucket": _to_int(row.get("sm_f1_bucket", "")),
                    "qa_p_bucket": _to_int(row.get("qa_p_bucket", "")),
                    "cosine_sim": metrics.get(doc_id, {}).get("cosine_sim"),
                    "qa_precision": metrics.get(doc_id, {}).get("qa_precision"),
                    "freq_distance": metrics.get(doc_id, {}).get("freq_distance"),
                    "gold_ner": {
                        lbl: gold.get(doc_id, {}).get(lbl, []) for lbl in QA_LABELS
                    },
                },
            }
            if warn:
                item["qa_parse_warning"] = True
            items.append(item)

    return {
        "version": PAYLOAD_VERSION,
        "label_scheme": LABEL_SCHEME,
        "items": items,
        "_warnings": warnings,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build blinded items.json for the human-validation web app")
    p.add_argument("--sample", default=str(REPO_ROOT / "notes/human_label_sample.csv"))
    p.add_argument("--metrics", default=str(REPO_ROOT / "notes/combined_scorer_demo/doc_metrics_labelfree.json"))
    p.add_argument("--gold", default=str(REPO_ROOT / "final_pipeline/datasets/ground_truth_ner.json"))
    p.add_argument("--output", default=str(REPO_ROOT / "webapp/data/items.json"))
    args = p.parse_args()

    payload = build_items(Path(args.sample), Path(args.metrics), Path(args.gold))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(payload["items"])
    warns = payload["_warnings"]
    print(f"[build_items] wrote {out_path}  ({n} items)")
    if warns:
        print(f"[build_items] WARNING: {len(warns)} rows did not parse to 5 QA answers: {warns}")
    else:
        print("[build_items] all rows parsed to exactly 5 QA answers")


if __name__ == "__main__":
    main()
