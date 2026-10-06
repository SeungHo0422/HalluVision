#!/usr/bin/env python3
"""Generic CyNER extraction with the paper's fine-tuned checkpoint.

Same recipe as [`run_thn_cyner.py`](run_thn_cyner.py) — that script is kept
verbatim for the THN run it produced; the extraction rules are imported from it
so the two can never drift — but with the corpus, text field and output path
given on the command line. Used for the Llama-3 summaries (Reviewer 2 #4), which
need summary-side NER produced by the *same* checkpoint as `original_ner.json`.

    CUDA_VISIBLE_DEVICES=0 python run_cyner_ner.py \
        --input pipeline_output/llama/llama_results_llama-3-8b-instruct.json \
        --text-key summary_text \
        --output pipeline_output/llama/llama_summary_ner.json
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import nltk  # noqa: E402

from run_thn_cyner import CKPT, extract, run_set  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="JSON list (or dict) of records carrying an id and the text field")
    ap.add_argument("--text-key", default="summary_text")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default=None, help="progress label")
    args = ap.parse_args()

    import cyner

    raw = json.loads(open(args.input, encoding="utf-8").read())
    records = list(raw.values()) if isinstance(raw, dict) else raw
    docs = [{"id": r["id"], args.text_key: r.get(args.text_key) or ""} for r in records]
    if args.limit:
        docs = docs[: args.limit]

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    model = cyner.CyNER(transformer_model=args.ckpt, use_heuristic=True, flair_model="ner")

    run_set(model, docs, args.text_key, args.output,
            args.label or os.path.basename(args.output))


if __name__ == "__main__":
    main()
