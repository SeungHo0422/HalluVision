"""
The Hacker News held-out loader (Reviewer 2 #5).

Loads articles from the bundled CSV
``datasets/thehackernews/TheHackerNews_Dataset.csv`` and converts them
into the same shape used by ``DataScraper.scrape_general_data`` so the
existing LLM / NER / analyzer pipeline runs without modification.

Schema produced::

    {
        "malware_name": "thn_<idx>",
        "article_title": <Title>,
        "article_url":   <Link>,
        "combined_content": "Title: ...\\n\\n<Article>",
        "sources": {"thehackernews": {...}},
        "metadata": {"source": "thehackernews", "label": <Label>, ...}
    }

CLI::

    python -m final_pipeline.data_scraper.thn_csv_loader \
        --csv datasets/thehackernews/TheHackerNews_Dataset.csv \
        --n-sample 500 --seed 42 \
        --output pipeline_output/thn_scraped.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_thn(
    csv_path: str,
    n_sample: int = 500,
    seed: int = 42,
    stratify_by_label: bool = True,
    min_chars: int = 200,
) -> List[Dict[str, Any]]:
    """
    Load and sample THN articles.

    Args:
        csv_path: TheHackerNews_Dataset.csv path
        n_sample: total sample target
        seed: RNG seed
        stratify_by_label: sample proportionally across the ``Label`` column
        min_chars: drop articles shorter than this (CSV has some empty rows)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(csv_path)

    rng = random.Random(seed)

    # csv has very large rows; bump field size limit
    import sys as _sys
    csv.field_size_limit(min(2 ** 31 - 1, _sys.maxsize))

    by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            article = (row.get("Article") or "").strip()
            if len(article) < min_chars:
                continue
            label = (row.get("Label") or "unknown").strip() or "unknown"
            row["_idx"] = str(idx)
            by_label[label].append(row)

    total = sum(len(v) for v in by_label.values())
    if total == 0:
        return []

    if stratify_by_label and len(by_label) > 1:
        # proportional quota; round-robin remainder
        quotas: Dict[str, int] = {}
        remainder = n_sample
        for label, rows in by_label.items():
            quota = int(round(n_sample * len(rows) / total))
            quotas[label] = min(quota, len(rows))
            remainder -= quotas[label]
        # distribute any leftover
        leftover_labels = sorted(by_label.keys(), key=lambda l: -len(by_label[l]))
        i = 0
        while remainder > 0 and leftover_labels:
            label = leftover_labels[i % len(leftover_labels)]
            if quotas[label] < len(by_label[label]):
                quotas[label] += 1
                remainder -= 1
            i += 1
            if i > 10 * len(leftover_labels):
                break
        picks: List[Dict[str, str]] = []
        for label, rows in by_label.items():
            rng.shuffle(rows)
            picks.extend(rows[: quotas[label]])
    else:
        flat = [r for rows in by_label.values() for r in rows]
        rng.shuffle(flat)
        picks = flat[:n_sample]

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(picks):
        title = (row.get("Title") or "").strip()
        article = (row.get("Article") or "").strip()
        combined = f"Title: {title}\n\n{article}" if title else article
        out.append({
            "malware_name": f"thn_{row['_idx']}",
            "article_title": title or "No title",
            "article_url": (row.get("Link") or "").strip(),
            "combined_content": combined,
            "sources": {
                "thehackernews": {
                    "url": (row.get("Link") or "").strip(),
                    "title": title,
                    "content_length": len(article),
                    "label": (row.get("Label") or "").strip(),
                }
            },
            "metadata": {
                "total_paragraphs": 1,
                "sources_used": ["thehackernews"],
                "article_index": i + 1,
                "source": "thehackernews",
                "label": (row.get("Label") or "").strip(),
                "original_row_idx": row["_idx"],
            },
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Load TheHackerNews CSV as scraped_data")
    p.add_argument("--csv", required=True)
    p.add_argument("--n-sample", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-stratify", action="store_true")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    docs = load_thn(
        args.csv,
        n_sample=args.n_sample,
        seed=args.seed,
        stratify_by_label=not args.no_stratify,
        min_chars=args.min_chars,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    print(f"[thn_csv_loader] {len(docs)} docs -> {args.output}")


if __name__ == "__main__":
    main()
