"""
Pull the annotators' submissions from AWS and fill them into a dated copy of
the human-validation sheet, ready for the agreement analysis.

Supports N annotators (anchor+disjoint design). Reads submissions from the
DynamoDB resume index (``--table``), the S3 data bucket (``--bucket``), or a
local dump of ``GET /admin/all`` (``--dump``, for hosts without AWS creds),
writes one ``label_<id>`` / ``reasoning_<id>`` column pair per annotator, and
joins the three auto metrics (``cosine_sim, qa_precision, freq_distance``).
The blank template ``notes/human_label_sample.csv`` is never overwritten.

Usage::

    python final_pipeline/analyzer/merge_human_labels.py \
        --table halluvision-human-labels --annotators a,b,c,d \
        --out notes/human_labels_2026-07-11.csv

Then::

    # N == 2  ->  the original 2-rater report
    python final_pipeline/analyzer/agreement.py \
        --labels notes/human_labels_2026-07-11.csv \
        --output notes/human_validation.md
    # N >= 3  ->  Fleiss kappa (anchor) + pairwise Cohen
    python final_pipeline/analyzer/agreement_multi.py \
        --labels notes/human_labels_2026-07-11.csv \
        --output notes/human_validation.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]

METRIC_COLS = ["cosine_sim", "qa_precision", "freq_distance"]
BASE_COLS = [
    "doc_id", "model", "sm_f1_bucket", "qa_p_bucket",
    "original_text_excerpt", "summary_text", "qa_text",
]


def fetch_from_dynamodb(table_name: str, annotators: list[str]) -> Dict[str, Dict[str, dict]]:
    import boto3

    table = boto3.resource("dynamodb").Table(table_name)
    out: Dict[str, Dict[str, dict]] = {a: {} for a in annotators}
    scan_kwargs: dict = {}
    while True:
        resp = table.scan(**scan_kwargs)
        for item in resp.get("Items", []):
            ann = item.get("annotator")
            if ann in out:
                out[ann][str(item["doc_id"])] = {
                    "label": item.get("label", ""),
                    "reasoning": item.get("reasoning", ""),
                }
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return out


def fetch_from_s3(bucket: str, annotators: list[str]) -> Dict[str, Dict[str, dict]]:
    import boto3

    s3 = boto3.client("s3")
    out: Dict[str, Dict[str, dict]] = {a: {} for a in annotators}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="submissions/"):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            rec = json.loads(body)
            ann = rec.get("annotator")
            if ann in out:
                out[ann][str(rec["doc_id"])] = {
                    "label": rec.get("label", ""),
                    "reasoning": rec.get("reasoning", ""),
                }
    return out


def fetch_from_dump(path: Path, annotators: list[str]) -> Dict[str, Dict[str, dict]]:
    """Read a saved ``GET /admin/all`` response — the escape hatch for machines
    without AWS credentials (``curl -H 'x-admin-key: ...' <ApiUrl>/admin/all``)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["submissions"] if isinstance(data, dict) else data
    out: Dict[str, Dict[str, dict]] = {a: {} for a in annotators}
    for rec in records:
        ann = rec.get("annotator")
        if ann in out:
            out[ann][str(rec["doc_id"])] = {
                "label": rec.get("label", ""),
                "reasoning": rec.get("reasoning", ""),
            }
    return out


def load_metrics(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["doc_id"]): r for r in data}


def merge(sample_path: Path, metrics_path: Path, subs: Dict[str, Dict[str, dict]], annotators: list[str]) -> list[dict]:
    metrics = load_metrics(metrics_path)
    rows_out = []
    with open(sample_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            doc_id = str(row["doc_id"]).strip()
            m = metrics.get(doc_id, {})
            out = {
                "doc_id": doc_id,
                "model": row.get("model", ""),
                "sm_f1_bucket": row.get("sm_f1_bucket", ""),
                "qa_p_bucket": row.get("qa_p_bucket", ""),
                "original_text_excerpt": row.get("original_text_excerpt", ""),
                "summary_text": row.get("summary_text", ""),
                "qa_text": row.get("qa_text", ""),
            }
            for a in annotators:
                rec = subs.get(a, {}).get(doc_id, {})
                out[f"label_{a}"] = rec.get("label", "")
                out[f"reasoning_{a}"] = rec.get("reasoning", "")
            for col in METRIC_COLS:
                out[col] = m.get(col, "")
            rows_out.append(out)
    return rows_out


def main() -> None:
    p = argparse.ArgumentParser(description="Merge S3/DynamoDB submissions into the human-validation CSV")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--table", help="DynamoDB table name (resume index)")
    src.add_argument("--bucket", help="S3 data bucket name")
    src.add_argument("--dump", help="Local JSON saved from GET /admin/all (no AWS credentials needed)")
    p.add_argument("--annotators", default="a,b,c,d", help="comma-separated ids to include as columns")
    p.add_argument("--sample", default=str(REPO_ROOT / "notes/human_label_sample.csv"))
    p.add_argument("--metrics", default=str(REPO_ROOT / "notes/combined_scorer_demo/doc_metrics_labelfree.json"))
    p.add_argument("--out", required=True, help="Output CSV (a dated copy; never overwrite the template)")
    args = p.parse_args()

    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    out_path = Path(args.out)
    if Path(args.sample).resolve() == out_path.resolve():
        raise SystemExit("Refusing to overwrite the blank template — choose a different --out path.")

    if args.table:
        subs = fetch_from_dynamodb(args.table, annotators)
    elif args.bucket:
        subs = fetch_from_s3(args.bucket, annotators)
    else:
        subs = fetch_from_dump(Path(args.dump), annotators)
    rows = merge(Path(args.sample), Path(args.metrics), subs, annotators)

    out_cols = BASE_COLS + [c for a in annotators for c in (f"label_{a}", f"reasoning_{a}")] + METRIC_COLS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

    counts = {a: sum(1 for r in rows if r.get(f"label_{a}")) for a in annotators}
    print(f"[merge] wrote {out_path}  ({len(rows)} rows)")
    for a in annotators:
        print(f"[merge]   {a}: {counts[a]} labeled")


if __name__ == "__main__":
    main()
