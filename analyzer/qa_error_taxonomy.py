"""
QA failure error taxonomy (Reviewer 2 minor #2).

Takes the existing ``failed_cases_*.csv`` exported by the QA analyzer
and classifies each (doc_id, label, prediction, gold_entities) row
into one of the following error types:

* ``format_error``      — prediction is empty / "no answer" / "n/a"
* ``ner_recall_miss``   — the prediction shows up in the original text
  but is absent from the NER gold list (NER missed an entity that the
  LLM nevertheless recovered)
* ``surface_variance``  — prediction shares ≥0.5 token-F1 with a gold
  entity (e.g. ``LockBit ransomware`` vs ``lockbit``). Soft hit that
  the span-level matching mode would have caught.
* ``ioc_token_form``    — prediction contains CVE-yyyy-#####, sha256/md5,
  IP / domain patterns but mismatches on punctuation only
* ``llm_hallucination`` — none of the above; prediction is genuinely
  unrelated to anything in the source text or the gold list

The classifier is heuristic — it gives us an actionable taxonomy
breakdown for the rebuttal table even before §7 human labelling.

Optionally, an ``--original-text`` JSON (``gpt_results_*.json``) lets
us check for ``ner_recall_miss`` via substring containment.

Usage::

    python -m final_pipeline.analyzer.qa_error_taxonomy \
        --failed-cases failed_cases_gpt4omini.csv \
        --gpt-results gpt_results_2503.json \
        --labels Indicator Vulnerability \
        --output notes/qa_error_taxonomy.md
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

try:
    from analyzer.span_match import token_f1, tokenize
except ImportError:  # pragma: no cover
    from final_pipeline.analyzer.span_match import token_f1, tokenize  # type: ignore


CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
HASH_RE = re.compile(r"\b[A-Fa-f0-9]{32,64}\b")
IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
DOMAIN_RE = re.compile(r"\b[A-Za-z0-9.-]+\.(?:com|net|org|io|biz|info|ru|cn|us|ly|info|tk|click|xyz)\b", re.IGNORECASE)


def _empty_or_no_answer(pred: str) -> bool:
    p = (pred or "").strip().lower()
    return not p or p in {"no answer", "n/a", "none", "null"}


def _has_ioc_pattern(s: str) -> bool:
    if not s:
        return False
    return bool(CVE_RE.search(s) or HASH_RE.search(s) or IP_RE.search(s) or DOMAIN_RE.search(s))


def _ioc_punct_mismatch(pred: str, golds: List[str]) -> bool:
    pred_norm = re.sub(r"[.\-_\s]", "", pred or "").lower()
    for g in golds:
        g_norm = re.sub(r"[.\-_\s]", "", g).lower()
        if pred_norm and pred_norm == g_norm and pred != g:
            return True
    return False


def _classify(pred: str, golds: List[str], original_text: Optional[str]) -> str:
    if _empty_or_no_answer(pred):
        return "format_error"
    if _ioc_punct_mismatch(pred, golds):
        return "ioc_token_form"
    if _has_ioc_pattern(pred):
        if any(_has_ioc_pattern(g) for g in golds):
            f1 = max((token_f1(pred, g) for g in golds), default=0.0)
            if f1 >= 0.3:
                return "ioc_token_form"
    best_f1 = max((token_f1(pred, g) for g in golds), default=0.0)
    if best_f1 >= 0.5:
        return "surface_variance"
    if original_text and pred.strip().lower() in original_text.lower():
        return "ner_recall_miss"
    return "llm_hallucination"


def _parse_gold(cell: str) -> List[str]:
    cell = (cell or "").strip()
    if not cell:
        return []
    try:
        v = ast.literal_eval(cell)
        if isinstance(v, list):
            return [str(x) for x in v]
    except (ValueError, SyntaxError):
        pass
    return [s.strip() for s in cell.strip("[]").split(",") if s.strip()]


def classify_rows(rows: List[dict], text_lookup: Dict[str, str],
                  labels: Optional[List[str]] = None) -> List[dict]:
    out = []
    for r in rows:
        if labels and r.get("label") not in labels:
            continue
        pred = r.get("prediction", "")
        golds = _parse_gold(r.get("gold_entities", ""))
        text = text_lookup.get(str(r.get("id", "")))
        out.append({
            "doc_id": str(r.get("id", "")),
            "label": r.get("label", ""),
            "prediction": pred,
            "gold_entities": golds,
            "error_type": _classify(pred, golds, text),
        })
    return out


def render_markdown(classified: List[dict], labels: List[str]) -> str:
    by_label_type: Dict[str, Counter] = {}
    for c in classified:
        by_label_type.setdefault(c["label"], Counter())[c["error_type"]] += 1
    overall = Counter(c["error_type"] for c in classified)

    error_types = [
        "format_error", "surface_variance", "ner_recall_miss",
        "ioc_token_form", "llm_hallucination",
    ]
    lines = [
        "# QA Failure Error Taxonomy (R2 minor)",
        "",
        f"Total classified failure rows: {len(classified)}",
        "",
        "## Overall distribution",
        "",
        "| error_type | count | % |",
        "|---|---:|---:|",
    ]
    total = sum(overall.values()) or 1
    for et in error_types:
        c = overall.get(et, 0)
        lines.append(f"| {et} | {c} | {c / total * 100:.1f}% |")

    lines += ["", "## Per-label breakdown", "",
              "| label | " + " | ".join(error_types) + " | total |",
              "|---" + "|---:" * (len(error_types) + 1) + "|"]
    for lbl in (labels or sorted(by_label_type.keys())):
        counts = by_label_type.get(lbl, Counter())
        row_total = sum(counts.values())
        cells = [str(counts.get(et, 0)) for et in error_types]
        lines.append(f"| {lbl} | " + " | ".join(cells) + f" | {row_total} |")

    lines += [
        "",
        "## Definitions",
        "",
        "- **format_error**: prediction empty / `no answer` / `n/a`.",
        "- **surface_variance**: prediction shares ≥0.5 token-F1 with a gold entity (e.g. *LockBit ransomware* vs *lockbit*). Span-level matching (§3) reclaims these.",
        "- **ner_recall_miss**: prediction appears in the source text but not in the NER gold list — NER side, not LLM side.",
        "- **ioc_token_form**: prediction matches a gold IoC up to punctuation (CVE, hash, IP, domain).",
        "- **llm_hallucination**: prediction neither matches gold tokens nor appears in source text — genuine LLM error.",
        "",
    ]
    return "\n".join(lines)


def _load_text_lookup(path: Optional[str]) -> Dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    if isinstance(data, list):
        for d in data:
            doc_id = str(d.get("id"))
            text = d.get("original_text") or d.get("text") or ""
            if doc_id and text:
                out[doc_id] = text
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Classify QA failure rows into error taxonomy")
    p.add_argument("--failed-cases", required=True)
    p.add_argument("--gpt-results", default=None,
                   help="Optional gpt_results JSON for ner_recall_miss check")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Restrict to specific labels (e.g. Indicator Vulnerability)")
    p.add_argument("--output", required=True)
    p.add_argument("--dump-rows", default=None,
                   help="Optional path to dump classified rows as JSON")
    args = p.parse_args()

    with open(args.failed_cases, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    text_lookup = _load_text_lookup(args.gpt_results)
    classified = classify_rows(rows, text_lookup, labels=args.labels)

    md = render_markdown(classified, args.labels or [])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    if args.dump_rows:
        Path(args.dump_rows).write_text(
            json.dumps(classified, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[qa_error_taxonomy] classified {len(classified)} rows → {args.output}")


if __name__ == "__main__":
    main()
