"""
Score the hand-filled NER validation sheets (Reviewer 2 #2a).

Consumes the two CSVs produced by ``analyzer.ner_manual_sheet`` after a human
has filled the ``verdict`` / ``missed_<Label>`` columns, and reports what the
reviewer asked for: how good the CyNER gold actually is, with confidence
intervals, per label and overall.

Precision comes from the entity sheet:

    strict   correct / (correct + partial + incorrect)
    lenient  (correct + partial) / (correct + partial + incorrect)

If the sheet was built with ``--max-entities-per-doc``, only a uniform sample of
each document's entities is judged and the ``weight`` column carries the inverse
inclusion probability (document entity count ÷ entities sampled). Every count
below is then weight-summed, which keeps precision an unbiased estimate of the
corpus figure instead of over-weighting entity-poor documents. Confidence
intervals use Kish's effective sample size, (Σw)² / Σw².

Both are reported because ``partial`` is dominated by truncated spans
(``UNC2`` ← ``UNC2165``), which hurt exact-match answer matching but do not
represent a fabricated entity — the two numbers bracket the true figure.

Recall comes from the doc sheet, over documents marked ``doc_done``:

    recall = est_true_positives / (est_true_positives + missed)

where ``est_true_positives`` is the weight-summed count of that document's
entities the annotator did not reject (``partial`` counted per strict/lenient as
above). Under a per-document cap this matters: the annotator lists *every*
missed entity but judges only a sample of the predicted ones, so an unweighted
numerator would understate recall badly.

Confidence intervals are Wilson score intervals at 95%; entity judgments within
a document are not independent, so they are reported as indicative rather than
exact. If ``verdict_b`` is filled, Cohen's κ between the two raters is added.

Usage::

    python -m final_pipeline.analyzer.ner_manual_score \
        --entities notes/csv/manual_ner_70/B_entities_precision.csv \
        --docs notes/csv/manual_ner_70/A_docs_recall.csv \
        --output notes/ner_manual_validation.md
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .agreement import _cohen_kappa
except ImportError:  # pragma: no cover - direct script execution
    from agreement import _cohen_kappa

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
VERDICTS = ["correct", "partial", "incorrect"]


def _wilson(successes: float, total: float, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _wilson_weighted(p: float, weights: List[float], z: float = 1.96) -> Tuple[float, float]:
    """Wilson interval on a weighted proportion via Kish effective sample size."""
    if not weights or p != p:
        return (float("nan"), float("nan"))
    sw = sum(weights)
    sw2 = sum(w * w for w in weights)
    if sw2 <= 0:
        return (float("nan"), float("nan"))
    n_eff = (sw * sw) / sw2
    return _wilson(p * n_eff, n_eff, z)


def _row_weight(row: dict) -> float:
    raw = (row.get("weight") or "").strip()
    if not raw:
        return 1.0
    try:
        w = float(raw)
    except ValueError:
        return 1.0
    return w if w > 0 else 1.0


def _fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.3f}"


def _fmt_ci(lo: float, hi: float) -> str:
    return "n/a" if lo != lo else f"[{lo:.3f}, {hi:.3f}]"


def _read_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _split_missed(cell: str) -> List[str]:
    return [e.strip() for e in (cell or "").replace(",", "|").split("|") if e.strip()]


def load_verdicts(
    rows: List[dict], column: str
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, int]], Dict[str, List[float]]]:
    """Per label: weighted verdict sums, raw verdict counts, and the weights used."""
    weighted: Dict[str, Dict[str, float]] = {l: defaultdict(float) for l in LABELS}
    raw: Dict[str, Dict[str, int]] = {l: defaultdict(int) for l in LABELS}
    weights: Dict[str, List[float]] = {l: [] for l in LABELS}
    for r in rows:
        verdict = (r.get(column) or "").strip().lower()
        label = (r.get("label") or "").strip()
        if verdict not in VERDICTS or label not in weighted:
            continue
        w = _row_weight(r)
        weighted[label][verdict] += w
        raw[label][verdict] += 1
        weights[label].append(w)
    return weighted, raw, weights


def find_bad_verdicts(rows: List[dict], column: str) -> List[Tuple[str, str]]:
    """(row_id, value) for non-empty verdicts outside the vocabulary.

    A typo silently drops the row from every figure, so surface it loudly.
    """
    bad: List[Tuple[str, str]] = []
    for r in rows:
        raw = (r.get(column) or "").strip()
        if raw and raw.lower() not in VERDICTS:
            bad.append((r.get("row_id", "?"), raw))
    return bad


def load_missed(doc_rows: List[dict]) -> Tuple[Dict[str, int], List[str], int]:
    """label -> missed count, over documents marked done. Also returns doc ids."""
    missed: Dict[str, int] = {l: 0 for l in LABELS}
    done_docs: List[str] = []
    for r in doc_rows:
        if (r.get("doc_done") or "").strip().lower() not in {"y", "yes", "1", "true", "o"}:
            continue
        done_docs.append((r.get("doc_id") or "").strip())
        for label in LABELS:
            missed[label] += len(_split_missed(r.get(f"missed_{label}", "")))
    return missed, done_docs, len(doc_rows)


def compute_report(
    entity_rows: List[dict],
    doc_rows: List[dict],
) -> dict:
    counts, raw_counts, weights = load_verdicts(entity_rows, "verdict")
    _, raw_counts_b, _ = load_verdicts(entity_rows, "verdict_b")
    missed, done_docs, n_docs_total = load_missed(doc_rows)
    done_set = set(done_docs)

    # entities belonging to fully-reviewed documents, for the recall denominator;
    # weight-summed so a per-document cap does not deflate the numerator
    done_correct_strict: Dict[str, float] = {l: 0.0 for l in LABELS}
    done_correct_lenient: Dict[str, float] = {l: 0.0 for l in LABELS}
    for r in entity_rows:
        if (r.get("doc_id") or "").strip() not in done_set:
            continue
        label = (r.get("label") or "").strip()
        verdict = (r.get("verdict") or "").strip().lower()
        if label not in LABELS:
            continue
        w = _row_weight(r)
        if verdict == "correct":
            done_correct_strict[label] += w
            done_correct_lenient[label] += w
        elif verdict == "partial":
            done_correct_lenient[label] += w

    per_label = []
    for label in LABELS:
        c = counts[label]
        rc = raw_counts[label]
        n = c["correct"] + c["partial"] + c["incorrect"]
        n_raw = rc["correct"] + rc["partial"] + rc["incorrect"]
        strict = c["correct"] / n if n else float("nan")
        lenient = (c["correct"] + c["partial"]) / n if n else float("nan")
        s_lo, s_hi = _wilson_weighted(strict, weights[label])
        l_lo, l_hi = _wilson_weighted(lenient, weights[label])

        m = missed[label]
        r_den_strict = done_correct_strict[label] + m
        r_den_lenient = done_correct_lenient[label] + m
        rec_strict = done_correct_strict[label] / r_den_strict if r_den_strict else float("nan")
        rec_lenient = done_correct_lenient[label] / r_den_lenient if r_den_lenient else float("nan")

        def _f1(p: float, r: float) -> float:
            if p != p or r != r or (p + r) == 0:
                return float("nan")
            return 2 * p * r / (p + r)

        per_label.append({
            "label": label,
            "n_judged": n_raw,
            "n_weighted": n,
            "correct": rc["correct"], "partial": rc["partial"], "incorrect": rc["incorrect"],
            "correct_w": c["correct"], "partial_w": c["partial"], "incorrect_w": c["incorrect"],
            "precision_strict": strict, "precision_strict_ci": (s_lo, s_hi),
            "precision_lenient": lenient, "precision_lenient_ci": (l_lo, l_hi),
            "missed": m,
            "est_tp_strict": done_correct_strict[label],
            "est_tp_lenient": done_correct_lenient[label],
            "recall_strict": rec_strict, "recall_lenient": rec_lenient,
            "f1_strict": _f1(strict, rec_strict), "f1_lenient": _f1(lenient, rec_lenient),
        })

    tot_c = sum(counts[l]["correct"] for l in LABELS)
    tot_p = sum(counts[l]["partial"] for l in LABELS)
    tot_i = sum(counts[l]["incorrect"] for l in LABELS)
    tot_n = tot_c + tot_p + tot_i
    tot_missed = sum(missed.values())
    tot_done_strict = sum(done_correct_strict.values())
    tot_done_lenient = sum(done_correct_lenient.values())

    all_weights = [w for l in LABELS for w in weights[l]]
    micro_p_strict = tot_c / tot_n if tot_n else float("nan")
    micro_p_lenient = (tot_c + tot_p) / tot_n if tot_n else float("nan")
    micro_r_strict = (tot_done_strict / (tot_done_strict + tot_missed)
                      if (tot_done_strict + tot_missed) else float("nan"))
    micro_r_lenient = (tot_done_lenient / (tot_done_lenient + tot_missed)
                       if (tot_done_lenient + tot_missed) else float("nan"))

    def _f1(p: float, r: float) -> float:
        if p != p or r != r or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    # error-type breakdown over rejected entities
    error_types: Dict[str, int] = defaultdict(int)
    for r in entity_rows:
        if (r.get("verdict") or "").strip().lower() in {"partial", "incorrect"}:
            error_types[(r.get("error_type") or "(blank)").strip() or "(blank)"] += 1

    # span_flag × verdict — does the pre-computed heuristic predict the human call?
    flag_x_verdict: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in entity_rows:
        v = (r.get("verdict") or "").strip().lower()
        if v in VERDICTS:
            flag_x_verdict[(r.get("span_flag") or "?").strip()][v] += 1

    kappa = None
    n_dual = 0
    if any(sum(raw_counts_b[l].values()) for l in LABELS):
        a_vals, b_vals = [], []
        for r in entity_rows:
            va = (r.get("verdict") or "").strip().lower()
            vb = (r.get("verdict_b") or "").strip().lower()
            if va in VERDICTS and vb in VERDICTS:
                a_vals.append(va)
                b_vals.append(vb)
        n_dual = len(a_vals)
        if n_dual:
            kappa = _cohen_kappa(a_vals, b_vals, VERDICTS)

    return {
        "n_entity_rows": len(entity_rows),
        "n_judged": len(all_weights),
        "weighted": any(abs(w - 1.0) > 1e-9 for w in all_weights),
        "n_docs_total": n_docs_total,
        "n_docs_done": len(done_docs),
        "per_label": per_label,
        "micro": {
            "correct": sum(raw_counts[l]["correct"] for l in LABELS),
            "partial": sum(raw_counts[l]["partial"] for l in LABELS),
            "incorrect": sum(raw_counts[l]["incorrect"] for l in LABELS),
            "n": len(all_weights),
            "correct_w": tot_c, "partial_w": tot_p, "incorrect_w": tot_i, "n_w": tot_n,
            "missed": tot_missed,
            "est_tp_strict": tot_done_strict,
            "est_tp_lenient": tot_done_lenient,
            "precision_strict": micro_p_strict,
            "precision_strict_ci": _wilson_weighted(micro_p_strict, all_weights),
            "precision_lenient": micro_p_lenient,
            "precision_lenient_ci": _wilson_weighted(micro_p_lenient, all_weights),
            "recall_strict": micro_r_strict,
            "recall_lenient": micro_r_lenient,
            "f1_strict": _f1(micro_p_strict, micro_r_strict),
            "f1_lenient": _f1(micro_p_lenient, micro_r_lenient),
        },
        "error_types": dict(error_types),
        "flag_x_verdict": {k: dict(v) for k, v in flag_x_verdict.items()},
        "kappa": kappa,
        "n_dual": n_dual,
    }


def render_markdown(report: dict, entities_path: str, docs_path: str) -> str:
    m = report["micro"]
    lines: List[str] = []
    lines.append("# NER Gold Manual Validation (R2 #2a)")
    lines.append("")
    lines.append(f"Sheets: `{entities_path}`, `{docs_path}`")
    lines.append("")
    lines.append(f"- Entity rows judged: **{report['n_judged']} / {report['n_entity_rows']}**")
    lines.append(f"- Documents reviewed for recall (`doc_done`): "
                 f"**{report['n_docs_done']} / {report['n_docs_total']}**")
    if report.get("weighted"):
        lines.append(f"- Within-document sampling in effect: counts are weight-summed "
                     f"(Σw = {m['n_w']:.0f} entities represented by {m['n']} judgments), "
                     f"CIs use Kish's effective sample size")
    if report["kappa"] is not None:
        lines.append(f"- Two-rater Cohen's κ (correct/partial/incorrect, n={report['n_dual']}): "
                     f"**{_fmt(report['kappa'])}**")
    lines.append("")

    lines.append("## Precision (entity sheet)")
    lines.append("")
    lines.append("| label | n | correct | partial | incorrect | P strict | 95% CI | P lenient | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|---|")
    for r in report["per_label"]:
        lines.append(
            f"| {r['label']} | {r['n_judged']} | {r['correct']} | {r['partial']} | {r['incorrect']} | "
            f"{_fmt(r['precision_strict'])} | {_fmt_ci(*r['precision_strict_ci'])} | "
            f"{_fmt(r['precision_lenient'])} | {_fmt_ci(*r['precision_lenient_ci'])} |"
        )
    lines.append(
        f"| **micro** | {m['n']} | {m['correct']} | {m['partial']} | {m['incorrect']} | "
        f"**{_fmt(m['precision_strict'])}** | {_fmt_ci(*m['precision_strict_ci'])} | "
        f"**{_fmt(m['precision_lenient'])}** | {_fmt_ci(*m['precision_lenient_ci'])} |"
    )
    lines.append("")
    lines.append("`strict` counts only `correct`; `lenient` also credits `partial` "
                 "(entity real, span truncated or label wrong). The true precision "
                 "for exact-match answer matching is the strict figure; lenient is "
                 "the ceiling if span normalization were applied.")
    lines.append("")

    lines.append("## Recall (document sheet)")
    lines.append("")
    lines.append("| label | est. true positives | missed | R strict | R lenient |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in report["per_label"]:
        lines.append(f"| {r['label']} | {r['est_tp_strict']:.1f} | {r['missed']} | "
                     f"{_fmt(r['recall_strict'])} | {_fmt(r['recall_lenient'])} |")
    lines.append(f"| **micro** | {m['est_tp_strict']:.1f} | {m['missed']} | "
                 f"**{_fmt(m['recall_strict'])}** | **{_fmt(m['recall_lenient'])}** |")
    lines.append("")
    lines.append("Recall is computed over documents marked `doc_done` only. "
                 "`est. true positives` is the weight-summed count of accepted "
                 "entities in those documents — under within-document sampling it "
                 "estimates how many of the document's predicted entities are real, "
                 "which is what the annotator-listed misses must be compared against.")
    lines.append("")

    lines.append("## F1")
    lines.append("")
    lines.append("| label | F1 strict | F1 lenient |")
    lines.append("|---|---:|---:|")
    for r in report["per_label"]:
        lines.append(f"| {r['label']} | {_fmt(r['f1_strict'])} | {_fmt(r['f1_lenient'])} |")
    lines.append(f"| **micro** | **{_fmt(m['f1_strict'])}** | **{_fmt(m['f1_lenient'])}** |")
    lines.append("")
    lines.append("Compare against the 76.23% F1 reported in §4.2 (held-out MITRE split): "
                 "this is the same model measured on the Threatpost corpus it is "
                 "actually applied to.")
    lines.append("")

    if report["error_types"]:
        lines.append("## Error types (rejected entities)")
        lines.append("")
        lines.append("| error_type | count |")
        lines.append("|---|---:|")
        for k, v in sorted(report["error_types"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v} |")
        lines.append("")

    if report["flag_x_verdict"]:
        lines.append("## `span_flag` vs human verdict")
        lines.append("")
        lines.append("| span_flag | correct | partial | incorrect |")
        lines.append("|---|---:|---:|---:|")
        for flag, d in sorted(report["flag_x_verdict"].items()):
            lines.append(f"| {flag} | {d.get('correct', 0)} | {d.get('partial', 0)} "
                         f"| {d.get('incorrect', 0)} |")
        lines.append("")
        lines.append("`substring_only` rows are the automatic truncated-span signal; "
                     "this table shows how often the human agreed with it.")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Score hand-filled NER validation sheets (R2 #2a)")
    p.add_argument("--entities", required=True, help="B_entities_precision.csv")
    p.add_argument("--docs", required=True, help="A_docs_recall.csv")
    p.add_argument("--output", required=True, help="markdown report path")
    p.add_argument("--json-output", default=None, help="optional raw json dump")
    args = p.parse_args()

    entity_rows = _read_csv(args.entities)
    doc_rows = _read_csv(args.docs)

    for column in ("verdict", "verdict_b"):
        bad = find_bad_verdicts(entity_rows, column)
        if bad:
            preview = ", ".join(f"row {rid}={val!r}" for rid, val in bad[:8])
            more = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
            print(f"[ner_manual_score] WARNING: {len(bad)} unrecognized `{column}` "
                  f"value(s), excluded from all figures — {preview}{more}")

    report = compute_report(entity_rows, doc_rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        render_markdown(report, args.entities, args.docs), encoding="utf-8"
    )
    if args.json_output:
        import json
        Path(args.json_output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"[ner_manual_score] judged {report['n_judged']}/{report['n_entity_rows']} entities, "
          f"{report['n_docs_done']}/{report['n_docs_total']} docs → {args.output}")


if __name__ == "__main__":
    main()
