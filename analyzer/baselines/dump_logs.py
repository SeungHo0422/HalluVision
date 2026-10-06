"""
Render FactScore / QAGS runner output as a readable audit log (R2 #6).

The runners already persist everything the judge produced — atomic claims
with their per-claim verdicts, and QAGS questions with the answer each of
the summary and the source gave. That detail is buried in a few hundred KB
of JSON, so this module flattens it into markdown you can actually read,
plus a per-item CSV for spot-checking in a spreadsheet.

Nothing here calls an LLM: it is a pure view over an existing run, so it
is safe to re-run at zero cost.

Usage::

    python -m analyzer.baselines.dump_logs \
        --factscore notes/factscore_scores.json \
        --qags notes/qags_scores.json \
        --output-dir notes/baseline_logs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

VERDICT_MARK = {"supported": "OK ", "unsupported": "BAD", "inconclusive": "?  "}


def _load(path: str) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _claim_rows(rec: dict) -> List[dict]:
    """Per-claim rows, tolerating runs made before ``claim_details`` existed."""
    details = rec.get("claim_details")
    if details:
        return details
    return [{"claim": c, "verdict": v, "raw_verdict": None}
            for c, v in zip(rec.get("claims", []), rec.get("verdicts", []))]


def factscore_markdown(records: List[dict], only_flagged: bool = False) -> str:
    out: List[str] = ["# FactScore — per-claim audit log", ""]
    scored = [r for r in records if r.get("factscore") is not None]
    n_claims = sum(r.get("n_claims", 0) for r in scored)
    bad = sum(1 for r in scored for d in _claim_rows(r) if d["verdict"] == "unsupported")
    inc = sum(1 for r in scored for d in _claim_rows(r) if d["verdict"] == "inconclusive")
    out += [f"- documents scored: **{len(scored)}** / {len(records)}",
            f"- atomic claims: **{n_claims}**",
            f"- unsupported: **{bad}**, inconclusive: **{inc}**", ""]
    if only_flagged:
        out.append("_Showing only documents with at least one non-supported claim._\n")

    for rec in records:
        rows = _claim_rows(rec)
        flagged = [d for d in rows if d["verdict"] != "supported"]
        if only_flagged and not flagged:
            continue
        score = rec.get("factscore")
        head = f"## doc {rec.get('doc_id')} — FactScore "
        head += "n/a" if score is None else f"{score:.3f}"
        head += f" ({rec.get('supported', 0)}/{rec.get('n_claims', 0)} supported)"
        out += [head, ""]
        if rec.get("note"):
            out += [f"> {rec['note']}", ""]
        if rec.get("summary_text"):
            out += ["**Summary under test**", "", "> " +
                    rec["summary_text"].strip().replace("\n", "\n> "), ""]
        for i, d in enumerate(rows, 1):
            mark = VERDICT_MARK.get(d["verdict"], "?  ")
            out.append(f"{i:>3}. `{mark}` {d['claim']}")
        out.append("")
        if rec.get("raw_decomposition"):
            out += ["<details><summary>raw decomposition response</summary>", "",
                    "```", rec["raw_decomposition"].strip(), "```", "", "</details>", ""]
    return "\n".join(out)


def qags_markdown(records: List[dict], only_disagree: bool = False) -> str:
    out: List[str] = ["# QAGS — per-question audit log", ""]
    scored = [r for r in records if r.get("qags") is not None]
    pairs = [p for r in scored for p in r.get("pairs", [])]
    exact = sum(1 for p in pairs if p["agreement"] == 1.0)
    zero = sum(1 for p in pairs if p["agreement"] == 0.0)
    out += [f"- documents scored: **{len(scored)}** / {len(records)}",
            f"- questions: **{len(pairs)}**",
            f"- full agreement: **{exact}**, zero agreement: **{zero}**", ""]
    if only_disagree:
        out.append("_Showing only documents with at least one imperfect answer pair._\n")

    for rec in records:
        prs = rec.get("pairs", [])
        if only_disagree and all(p["agreement"] == 1.0 for p in prs):
            continue
        score = rec.get("qags")
        head = f"## doc {rec.get('doc_id')} — QAGS "
        head += "n/a" if score is None else f"{score:.3f}"
        head += f" ({len(prs)} questions)"
        out += [head, ""]
        if rec.get("note"):
            out += [f"> {rec['note']}", ""]
        for i, p in enumerate(prs, 1):
            out += [f"**Q{i}.** {p['question']}  _(agreement {p['agreement']:.3f}"
                    + (f", {p['match']}" if p.get("match") else "") + ")_",
                    "",
                    f"- summary → `{p['summary_answer']}`",
                    f"- source  → `{p['source_answer']}`",
                    ""]
    return "\n".join(out)


def factscore_csv(records: List[dict], path: Path) -> int:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "factscore", "claim_index", "verdict", "claim"])
        n = 0
        for rec in records:
            for i, d in enumerate(_claim_rows(rec), 1):
                w.writerow([rec.get("doc_id"), rec.get("factscore"), i,
                            d["verdict"], d["claim"]])
                n += 1
    return n


def qags_csv(records: List[dict], path: Path) -> int:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "qags", "q_index", "agreement", "match",
                    "question", "summary_answer", "source_answer"])
        n = 0
        for rec in records:
            for i, p in enumerate(rec.get("pairs", []), 1):
                w.writerow([rec.get("doc_id"), rec.get("qags"), i,
                            p.get("agreement"), p.get("match", ""),
                            p.get("question"), p.get("summary_answer"),
                            p.get("source_answer")])
                n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Render baseline runner output as readable logs")
    p.add_argument("--factscore", help="FactScore scores JSON")
    p.add_argument("--qags", help="QAGS scores JSON")
    p.add_argument("--output-dir", default="notes/baseline_logs")
    p.add_argument("--prefix", default="", help="Filename prefix (e.g. 'origprompt_')")
    p.add_argument("--only-flagged", action="store_true",
                   help="Markdown keeps only docs with a non-supported claim / imperfect answer")
    args = p.parse_args()

    if not args.factscore and not args.qags:
        p.error("give at least one of --factscore / --qags")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.factscore:
        recs = _load(args.factscore)
        md = out_dir / f"{args.prefix}factscore_log.md"
        cs = out_dir / f"{args.prefix}factscore_claims.csv"
        md.write_text(factscore_markdown(recs, args.only_flagged), encoding="utf-8")
        n = factscore_csv(recs, cs)
        print(f"[dump] factscore: {len(recs)} docs / {n} claims → {md}, {cs}")

    if args.qags:
        recs = _load(args.qags)
        md = out_dir / f"{args.prefix}qags_log.md"
        cs = out_dir / f"{args.prefix}qags_questions.csv"
        md.write_text(qags_markdown(recs, args.only_flagged), encoding="utf-8")
        n = qags_csv(recs, cs)
        print(f"[dump] qags: {len(recs)} docs / {n} questions → {md}, {cs}")


if __name__ == "__main__":
    main()
