"""
Per-article table for the R2 #3 human validation: article, LLM output, the three
HalluVision metrics, and what the annotators said.

Emits two views of the same 100 rows:

  notes/human_validation_table.csv  full — article excerpt, summary, QA answers,
                                    every rater's label + reasoning, all metrics
  notes/human_validation_table.md   compact — metrics and labels only, sorted by
                                    cosine, plus the two tails written out

Metric sources are the paper implementations (see notes/metric_vs_human.py).

Usage::

    python scripts/make_validation_table.py
"""

from __future__ import annotations

import argparse
import os
import csv
import importlib.util
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(os.environ.get("HALLUVISION_WORKSPACE", Path.cwd())).resolve()  # data workspace


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_am = _load("_agreement_multi", Path(__file__).resolve().parents[1] / "analyzer/agreement_multi.py")
_mvh = _load("_metric_vs_human", Path(__file__).resolve().parent / "metric_vs_human.py")

LABEL_LEVELS = _am.LABEL_LEVELS
SEVERITY = _am.SEVERITY
KO = {"faithful": "faithful", "partial": "partial", "hallucinated": "hallucinated"}


def build(labels_path: Path, sample_path: Path) -> tuple[list[dict], list[str]]:
    metrics = _mvh.load_metrics()
    sample = {r["doc_id"]: r for r in csv.DictReader(open(sample_path, encoding="utf-8"))}

    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        annotators = _am.detect_annotators(reader.fieldnames)
        rows = []
        for r in reader:
            doc_id = r["doc_id"]
            labs, reasons = {}, {}
            for a in annotators:
                v = (r.get(f"label_{a}", "") or "").strip().lower()
                if v in LABEL_LEVELS:
                    labs[a] = v
                    reasons[a] = (r.get(f"reasoning_{a}", "") or "").strip()
            if not labs:
                continue
            cons = _am._majority(list(labs.values()))
            m = metrics.get(doc_id, {})
            src = sample.get(doc_id, {})
            rows.append({
                "doc_id": doc_id,
                "consensus": cons,
                "severity": SEVERITY[cons],
                "soft": mean(SEVERITY[v] for v in labs.values()) / 2.0,
                "unanimous": len(set(labs.values())) == 1,
                "raters": labs,
                "reasons": reasons,
                "cosine_sim": m.get("cosine_sim"),
                "qa_precision": m.get("qa_precision"),
                "freq_distance": m.get("freq_distance"),
                "article": src.get("original_text_excerpt", ""),
                "summary": src.get("summary_text", ""),
                "qa": src.get("qa_text", ""),
            })
    return rows, annotators


def write_csv(rows: list[dict], annotators: list[str], out: Path) -> None:
    cols = (["doc_id", "consensus", "soft_score", "unanimous", "n_raters",
             "cosine_sim", "qa_precision", "freq_distance", "rater_labels"]
            + [f"label_{a}" for a in annotators]
            + [f"reasoning_{a}" for a in annotators]
            + ["article_excerpt", "llm_summary", "llm_qa_answers"])
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            row = {
                "doc_id": r["doc_id"],
                "consensus": r["consensus"],
                "soft_score": f"{r['soft']:.3f}",
                "unanimous": "yes" if r["unanimous"] else "no",
                "n_raters": len(r["raters"]),
                "cosine_sim": "" if r["cosine_sim"] is None else f"{r['cosine_sim']:.4f}",
                "qa_precision": "" if r["qa_precision"] is None else f"{r['qa_precision']:.4f}",
                "freq_distance": "" if r["freq_distance"] is None else f"{r['freq_distance']:.4f}",
                "rater_labels": ", ".join(f"{a}:{l}" for a, l in sorted(r["raters"].items())),
                "article_excerpt": r["article"],
                "llm_summary": r["summary"],
                "llm_qa_answers": r["qa"],
            }
            for a in annotators:
                row[f"label_{a}"] = r["raters"].get(a, "")
                row[f"reasoning_{a}"] = r["reasons"].get(a, "")
            w.writerow(row)


def _fmt(v, nd=3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _labels_cell(r: dict) -> str:
    order = {"faithful": 0, "partial": 1, "hallucinated": 2}
    short = {"faithful": "F", "partial": "P", "hallucinated": "H"}
    return " ".join(short[l] for l in sorted(r["raters"].values(), key=lambda x: order[x]))


def write_md(rows: list[dict], out: Path, csv_name: str) -> None:
    by_cos = sorted(rows, key=lambda r: (r["cosine_sim"] is None, r["cosine_sim"]))

    L = ["# human validation 100건 — 기사별 메트릭 vs 사람 판정", ""]
    L.append(f"기사 원문·LLM 요약·QA 답변·rater별 사유까지 들어간 전체 표는 `{csv_name}` 참조. "
             "이 문서는 수치만 추린 압축본이다.")
    L.append("")
    L.append("`raters` 열은 3인의 판정을 F(faithful) / P(partial) / H(hallucinated)로 표기했다. "
             "`consensus`는 다수결(동점이면 더 심각한 쪽). 코사인 오름차순 정렬 — 위쪽이 "
             "프레임워크가 가장 의심한 기사다.")
    L.append("")

    # group summary
    L.append("## consensus별 메트릭 요약")
    L.append("")
    L.append("| consensus | n | cosine_sim | qa_precision | freq_distance |")
    L.append("|---|---:|---:|---:|---:|")
    for lvl in LABEL_LEVELS:
        grp = [r for r in rows if r["consensus"] == lvl]
        def avg(key):
            vals = [r[key] for r in grp if r[key] is not None]
            return f"{mean(vals):.3f}" if vals else "—"
        L.append(f"| {lvl} | {len(grp)} | {avg('cosine_sim')} | {avg('qa_precision')} | {avg('freq_distance')} |")
    L.append("")

    L.append("## 전체 100건 (코사인 오름차순)")
    L.append("")
    L.append("| # | doc_id | cosine_sim | qa_precision | freq_distance | raters | consensus |")
    L.append("|---:|---|---:|---:|---:|---|---|")
    for i, r in enumerate(by_cos, 1):
        L.append(f"| {i} | {r['doc_id']} | {_fmt(r['cosine_sim'])} | {_fmt(r['qa_precision'])} | "
                 f"{_fmt(r['freq_distance'])} | {_labels_cell(r)} | {r['consensus']}"
                 f"{' ★' if r['unanimous'] else ''} |")
    L.append("")
    L.append("★ = 3인 만장일치.")
    L.append("")

    # the two tails, written out
    for title, subset in (("코사인 최하위 10건 — 프레임워크가 flag한 기사", by_cos[:10]),
                          ("코사인 최상위 10건 — 프레임워크가 pass한 기사", list(reversed(by_cos[-10:])))):
        L.append(f"## {title}")
        L.append("")
        for r in subset:
            L.append(f"### doc {r['doc_id']} — consensus **{r['consensus']}** "
                     f"(raters {_labels_cell(r)}), cosine {_fmt(r['cosine_sim'], 4)}, "
                     f"qa_precision {_fmt(r['qa_precision'])}, freq_distance {_fmt(r['freq_distance'])}")
            L.append("")
            L.append(f"- **LLM 요약**: {r['summary'][:400]}{'…' if len(r['summary']) > 400 else ''}")
            L.append(f"- **QA 답변**: {r['qa'][:200]}{'…' if len(r['qa']) > 200 else ''}")
            reasons = [f"{a}: {t}" for a, t in sorted(r["reasons"].items()) if t]
            if reasons:
                L.append(f"- **rater 사유**: {' / '.join(reasons)[:400]}")
            L.append("")

    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=str(REPO_ROOT / "notes/human_labels_2026-08-01.csv"))
    p.add_argument("--sample", default=str(REPO_ROOT / "notes/human_label_sample.csv"))
    p.add_argument("--out-csv", default=str(REPO_ROOT / "notes/human_validation_table.csv"))
    p.add_argument("--out-md", default=str(REPO_ROOT / "notes/human_validation_table.md"))
    args = p.parse_args()

    rows, annotators = build(Path(args.labels), Path(args.sample))
    write_csv(rows, annotators, Path(args.out_csv))
    write_md(rows, Path(args.out_md), Path(args.out_csv).name)
    print(f"[table] {len(rows)} rows -> {args.out_csv}")
    print(f"[table] {len(rows)} rows -> {args.out_md}")


if __name__ == "__main__":
    main()
