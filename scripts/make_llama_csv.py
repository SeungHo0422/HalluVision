#!/usr/bin/env python3
"""Export the Llama-3-8B evaluation (R2 #4) as CSV tables.

Reads the JSON artefacts produced by `run_llama_inference.py` + `run_llama_eval.sh`
and writes flat, spreadsheet-ready tables under `notes/csv/`. Files are UTF-8 with
BOM so Excel opens them with the right encoding.

    python make_llama_csv.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(os.environ.get("HALLUVISION_WORKSPACE", Path.cwd())).resolve()  # data workspace
OUT_DIR = ROOT / "notes" / "csv"
LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]
MODELS = ["llama-3-8b", "gpt-4o-mini", "gpt-3.5-turbo"]

# doc-stats / cosine files per model
QA_STATS = {
    "llama-3-8b": "notes/doc_stats_qa_llama_tp500.json",
    "gpt-4o-mini": "notes/doc_stats_qa_gpt4omini_tp500.json",
    "gpt-3.5-turbo": "notes/doc_stats_qa_gpt35_tp500.json",
}
COSINE = {
    "llama-3-8b": "notes/doc_cosine_llama_tp500.json",
    "llama-3-8b (preamble stripped)": "notes/doc_cosine_llama_stripped_tp500.json",
    "gpt-4o-mini": "notes/doc_cosine_gpt4omini_tp500.json",
    "gpt-3.5-turbo": "notes/doc_cosine_gpt35_tp500.json",
}
GENERATIONS = {
    "llama-3-8b": "pipeline_output/llama/llama_results_llama-3-8b-instruct.json",
    "gpt-4o-mini": "pipeline_output/llama/gpt_subset_gpt4omini.json",
    "gpt-3.5-turbo": "pipeline_output/llama/gpt_subset_gpt35.json",
}


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_csv(name: str, header: list, rows: list) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{path}  ({len(rows)} rows)")


def r(x, n=4):
    return "" if x is None else round(float(x), n)


# ── 1. headline: one row per model per axis ──────────────────────────────
def table_overview() -> None:
    qa4 = load("notes/stats_llama_vs_gpt4omini_qa.json")["bootstrap"]
    qa35 = load("notes/stats_llama_vs_gpt35_qa.json")["bootstrap"]
    cos4 = load("notes/stats_llama_vs_gpt4omini_cosine.json")["bootstrap"]
    cos35 = load("notes/stats_llama_vs_gpt35_cosine.json")["bootstrap"]
    coss = load("notes/stats_llama_stripped_vs_gpt4omini_cosine.json")["bootstrap"]
    summ = load("notes/llama_summary_eval.json")

    qa_boot = {"llama-3-8b": qa4["llama-3-8b"], "gpt-4o-mini": qa4["gpt-4o-mini"],
               "gpt-3.5-turbo": qa35["gpt-3.5-turbo"]}
    cos_boot = {"llama-3-8b": cos4["llama-3-8b"], "gpt-4o-mini": cos4["gpt-4o-mini"],
                "gpt-3.5-turbo": cos35["gpt-3.5-turbo"],
                "llama-3-8b (preamble stripped)": coss["llama-3-8b-stripped"]}
    summ_key = {"llama-3-8b": "llama-3-8b-instruct", "gpt-4o-mini": "gpt-4o-mini",
                "gpt-3.5-turbo": "gpt-3.5-turbo"}

    rows = []
    for m in MODELS:
        q = qa_boot[m]["_overall_"]
        c = cos_boot[m]["_overall_"]["cosine_sim"]
        s = summ[summ_key[m]]
        rows.append([
            m,
            r(q["precision"]["mean"]), r(q["recall"]["mean"]), r(q["f1"]["mean"]),
            q["precision"]["n"],
            r(c["mean"]), f'[{r(c["ci_lower"])}, {r(c["ci_upper"])}]',
            r(s["micro"]["precision"]), r(s["micro"]["recall"]), r(s["micro"]["f1"]),
            r(s["l1_distance_mean"]),
        ])
    c = cos_boot["llama-3-8b (preamble stripped)"]["_overall_"]["cosine_sim"]
    rows.append(["llama-3-8b (preamble stripped)", "", "", "", "",
                 r(c["mean"]), f'[{r(c["ci_lower"])}, {r(c["ci_upper"])}]',
                 "", "", "", ""])

    write_csv(
        "llama_overview.csv",
        ["model", "qa_precision", "qa_recall", "qa_f1", "qa_n_docs",
         "cosine_mean", "cosine_95ci",
         "summary_micro_precision", "summary_micro_recall", "summary_micro_f1",
         "summary_l1_distance"],
        rows,
    )


# ── 2. QA answer matching, per label ─────────────────────────────────────
def table_qa_by_label() -> None:
    qa4 = load("notes/stats_llama_vs_gpt4omini_qa.json")["bootstrap"]
    qa35 = load("notes/stats_llama_vs_gpt35_qa.json")["bootstrap"]
    boot = {"llama-3-8b": qa4["llama-3-8b"], "gpt-4o-mini": qa4["gpt-4o-mini"],
            "gpt-3.5-turbo": qa35["gpt-3.5-turbo"]}
    rows = []
    for m in MODELS:
        for lbl in ["_overall_"] + LABELS:
            for metric in ("precision", "recall", "f1"):
                d = boot[m][lbl][metric]
                rows.append([m, lbl, metric, r(d["mean"]),
                             r(d["ci_lower"]), r(d["ci_upper"]), d["n"]])
    write_csv("llama_qa_by_label.csv",
              ["model", "label", "metric", "mean", "ci_lower", "ci_upper", "n_docs"],
              rows)


# ── 3. paired bootstrap contrasts ────────────────────────────────────────
def table_paired() -> None:
    rows = []
    sources = [
        ("QA answer matching", "gpt-4o-mini - llama-3-8b",
         "notes/stats_llama_vs_gpt4omini_qa.json"),
        ("QA answer matching", "gpt-3.5-turbo - llama-3-8b",
         "notes/stats_llama_vs_gpt35_qa.json"),
        ("Cosine similarity", "gpt-4o-mini - llama-3-8b",
         "notes/stats_llama_vs_gpt4omini_cosine.json"),
        ("Cosine similarity", "gpt-3.5-turbo - llama-3-8b",
         "notes/stats_llama_vs_gpt35_cosine.json"),
        ("Cosine similarity", "gpt-4o-mini - llama-3-8b (preamble stripped)",
         "notes/stats_llama_stripped_vs_gpt4omini_cosine.json"),
    ]
    for axis, contrast, path in sources:
        paired = load(path)["paired"]
        for lbl, metrics in paired.items():
            for metric, d in metrics.items():
                rows.append([
                    axis, contrast, lbl, metric,
                    r(d["delta_mean"]), r(d["ci_lower"]), r(d["ci_upper"]),
                    r(d["p_value"], 6), d["n"],
                    "yes" if d["p_value"] < 0.05 else "no",
                ])
    write_csv("llama_paired_bootstrap.csv",
              ["axis", "contrast", "label", "metric", "delta_mean",
               "ci_lower", "ci_upper", "p_value", "n_docs", "significant_p<0.05"],
              rows)


# ── 4. summary axis, per label ───────────────────────────────────────────
def table_summary_axis() -> None:
    summ = load("notes/llama_summary_eval.json")
    rows = []
    for name, res in summ.items():
        for lbl in LABELS:
            m = res["per_label"][lbl]
            rows.append([name, lbl, r(m["precision"]), r(m["recall"]), r(m["f1"]),
                         m["tp"], m["fp"], m["fn"], "", res["n_docs"]])
        mi = res["micro"]
        rows.append([name, "MICRO", r(mi["precision"]), r(mi["recall"]), r(mi["f1"]),
                     mi["tp"], mi["fp"], mi["fn"],
                     r(res["l1_distance_mean"]), res["n_docs"]])
    write_csv("llama_summary_axis.csv",
              ["model", "label", "precision", "recall", "f1", "tp", "fp", "fn",
               "l1_distance_mean", "n_docs"],
              rows)


# ── 5. QA output format compliance ───────────────────────────────────────
def table_compliance() -> None:
    from analyzer.qa_format import compliance_report
    rows = []
    for m, path in GENERATIONS.items():
        rep = compliance_report(load(path))
        rows.append([m, rep["n"], rep["canonical"], rep["recovered"],
                     rep["unrecoverable"],
                     r(100 * rep["canonical"] / rep["n"], 2),
                     r(100 * rep["unrecoverable"] / rep["n"], 2)])
    write_csv("llama_qa_format_compliance.csv",
              ["model", "n_docs", "canonical", "recovered_by_normaliser",
               "unrecoverable", "canonical_pct", "unrecoverable_pct"],
              rows)


# ── 6. per-document metrics (long format) ────────────────────────────────
def table_per_doc() -> None:
    qa = {m: load(p) for m, p in QA_STATS.items()}
    cos = {m: load(p) for m, p in COSINE.items()}
    ids = [str(i) for i in load("pipeline_output/llama/tp_subset_500_ids.json")]
    gold = load("original_ner.json")

    rows = []
    for m in MODELS:
        for doc_id in ids:
            q = qa[m].get(doc_id)
            c = cos[m].get(doc_id, {}).get("_overall_", {}).get("cosine_sim")
            g = gold.get(doc_id, {})
            row = [doc_id, m,
                   sum(len(g.get(l, []) or []) for l in LABELS),
                   r(c) if c is not None else "",
                   r(q["_overall_"]["precision"]) if q else "",
                   r(q["_overall_"]["recall"]) if q else "",
                   r(q["_overall_"]["f1"]) if q else ""]
            for lbl in LABELS:
                row.append(r(q[lbl]["precision"]) if q else "")
            row.append("yes" if q else "no")
            rows.append(row)
    write_csv("llama_per_doc_metrics.csv",
              ["doc_id", "model", "gold_entity_count", "cosine_sim",
               "qa_precision", "qa_recall", "qa_f1"]
              + [f"qa_precision_{l}" for l in LABELS]
              + ["qa_scored"],
              rows)


# ── 7. raw generations ───────────────────────────────────────────────────
def table_generations() -> None:
    from analyzer.qa_format import normalize_qa_text
    subset = {str(x["id"]): x["original_text"]
              for x in load("pipeline_output/llama/tp_subset_500.json")}
    rows = []
    for m, path in GENERATIONS.items():
        for rec in load(path):
            doc_id = str(rec["id"])
            qa_text = rec.get("qa_text", "") or ""
            rows.append([
                doc_id, m,
                len(subset.get(doc_id, "").split()),
                rec.get("summary_text", "") or "",
                len((rec.get("summary_text") or "").split()),
                qa_text,
                normalize_qa_text(qa_text),
            ])
    write_csv("llama_generations.csv",
              ["doc_id", "model", "source_words", "summary_text", "summary_words",
               "qa_text_raw", "qa_text_normalised"],
              rows)


if __name__ == "__main__":
    table_overview()
    table_qa_by_label()
    table_paired()
    table_summary_axis()
    table_compliance()
    table_per_doc()
    table_generations()
