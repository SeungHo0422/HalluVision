#!/usr/bin/env python3
"""R2 #6 — head-to-head: FactScore / QAGS / HalluVision against the same human labels.

`notes/baseline_run_r2_6.md` deferred the decisive comparison until §7 labels existed.
They do now, and all 100 labelled docs are inside the 200-doc baseline subset, so this
runs offline (no judge-LLM calls).

    python3 notes/baseline_vs_human.py --output notes/baseline_vs_human.md
"""

from __future__ import annotations

import argparse
import os
import importlib.util
import json
import random
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(os.environ.get("HALLUVISION_WORKSPACE", Path.cwd())).resolve()  # data workspace

_spec = importlib.util.spec_from_file_location("_mvh", Path(__file__).resolve().parent / "metric_vs_human.py")
mvh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mvh)

# sign = +1 when a HIGHER raw value means MORE hallucination. FactScore and QAGS are
# both faithfulness scores, so they flip like cosine does.
METRICS = [
    ("cosine_sim", -1.0, "HV — jina-v3 cosine"),
    ("qa_precision", -1.0, "HV — CyNER answer-matching precision"),
    ("freq_distance", +1.0, "HV — entity-label TV distance (Eq. 5)"),
    ("S", +1.0, "HV — combined z-score"),
    ("factscore", -1.0, "FactScore (gpt-4o-mini judge)"),
    ("qags", -1.0, "QAGS (gpt-4o-mini QG/QA, token-F1)"),
]


def attach_baselines(rows: list[dict]) -> None:
    fs = {d["doc_id"]: d for d in json.loads((REPO_ROOT / "notes/factscore_scores.json").read_text())}
    qg = {d["doc_id"]: d for d in json.loads((REPO_ROOT / "notes/qags_scores.json").read_text())}
    for r in rows:
        d = fs.get(r["doc_id"])
        r["factscore"] = d["factscore"] if d else None
        r["n_claims"] = d["n_claims"] if d else None
        r["unsupported"] = (d["n_claims"] - d["supported"]) if d else None
        r["qags"] = qg[r["doc_id"]]["qags"] if r["doc_id"] in qg else None


def table(blocks: list[dict]) -> list[str]:
    out = ["| metric | n | rho (soft) | 95% CI | p | AUC (any) | 95% CI | p | AUC (severe) | 95% CI | p |",
           "|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|"]
    for b in blocks:
        out.append(
            f"| `{b['key']}` | {b['n']} | {b['rho_soft']:.3f} | {mvh.fmt_ci(b['rho_ci'])} | {b['rho_p']:.3f} "
            f"| {b['auc_any']:.3f} | {mvh.fmt_ci(b['auc_any_ci'])} | {b['auc_any_p']:.3f} "
            f"| {b['auc_severe']:.3f} | {mvh.fmt_ci(b['auc_severe_ci'])} | {b['auc_severe_p']:.3f} |"
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=str(REPO_ROOT / "notes/human_labels_2026-08-01.csv"))
    p.add_argument("--output", default=str(REPO_ROOT / "notes/baseline_vs_human.md"))
    args = p.parse_args()

    rows, annotators, alpha = mvh.load_rows(Path(args.labels))
    attach_baselines(rows)
    mvh.METRICS = METRICS

    rng = random.Random(mvh.SEED)
    lines = [
        "# R2 #6 — FactScore / QAGS / HalluVision vs the same human labels",
        "",
        f"Labels: `{Path(args.labels).name}` · {len(rows)} docs · {len(annotators)} annotators · "
        f"Krippendorff alpha = {alpha:.3f}",
        "",
        "All scores oriented so that HIGHER = MORE hallucination; AUC > 0.5 and rho > 0 mean the "
        "metric agrees with the human consensus. `any` = consensus in {partial, hallucinated}; "
        "`severe` = consensus == hallucinated. CIs and p: 2,000 bootstrap / permutation draws, seed 42.",
        "",
        "## Full sample",
        "",
    ]
    lines += table(mvh.metric_block(rows, rng))

    uni = [r for r in rows if r["unanimous"]]
    lines += ["", f"## Unanimous subset (n={len(uni)})", ""]
    lines += table(mvh.metric_block(uni, random.Random(mvh.SEED)))

    # baseline discrimination: how much signal is even there to use
    fs_vals = [r["factscore"] for r in rows if r["factscore"] is not None]
    qg_vals = [r["qags"] for r in rows if r["qags"] is not None]
    lines += [
        "", "## Baseline score distributions on the labelled 100", "",
        "| baseline | n | mean | sd | distinct values | at ceiling (== max) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| FactScore | {len(fs_vals)} | {mean(fs_vals):.3f} | {pstdev(fs_vals):.3f} | "
        f"{len(set(fs_vals))} | {sum(1 for v in fs_vals if v >= 1.0)} |",
        f"| QAGS | {len(qg_vals)} | {mean(qg_vals):.3f} | {pstdev(qg_vals):.3f} | "
        f"{len(set(qg_vals))} | {sum(1 for v in qg_vals if v >= max(qg_vals))} |",
    ]

    # per-human-class means: does the baseline move at all across human severity?
    lines += ["", "### Baseline mean by human consensus class", "",
              "| baseline | faithful | partial | hallucinated |", "|---|---|---|---|"]
    for key, label in (("factscore", "FactScore"), ("qags", "QAGS")):
        cells = []
        for lvl in mvh.LABEL_LEVELS:
            vals = [r[key] for r in rows if r["consensus"] == lvl and r.get(key) is not None]
            cells.append(f"{len(vals)}, {mean(vals):.3f}±{pstdev(vals):.3f}" if len(vals) > 1 else "n/a")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # the qualitative pool: baseline says clean, humans + HV disagree
    clean_fs = [r for r in rows if r.get("factscore") == 1.0]
    missed = [r for r in clean_fs if r["consensus"] == "hallucinated"]
    missed.sort(key=lambda r: (r.get("cosine_sim") if r.get("cosine_sim") is not None else 9))
    lines += [
        "", "## Case pool — FactScore == 1.0 but humans said hallucinated", "",
        f"{len(clean_fs)} of {len(rows)} labelled docs score a perfect FactScore; "
        f"**{len(missed)}** of those are consensus-hallucinated by human annotators.",
        "",
        "| doc_id | n_claims | QAGS | HV cosine | HV qa_prec | HV freq_dist | raters |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    def f(v):
        return "n/a" if v is None else f"{v:.3f}"

    for r in missed[:20]:
        lines.append(
            f"| {r['doc_id']} | {r['n_claims']} | {f(r.get('qags'))} | {f(r.get('cosine_sim'))} | "
            f"{f(r.get('qa_precision'))} | {f(r.get('freq_distance'))} | {r['n_raters']} |"
        )

    # reverse direction: what the baselines DO flag, and whether humans agree
    flagged_fs = [r for r in rows if r.get("factscore") is not None and r["factscore"] < 1.0]
    lines += [
        "", "## Reverse direction — the 8 docs FactScore does flag", "",
        "| doc_id | FactScore | unsupported/claims | QAGS | HV cosine | human consensus |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in sorted(flagged_fs, key=lambda r: r["factscore"]):
        lines.append(
            f"| {r['doc_id']} | {r['factscore']:.3f} | {r['unsupported']}/{r['n_claims']} | "
            f"{f(r.get('qags'))} | {f(r.get('cosine_sim'))} | {r['consensus']} |"
        )

    # QAGS low tail vs humans — QAGS has real variance, so its tail is a fair test
    qags_sorted = [r for r in rows if r.get("qags") is not None]
    qags_sorted.sort(key=lambda r: r["qags"])
    tail = qags_sorted[:20]
    tail_dist = {lvl: sum(1 for r in tail if r["consensus"] == lvl) for lvl in mvh.LABEL_LEVELS}
    cos_sorted = sorted((r for r in rows if r.get("cosine_sim") is not None), key=lambda r: r["cosine_sim"])
    cos_tail = cos_sorted[:20]
    cos_dist = {lvl: sum(1 for r in cos_tail if r["consensus"] == lvl) for lvl in mvh.LABEL_LEVELS}
    lines += [
        "", "## Bottom-20 tail composition (what each metric flags first)", "",
        "| metric (worst 20 docs) | faithful | partial | hallucinated |",
        "|---|---:|---:|---:|",
        f"| QAGS | {tail_dist['faithful']} | {tail_dist['partial']} | {tail_dist['hallucinated']} |",
        f"| HV cosine | {cos_dist['faithful']} | {cos_dist['partial']} | {cos_dist['hallucinated']} |",
        "",
        "Base rate in the labelled 100: 25 faithful / 50 partial / 25 hallucinated, so a "
        "non-discriminative metric yields roughly 5 / 10 / 5.",
    ]

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
