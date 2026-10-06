"""
Contrast the FactScore / QAGS baselines with the §7 human labels (R2 #6).

FactScore calls almost every gpt-4o-mini summary fully supported (mean 0.99,
median 1.0). The annotators do not: most of the same documents come back as
``partial`` or ``hallucinated``. This module quantifies that gap and, more
usefully, pulls out the documents where the two disagree hardest so the
disagreement can be characterised rather than merely reported.

Produces:
  * majority vote per document (3 raters each, ties kept as ``tie``)
  * human-label x baseline-score contingency
  * ROC AUC of every metric against the human labels, under a strict target
    (hallucinated only) and a lenient one (hallucinated or partial)
  * the FactScore == 1.0 but human == hallucinated documents, with the atomic
    claims and the annotators' free-text reasoning side by side

No LLM calls: this is a join over artifacts that already exist.

Usage::

    python -m analyzer.baselines.human_vs_baseline \
        --labels notes/human_labels_2026-08-07.csv \
        --factscore notes/factscore_scores.json \
        --factscore-orig notes/factscore_scores_origprompt.json \
        --qags notes/qags_scores.json \
        --hv notes/hv_metrics_baseline_subset.json \
        --output notes/human_vs_baseline.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LABELS = ("faithful", "partial", "hallucinated")
# Metrics whose *low* values mean "more hallucinated". freq_distance is the
# one metric that points the other way.
LOWER_IS_WORSE = {"factscore", "factscore_orig", "qags", "cosine_sim", "qa_precision"}


# --------------------------------------------------------------- loading

def load_human(path: str) -> Dict[str, dict]:
    """doc_id -> {votes, majority, reasoning[]} from the merged label CSV."""
    out: Dict[str, dict] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        ann_cols = [c[len("label_"):] for c in reader.fieldnames or []
                    if c.startswith("label_")]
        for row in reader:
            doc_id = str(row["doc_id"]).strip()
            votes = [row.get(f"label_{a}", "").strip() for a in ann_cols]
            votes = [v for v in votes if v in LABELS]
            if not votes:
                continue
            reasons = [(a, row.get(f"label_{a}", "").strip(),
                        row.get(f"reasoning_{a}", "").strip())
                       for a in ann_cols
                       if row.get(f"label_{a}", "").strip() in LABELS
                       and row.get(f"reasoning_{a}", "").strip()]
            counts = Counter(votes)
            top, n_top = counts.most_common(1)[0]
            majority = top if n_top > len(votes) / 2 else "tie"
            out[doc_id] = {
                "votes": votes,
                "counts": dict(counts),
                "majority": majority,
                "n_hallucinated": counts.get("hallucinated", 0),
                "reasoning": reasons,
                "summary_text": row.get("summary_text", ""),
                "qa_text": row.get("qa_text", ""),
                "excerpt": row.get("original_text_excerpt", "") or "",
                # What the annotator was actually shown. The webapp truncates
                # the article; the judge runners do not, so the two sides do
                # not always see the same evidence.
                "excerpt_len": len(row.get("original_text_excerpt", "") or ""),
            }
    return out


def load_scores(path: Optional[str], key: str) -> Dict[str, dict]:
    if not path:
        return {}
    recs = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(r["doc_id"]): r for r in recs if r.get(key) is not None}


def load_hv(path: Optional[str]) -> Dict[str, dict]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}
    return {str(r["doc_id"]): r for r in data}


# --------------------------------------------------------------- stats

def fleiss_kappa(rows: List[Counter]) -> Optional[float]:
    """Fleiss' kappa for a constant number of ratings per item.

    The §7 design gives every document exactly 3 ratings but draws the raters
    from a pool of 10, which is precisely the case Fleiss covers (and which
    the anchor-based path in ``agreement_multi`` cannot report).
    """
    rows = [r for r in rows if sum(r.values()) > 1]
    if not rows:
        return None
    n = sum(rows[0].values())
    if any(sum(r.values()) != n for r in rows):
        return None  # unequal rating counts: use Krippendorff's alpha instead
    N = len(rows)
    p_j = {c: sum(r.get(c, 0) for r in rows) / (N * n) for c in LABELS}
    P_i = [(sum(v * v for v in r.values()) - n) / (n * (n - 1)) for r in rows]
    P_bar = sum(P_i) / N
    P_e = sum(v * v for v in p_j.values())
    if P_e >= 1.0:
        return None
    return (P_bar - P_e) / (1 - P_e)


def roc_auc(scores: List[float], positives: List[int]) -> Optional[float]:
    """AUC via the Mann-Whitney U identity, with ties counted at half weight."""
    pos = [s for s, y in zip(scores, positives) if y == 1]
    neg = [s for s, y in zip(scores, positives) if y == 0]
    if not pos or not neg:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def auc_for_metric(pairs: List[Tuple[float, int]], name: str) -> Optional[float]:
    """AUC oriented so that 'better at detecting hallucination' is > 0.5."""
    if not pairs:
        return None
    scores = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if name in LOWER_IS_WORSE:
        scores = [-s for s in scores]
    return roc_auc(scores, ys)


# --------------------------------------------------------------- report

def evidence_parity(human: Dict[str, dict], source_len: Dict[str, int]) -> List[str]:
    """Did the annotators and the judge read the same article?

    The §7 webapp shows a truncated excerpt while the runners feed the judge up
    to 8000 characters. Where the excerpt is short, an annotator can only call
    content 'not in the article' — so part of the human signal is truncation,
    not hallucination. This section measures how much.
    """
    pairs = [(d, h["excerpt_len"] / source_len[d]) for d, h in human.items()
             if source_len.get(d)]
    if not pairs:
        return []
    full = [d for d, r in pairs if r >= 0.999]
    trunc = [d for d, r in pairs if r < 0.999]
    ratios = sorted(r for _, r in pairs)
    median = ratios[len(ratios) // 2]

    def rate(ids: List[str]) -> str:
        n = sum(1 for d in ids if human[d]["majority"] == "hallucinated")
        return f"{n}/{len(ids)} ({n / len(ids):.0%})" if ids else "—"

    out = ["## 3b. Evidence parity — did both sides read the same article?", "",
           "The annotation webapp truncates the article; the FactScore / QAGS "
           "runners pass up to 8000 characters. Where those differ, an annotator "
           "can flag as fabricated something the judge correctly found further "
           "down the page.", "",
           f"- excerpt / full-article length, median: **{median:.0%}**",
           f"- documents where the annotator saw the whole article: "
           f"**{len(full)}** of {len(pairs)}", "",
           "| annotator saw | n | majority *hallucinated* |", "|---|---:|---:|",
           f"| whole article | {len(full)} | {rate(full)} |",
           f"| truncated excerpt | {len(trunc)} | {rate(trunc)} |", "",
           "A gap here means the human labels partly encode *how much of the "
           "article was shown*, and any AUC computed against them inherits that.",
           ""]
    return out


def build(human: Dict[str, dict], metrics: Dict[str, Dict[str, float]],
          factscore_recs: Dict[str, dict], top_n: int,
          source_len: Optional[Dict[str, int]] = None) -> str:
    doc_ids = [d for d in human if any(d in m for m in metrics.values())]
    out: List[str] = ["# FactScore / QAGS vs human labels (R2 #6)", ""]

    # ---- label distribution
    maj = Counter(human[d]["majority"] for d in human)
    votes = Counter(v for d in human for v in human[d]["votes"])
    kappa = fleiss_kappa([Counter(human[d]["votes"]) for d in human])
    out += ["## 1. Human labels", "",
            f"- documents: **{len(human)}**, ratings: **{sum(votes.values())}** "
            f"({sum(votes.values()) // max(len(human), 1)} per document)",
            f"- rating distribution: " +
            ", ".join(f"{k} **{votes.get(k, 0)}**" for k in LABELS),
            f"- majority vote: " +
            ", ".join(f"{k} **{maj.get(k, 0)}**" for k in list(LABELS) + ["tie"]),
            f"- Fleiss' κ (3 raters/item, 10-rater pool): "
            f"**{kappa:.3f}**" if kappa is not None else "- Fleiss' κ: n/a", ""]

    # ---- per-metric summary by human label
    out += ["## 2. Metric value by human majority label", ""]
    header = "| metric | " + " | ".join(
        f"{k} (n)" for k in list(LABELS) + ["tie"]) + " |"
    out += [header, "|---|" + "---:|" * 4]
    for name, table in metrics.items():
        cells = []
        for lab in list(LABELS) + ["tie"]:
            vals = [table[d] for d in doc_ids
                    if human[d]["majority"] == lab and d in table]
            cells.append(f"{sum(vals) / len(vals):.3f} ({len(vals)})" if vals else "— (0)")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out.append("")

    # ---- AUC
    out += ["## 3. ROC AUC against the human labels", "",
            "Oriented so >0.5 means the metric ranks human-flagged documents as "
            "more hallucinated. `strict` = majority *hallucinated*; "
            "`any` = majority *hallucinated or partial*.", "",
            "| metric | n | AUC (strict) | AUC (any) |", "|---|---:|---:|---:|"]
    for name, table in metrics.items():
        strict, lenient = [], []
        for d in doc_ids:
            if d not in table:
                continue
            m = human[d]["majority"]
            if m != "tie":
                strict.append((table[d], 1 if m == "hallucinated" else 0))
                lenient.append((table[d], 1 if m in ("hallucinated", "partial") else 0))
        a1 = auc_for_metric(strict, name)
        a2 = auc_for_metric(lenient, name)
        out.append(f"| {name} | {len(strict)} | "
                   + (f"{a1:.3f}" if a1 is not None else "—") + " | "
                   + (f"{a2:.3f}" if a2 is not None else "—") + " |")
    out.append("")

    # ---- evidence parity
    if source_len:
        out += evidence_parity(human, source_len)

    # ---- the disagreement that matters
    fs = metrics.get("factscore", {})
    perfect = [d for d in doc_ids if fs.get(d) == 1.0]
    out += ["## 4. Where FactScore and the annotators part ways", "",
            f"FactScore scores **{len(perfect)}** of {len(doc_ids)} human-labelled "
            f"documents at exactly 1.0 (every atomic claim supported). "
            "The annotators' verdicts on those same documents:", ""]
    pm = Counter(human[d]["majority"] for d in perfect)
    out += ["| human majority on FactScore==1.0 docs | n |", "|---|---:|"]
    for lab in list(LABELS) + ["tie"]:
        out.append(f"| {lab} | {pm.get(lab, 0)} |")
    out += ["", f"So {pm.get('hallucinated', 0) + pm.get('partial', 0)} documents "
            "that FactScore certifies as fully supported are judged partially or "
            "wholly hallucinated by the people who read them.", ""]

    # ---- case dump
    cases = sorted(
        [d for d in perfect if human[d]["n_hallucinated"] > 0],
        key=lambda d: -human[d]["n_hallucinated"])[:top_n]
    out += ["## 5. Case files — FactScore 1.000, humans disagree", "",
            "Every atomic claim below was judged SUPPORTED against the source "
            "article. The annotators still flagged the summary.", ""]
    for d in cases:
        h = human[d]
        out += [f"### doc {d} — human votes: "
                + ", ".join(f"{k}×{v}" for k, v in sorted(h["counts"].items()))
                + f" (majority: {h['majority']})", ""]
        for name, table in metrics.items():
            if d in table:
                out.append(f"- {name}: {table[d]:.3f}")
        out.append("")
        if h["summary_text"]:
            out += ["**Summary**", "",
                    "> " + h["summary_text"].strip().replace("\n", "\n> "), ""]
        rec = factscore_recs.get(d, {})
        claims = rec.get("claims") or [c["claim"] for c in rec.get("claim_details", [])]
        if claims:
            out += [f"**Atomic claims ({len(claims)}, all SUPPORTED)**", ""]
            out += [f"{i}. {c}" for i, c in enumerate(claims, 1)]
            out.append("")
        if h["reasoning"]:
            out += ["**Annotator reasoning**", ""]
            out += [f"- `{a}` ({lab}): {r}" for a, lab, r in h["reasoning"]]
            out.append("")
    return "\n".join(out)


def build_cases(human: Dict[str, dict], metrics: Dict[str, Dict[str, float]],
                factscore_recs: Dict[str, dict],
                source_len: Optional[Dict[str, int]] = None) -> str:
    """
    Every document where FactScore says 1.000, the annotators say otherwise,
    and at least one of them wrote down why.

    These are the cases that decide how §4.7 gets framed, so each one carries
    what both sides actually saw: the atomic claims the judge certified, and
    how much of the article the annotator was shown. The second number matters
    because the webapp truncates the article, so a complaint of the form
    "that is not in the source" may be describing the truncation.
    """
    fs = metrics.get("factscore", {})
    docs = [d for d, h in human.items()
            if fs.get(d) == 1.0
            and h["majority"] in ("partial", "hallucinated")
            and h["reasoning"]]
    docs.sort(key=lambda d: (-human[d]["n_hallucinated"], d))

    n_h = sum(1 for d in docs if human[d]["majority"] == "hallucinated")
    out: List[str] = [
        "# FactScore 1.000 but the annotators disagreed", "",
        "Selection: FactScore == 1.000 (every atomic claim SUPPORTED) **and** "
        "human majority is *partial* or *hallucinated* **and** at least one "
        "annotator left a written reason.", "",
        f"**{len(docs)} documents** ({n_h} majority-hallucinated, "
        f"{len(docs) - n_h} majority-partial).", "",
        "`excerpt coverage` is how much of the article the annotator was shown. "
        "Below 100%, a *not in the source* complaint may be a truncation "
        "artifact rather than a hallucination, so it is printed for every case.",
        ""]
    if source_len:
        clean = [d for d in docs
                 if source_len.get(d)
                 and len(human[d]["excerpt"]) >= source_len[d] * 0.999]
        out += [f"Of these, **{len(clean)}** annotators read the whole article "
                f"and **{len(docs) - len(clean)}** saw a truncated excerpt. "
                "The full-article cases are the ones to trust first: "
                + (", ".join(f"`doc {d}`" for d in clean) if clean else "none"),
                ""]
    out += ["---", ""]

    for d in docs:
        h = human[d]
        cov = None
        if source_len and source_len.get(d):
            cov = len(h["excerpt"]) / source_len[d]
        out += [f"## doc {d} — majority **{h['majority']}** "
                + "(" + ", ".join(f"{k}×{v}" for k, v in sorted(h["counts"].items())) + ")",
                ""]
        bits = []
        for name in ("factscore", "factscore_orig", "qags", "cosine_sim",
                     "qa_precision", "freq_distance"):
            t = metrics.get(name, {})
            if d in t:
                bits.append(f"{name} {t[d]:.3f}")
        if bits:
            out += ["- " + " · ".join(bits)]
        if cov is not None:
            flag = "" if cov >= 0.999 else "  ← truncated"
            out += [f"- excerpt coverage: **{cov:.0%}** "
                    f"({len(h['excerpt'])} of {source_len[d]} chars){flag}"]
        out.append("")

        if h["summary_text"]:
            out += ["**Summary under test**", "",
                    "> " + h["summary_text"].strip().replace("\n", "\n> "), ""]
        if h["qa_text"]:
            out += ["**QA answers shown to the annotator**", "",
                    "> " + h["qa_text"].strip().replace("\n", "\n> "), ""]

        rec = factscore_recs.get(d, {})
        claims = rec.get("claims") or [c["claim"] for c in rec.get("claim_details", [])]
        if claims:
            out += [f"**Atomic claims certified SUPPORTED ({len(claims)})**", ""]
            out += [f"{i}. {c}" for i, c in enumerate(claims, 1)]
            out.append("")
        out += ["**Annotator reasoning**", ""]
        out += [f"- `{a}` ({lab}): {r}" for a, lab, r in h["reasoning"]]
        out += ["", "---", ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Contrast baselines with §7 human labels")
    p.add_argument("--labels", required=True, help="merged human label CSV")
    p.add_argument("--factscore", required=True)
    p.add_argument("--factscore-orig", help="FactScore run with the verbatim paper prompt")
    p.add_argument("--qags")
    p.add_argument("--hv", help="hv_metrics_baseline_subset.json")
    p.add_argument("--output", required=True)
    p.add_argument("--top-n", type=int, default=8, help="case files to dump")
    p.add_argument("--subset", help="baseline_subset.json, for the evidence-parity check")
    p.add_argument("--cases-output",
                   help="Write the FactScore==1.0 / human-disagrees case files here")
    args = p.parse_args()

    human = load_human(args.labels)
    fs_recs = load_scores(args.factscore, "factscore")
    metrics: Dict[str, Dict[str, float]] = {
        "factscore": {d: r["factscore"] for d, r in fs_recs.items()},
    }
    if args.factscore_orig:
        metrics["factscore_orig"] = {
            d: r["factscore"] for d, r in load_scores(args.factscore_orig, "factscore").items()}
    if args.qags:
        metrics["qags"] = {d: r["qags"] for d, r in load_scores(args.qags, "qags").items()}
    if args.hv:
        hv = load_hv(args.hv)
        for col in ("cosine_sim", "qa_precision", "freq_distance"):
            table = {d: v[col] for d, v in hv.items()
                     if isinstance(v, dict) and v.get(col) is not None}
            if table:
                metrics[col] = table

    source_len = None
    if args.subset:
        docs = json.loads(Path(args.subset).read_text(encoding="utf-8"))["docs"]
        source_len = {str(d["id"]): len(d.get("original_text") or "") for d in docs}

    report = build(human, metrics, fs_recs, args.top_n, source_len)
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"[human_vs_baseline] {len(human)} labelled docs, "
          f"{len(metrics)} metrics → {args.output}")

    if args.cases_output:
        cases = build_cases(human, metrics, fs_recs, source_len)
        Path(args.cases_output).write_text(cases, encoding="utf-8")
        print(f"[human_vs_baseline] case files → {args.cases_output}")


if __name__ == "__main__":
    main()
