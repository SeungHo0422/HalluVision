"""
Diagnostics on top of the R2#3 human-validation labels: does the low 3-class
Krippendorff's alpha come from the `partial` boundary, and which annotators
drive it?

Reuses the alpha/kappa implementations in ``agreement_multi`` so the numbers
stay comparable with ``notes/human_validation.md``.

Usage::

    python scripts/human_validation_diagnostics.py \
        --labels notes/human_labels_2026-08-01.csv \
        --output notes/human_validation_diagnostics.md
"""

from __future__ import annotations

import argparse
import os
import csv
import importlib.util
import itertools
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(os.environ.get("HALLUVISION_WORKSPACE", Path.cwd())).resolve()  # data workspace

# Load agreement_multi straight from its path: importing it as `analyzer.*`
# would pull in the package __init__, which needs numpy/torch.
_spec = importlib.util.spec_from_file_location(
    "_agreement_multi", Path(__file__).resolve().parents[1] / "analyzer/agreement_multi.py"
)
_am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_am)

LABEL_LEVELS = _am.LABEL_LEVELS
SEVERITY = _am.SEVERITY
_cohen_kappa = _am._cohen_kappa
_krippendorff_alpha = _am._krippendorff_alpha
_majority = _am._majority
detect_annotators = _am.detect_annotators

# The two ways to collapse the 3-class scale to a binary one.
COLLAPSE = {
    "3-class (as reported)": lambda l: l,
    "faithful vs {partial, hallucinated}": lambda l: "faithful" if l == "faithful" else "not_faithful",
    "{faithful, partial} vs hallucinated": lambda l: "hallucinated" if l == "hallucinated" else "not_hallucinated",
}


def load_per_item(labels_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        annotators = detect_annotators(reader.fieldnames)
        per_item = []
        for row in reader:
            labs = {}
            for a in annotators:
                v = (row.get(f"label_{a}", "") or "").strip().lower()
                if v in LABEL_LEVELS:
                    labs[a] = v
            per_item.append(labs)
    return per_item, annotators


def alpha_for(per_item, mapper, levels) -> tuple[float, int]:
    items = [[mapper(v) for v in labs.values()] for labs in per_item if len(labs) >= 2]
    return _krippendorff_alpha(items, levels), len(items)


def loo_deviation(per_item, annotators) -> dict[str, dict]:
    """For each annotator: how often they differ from the majority of the OTHER
    raters on the same item (leave-one-out, so the rater cannot vote for their
    own consensus), plus the direction of the disagreement in severity terms."""
    out = {}
    for a in annotators:
        n = tie = disagree = 0
        harsher = softer = 0
        for labs in per_item:
            if a not in labs:
                continue
            others = [v for k, v in labs.items() if k != a]
            if len(others) < 2:
                continue
            counts = Counter(others)
            top = max(counts.values())
            tied = [l for l, c in counts.items() if c == top]
            if len(tied) > 1:
                tie += 1  # others split -> genuinely ambiguous item, not this rater's fault
                continue
            n += 1
            ref = tied[0]
            if labs[a] != ref:
                disagree += 1
                if SEVERITY[labs[a]] > SEVERITY[ref]:
                    harsher += 1
                else:
                    softer += 1
        out[a] = {
            "n_decisive": n,
            "n_tied": tie,
            "disagree": disagree,
            "rate": disagree / n if n else float("nan"),
            "harsher": harsher,
            "softer": softer,
        }
    return out


def alpha_without(per_item, annotators) -> dict[str, float]:
    """Leave-one-annotator-out alpha: how much reliability recovers if a rater
    is dropped. A large jump = that rater is the main source of noise."""
    out = {}
    for a in annotators:
        items = [
            [v for k, v in labs.items() if k != a]
            for labs in per_item
            if len([k for k in labs if k != a]) >= 2
        ]
        out[a] = _krippendorff_alpha(items, LABEL_LEVELS)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=str(REPO_ROOT / "notes/human_labels_2026-08-01.csv"))
    p.add_argument("--output", default=str(REPO_ROOT / "notes/human_validation_diagnostics.md"))
    args = p.parse_args()

    labels_path = Path(args.labels)
    per_item, annotators = load_per_item(labels_path)
    multi = [labs for labs in per_item if len(labs) >= 2]

    L = ["# Human Validation Diagnostics (R2#3)", ""]
    L.append(f"Source labels: `{labels_path.relative_to(REPO_ROOT)}`")
    L.append(f"Annotators ({len(annotators)}): {', '.join(annotators)}  ·  multi-rated items: {len(multi)}")
    L.append("")

    # ---- 1. does collapsing the scale rescue alpha? ----
    L.append("## 1. Krippendorff's α under scale collapsing")
    L.append("")
    L.append("| scale | levels | n items | α |")
    L.append("|---|---|---:|---:|")
    for name, mapper in COLLAPSE.items():
        levels = sorted({mapper(l) for l in LABEL_LEVELS})
        a, n = alpha_for(per_item, mapper, levels)
        L.append(f"| {name} | {' / '.join(levels)} | {n} | {a:.3f} |")
    L.append("")

    # ---- 2. where the disagreements sit ----
    unanimous = sum(1 for labs in multi if len(set(labs.values())) == 1)
    adjacent = 0
    extreme = 0
    for labs in multi:
        sev = sorted(SEVERITY[v] for v in labs.values())
        if len(set(sev)) == 1:
            continue
        if sev[-1] - sev[0] == 1:
            adjacent += 1
        else:
            extreme += 1
    L.append("## 2. Disagreement structure (per item, severity distance)")
    L.append("")
    L.append(f"- unanimous: **{unanimous}** / {len(multi)}")
    L.append(f"- adjacent only (faithful↔partial or partial↔hallucinated): **{adjacent}**")
    L.append(f"- spans both ends (faithful & hallucinated on the same item): **{extreme}**")
    L.append("")

    # ---- 3. per-annotator leave-one-out behaviour ----
    dev = loo_deviation(per_item, annotators)
    awo = alpha_without(per_item, annotators)
    L.append("## 3. Per-annotator deviation from the leave-one-out consensus")
    L.append("")
    L.append("`rate` = share of items where the rater disagrees with the majority of the *other* "
             "raters (items where the others split are excluded and counted under `tied`). "
             "`α without` = 3-class α recomputed with that rater dropped (baseline "
             f"α = {_krippendorff_alpha([list(l.values()) for l in multi], LABEL_LEVELS):.3f}).")
    L.append("")
    L.append("| annotator | decisive n | tied | disagree | rate | harsher | softer | α without |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in sorted(annotators, key=lambda x: -dev[x]["rate"]):
        d = dev[a]
        L.append(
            f"| {a} | {d['n_decisive']} | {d['n_tied']} | {d['disagree']} | {d['rate']:.1%} | "
            f"{d['harsher']} | {d['softer']} | {awo[a]:.3f} |"
        )
    L.append("")

    # ---- 4. pairwise Cohen on the binarised scale ----
    L.append("## 4. Pairwise Cohen's κ, faithful vs {partial, hallucinated}")
    L.append("")
    mapper = COLLAPSE["faithful vs {partial, hallucinated}"]
    levels = ["faithful", "not_faithful"]
    rows = []
    for a, b in itertools.combinations(annotators, 2):
        xa, xb = [], []
        for labs in per_item:
            if a in labs and b in labs:
                xa.append(mapper(labs[a]))
                xb.append(mapper(labs[b]))
        if xa:
            rows.append((f"{a}-{b}", len(xa), _cohen_kappa(xa, xb, levels),
                         sum(1 for x, y in zip(xa, xb) if x == y) / len(xa)))
    L.append("| pair | n | κ | % agree |")
    L.append("|---|---:|---:|---:|")
    for name, n, k, pa in rows:
        L.append(f"| {name} | {n} | {k:.3f} | {pa:.1%} |")
    L.append("")
    valid = [k for _, _, k, _ in rows if k == k]
    if valid:
        L.append(f"Mean pairwise κ (binary): **{sum(valid) / len(valid):.3f}** over {len(valid)} overlapping pairs.")
        L.append("")

    # ---- 5. consensus under each scale ----
    L.append("## 5. Consensus label distribution")
    L.append("")
    cons = Counter(_majority(list(labs.values())) for labs in multi)
    L.append("| label | count |")
    L.append("|---|---:|")
    for l in LABEL_LEVELS:
        L.append(f"| {l} | {cons.get(l, 0)} |")
    L.append("")

    out_path = Path(args.output)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[diagnostics] wrote {out_path}")


if __name__ == "__main__":
    main()
