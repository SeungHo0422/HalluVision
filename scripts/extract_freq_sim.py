"""
Per-document frequency-based entity *similarity* (paper Eq. 5, F = 1 - TV),
extracted into the doc-stats JSON shape `analyzer.stats` expects, so the
gpt-4o-mini vs gpt-3.5-turbo comparison on this metric goes through the same
paired-bootstrap pipeline as cosine and answer-matching (R2 #8).

TV distance here is identical to `combined_score_corpus.tv_distance` (which
already backs the R2 #1 combined-score work) — this script only adds the
model-vs-model framing that combined_score_corpus.py doesn't need.

Usage:
    python scripts/extract_freq_sim.py \
        --original original_ner.json --summary-ner summary_ner.json \
        --output notes/doc_freq_sim_gpt4omini.json
    python scripts/extract_freq_sim.py \
        --original original_ner.json --summary-ner cyner_sum_results_250427.json \
        --output notes/doc_freq_sim_gpt35.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]


def tv_distance(c_o: list[int], c_g: list[int]) -> float | None:
    so, sg = sum(c_o), sum(c_g)
    if so == 0 or sg == 0:
        return None
    p_o = [x / so for x in c_o]
    p_g = [x / sg for x in c_g]
    return 0.5 * sum(abs(a - b) for a, b in zip(p_o, p_g))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--summary-ner", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    orig = json.loads(Path(args.original).read_text())
    summ = json.loads(Path(args.summary_ner).read_text())

    out = {}
    for did, s in summ.items():
        o = orig.get(did)
        if o is None:
            continue
        tv = tv_distance(
            [len(set(o.get(l, []))) for l in LABELS],
            [len(set(s.get(l, []))) for l in LABELS],
        )
        if tv is not None:
            out[did] = {"_overall_": {"freq_sim": 1.0 - tv}}

    Path(args.output).write_text(json.dumps(out))
    print(f"{len(out)} docs -> {args.output}")


if __name__ == "__main__":
    main()
