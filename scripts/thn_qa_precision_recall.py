"""THN Q&A answer matching: precision AND recall, using the exact formulation
reproduced from the paper's Table 7 (analysis_qa.py precision; recall = TP /
unique gold entities per label). Prints Threatpost alongside as a parity check.
"""
import json
from collections import defaultdict

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]


def score(orig, qa_list):
    id_to_qa = {i["id"]: i["qa_text"] for i in qa_list if "id" in i and "qa_text" in i}
    tp = defaultdict(int)
    answered = defaultdict(int)
    gold_uniq = defaultdict(int)
    n_doc = 0
    for id_, qa_str in id_to_qa.items():
        entry = orig.get(str(id_))
        if not entry:
            continue
        items = [x.strip() for x in qa_str.strip("{} ").split(",") if x.strip()]
        n_doc += 1
        try:
            for idx, label in enumerate(LABELS):
                pred = items[3] if idx == 2 else items[2] if idx == 3 else items[idx]
                gold_set = {str(g).lower() for g in entry.get(label, [])}
                gold_uniq[label] += len(gold_set)
                pred_set = set() if pred.lower() == "no answer" else {pred.lower()}
                if pred_set:
                    answered[label] += 1
                if pred_set & gold_set:
                    tp[label] += 1
        except Exception:
            continue
    out = {}
    for label in LABELS:
        out[label] = {
            "precision": tp[label] / answered[label] if answered[label] else 0.0,
            "recall": tp[label] / gold_uniq[label] if gold_uniq[label] else 0.0,
            "tp": tp[label], "answered": answered[label], "gold_uniq": gold_uniq[label],
        }
    out["_n_doc"] = n_doc
    return out


RUNS = [
    ("Threatpost", "gpt-4o-mini", "original_ner.json", "gpt_results_2503.json"),
    ("Threatpost", "gpt-3.5-turbo", "original_ner.json", "gpt_results_250427.json"),
    ("THN", "gpt-4o-mini", "pipeline_output/thn/thn_original_ner.json",
     "pipeline_output/thn/gpt_results_thn_gpt-4o-mini.json"),
    ("THN", "gpt-3.5-turbo", "pipeline_output/thn/thn_original_ner.json",
     "pipeline_output/thn/gpt_results_thn_gpt-3.5-turbo.json"),
]

results = {}
for corpus, model, orig_path, qa_path in RUNS:
    r = score(json.load(open(orig_path)), json.load(open(qa_path)))
    results[f"{corpus}|{model}"] = r
    print(f"\n=== {corpus} / {model}  (n={r['_n_doc']}) ===")
    print(f"{'label':15s} {'P':>8} {'R':>8}   {'TP':>6} {'answered':>9} {'goldUniq':>9}")
    for label in LABELS:
        d = r[label]
        print(f"{label:15s} {d['precision']:8.4f} {d['recall']:8.4f}   "
              f"{d['tp']:6d} {d['answered']:9d} {d['gold_uniq']:9d}")

json.dump(results, open("notes/thn_qa_precision_recall.json", "w"), indent=1)
