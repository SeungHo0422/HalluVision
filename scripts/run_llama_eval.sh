#!/usr/bin/env bash
# Evaluate the Llama-3-8B run against GPT-3.5-turbo / GPT-4o-mini on the same
# stratified 500-doc Threatpost subset (Reviewer 2 #4 / Reviewer 3).
#
# Prerequisite: run_llama_inference.py (repo root) has produced
#   pipeline_output/llama/llama_results_llama-3-8b-instruct.json
#
#   scripts/run_llama_eval.sh            # all steps
#   SKIP_CYNER=1 scripts/run_llama_eval.sh   # skip summary-side NER (needs the GPU ckpt)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Data layout: run from (or point HALLUVISION_WORKSPACE at) a workspace built by
#   python scripts/prepare_workspace.py --mendeley <download> --workspace <dir>
WORKSPACE="${HALLUVISION_WORKSPACE:-$PWD}"
cd "$WORKSPACE"
export PYTHONPATH="$REPO_DIR:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

LLAMA=pipeline_output/llama/llama_results_llama-3-8b-instruct.json
IDS=pipeline_output/llama/tp_subset_500_ids.json
OUT=pipeline_output/llama

echo "=== 1. restrict the GPT runs to the same 500 ids ==="
python - <<'PY'
import json
ids = {str(i) for i in json.load(open('pipeline_output/llama/tp_subset_500_ids.json'))}
for tag, path in (('gpt4omini', 'gpt_results_2503.json'), ('gpt35', 'gpt_results_250427.json')):
    rows = [r for r in json.load(open(path)) if str(r['id']) in ids]
    json.dump(rows, open(f'pipeline_output/llama/gpt_subset_{tag}.json', 'w'), ensure_ascii=False)
    print(f'{tag}: {len(rows)} docs')
# summary-side NER of the GPT runs, restricted to the subset
for tag, path in (('gpt4omini', 'summary_ner.json'), ('gpt35', 'cyner_sum_results_250427.json')):
    d = json.load(open(path))
    sub = {k: v for k, v in d.items() if k in ids}
    json.dump(sub, open(f'pipeline_output/llama/summary_ner_subset_{tag}.json', 'w'), ensure_ascii=False)
    print(f'{tag} summary NER: {len(sub)} docs')
PY

echo "=== 2. QA answer-matching, per-doc stats (exact match) ==="
python - <<'PY'
import json
from analyzer.qa_format import compliance_report
for tag, path in (('llama-3-8b', 'pipeline_output/llama/llama_results_llama-3-8b-instruct.json'),
                  ('gpt-4o-mini', 'pipeline_output/llama/gpt_subset_gpt4omini.json'),
                  ('gpt-3.5-turbo', 'pipeline_output/llama/gpt_subset_gpt35.json')):
    print(f'  format compliance {tag}: {compliance_report(json.load(open(path)))}')
PY
for spec in "$LLAMA:llama" "$OUT/gpt_subset_gpt4omini.json:gpt4omini" "$OUT/gpt_subset_gpt35.json:gpt35"; do
    python -m analyzer.extract_doc_stats --task qa --original original_ner.json \
        --predictions "${spec%%:*}" --normalize-qa \
        --output "notes/doc_stats_qa_${spec##*:}_tp500.json"
done

echo "=== 3. paired bootstrap, QA ==="
python -m analyzer.stats --model-a llama-3-8b --model-b gpt-4o-mini \
    --metrics-a notes/doc_stats_qa_llama_tp500.json \
    --metrics-b notes/doc_stats_qa_gpt4omini_tp500.json \
    --output notes/stats_llama_vs_gpt4omini_qa.md
python -m analyzer.stats --model-a llama-3-8b --model-b gpt-3.5-turbo \
    --metrics-a notes/doc_stats_qa_llama_tp500.json \
    --metrics-b notes/doc_stats_qa_gpt35_tp500.json \
    --output notes/stats_llama_vs_gpt35_qa.md

echo "=== 4. cosine similarity (GPU) ==="
# The Llama runner records only id/summary_text/qa_text, so the source article
# has to be joined back in from the subset file before embedding.
python - <<'PY'
import json
subset = {str(r['id']): r['original_text']
          for r in json.load(open('pipeline_output/llama/tp_subset_500.json'))}
rows = json.load(open('pipeline_output/llama/llama_results_llama-3-8b-instruct.json'))
merged = [{'id': str(r['id']),
           'original_text': subset[str(r['id'])],
           'summary_text': r.get('summary_text', '')}
          for r in rows if str(r['id']) in subset]
json.dump(merged, open('pipeline_output/llama/cosine_input_llama.json', 'w'), ensure_ascii=False)
print(f'  cosine input: {len(merged)} pairs')
PY
for f in "$OUT/cosine_input_llama.json:llama" "$OUT/gpt_subset_gpt4omini.json:gpt4omini" "$OUT/gpt_subset_gpt35.json:gpt35"; do
    src="${f%%:*}"; tag="${f##*:}"
    [ -f "notes/doc_cosine_${tag}_tp500.json" ] && { echo "  skip $tag (exists)"; continue; }
    python -m analyzer.cosine_similarity --input "$src" \
        --output "notes/doc_cosine_${tag}_tp500.json" \
        --raw-output "notes/cosine_raw_${tag}_tp500.json" \
        --pooling cls --batch-size 1 --device cuda
done
python -m analyzer.stats --model-a llama-3-8b --model-b gpt-4o-mini \
    --metrics-a notes/doc_cosine_llama_tp500.json \
    --metrics-b notes/doc_cosine_gpt4omini_tp500.json \
    --metrics cosine_sim --labels _overall_ \
    --output notes/stats_llama_vs_gpt4omini_cosine.md
python -m analyzer.stats --model-a llama-3-8b --model-b gpt-3.5-turbo \
    --metrics-a notes/doc_cosine_llama_tp500.json \
    --metrics-b notes/doc_cosine_gpt35_tp500.json \
    --metrics cosine_sim --labels _overall_ \
    --output notes/stats_llama_vs_gpt35_cosine.md

if [ "${SKIP_CYNER:-0}" != "1" ]; then
    echo "=== 5. summary-side NER for Llama (paper's CyNER checkpoint) ==="
    if [ ! -f $OUT/llama_summary_ner.json ]; then
        CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/run_cyner_ner.py" --input "$LLAMA" \
            --text-key summary_text --output $OUT/llama_summary_ner.json
    fi
    echo "=== 6. summary axis, three models on the same 500 docs ==="
    python "$SCRIPT_DIR/run_thn_summary_eval.py" \
        --pair "llama-3-8b-instruct" original_ner.json $OUT/llama_summary_ner.json \
        --pair "gpt-4o-mini"   original_ner.json $OUT/summary_ner_subset_gpt4omini.json \
        --pair "gpt-3.5-turbo" original_ner.json $OUT/summary_ner_subset_gpt35.json \
        --output notes/llama_summary_eval.md --json-output notes/llama_summary_eval.json
fi

echo "=== done ==="
