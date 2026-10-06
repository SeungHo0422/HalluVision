#!/usr/bin/env bash
# HalluVision — summary-task cosine similarity bootstrap (Reviewer 2 #8)
#
# Regenerates the FULL per-document original-vs-summary cosine similarity for
# both models, then runs the paired bootstrap on it.
#
# Why this exists: the published pipeline only persisted the low-similarity
# tail (low_cosine_similarity*.json, cosine <= 0.8), so the R2 #8 paired test
# could only be run on the NER answer-matching P/R/F1 -- not on the cosine
# distribution that Figure 3 actually plots and that the reviewer points at.
#
# Inputs (relative to the workspace):
#   - gpt_results_250427.json  : gpt-3.5-turbo  (original_text + summary_text)
#   - gpt_results_2503.json    : gpt-4o-mini
#
# Outputs:
#   - notes/doc_cosine_gpt35.json / notes/doc_cosine_gpt4omini.json  (input to stats)
#   - notes/cosine_raw_gpt35.json / notes/cosine_raw_gpt4omini.json  (flat values)
#   - notes/stats_results_summary_cosine.md / .json                  (final table)
#
# Resumable: re-running continues from the per-document checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Data layout: run from (or point HALLUVISION_WORKSPACE at) a workspace built by
#   python scripts/prepare_workspace.py --mendeley <download> --workspace <dir>
WORKSPACE="${HALLUVISION_WORKSPACE:-$PWD}"
cd "$WORKSPACE"
export PYTHONPATH="$REPO_DIR:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
QA_RESULTS_35="${QA_RESULTS_35:-gpt_results_250427.json}"
QA_RESULTS_4O="${QA_RESULTS_4O:-gpt_results_2503.json}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-8}"
POOLING="${POOLING:-cls}"
N_BOOT="${N_BOOT:-10000}"
LIMIT="${LIMIT:-}"          # e.g. LIMIT=50 for a quick end-to-end rehearsal
CKPT_DIR="${CKPT_DIR:-.cosine_ckpt}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --device D        auto|cuda|mps|cpu   (default: $DEVICE)
  --batch-size N    embedding batch size (default: $BATCH_SIZE)
  --pooling P       cls|mean; cls matches published Figure 3 (default: $POOLING)
  --n-boot N        bootstrap resamples (default: $N_BOOT)
  --limit N         only process first N docs per model (rehearsal)
  --skip-embed      reuse existing notes/doc_cosine_*.json, run stats only
  -h, --help        Show this help

Environment overrides: PYTHON, QA_RESULTS_35, QA_RESULTS_4O, CKPT_DIR
EOF
}

SKIP_EMBED=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --pooling) POOLING="$2"; shift 2 ;;
        --n-boot) N_BOOT="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --skip-embed) SKIP_EMBED=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

mkdir -p notes "$CKPT_DIR"

embed() {
    local input="$1" tag="$2"
    echo "=================================================="
    echo "STEP: cosine embedding — $tag  ($input)"
    echo "=================================================="
    local extra=()
    [[ -n "$LIMIT" ]] && extra+=(--limit "$LIMIT")
    "$PYTHON" "$REPO_DIR/analyzer/cosine_similarity.py" \
        --input "$input" \
        --output "notes/doc_cosine_${tag}.json" \
        --raw-output "notes/cosine_raw_${tag}.json" \
        --checkpoint "${CKPT_DIR}/${tag}.json" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --pooling "$POOLING" \
        "${extra[@]}"
}

if [[ "$SKIP_EMBED" == "0" ]]; then
    for f in "$QA_RESULTS_35" "$QA_RESULTS_4O"; do
        if [[ ! -f "$f" ]]; then
            echo "Error: required input not found: $f" >&2
            exit 1
        fi
    done
    embed "$QA_RESULTS_35" "gpt35"
    embed "$QA_RESULTS_4O" "gpt4omini"
fi

echo "=================================================="
echo "STEP: paired bootstrap on cosine_sim"
echo "=================================================="
"$PYTHON" "$REPO_DIR/analyzer/stats.py" \
    --model-a gpt-3.5-turbo \
    --model-b gpt-4o-mini \
    --metrics-a notes/doc_cosine_gpt35.json \
    --metrics-b notes/doc_cosine_gpt4omini.json \
    --metrics cosine_sim \
    --labels _overall_ \
    --n-boot "$N_BOOT" \
    --output notes/stats_results_summary_cosine.md

echo "[done] outputs:"
echo "  - notes/doc_cosine_{gpt35,gpt4omini}.json"
echo "  - notes/cosine_raw_{gpt35,gpt4omini}.json"
echo "  - notes/stats_results_summary_cosine.md / .json"
