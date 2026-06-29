#!/usr/bin/env bash
# HalluVision end-to-end pipeline runner
# Stages: data scraping -> LLM (summary + QA) -> analysis -> visualization
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Defaults (override via flags or env) ---
ORIGINAL_NER="${ORIGINAL_NER:-datasets/ground_truth_ner.json}"
THREATPOST_URLS="${THREATPOST_URLS:-datasets/threatpost_urls.txt}"
MALWARE_FILE="${MALWARE_FILE:-datasets/wiki_malware_list.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-pipeline_output}"
SOURCES="${SOURCES:-threatpost}"
TASKS="${TASKS:-summarization qa}"
WIKI_PARAGRAPHS="${WIKI_PARAGRAPHS:-3}"
PROMPT_VERSION="${PROMPT_VERSION:-v2}"
CONDA_ENV="${CONDA_ENV:-halluvision-revision}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SKIP_INSTALL="${SKIP_INSTALL:-1}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --original-ner PATH       Ground truth NER json (default: $ORIGINAL_NER)
  --threatpost-urls PATH    Threatpost URLs txt (default: $THREATPOST_URLS)
  --malware-file PATH       Malware list txt (default: $MALWARE_FILE)
  --output-dir PATH         Pipeline output dir (default: $OUTPUT_DIR)
  --sources "a b"           Data sources (default: $SOURCES)
  --tasks "a b"             LLM tasks (default: $TASKS)
  --wiki-paragraphs N       Wikipedia paragraph level (default: $WIKI_PARAGRAPHS)
  --prompt-version V        Summarization prompt version (default: $PROMPT_VERSION)
  --conda-env NAME          Conda env to activate (default: $CONDA_ENV)
  --python-version VER      Python version when creating env (default: $PYTHON_VERSION)
  --install                 pip install -r requirements.txt into the env
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --original-ner) ORIGINAL_NER="$2"; shift 2 ;;
        --threatpost-urls) THREATPOST_URLS="$2"; shift 2 ;;
        --malware-file) MALWARE_FILE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --sources) SOURCES="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --wiki-paragraphs) WIKI_PARAGRAPHS="$2"; shift 2 ;;
        --prompt-version) PROMPT_VERSION="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --install) SKIP_INSTALL=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# --- Load .env if present ---
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY is not set. Export it or put it in final_pipeline/.env" >&2
    exit 1
fi

# --- Activate conda env (create if missing) ---
CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/anaconda3)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "[setup] creating conda env '$CONDA_ENV' (python=$PYTHON_VERSION)"
    conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
    SKIP_INSTALL=0  # new env always needs deps
fi
conda activate "$CONDA_ENV"

# --- Install requirements ---
if [[ "$SKIP_INSTALL" != "1" ]]; then
    echo "[setup] installing requirements into '$CONDA_ENV'"
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
fi

# --- Validate inputs ---
for f in "$ORIGINAL_NER" "$THREATPOST_URLS" "$MALWARE_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: input file not found: $f" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

echo "[run] HalluVision pipeline starting"
echo "       original-ner    : $ORIGINAL_NER"
echo "       threatpost-urls : $THREATPOST_URLS"
echo "       malware-file    : $MALWARE_FILE"
echo "       output-dir      : $OUTPUT_DIR"
echo "       sources         : $SOURCES"
echo "       tasks           : $TASKS"

# shellcheck disable=SC2086
python3 main.py \
    --original-ner "$ORIGINAL_NER" \
    --threatpost-urls "$THREATPOST_URLS" \
    --malware-file "$MALWARE_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --sources $SOURCES \
    --tasks $TASKS \
    --wiki-paragraphs "$WIKI_PARAGRAPHS" \
    --prompt-version "$PROMPT_VERSION"

echo "[done] results in $OUTPUT_DIR (see PIPELINE_REPORT.md)"
