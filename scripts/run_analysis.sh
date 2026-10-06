#!/usr/bin/env bash
# HalluVision analysis runner
# Skips scraping and LLM generation; evaluates and plots from the stored outputs.
#
# Inputs (relative to the workspace, see scripts/prepare_workspace.py):
#   - original_ner.json                : 원문 CyNER ground truth
#   - summary_ner.json                 : gpt-4o-mini summary NER 결과
#   - cyner_sum_results_250427.json    : gpt-3.5-turbo summary CyNER 결과
#   - gpt_results_2503.json            : gpt-4o-mini QA 결과
#   - gpt_results_250427.json          : gpt-3.5-turbo QA 결과 (후처리됨, qa_text 유효)
# (gpt_results_3.5.json 은 3.5 raw — qa_text 비어 있어 사용하지 않음)
#
# Outputs:
#   - figures/figure6.pdf                              (eval_summary.py; figure5 is produced by notebooks/eval_summary_prof.ipynb)
#   - false_positives_gpt4o.json, false_positives_gpt35.json
#   - failed_cases_gpt4omini.csv, failed_cases_gpt3.5.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Data layout: run from (or point HALLUVISION_WORKSPACE at) a workspace built by
#   python scripts/prepare_workspace.py --mendeley <download> --workspace <dir>
WORKSPACE="${HALLUVISION_WORKSPACE:-$PWD}"
cd "$WORKSPACE"
export PYTHONPATH="$REPO_DIR:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CONDA_ENV="${CONDA_ENV:-halluvision-revision}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SKIP_INSTALL="${SKIP_INSTALL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_QA="${RUN_QA:-1}"

ORIGINAL_NER="${ORIGINAL_NER:-original_ner.json}"
SUMMARY_NER_4O="${SUMMARY_NER_4O:-summary_ner.json}"
SUMMARY_NER_35="${SUMMARY_NER_35:-cyner_sum_results_250427.json}"
QA_RESULTS_4O="${QA_RESULTS_4O:-gpt_results_2503.json}"
QA_RESULTS_35="${QA_RESULTS_35:-gpt_results_250427.json}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --conda-env NAME    Conda env to activate (default: $CONDA_ENV)
  --python-version V  Python version when creating env (default: $PYTHON_VERSION)
  --install           pip install -r <repo>/requirements.txt
  --skip-summary      Skip eval_summary.py
  --skip-qa           Skip analysis_qa.py
  --original PATH     Ground truth NER (default: $ORIGINAL_NER)
  --summary-4o PATH   gpt-4o-mini summary NER (default: $SUMMARY_NER_4O)
  --summary-35 PATH   gpt-3.5-turbo summary NER (default: $SUMMARY_NER_35)
  --qa-4o PATH        gpt-4o-mini QA results (default: $QA_RESULTS_4O)
  --qa-35 PATH        gpt-3.5-turbo QA results (default: $QA_RESULTS_35)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --install) SKIP_INSTALL=0; shift ;;
        --skip-summary) RUN_SUMMARY=0; shift ;;
        --skip-qa) RUN_QA=0; shift ;;
        --original) ORIGINAL_NER="$2"; shift 2 ;;
        --summary-4o) SUMMARY_NER_4O="$2"; shift 2 ;;
        --summary-35) SUMMARY_NER_35="$2"; shift 2 ;;
        --qa-4o) QA_RESULTS_4O="$2"; shift 2 ;;
        --qa-35) QA_RESULTS_35="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# --- conda env ---
if command -v conda >/dev/null 2>&1 && [[ "${USE_CONDA:-1}" == "1" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/anaconda3)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        echo "[setup] creating conda env '$CONDA_ENV' (python=$PYTHON_VERSION)"
        conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
        SKIP_INSTALL=0  # new env always needs deps
    fi
    conda activate "$CONDA_ENV"
else
    echo "[setup] conda not used; running with $(command -v python3)"
fi

if [[ "$SKIP_INSTALL" == "0" ]]; then
    echo "[setup] installing requirements into '$CONDA_ENV'"
    pip install --quiet --upgrade pip
    pip install --quiet -r "$REPO_DIR/requirements.txt"
fi

mkdir -p figures

# --- input validation ---
needed=("$ORIGINAL_NER")
[[ "$RUN_SUMMARY" == "1" ]] && needed+=("$SUMMARY_NER_4O" "$SUMMARY_NER_35")
[[ "$RUN_QA" == "1" ]] && needed+=("$QA_RESULTS_4O" "$QA_RESULTS_35")
for f in "${needed[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: required input not found: $f" >&2
        exit 1
    fi
done

# --- step 1: summary evaluation (both models) ---
if [[ "$RUN_SUMMARY" == "1" ]]; then
    echo "=================================================="
    echo "STEP: Summary NER evaluation (gpt-4o-mini + gpt-3.5-turbo)"
    echo "=================================================="
    # eval_summary.py 는 두 모델을 모두 처리하지만 경로가 하드코딩되어 있어
    # 외부 변수와 다를 경우 임시로 치환 후 실행한다.
    python3 - <<PYEOF
import runpy, re, sys
src = open("$SCRIPT_DIR/eval_summary.py", encoding="utf-8").read()
src = re.sub(r'original_path\s*=\s*".*?"', f'original_path = "{r"""$ORIGINAL_NER"""}"', src, count=1)
src = re.sub(r'summary_path_4o\s*=\s*".*?"', f'summary_path_4o = "{r"""$SUMMARY_NER_4O"""}"', src, count=1)
src = re.sub(r'summary_path_35\s*=\s*".*?"', f'summary_path_35 = "{r"""$SUMMARY_NER_35"""}"', src, count=1)
exec(compile(src, "eval_summary.py", "exec"), {"__name__": "__main__", "__file__": "eval_summary.py"})
PYEOF
fi

# --- step 2: QA evaluation per model ---
run_qa_for_model() {
    local qa_path="$1"
    local model_tag="$2"   # "gpt4omini" or "gpt3.5"
    echo "--------------------------------------------------"
    echo "QA eval: $qa_path  (model=$model_tag)"
    echo "--------------------------------------------------"
    local fp_csv="failed_cases_${model_tag}.csv"
    python3 - <<PYEOF
import re
src = open("$SCRIPT_DIR/analysis_qa.py", encoding="utf-8").read()
src = re.sub(r'original_path\s*=\s*".*?"', f'original_path = "{r"""$ORIGINAL_NER"""}"', src, count=1)
src = re.sub(r'llm_qa_data_path\s*=\s*".*?"', f'llm_qa_data_path = "{r"""$qa_path"""}"', src, count=1)
src = re.sub(r'output_path\s*=\s*"failed_cases_[^"]*\.csv"', f'output_path = "{r"""$fp_csv"""}"', src, count=1)
exec(compile(src, "analysis_qa.py", "exec"), {"__name__": "__main__", "__file__": "analysis_qa.py"})
PYEOF
}

if [[ "$RUN_QA" == "1" ]]; then
    echo "=================================================="
    echo "STEP: QA evaluation"
    echo "=================================================="
    run_qa_for_model "$QA_RESULTS_4O" "gpt4omini"
    run_qa_for_model "$QA_RESULTS_35" "gpt3.5"
fi

echo "[done] analysis complete. outputs:"
echo "  - figures/figure6.pdf (publication CDF figures: notebooks/eval_summary_prof.ipynb, notebooks/analysis_qa.ipynb)"
echo "  - false_positives_gpt4o.json, false_positives_gpt35.json"
echo "  - failed_cases_gpt4omini.csv, failed_cases_gpt3.5.csv"
