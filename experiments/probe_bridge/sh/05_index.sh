#!/usr/bin/env bash
# Bridge probe step 05: index. Run from anywhere; executes at the repo root.
# Flags pass through to the Python step (--force; step 05 also takes
# --tensor-parallel-size N, --gpus 0,1,2,3 and --dry-run).
# The LAST line printed is the path of this step's report.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
source scripts/env.sh
# Server defaults of the main study's run scripts. These come after env.sh,
# which otherwise points the Hugging Face cache at the repository's .cache.
export HF_HOME="${PROBE_HF_HOME:-/mnt/cache/taghavi}"
export SAADI_HF_CACHE_ROOT="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GLOBAL_VISIBLE_DEVICES="${GLOBAL_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
# Keep the NovelHopQA loader's match reports inside the probe folder.
export NOVELHOPQA_REPORT_DIR="${ROOT_DIR}/experiments/probe_bridge/logs/dataset_reports"
# The corpus has texts for a subset of the QA titles; the paper's saved run
# used the same subset, so missing titles are skipped, not fatal.
export NOVELHOPQA_SUBSET_MODE="${NOVELHOPQA_SUBSET_MODE:-1}"
# The same virtual environment as the main study, when present.
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p experiments/probe_bridge/logs experiments/probe_bridge/reports
TS="$(date +%Y%m%d_%H%M%S)"
export PB_LOG_FILE="experiments/probe_bridge/logs/05_index_${TS}.log"
REPORT="experiments/probe_bridge/reports/step_05.md"
# Keep the previous report aside so the path below always shows THIS run.
if [[ -f "${REPORT}" ]]; then
  mv "${REPORT}" "${REPORT%.md}.prev.md"
fi

set +e
"${PYTHON:-python3}" experiments/probe_bridge/steps/step05_index.py "$@" 2>&1 | tee "${PB_LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

# A step that died before reporting still leaves a report with the log tail.
if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# Bridge probe - step 05: FAILED before writing its report"
    echo
    echo "Exit code: ${status}. Last 50 log lines:"
    echo
    echo '```'
    tail -n 50 "${PB_LOG_FILE}" || true
    echo '```'
  } > "${REPORT}"
fi

echo "${REPORT}"
exit "${status}"
