#!/usr/bin/env bash
# Bridge probe step 07: retrieve_and_score. Run from anywhere; executes at the repo root.
# Flags pass through to the Python step (--force; step 05 also takes
# --tensor-parallel-size N, --gpus 0,1,2,3 and --dry-run).
# The LAST line printed is the path of this step's report.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
source scripts/env.sh
# Keep the NovelHopQA loader's match reports inside the probe folder.
export NOVELHOPQA_REPORT_DIR="${ROOT_DIR}/experiments/probe_bridge/logs/dataset_reports"
# The same virtual environment as the main study, when present.
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

mkdir -p experiments/probe_bridge/logs experiments/probe_bridge/reports
TS="$(date +%Y%m%d_%H%M%S)"
export PB_LOG_FILE="experiments/probe_bridge/logs/07_retrieve_and_score_${TS}.log"
REPORT="experiments/probe_bridge/reports/step_07.md"
# Keep the previous report aside so the path below always shows THIS run.
if [[ -f "${REPORT}" ]]; then
  mv "${REPORT}" "${REPORT%.md}.prev.md"
fi

set +e
"${PYTHON:-python3}" experiments/probe_bridge/steps/step07_retrieve_and_score.py "$@" 2>&1 | tee "${PB_LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

# A step that died before reporting still leaves a report with the log tail.
if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# Bridge probe - step 07: FAILED before writing its report"
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
