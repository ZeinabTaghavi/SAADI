#!/usr/bin/env bash
# C3 step 01: find the inputs, compute the BM25 channel scores (CPU), run check 1
# (all three = the paper's saved rankings, every query) and check 2 (text only vs Vanilla).
# Run from anywhere; executes at the repository root. CPU only, no GPU, no network.
#
#   bash experiments/rebuttal/c3_channel_ablation/sh/01_check.sh            # all 20 pairs
#   bash experiments/rebuttal/c3_channel_ablation/sh/01_check.sh --force    # recompute BM25 scores
#   C3_STREAMS_DIR=/path/to/streams bash .../sh/01_check.sh                 # other dense channel scores
#   C3_RETRIEVERS=bm25 bash .../sh/01_check.sh                              # BM25 only
#
# Never asks for input; nohup-safe. The LAST line printed is the path of the report.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/experiments/rebuttal/c3_channel_ablation/env.sh"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${C3_LOG_DIR}/01_check_${TS}.log"
REPORT="${C3_REPORT_DIR}/step_01.md"
# Keep the previous report aside so the path below always shows THIS run.
if [[ -f "${REPORT}" ]]; then
  mv "${REPORT}" "${REPORT%.md}.prev.md"
fi

set +e
"${PYTHON}" "${C3_EXP_DIR}/scripts/step01_check.py" "$@" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

# A step that died before reporting still leaves a report with the log tail.
if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# C3 channel ablation - step 01: FAILED before writing its report"
    echo
    echo "Exit code: ${status}. Command: bash experiments/rebuttal/c3_channel_ablation/sh/01_check.sh $*"
    echo
    echo "Last 50 log lines (${LOG_FILE#"${ROOT_DIR}/"}):"
    echo
    echo '```'
    tail -n 50 "${LOG_FILE}" || true
    echo '```'
  } > "${REPORT}"
fi

echo "${REPORT}"
exit "${status}"
