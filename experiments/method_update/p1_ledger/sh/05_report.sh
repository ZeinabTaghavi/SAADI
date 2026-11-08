#!/usr/bin/env bash
# p1_ledger step 05: report. Run from anywhere; executes at the repository root.
# Flags pass through to scripts/step05_report.py (--force reruns a finished step; see the
# script's --help for its own flags). Refuses to run while the previous step's
# done marker is missing; never asks for input; nohup-safe.
# The LAST line printed is the path of this step's report.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/experiments/method_update/p1_ledger/env.sh"

TS="$(date +%Y%m%d_%H%M%S)"
export P1_LOG_FILE="${P1_LOG_DIR}/05_report_${TS}.log"
REPORT="${P1_REPORT_DIR}/step_05.md"
# Keep the previous report aside so the path below always shows THIS run.
if [[ -f "${REPORT}" ]]; then
  mv "${REPORT}" "${REPORT%.md}.prev.md"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

set +e
"${PYTHON}" "${P1_EXP_DIR}/scripts/step05_report.py" "$@" 2>&1 | tee "${P1_LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

# A step that died before reporting still leaves a report with the log tail.
if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# p1_ledger - step 05: FAILED before writing its report"
    echo
    echo "Exit code: ${status}. Last 50 log lines:"
    echo
    echo '```'
    tail -n 50 "${P1_LOG_FILE}" || true
    echo '```'
  } > "${REPORT}"
fi

echo "${REPORT}"
exit "${status}"
