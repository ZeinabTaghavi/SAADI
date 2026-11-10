#!/usr/bin/env bash
# C3 step 04: tables, paired comparisons, REPORT.md and table_paper.tex (CPU, seconds).
# Reads only results/rebuttal/c3_channel_ablation/per_query/*.csv.gz (steps 02, 03) and the paper's numbers;
# runs on the laptop or the server, after both are there. Re-scores nothing.
#
#   bash experiments/rebuttal/c3_channel_ablation/sh/04_report.sh
#
# Never asks for input; nohup-safe. The LAST line printed is the path of the report.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/experiments/rebuttal/c3_channel_ablation/env.sh"

STEP=04
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${C3_LOG_DIR}/${STEP}_report_${TS}.log"
REPORT="${C3_REPORT_DIR}/step_${STEP}.md"
if [[ -f "${REPORT}" ]]; then
  mv "${REPORT}" "${REPORT%.md}.prev.md"
fi

set +e
"${PYTHON}" "${C3_EXP_DIR}/scripts/step04_report.py" "$@" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# C3 channel ablation - step ${STEP}: FAILED before writing its report"
    echo
    echo "Exit code: ${status}. Last 50 log lines (${LOG_FILE#"${ROOT_DIR}/"}):"
    echo
    echo '```'
    tail -n 50 "${LOG_FILE}" || true
    echo '```'
  } > "${REPORT}"
fi

echo "${REPORT}"
exit "${status}"
