#!/usr/bin/env bash
# C4 step 05: paired bootstrap (10,000 resamples over queries, seed 42), cross-pair summary,
# results/rebuttal/c4_subdoc_control/{table.csv,paired.csv,REPORT.md,table_paper.tex}. CPU, about a minute.
# Needs steps 01-03 OK; step 04 is included when it has run (OK).
# Main query set: C4_MAIN_SET=R (default) or S.
# The LAST line printed is the report path; the exit code is non-zero when the step failed.
set -uo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${C4_LOG_DIR}/05_report_${TS}.log"
REPORT="${C4_REPORT_DIR}/step_05.md"
[[ -f "${REPORT}" ]] && mv "${REPORT}" "${REPORT%.md}.prev.md"
rm -f "${C4_REPORT_DIR}/step_05.json"

"${PYTHON}" "${C4_EXP_DIR}/scripts/step05_report.py" "$@" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# C4 sub-document control - step 05: FAILED before writing its report"
    echo
    echo "Command: bash experiments/rebuttal/c4_subdoc_control/sh/05_report.sh $*"
    echo
    echo "Exit code: ${status}. Last 60 log lines (${LOG#"${C4_ROOT_DIR}/"}):"
    echo
    echo '```'
    tail -n 60 "${LOG}" || true
    echo '```'
  } > "${REPORT}"
  [[ "${status}" -eq 0 ]] && status=1
fi

echo "${REPORT}"
exit "${status}"
