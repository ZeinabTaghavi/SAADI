#!/usr/bin/env bash
# C4 step 06: which SAADI channel carries the gain inside the gold sub-document (text + summary,
# text + memory, full SAADI, each against text alone). CPU, a few minutes. Needs steps 01-03 OK
# (and 04 when it ran). Writes results/rebuttal/c4_subdoc_control/{channels.csv,CHANNELS.md}.
# The LAST line printed is the report path; the exit code is non-zero when the step failed.
set -uo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${C4_LOG_DIR}/06_channels_${TS}.log"
REPORT="${C4_REPORT_DIR}/step_06.md"
[[ -f "${REPORT}" ]] && mv "${REPORT}" "${REPORT%.md}.prev.md"
rm -f "${C4_REPORT_DIR}/step_06.json"

"${PYTHON}" "${C4_EXP_DIR}/scripts/step06_channels.py" "$@" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# C4 sub-document control - step 06: FAILED before writing its report"
    echo
    echo "Command: bash experiments/rebuttal/c4_subdoc_control/sh/06_channels.sh $*"
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
