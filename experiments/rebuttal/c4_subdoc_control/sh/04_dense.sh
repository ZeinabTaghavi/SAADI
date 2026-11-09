#!/usr/bin/env bash
# C4 step 04: the same for the dense-retriever pairs whose scores passed step 02 (CPU).
# Needs steps 01 and 02 OK. Pairs not ready in step 02 are listed, not measured.
# Optional flags: --settings ...  --retrievers ...  --workers N (default 4)
# Main query set: C4_MAIN_SET=R (default) or S; both sets are always computed.
# The LAST line printed is the report path; the exit code is non-zero when the step failed.
set -uo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${C4_LOG_DIR}/04_dense_${TS}.log"
REPORT="${C4_REPORT_DIR}/step_04.md"
[[ -f "${REPORT}" ]] && mv "${REPORT}" "${REPORT%.md}.prev.md"
rm -f "${C4_REPORT_DIR}/step_04.json"

"${PYTHON}" "${C4_EXP_DIR}/scripts/step03_measures.py" --family dense "$@" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# C4 sub-document control - step 04: FAILED before writing its report"
    echo
    echo "Command: bash experiments/rebuttal/c4_subdoc_control/sh/04_dense.sh $*"
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
