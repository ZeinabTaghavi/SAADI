#!/usr/bin/env bash
# C4 step 03: per-query ranks inside the gold sub-document for the 8 BM25 pairs (CPU, a few minutes).
# Needs steps 01 and 02 OK. Pairs not ready in step 02 are listed, not measured.
# Optional flags: --settings ...  --retrievers ...  --workers N (default 4)
# Main query set: C4_MAIN_SET=R (default) or S; both sets are always computed.
# The LAST line printed is the report path; the exit code is non-zero when the step failed.
set -uo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${C4_LOG_DIR}/03_bm25_${TS}.log"
REPORT="${C4_REPORT_DIR}/step_03.md"
[[ -f "${REPORT}" ]] && mv "${REPORT}" "${REPORT%.md}.prev.md"
rm -f "${C4_REPORT_DIR}/step_03.json"

"${PYTHON}" "${C4_EXP_DIR}/scripts/step03_measures.py" --family bm25 "$@" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# C4 sub-document control - step 03: FAILED before writing its report"
    echo
    echo "Command: bash experiments/rebuttal/c4_subdoc_control/sh/03_bm25.sh $*"
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
