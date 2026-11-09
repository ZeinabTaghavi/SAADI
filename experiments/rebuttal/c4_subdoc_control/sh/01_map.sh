#!/usr/bin/env bash
# C4 step 01: chunk -> sub-document map, gold sub-document checks, sub-document sizes.
# CPU only, reads only; a few minutes. Run from anywhere (executes at the repository root):
#   bash experiments/rebuttal/c4_subdoc_control/sh/01_map.sh
# Optional flags pass through: --settings musique_32k,qasper_64k  --retrievers bm25
# Writes results/rebuttal/c4_subdoc_control/subdoc_map/<setting>.json and reports/step_01.{md,json}.
# The LAST line printed is the report path; the exit code is non-zero when the step failed.
set -uo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${C4_LOG_DIR}/01_map_${TS}.log"
REPORT="${C4_REPORT_DIR}/step_01.md"
# Keep the previous report aside so the path below always shows THIS run.
[[ -f "${REPORT}" ]] && mv "${REPORT}" "${REPORT%.md}.prev.md"
rm -f "${C4_REPORT_DIR}/step_01.json"

"${PYTHON}" "${C4_EXP_DIR}/scripts/step01_map.py" "$@" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

# A step that died before reporting still leaves a report with the log tail.
if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# C4 sub-document control - step 01: FAILED before writing its report"
    echo
    echo "Command: bash experiments/rebuttal/c4_subdoc_control/sh/01_map.sh $*"
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
