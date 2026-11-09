#!/usr/bin/env bash
# C4 step 02: input completeness, BM25 full scores (CPU), reproduction of the paper's numbers.
# CPU only. The BM25 re-scoring of the 8 settings is the long part (resumable: per-document
# checkpoints under work/streams_work/); run it with nohup:
#   nohup bash experiments/rebuttal/c4_subdoc_control/sh/02_check.sh > /dev/null 2>&1 &
# Optional flags: --workers N (parallel settings, default 4)  --settings ...  --retrievers ...  --force
# Dense full scores are read from $C4_DENSE_STREAMS_DIR (default: C2's streams).
# The LAST line printed is the report path; the exit code is non-zero when the step failed.
set -uo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${C4_LOG_DIR}/02_check_${TS}.log"
REPORT="${C4_REPORT_DIR}/step_02.md"
[[ -f "${REPORT}" ]] && mv "${REPORT}" "${REPORT%.md}.prev.md"
rm -f "${C4_REPORT_DIR}/step_02.json"

"${PYTHON}" "${C4_EXP_DIR}/scripts/step02_check.py" "$@" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# C4 sub-document control - step 02: FAILED before writing its report"
    echo
    echo "Command: bash experiments/rebuttal/c4_subdoc_control/sh/02_check.sh $*"
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
