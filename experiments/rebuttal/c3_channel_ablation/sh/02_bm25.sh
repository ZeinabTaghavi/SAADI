#!/usr/bin/env bash
# C3 step 02: all six variants for BM25 (CPU), on the BM25 pairs that passed step 01's check 1.
# Runs on the laptop or the server. Per-query ranks go to
# results/rebuttal/c3_channel_ablation/per_query/ (committed, so step 04 can combine machines).
#
#   bash experiments/rebuttal/c3_channel_ablation/sh/02_bm25.sh
#
# Never asks for input; nohup-safe. The LAST line printed is the path of the report.
set -euo pipefail

export C3_RETRIEVERS_USER="${C3_RETRIEVERS:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/experiments/rebuttal/c3_channel_ablation/env.sh"

STEP=02
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${C3_LOG_DIR}/${STEP}_bm25_${TS}.log"
REPORT="${C3_REPORT_DIR}/step_${STEP}.md"
if [[ -f "${REPORT}" ]]; then
  mv "${REPORT}" "${REPORT%.md}.prev.md"
fi

set +e
"${PYTHON}" "${C3_EXP_DIR}/scripts/step02_variants.py" --step 2 "$@" 2>&1 | tee "${LOG_FILE}"
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
