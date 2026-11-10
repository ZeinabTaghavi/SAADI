#!/usr/bin/env bash
# C8 step 99 (CPU only): merge the datasets into results/rebuttal/c8_long_context/{table.csv,
# paired.csv, cost.csv, REPORT.md, table_paper.tex}, write reports/step_99_report.{md,json}, push.
# run_gpus_4567.sh calls it at the end; run it again by hand at any time, e.g. after C5's reader
# answers arrive (the SAADI-with-C5-reader row is read from C5's results):
#   bash experiments/rebuttal/c8_long_context/sh/99_report.sh
# With --rescore it first re-runs step 03 for every dataset (CPU) from the saved answers.
# The LAST line printed is the report path; the exit code is non-zero when the step failed.
set -uo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
# shellcheck disable=SC1091
source "${C8_EXP_DIR}/sh/lib.sh"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "${C8_LOG_DIR}" "${C8_REPORT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${C8_LOG_DIR}/99_report_${TS}.log"
status=0
if [[ "${1:-}" == "--rescore" ]]; then
  IFS=',' read -r -a DSETS <<< "${C8_DATASETS}"
  for DS in "${DSETS[@]}"; do
    R3="${C8_REPORT_DIR}/step_03_${DS}_score.md"
    c8_run_step 03 "step 03 scores, ${DS}" "${R3}" "${LOG}" "${PYTHON}" "${C8_EXP_DIR}/scripts/step03_score.py" --dataset "${DS}" || status=1
    c8_push "${DS}" 03_score "${R3}"
  done
fi
REPORT="${C8_REPORT_DIR}/step_99_report.md"
c8_run_step 99 "step 99 (final report)" "${REPORT}" "${LOG}" "${PYTHON}" "${C8_EXP_DIR}/scripts/step99_report.py" || status=1
c8_push all 99_report "${REPORT}"
echo "${REPORT}"
exit "${status}"
