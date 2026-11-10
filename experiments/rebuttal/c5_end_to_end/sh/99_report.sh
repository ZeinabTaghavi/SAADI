#!/usr/bin/env bash
# C5 step 99 (CPU only): merge both datasets into results/rebuttal/c5_end_to_end/
# {table.csv, paired.csv, table_paper.tex, REPORT.md}, write reports/step_99_report.{md,json}, push.
# Run after both GPU scripts finished:
#   bash experiments/rebuttal/c5_end_to_end/sh/99_report.sh
# The LAST line printed is the report path; the exit code is non-zero when the step failed.
set -uo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
# shellcheck disable=SC1091
source "${C5_EXP_DIR}/sh/lib.sh"
export CUDA_VISIBLE_DEVICES=""
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${C5_LOG_DIR}/99_report_${TS}.log"
REPORT="${C5_REPORT_DIR}/step_99_report.md"
c5_run_step 99 "step 99 (final report)" "${REPORT}" "${LOG}" "${PYTHON}" "${C5_EXP_DIR}/scripts/step99_report.py"
status=$?
c5_push all 99_report "${REPORT}"
echo "${REPORT}"
exit "${status}"
