#!/usr/bin/env bash
# C5 step 00 (CPU only, no model is called): find and check every saved retrieval output,
# reproduce the paper's Vanilla and SAADI top-10 lists (stops on a mismatch), and build one
# contexts file per dataset and condition. Run on the server, from anywhere:
#   bash experiments/rebuttal/c5_end_to_end/sh/00_prepare.sh
# Optional: --datasets musique_32k   (default: musique_32k,novelhopqa)
# Paths are in env.sh (e.g. C5_RAPTOR_DIR=..., C5_ARAG_DIR=..., C5_C1_DIR=...).
# Writes outputs/rebuttal/c5_end_to_end/contexts/<dataset>/<condition>.jsonl.gz (+ manifest.json),
# results/rebuttal/c5_end_to_end/prepare/context_stats.csv and reports/step_00.{md,json};
# then commits and pushes the C5 folders. The LAST line printed is the report path; the exit
# code is non-zero when the step failed.
set -uo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
# shellcheck disable=SC1091
source "${C5_EXP_DIR}/sh/lib.sh"

# CPU only: hide every GPU from this step.
export CUDA_VISIBLE_DEVICES=""

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${C5_LOG_DIR}/00_prepare_${TS}.log"
# --tag <name> (e.g. --datasets novelhopqa --tag novelhopqa): the report is step_00_<name>.md.
TAG=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --tag) TAG="${args[i + 1]:-}" ;;
    --tag=*) TAG="${args[i]#--tag=}" ;;
  esac
done
REPORT="${C5_REPORT_DIR}/step_00${TAG:+_${TAG}}.md"
PUSH_NAME="00_prepare${TAG:+_${TAG}}"

{
  echo "[c5] host $(hostname); repo ${C5_ROOT_DIR}; python ${PYTHON}"
  echo "[c5] RAPTOR ${C5_RAPTOR_DIR}"
  echo "[c5] A-RAG  ${C5_ARAG_DIR}"
  echo "[c5] C1     ${C5_C1_DIR}"
  echo "[c5] HF_HOME ${HF_HOME}"
} | tee -a "${LOG}"

c5_check_python "${REPORT}" "step 00 (prepare)" 2>&1 | tee -a "${LOG}"
if [[ "${PIPESTATUS[0]}" -eq 0 ]]; then
  c5_run_step 00 "step 00 (prepare)" "${REPORT}" "${LOG}" \
    "${PYTHON}" "${C5_EXP_DIR}/scripts/step00_prepare.py" "$@"
  status=$?
else
  status=1
fi

c5_push all "${PUSH_NAME}" "${REPORT}"

echo "${REPORT}"
exit "${status}"
