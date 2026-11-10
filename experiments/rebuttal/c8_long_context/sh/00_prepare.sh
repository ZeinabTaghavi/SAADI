#!/usr/bin/env bash
# C8 step 00 (CPU only, no model is called): check C5's outputs, build every full document from
# SAADI's chunks, count tokens with the long-context model's tokenizer, place the gold evidence in
# each document, count the prompt tokens of all four conditions, and choose the vLLM settings
# (context window, memory fit, attention backend). Run on the server, from anywhere:
#   bash experiments/rebuttal/c8_long_context/sh/00_prepare.sh
# Optional: --datasets musique_32k   (default: musique_32k,novelhopqa)
# Writes outputs/rebuttal/c8_long_context/prepared/ (documents, query index, vllm_settings.env),
# results/rebuttal/c8_long_context/prepare/*.csv and reports/step_00.{md,json}; then commits and
# pushes the C8 folders. The LAST line printed is the report path; the exit code is non-zero
# when the step failed.
set -uo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
# shellcheck disable=SC1091
source "${C8_EXP_DIR}/sh/lib.sh"

# CPU only: hide every GPU from this step (nvidia-smi is only read for the memory sizes).
export CUDA_VISIBLE_DEVICES=""

mkdir -p "${C8_LOG_DIR}" "${C8_REPORT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${C8_LOG_DIR}/00_prepare_${TS}.log"
REPORT="${C8_REPORT_DIR}/step_00.md"

{
  echo "[c8] host $(hostname); repo ${C8_ROOT_DIR}; python ${PYTHON}; vLLM python ${C8_VLLM_PY}"
  echo "[c8] model ${C8_MODEL}; C5 contexts ${C8_C5_CONTEXTS_DIR}"
  echo "[c8] HF_HOME ${HF_HOME}"
} | tee -a "${LOG}"

c8_check_python "${REPORT}" "step 00 (prepare)" 2>&1 | tee -a "${LOG}"
if [[ "${PIPESTATUS[0]}" -eq 0 ]]; then
  c8_run_step 00 "step 00 (prepare)" "${REPORT}" "${LOG}" \
    "${PYTHON}" "${C8_EXP_DIR}/scripts/step00_prepare.py" "$@"
  status=$?
else
  status=1
fi

c8_push all 00_prepare "${REPORT}"

echo "${REPORT}"
exit "${status}"
