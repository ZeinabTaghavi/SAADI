#!/usr/bin/env bash
# Probe baselines, end to end and unattended, on GPUs 4-7:
#   00_check -> 01_b0_vanilla -> 02_b1_oracle -> 03_b2_contextual -> 04_b3_hipporag
#   -> 05_b4_lqca -> 06_score -> 07_report
# CPU stages first, then the GPU stages in order of expected value (B2, B3, B4).
# Each stage writes reports/<stage>.md (+ .json) and a marker in
# outputs/probe_baselines/markers/ (<stage>.done or <stage>.failed). A finished stage is
# skipped on a rerun unless --force. A failing system stage writes its report (traceback and
# the last 50 log lines) and the script CONTINUES with the next system; 06_score and
# 07_report run on whatever finished. Only a changed probe spec stops everything.
# LLM stages resume per book (B2 per chunk, B3 from HippoRAG's caches, B4 per segment).
#
# Usage (from anywhere; nohup-safe, never asks for input):
#   nohup bash experiments/probe_baselines/run_all.sh > experiments/probe_baselines/logs/run_all_nohup.out 2>&1 &
#   bash experiments/probe_baselines/run_all.sh --from 04_b3_hipporag
#   bash experiments/probe_baselines/run_all.sh --from 03_b2_contextual --force
# The LAST line printed is the path of reports/FINAL.md.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${EXP_DIR}/env.sh"

STAGES=(00_check 01_b0_vanilla 02_b1_oracle 03_b2_contextual 04_b3_hipporag 05_b4_lqca 06_score 07_report)
FROM=0
FORCE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      [[ $# -ge 2 ]] || { echo "--from needs a stage: ${STAGES[*]}"; exit 2; }
      found=""
      for i in "${!STAGES[@]}"; do
        if [[ "${STAGES[$i]}" == "$2" || "${STAGES[$i]:0:2}" == "$2" ]]; then FROM=$i; found=1; fi
      done
      [[ -n "${found}" ]] || { echo "Unknown stage $2; expected one of ${STAGES[*]}"; exit 2; }
      shift 2 ;;
    --force) FORCE="--force"; shift ;;
    *) echo "Unknown argument $1 (use --from <stage> and --force)"; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${PBL_LOG_DIR}/run_all_${TS}.log"
SERVER_LOG="${PBL_LOG_DIR}/vllm_server_${TS}.log"
FINAL="${PBL_REPORT_DIR}/FINAL.md"
exec > >(tee -a "${LOG}") 2>&1
export PBL_LOG_FILE="${LOG}"

echo "== probe baselines started $(date '+%Y-%m-%d %H:%M:%S'); from ${STAGES[$FROM]}; force='${FORCE}'"
echo "== commit $(git -C "${PBL_ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown); log ${LOG}"
echo "== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} http ${PBL_VLLM_URL} VLLM_PORT=${VLLM_PORT} window ${PBL_MAX_MODEL_LEN}"

stage_done() { [[ -f "${PBL_OUT_DIR}/markers/$1.done" ]]; }
will_run() {  # stage index -> true when this invocation runs it
  local i="$1"
  [[ "${i}" -ge "${FROM}" ]] && { [[ -n "${FORCE}" ]] || ! stage_done "${STAGES[$i]}"; }
}

# ---------------------------------------------------------------------------
# The vLLM server (B2 and B3). Started first so the model loads while the third-party
# environments install; stopped before B4, which needs the GPUs for Maverick.
# ---------------------------------------------------------------------------
SERVER_PID=""
start_server() {
  echo "== starting the vLLM server (log ${SERVER_LOG})"
  setsid "${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${PBL_MODEL_ID}" --served-model-name "${PBL_MODEL}" \
    --host 127.0.0.1 --port "${PBL_HTTP_PORT}" \
    --tensor-parallel-size "${PBL_TP}" --dtype bfloat16 \
    --max-model-len "${PBL_MAX_MODEL_LEN}" --gpu-memory-utilization "${PBL_GPU_UTIL}" \
    --max-num-seqs 16 --max-num-batched-tokens 16384 --disable-custom-all-reduce \
    --enable-prefix-caching --seed "${PBL_SEED}" --generation-config vllm \
    > "${SERVER_LOG}" 2>&1 < /dev/null &
  SERVER_PID=$!
}
wait_server() {
  local waited=0
  while [[ ${waited} -lt 3600 ]]; do
    if curl -sf "${PBL_VLLM_URL}/models" > /dev/null 2>&1; then
      echo "== vLLM server ready after ${waited} s"
      return 0
    fi
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "== the vLLM server exited during start-up; last lines of ${SERVER_LOG}:"
      tail -n 30 "${SERVER_LOG}" || true
      return 1
    fi
    sleep 15
    waited=$((waited + 15))
  done
  echo "== the vLLM server was not ready after 60 min"
  return 1
}
stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "== stopping the vLLM server (pid ${SERVER_PID})"
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 2; done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

# ---------------------------------------------------------------------------
# Third-party environments (idempotent; failures are reported by 00_check and make
# only the stage that needs them fail).
# ---------------------------------------------------------------------------
# A virtual environment with a working pip. `python -m venv` needs ensurepip, which some
# system Pythons lack (it then leaves a venv without pip: the first night's B3/B4 failure);
# virtualenv, installed with the study venv's own pip, seeds pip itself.
safe_clear() {  # removes a folder only when it lies inside third_party/
  local dir="$1"
  case "${dir}" in
    "${PBL_TP_DIR}"/?*) [[ -n "${PBL_TP_DIR}" ]] && rm -rf -- "${dir}" ;;
    *) echo "   refusing to remove ${dir} (outside ${PBL_TP_DIR})"; return 1 ;;
  esac
}
make_venv() {  # <dir> <log>
  local dir="$1" log="$2"
  safe_clear "${dir}" || return 1
  if "${PBL_BASE_PYTHON}" -m venv "${dir}" >> "${log}" 2>&1 && "${dir}/bin/python" -m pip --version >> "${log}" 2>&1; then
    return 0
  fi
  echo "   python -m venv gave no working pip; creating ${dir} with virtualenv instead" | tee -a "${log}"
  safe_clear "${dir}" || return 1
  "${PBL_BASE_PYTHON}" -m pip install -q --target "${PBL_TP_DIR}/_virtualenv" "virtualenv>=20.26" >> "${log}" 2>&1 || return 1
  PYTHONPATH="${PBL_TP_DIR}/_virtualenv" "${PBL_BASE_PYTHON}" -m virtualenv -q "${dir}" >> "${log}" 2>&1 || return 1
  "${dir}/bin/python" -m pip --version >> "${log}" 2>&1
}
setup_failed() {  # <name> <log>
  echo "== $1 setup FAILED; last lines of $2:"
  tail -n 25 "$2" || true
}
setup_hipporag() {
  [[ -f "${PBL_TP_DIR}/.hipporag_ok" ]] && return 0
  local log="${PBL_LOG_DIR}/setup_hipporag_${TS}.log"
  echo "== setting up ${PBL_TP_DIR}/venv_hipporag (hipporag 2.0.0a4 pins torch 2.5.1, transformers 4.45.2, vllm 0.6.6); log ${log}"
  make_venv "${PBL_TP_DIR}/venv_hipporag" "${log}" \
    && "${PBL_HIPPORAG_PY}" -m pip install -q --upgrade pip >> "${log}" 2>&1 \
    && "${PBL_HIPPORAG_PY}" -m pip install "hipporag==2.0.0a4" >> "${log}" 2>&1 \
    && "${PBL_HIPPORAG_PY}" -c "import hipporag, huggingface_hub" >> "${log}" 2>&1 \
    && touch "${PBL_TP_DIR}/.hipporag_ok" \
    && echo "== hipporag venv ready" \
    || setup_failed hipporag "${log}"
}
setup_lqca() {
  [[ -f "${PBL_TP_DIR}/.lqca_ok" ]] && return 0
  echo "== setting up LQCA: clone @ ${PBL_LQCA_COMMIT:0:12} and ${PBL_TP_DIR}/venv_lqca"
  {
    if [[ ! -d "${PBL_LQCA_DIR}/.git" ]]; then git clone -q "${PBL_LQCA_URL}" "${PBL_LQCA_DIR}"; fi
    git -C "${PBL_LQCA_DIR}" fetch -q origin && git -C "${PBL_LQCA_DIR}" checkout -q "${PBL_LQCA_COMMIT}"
  } || echo "== LQCA clone FAILED"
  # maverick-coref 1.0.7 (the coreference model LQCA imports as `maverick`) needs torch < 2.6
  # to load its checkpoint, and transformers 4.45.2 is the last that runs on torch 2.5.
  local log="${PBL_LOG_DIR}/setup_lqca_${TS}.log"
  echo "   venv log ${log}"
  make_venv "${PBL_TP_DIR}/venv_lqca" "${log}" \
    && "${PBL_LQCA_PY}" -m pip install -q --upgrade pip >> "${log}" 2>&1 \
    && "${PBL_LQCA_PY}" -m pip install "torch==2.5.1" "transformers==4.45.2" "maverick-coref==1.0.7" tiktoken requests numpy >> "${log}" 2>&1 \
    && "${PBL_LQCA_PY}" -m spacy download en_core_web_sm >> "${log}" 2>&1 \
    && "${PBL_LQCA_PY}" -c "import maverick, tiktoken; tiktoken.get_encoding('cl100k_base')" >> "${log}" 2>&1 \
    && touch "${PBL_TP_DIR}/.lqca_ok" \
    && echo "== LQCA venv ready" \
    || setup_failed LQCA "${log}"
}

write_abort_final() {
  {
    echo "# Probe baselines: stopped before any system ran"
    echo
    echo "$1"
    echo
    echo "Last 50 lines of the run log:"
    echo
    echo '```'
    tail -n 50 "${LOG}" || true
    echo '```'
  } > "${FINAL}"
}

run_stage() {  # index, command words separated by ':::'
  local i="$1"; shift
  local flag=""
  [[ "${i}" -ge "${FROM}" ]] && flag="${FORCE}"
  # Scoring and the report are cheap and must see every system and SAADI run that exists now.
  [[ "${i}" -ge 6 ]] && flag="--force"
  if [[ "${i}" -lt "${FROM}" ]]; then
    echo "== ${STAGES[$i]}: before --from, not run"
    return 0
  fi
  echo
  echo "== stage ${STAGES[$i]} $(date '+%H:%M:%S')"
  set +e
  "${PYTHON}" "${PBL_EXP_DIR}/common/report_utils.py" stage "${STAGES[$i]}" ${flag} -- "$@"
  local code=$?
  set -e
  echo "== stage ${STAGES[$i]} exit ${code} $(date '+%H:%M:%S')"
  return 0  # baselines are independent: a failed stage never stops the next one
}

# The frozen spec first: nothing runs on a changed spec.
if ! "${PYTHON}" "${PBL_EXP_DIR}/common/load_probe.py" --check > "${PBL_LOG_DIR}/spec_check_${TS}.json" 2>&1; then
  write_abort_final "The frozen probe spec check failed (see ${PBL_LOG_DIR}/spec_check_${TS}.json). Nothing was run."
  echo "== the frozen spec check FAILED; nothing was run"
  echo "${FINAL}"
  exit 3
fi
echo "== frozen spec verified"

NEED_SERVER=""
if will_run 3 || will_run 4; then NEED_SERVER=1; fi
if [[ -n "${NEED_SERVER}" ]] || will_run 0; then start_server; fi
if will_run 0 || will_run 4; then setup_hipporag; fi
if will_run 0 || will_run 5; then setup_lqca; fi
if [[ -n "${SERVER_PID}" ]]; then wait_server || echo "== continuing without a server: 03 and 04 will fail and say so"; fi

P="${PYTHON}"
X="${PBL_EXP_DIR}"
run_stage 0 "$P" "$X/common/check.py"
run_stage 1 "$P" "$X/systems/b0_vanilla/build.py" ::: "$P" "$X/systems/b0_vanilla/retrieve.py"
run_stage 2 "$P" "$X/systems/b1_oracle/build.py" ::: "$P" "$X/systems/b1_oracle/retrieve.py"
run_stage 3 "$P" "$X/systems/b2_contextual/build.py" ::: "$P" "$X/systems/b2_contextual/retrieve.py"
run_stage 4 "${PBL_HIPPORAG_PY}" "$X/systems/b3_hipporag/build.py" ::: "${PBL_HIPPORAG_PY}" "$X/systems/b3_hipporag/retrieve.py"
stop_server
run_stage 5 "${PBL_LQCA_PY}" "$X/systems/b4_lqca/build.py" ::: "$P" "$X/systems/b4_lqca/retrieve.py"
run_stage 6 "$P" "$X/common/scorer.py" score
run_stage 7 "$P" "$X/common/final_report.py"

echo
"${PYTHON}" "${PBL_EXP_DIR}/common/report_utils.py" status
echo "== probe baselines finished $(date '+%Y-%m-%d %H:%M:%S')"
if [[ ! -f "${FINAL}" ]]; then
  write_abort_final "07_report did not write FINAL.md; see reports/07_report.md."
fi
echo "${FINAL}"
