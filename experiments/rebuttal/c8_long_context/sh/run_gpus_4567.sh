#!/usr/bin/env bash
# C8 GPU script: everything after step 00, unattended, on GPUs 4,5,6,7 only. Start it once:
#   nohup bash experiments/rebuttal/c8_long_context/sh/run_gpus_4567.sh > /dev/null 2>&1 &
# It
#   1. checks step 00 passed and reads its vLLM settings;
#   2. WAITS until C5's run on GPUs 4-7 has finished (its lock is gone) and GPUs 4-7 are free
#      (up to C8_WAIT_MAX_HOURS, default 24), so it can be started right away;
#   3. starts its own vLLM server (port C8_PORT) with dual chunk attention when step 00 found it,
#      else (or if that server does not start) with the default backend and a 262,144-token window;
#   4. step 01: times C8_TIMING_QUERIES queries per dataset and estimates the run (reported; over
#      C8_MAX_GPU_HOURS it still runs unless C8_ON_OVER_BUDGET=stop);
#   5. step 02 + 03 for MuSiQue-32K, then NovelHopQA (reader, scores); no new full document is
#      started after C8_DEADLINE_HOURS (default 14 h after the server is up);
#   6. stops the server (also on failure or Ctrl-C) and runs 99_report.sh.
# Each step writes its report and pushes. Restart the same command to resume (answers are saved per
# query). Logs: experiments/rebuttal/c8_long_context/logs/run_gpus_4567_*.log; vLLM log under
# outputs/rebuttal/c8_long_context/vllm_logs/. Exit code non-zero if a step failed.
set -uo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
# shellcheck disable=SC1091
source "${C8_EXP_DIR}/sh/lib.sh"

# GPUs 4-7 only, whatever the shell had.
export CUDA_VISIBLE_DEVICES="4,5,6,7"
GPUS="${CUDA_VISIBLE_DEVICES}"
PORT="${C8_PORT}"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${C8_LOG_DIR}" "${C8_REPORT_DIR}" "${C8_OUT_DIR}/vllm_logs"
LOG="${C8_LOG_DIR}/run_gpus_4567_${TS}.log"
export VLLM_CACHE_ROOT="${C8_OUT_DIR}/cache/vllm"
export TRITON_CACHE_DIR="${C8_OUT_DIR}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${C8_OUT_DIR}/cache/inductor"
# Short temp folder: vLLM's ZeroMQ socket paths must stay under 107 characters.
export TMPDIR="${C8_TMP_BASE:-/tmp}/c8_${USER:-user}"
export VLLM_RPC_BASE_PATH="${TMPDIR}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"
chmod 700 "${TMPDIR}" 2>/dev/null
# /reset_prefix_cache (used to measure each document's uncached read) is a dev endpoint.
export VLLM_SERVER_DEV_MODE=1

say() { echo "[c8] $(date '+%F %T') $*" | tee -a "${LOG}"; }

fail_report() {  # <report.md> <title> <message>
  {
    echo "# C8 long-context reader - $2: FAILED"
    echo
    echo "Written $(date '+%F %T') on $(hostname); GPUs ${GPUS}; port ${PORT}."
    echo
    echo "$3"
    echo
    echo "Last 60 lines of ${LOG#"${C8_ROOT_DIR}/"}:"
    echo
    echo '```'
    tail -n 60 "${LOG}" 2>/dev/null
    echo '```'
  } > "$1"
  echo "FAILED: $3" | cut -c1-150 > "${1%.md}.summary.txt"
}

LOCK="${C8_OUT_DIR}/run_gpus_4567.lock.d"
mkdir -p "${C8_OUT_DIR}"
if ! mkdir "${LOCK}" 2>/dev/null; then
  old="$(cat "${LOCK}/pid" 2>/dev/null)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[c8] another C8 run is going (pid ${old}); stop it first." >&2
    exit 2
  fi
  rm -rf "${LOCK}" && mkdir "${LOCK}"
fi
echo "$$" > "${LOCK}/pid"

SERVER_PID=""
stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    say "stopping vLLM server (pid ${SERVER_PID})"
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null
    for _ in $(seq 1 60); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null
    wait "${SERVER_PID}" 2>/dev/null
  fi
  SERVER_PID=""
}
cleanup() {
  stop_server
  rm -rf "${LOCK}"
}
trap cleanup EXIT
trap 'say "interrupted"; exit 130' INT TERM

server_ready() {
  "${PYTHON}" - "${PORT}" "${C8_MODEL}" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=5) as r:
        ids = [m["id"] for m in json.loads(r.read())["data"]]
    sys.exit(0 if sys.argv[2] in ids else 1)
except Exception:
    sys.exit(1)
PY
}

port_busy() {  # something already accepts connections on the port (sockets left in TIME_WAIT do not count)
  "${PYTHON}" - "${PORT}" <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(3)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
sys.exit(0)
PY
}

gpu_busy() {  # prints the busy GPUs among 4-7 (more than C8_GPU_BUSY_MB in use)
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "${GPUS}" 2>/dev/null \
    | awk -F', *' -v lim="${C8_GPU_BUSY_MB}" '$2 > lim {printf "GPU %s: %s MB; ", $1, $2}'
}

overall=0
# Problems before the server is up go to their own report (a restart never overwrites step 01).
PRE_REPORT="${C8_REPORT_DIR}/step_01_preflight.md"
say "host $(hostname); GPUs ${GPUS}; port ${PORT}; model ${C8_MODEL}; vLLM python ${C8_VLLM_PY}"

# ---------------------------------------------------------------- 1. step 00 and its settings
s00_status="$("${PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1])).get('status','?'))" "${C8_REPORT_DIR}/step_00.json" 2>/dev/null || echo missing)"
if [[ "${s00_status}" != OK* || ! -f "${C8_SETTINGS_FILE}" ]]; then
  fail_report "${PRE_REPORT}" "GPU run (preflight)" "Step 00 has not passed (status: ${s00_status}; settings ${C8_SETTINGS_FILE#"${C8_ROOT_DIR}/"}). Run sh/00_prepare.sh and fix what its report says."
  c8_push all 01_preflight "${PRE_REPORT}"
  exit 1
fi
# shellcheck disable=SC1090
source "${C8_SETTINGS_FILE}"
MAX_LEN="${C8_MAX_MODEL_LEN:-${C8_STEP00_MAX_MODEL_LEN}}"
NATIVE_LEN=262144
if ! out="$("${C8_VLLM_PY}" -c "import vllm, sys; print(vllm.__version__)" 2>&1)"; then
  fail_report "${PRE_REPORT}" "GPU run (preflight)" "The vLLM Python ${C8_VLLM_PY} cannot import vllm: $(echo "${out}" | tail -n 2 | tr '\n' ' '). Set C8_VLLM_PYTHON."
  c8_push all 01_preflight "${PRE_REPORT}"
  exit 1
fi
VLLM_VERSION="$(echo "${out}" | tail -n 1)"
say "vLLM ${VLLM_VERSION}; step 00: max_model_len ${C8_STEP00_MAX_MODEL_LEN}, dual chunk attention available ${C8_STEP00_DCA_AVAILABLE}, ${C8_STEP00_OVER_NATIVE_QUERIES} queries beyond the native window"

# ---------------------------------------------------------------- 2. wait for C5 and free GPUs
C5_LOCK="${C5_OUT_DIR}/run_novelhopqa.lock.d"
wait_max=$(( ${C8_WAIT_MAX_HOURS:-24} * 3600 ))
waited=0
while true; do
  why=""
  if [[ -f "${C5_LOCK}/pid" ]]; then
    c5pid="$(cat "${C5_LOCK}/pid" 2>/dev/null)"
    [[ -n "${c5pid}" ]] && kill -0 "${c5pid}" 2>/dev/null && why="C5's run on GPUs 4-7 is going (pid ${c5pid}); "
  fi
  busy="$(gpu_busy)"
  [[ -n "${busy}" ]] && why="${why}busy: ${busy}"
  [[ -z "${why}" ]] && break
  if [[ "${waited}" -ge "${wait_max}" ]]; then
    fail_report "${PRE_REPORT}" "GPU run (waiting for GPUs 4-7)" "Still waiting after ${waited} s: ${why}. Nothing was started."
    c8_push all 01_preflight "${PRE_REPORT}"
    exit 1
  fi
  (( waited % 600 == 0 )) && say "waiting: ${why}"
  sleep 60
  waited=$((waited + 60))
done
[[ "${waited}" -gt 0 ]] && say "GPUs 4-7 free after waiting ${waited} s"
if port_busy; then
  fail_report "${PRE_REPORT}" "GPU run (preflight)" "Port ${PORT} is already in use; set C8_PORT to a free port."
  c8_push all 01_preflight "${PRE_REPORT}"
  exit 1
fi

# ---------------------------------------------------------------- 3. vLLM server
# Newer vLLM lists every flag only with --help=all.
HELP="$("${C8_VLLM_PY}" -m vllm.entrypoints.openai.api_server --help=all 2>&1)"
grep -q -- "--max-model-len" <<< "${HELP}" || HELP="$("${C8_VLLM_PY}" -m vllm.entrypoints.openai.api_server --help 2>&1)"
has_flag() { grep -q -- "$1" <<< "${HELP}"; }
EXTRA=()
[[ "${C8_ENFORCE_EAGER}" == "1" ]] && EXTRA+=(--enforce-eager)
[[ "${C8_ENABLE_PREFIX_CACHING}" == "1" ]] && EXTRA+=(--enable-prefix-caching)
has_flag "--enable-chunked-prefill" && EXTRA+=(--enable-chunked-prefill)
has_flag "--enable-prompt-tokens-details" && EXTRA+=(--enable-prompt-tokens-details)
has_flag "--disable-custom-all-reduce" && EXTRA+=(--disable-custom-all-reduce)

start_server() {  # <mode: dca|native>
  local mode="$1" len slog
  if [[ "${mode}" == "dca" ]]; then
    len="${MAX_LEN}"
  else
    len=$(( MAX_LEN < NATIVE_LEN ? MAX_LEN : NATIVE_LEN ))
  fi
  slog="${C8_OUT_DIR}/vllm_logs/server_${mode}_${TS}.log"
  local launcher=()
  command -v setsid >/dev/null 2>&1 && launcher=(setsid)
  say "starting vLLM (${mode}) on GPUs ${GPUS}, port ${PORT}, --max-model-len ${len}; log ${slog#"${C8_ROOT_DIR}/"}"
  local mode_args=()
  # Without the dual chunk attention backend the model's config must not ask for it: standard attention,
  # identical to dual chunk attention inside one 262,144-token chunk.
  [[ "${mode}" == "native" ]] && mode_args=(--hf-overrides '{"dual_chunk_attention_config": null}')
  (
    if [[ "${mode}" == "dca" ]]; then
      export VLLM_ATTENTION_BACKEND="${C8_ATTENTION_BACKEND}" VLLM_USE_V1="${C8_VLLM_USE_V1}"
    else
      unset VLLM_ATTENTION_BACKEND VLLM_USE_V1
    fi
    # shellcheck disable=SC2086
    exec ${launcher[@]+"${launcher[@]}"} "${C8_VLLM_PY}" -m vllm.entrypoints.openai.api_server \
      --model "${C8_STEP00_MODEL_PATH:-${C8_MODEL}}" --served-model-name "${C8_MODEL}" --host 127.0.0.1 --port "${PORT}" \
      --tensor-parallel-size "${C8_TENSOR_PARALLEL}" --dtype bfloat16 --max-model-len "${len}" \
      --gpu-memory-utilization "${C8_GPU_MEMORY_UTILIZATION}" --max-num-seqs "${C8_MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${C8_MAX_NUM_BATCHED_TOKENS}" --seed "${C8_SEED}" \
      "${EXTRA[@]}" ${mode_args[@]+"${mode_args[@]}"} ${C8_VLLM_EXTRA_ARGS}
  ) > "${slog}" 2>&1 &
  SERVER_PID=$!
  local w=0
  while ! server_ready; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      say "vLLM (${mode}) exited during start-up; last lines:"
      tail -n 40 "${slog}" | tee -a "${LOG}"
      SERVER_PID=""
      return 1
    fi
    if [[ "${w}" -ge "${C8_VLLM_STARTUP_TIMEOUT}" ]]; then
      say "vLLM (${mode}) not ready after ${w} s; last lines:"
      tail -n 40 "${slog}" | tee -a "${LOG}"
      stop_server
      return 1
    fi
    sleep 10
    w=$((w + 10))
  done
  say "vLLM (${mode}) ready after ${w} s"
  export C8_SERVED_MODE="${mode}" C8_SERVED_MAX_MODEL_LEN="${len}" C8_SERVER_LOG="${slog}"
}

MODE="native"
[[ "${C8_STEP00_DCA_AVAILABLE}" == "1" && "${C8_FORCE_MODE:-}" != "native" ]] && MODE="dca"
if ! start_server "${MODE}"; then
  if [[ "${MODE}" == "dca" ]]; then
    say "falling back to the default attention backend (documents beyond 262,144 tokens will be skipped and listed)"
    MODE="native"
    start_server native || MODE=""
  else
    MODE=""
  fi
fi
if [[ -z "${MODE}" ]]; then
  fail_report "${PRE_REPORT}" "GPU run (vLLM start)" "The vLLM server did not start (logs in ${C8_OUT_DIR#"${C8_ROOT_DIR}/"}/vllm_logs/)."
  c8_push all 01_preflight "${PRE_REPORT}"
  exit 1
fi
rm -f "${PRE_REPORT}" "${PRE_REPORT%.md}.summary.txt"
export C8_DEADLINE_EPOCH=$(( $(date +%s) + ${C8_DEADLINE_HOURS:-14} * 3600 ))
say "deadline for starting new full documents: $(date -d "@${C8_DEADLINE_EPOCH}" '+%F %T' 2>/dev/null || echo "${C8_DEADLINE_EPOCH}")"

ensure_server() {  # restart the server once if it died
  server_ready && return 0
  say "server not answering; restarting it (${MODE})"
  stop_server
  start_server "${MODE}"
}

run_with_retry() {  # <step> <title> <report> <python script> <args...>: one retry after a server restart
  local step="$1" title="$2" report="$3"
  shift 3
  c8_run_step "${step}" "${title}" "${report}" "${LOG}" "${PYTHON}" "$@" && return 0
  if ! server_ready; then
    ensure_server || return 1
    say "retrying ${step} after the server restart"
    c8_run_step "${step}" "${title}" "${report}" "${LOG}" "${PYTHON}" "$@"
    return $?
  fi
  return 1
}

# ---------------------------------------------------------------- 4. timing
# A restart keeps a passed timing of the same server mode (C8_RETIME=1 times again).
t_ok="$("${PYTHON}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(1 if d.get('status')=='OK' and d['server']['mode']==sys.argv[2] else 0)" "${C8_REPORT_DIR}/step_01_timing.json" "${MODE}" 2>/dev/null || echo 0)"
if [[ "${t_ok}" == "1" && "${C8_RETIME:-0}" != "1" ]]; then
  say "step 01 timing already passed in ${MODE} mode; kept (C8_RETIME=1 to time again)"
else
  run_with_retry 01 "step 01 timing" "${C8_REPORT_DIR}/step_01_timing.md" "${C8_EXP_DIR}/scripts/step01_timing.py" --port "${PORT}" || overall=1
  c8_push all 01_timing "${C8_REPORT_DIR}/step_01_timing.md"
fi
STOP_DS="$("${PYTHON}" -c "import json,sys; print(','.join(json.load(open(sys.argv[1])).get('stop_datasets', [])))" "${C8_REPORT_DIR}/step_01_timing.json" 2>/dev/null)"

# ---------------------------------------------------------------- 5. datasets
IFS=',' read -r -a DSETS <<< "${C8_DATASETS}"
for DS in "${DSETS[@]}"; do
  R2="${C8_REPORT_DIR}/step_02_${DS}_reader.md"
  if [[ ",${STOP_DS}," == *",${DS},"* ]]; then
    fail_report "${R2}" "step 02 reader, ${DS}" "Not run: the timing estimate is above C8_MAX_GPU_HOURS=${C8_MAX_GPU_HOURS} and C8_ON_OVER_BUDGET=stop (see step_01_timing.md)."
    c8_push "${DS}" 02_reader "${R2}"
    continue
  fi
  ensure_server || { fail_report "${R2}" "step 02 reader, ${DS}" "The vLLM server is down and did not restart."; c8_push "${DS}" 02_reader "${R2}"; overall=1; break; }
  run_with_retry 02 "step 02 reader, ${DS}" "${R2}" "${C8_EXP_DIR}/scripts/step02_reader.py" --dataset "${DS}" --port "${PORT}" || overall=1
  c8_push "${DS}" 02_reader "${R2}"
  R3="${C8_REPORT_DIR}/step_03_${DS}_score.md"
  c8_run_step 03 "step 03 scores, ${DS}" "${R3}" "${LOG}" "${PYTHON}" "${C8_EXP_DIR}/scripts/step03_score.py" --dataset "${DS}" || overall=1
  c8_push "${DS}" 03_score "${R3}"
done

# ---------------------------------------------------------------- 6. stop, final report
stop_server
CUDA_VISIBLE_DEVICES="" bash "${C8_EXP_DIR}/sh/99_report.sh" || overall=1
say "done (exit ${overall})"
exit "${overall}"
