#!/usr/bin/env bash
# One C5 GPU group: one dataset on four GPUs with its own vLLM server. Called by
# run_gpus_0123.sh (MuSiQue-32K) and run_gpus_4567.sh (NovelHopQA); not meant to be run directly:
#   bash experiments/rebuttal/c5_end_to_end/sh/run_group.sh <dataset> <gpus> <port>
# Stages (each writes its report and pushes):
#   01 RAPTOR re-run (MuSiQue-32K only, when step 00 found no RAPTOR contexts), then the
#      re-prepare of that dataset's contexts (step_00_<dataset>);
#   02 reader: start the vLLM server, wait until ready, answer every condition, stop the server;
#   03 scores.
# Resumable: a restart skips the RAPTOR trees/queries already done and the answers already saved.
# The server is stopped at the end, also on failure or Ctrl-C. Exit code non-zero if a stage failed.
set -uo pipefail

DS="$1"
GPUS="$2"
PORT="$3"

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
# shellcheck disable=SC1091
source "${C5_EXP_DIR}/sh/lib.sh"

export CUDA_VISIBLE_DEVICES="${GPUS}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${C5_LOG_DIR}/run_${DS}_${TS}.log"
# Nothing shared with the other group: own vLLM/torch/triton caches and temp dir.
export VLLM_CACHE_ROOT="${C5_OUT_DIR}/cache/${DS}/vllm"
export TRITON_CACHE_DIR="${C5_OUT_DIR}/cache/${DS}/triton"
export TORCHINDUCTOR_CACHE_DIR="${C5_OUT_DIR}/cache/${DS}/inductor"
# The temp folder must have a SHORT path: vLLM puts its ZeroMQ socket files there, and a Unix socket
# path is limited to 107 characters (a folder inside the repo is too long).
export TMPDIR="${C5_TMP_BASE:-/tmp}/c5_${USER:-user}_${DS}"
export VLLM_RPC_BASE_PATH="${TMPDIR}"
if [[ "${#TMPDIR}" -gt 60 ]]; then
  echo "[c5-${DS}] temp folder path ${TMPDIR} is too long for vLLM's sockets (max 60 characters here); set C5_TMP_BASE to a short folder" >&2
  exit 2
fi
mkdir -p "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}" "${C5_OUT_DIR}/vllm_logs"
chmod 700 "${TMPDIR}" 2>/dev/null

say() { echo "[c5-${DS}] $(date '+%F %T') $*" | tee -a "${LOG}"; }

fail_report() {  # <report.md> <title> <message>
  {
    echo "# C5 end-to-end QA - $2: FAILED"
    echo
    echo "Written $(date '+%F %T') on $(hostname); GPUs ${GPUS}; port ${PORT}."
    echo
    echo "$3"
    echo
    echo "Last 60 lines of ${LOG#"${C5_ROOT_DIR}/"}:"
    echo
    echo '```'
    tail -n 60 "${LOG}" 2>/dev/null
    echo '```'
  } > "$1"
  echo "FAILED: $3" | cut -c1-150 > "${1%.md}.summary.txt"
}

# One run of this group at a time.
LOCK="${C5_OUT_DIR}/run_${DS}.lock.d"
if ! mkdir "${LOCK}" 2>/dev/null; then
  old="$(cat "${LOCK}/pid" 2>/dev/null)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[c5-${DS}] another run of this group is going (pid ${old}); stop it first." >&2
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

port_busy() {
  "${PYTHON}" - "$1" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(0)  # busy
sys.exit(1)
PY
}

server_ready() {
  "${PYTHON}" - "$1" "${C5_READER_MODEL}" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=5) as r:
        ids = [m["id"] for m in json.loads(r.read())["data"]]
    sys.exit(0 if sys.argv[2] in ids else 1)
except Exception:
    sys.exit(1)
PY
}

start_server() {
  local slog="${C5_OUT_DIR}/vllm_logs/${DS}_${TS}.log"
  local launcher=()
  command -v setsid >/dev/null 2>&1 && launcher=(setsid)
  say "starting vLLM on GPUs ${GPUS}, port ${PORT}; log ${slog#"${C5_ROOT_DIR}/"}"
  # shellcheck disable=SC2086
  ${launcher[@]+"${launcher[@]}"} "${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${C5_READER_MODEL}" --served-model-name "${C5_READER_MODEL}" \
    --host 127.0.0.1 --port "${PORT}" \
    --tensor-parallel-size "${C5_TENSOR_PARALLEL}" --dtype bfloat16 \
    --max-model-len "${C5_MAX_MODEL_LEN}" --gpu-memory-utilization "${C5_GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${C5_MAX_NUM_SEQS}" --seed "${C5_SEED}" --disable-custom-all-reduce \
    ${C5_VLLM_EXTRA_ARGS} > "${slog}" 2>&1 &
  SERVER_PID=$!
  local waited=0
  while ! server_ready "${PORT}"; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      say "vLLM server exited during start-up; last lines:"
      tail -n 40 "${slog}" | tee -a "${LOG}"
      SERVER_PID=""
      return 1
    fi
    if [[ "${waited}" -ge "${C5_VLLM_STARTUP_TIMEOUT}" ]]; then
      say "vLLM server not ready after ${waited} s; last lines:"
      tail -n 40 "${slog}" | tee -a "${LOG}"
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
  say "vLLM server ready after ${waited} s"
}

overall=0
say "host $(hostname); dataset ${DS}; GPUs ${GPUS}; port ${PORT}; reader ${C5_READER_MODEL}"

# ---------------------------------------------------------------- preflight
MANIFEST="${C5_CONTEXTS_DIR}/${DS}/manifest.json"
PRE_REPORT="${C5_REPORT_DIR}/step_02_${DS}_reader.md"
c5_check_python "${PRE_REPORT}" "step 02 reader, ${DS}" vllm 2>&1 | tee -a "${LOG}"
if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
  c5_push "${DS}" 02_reader "${PRE_REPORT}"
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  fail_report "${PRE_REPORT}" "step 02 reader, ${DS}" "No contexts for ${DS} (${MANIFEST#"${C5_ROOT_DIR}/"} missing): run sh/00_prepare.sh first and check its report."
  c5_push "${DS}" 02_reader "${PRE_REPORT}"
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "${GPUS}" 2>/dev/null | awk -F', *' -v lim="${C5_GPU_BUSY_MB}" '$2 > lim {print "GPU " $1 ": " $2 " MB"}')"
  if [[ -n "${busy}" ]]; then
    fail_report "${PRE_REPORT}" "step 02 reader, ${DS}" "GPUs ${GPUS} are in use (more than ${C5_GPU_BUSY_MB} MB): ${busy//$'\n'/; }. Nothing was started."
    c5_push "${DS}" 02_reader "${PRE_REPORT}"
    exit 1
  fi
fi
if port_busy "${PORT}"; then
  fail_report "${PRE_REPORT}" "step 02 reader, ${DS}" "Port ${PORT} is already in use; set C5_PORT_* in env.sh to a free port."
  c5_push "${DS}" 02_reader "${PRE_REPORT}"
  exit 1
fi

# ---------------------------------------------------------------- 01 RAPTOR re-run
raptor_available="$("${PYTHON}" -c "import json,sys; print(int(bool(json.load(open(sys.argv[1]))['conditions'].get('raptor',{}).get('available'))))" "${MANIFEST}")"
if [[ "${DS}" == "musique_32k" && "${raptor_available}" != "1" && "${C5_RAPTOR_RERUN}" != "0" ]]; then
  REP="${C5_REPORT_DIR}/step_01_${DS}_raptor.md"
  say "stage 01: RAPTOR re-run on SAADI's documents (RAPTOR's own vLLM engine on GPUs ${GPUS})"
  c5_run_step 01 "step 01 RAPTOR re-run, ${DS}" "${REP}" "${LOG}" "${PYTHON}" "${C5_EXP_DIR}/scripts/step01_raptor_rerun.py"
  st=$?
  c5_push "${DS}" 01_raptor "${REP}"
  if [[ "${st}" -eq 0 ]]; then
    REP0="${C5_REPORT_DIR}/step_00_${DS}.md"
    say "stage 01b: rebuild ${DS} contexts with the RAPTOR re-run (CPU)"
    CUDA_VISIBLE_DEVICES="" c5_run_step 00 "step 00 re-prepare, ${DS}" "${REP0}" "${LOG}" \
      "${PYTHON}" "${C5_EXP_DIR}/scripts/step00_prepare.py" --datasets "${DS}" --tag "${DS}"
    st=$?
    c5_push "${DS}" 00_prepare_after_raptor "${REP0}"
    if [[ "${st}" -ne 0 ]]; then
      say "re-prepare failed; stopping (the contexts may be inconsistent)"
      exit 1
    fi
  else
    say "RAPTOR re-run failed; the reader goes on without RAPTOR (start the script again to resume RAPTOR)"
    overall=1
  fi
fi

# ---------------------------------------------------------------- 02 reader
REP="${C5_REPORT_DIR}/step_02_${DS}_reader.md"
if start_server; then
  c5_run_step 02 "step 02 reader, ${DS}" "${REP}" "${LOG}" "${PYTHON}" "${C5_EXP_DIR}/scripts/step02_reader.py" --dataset "${DS}" --port "${PORT}"
  st=$?
else
  fail_report "${REP}" "step 02 reader, ${DS}" "The vLLM server did not start (log in outputs/rebuttal/c5_end_to_end/vllm_logs/${DS}_${TS}.log; tail above)."
  st=1
fi
stop_server
c5_push "${DS}" 02_reader "${REP}"
[[ "${st}" -ne 0 ]] && overall=1

# ---------------------------------------------------------------- 03 scores (CPU; also on partial answers)
REP="${C5_REPORT_DIR}/step_03_${DS}_score.md"
c5_run_step 03 "step 03 scores, ${DS}" "${REP}" "${LOG}" "${PYTHON}" "${C5_EXP_DIR}/scripts/step03_score.py" --dataset "${DS}"
st=$?
c5_push "${DS}" 03_score "${REP}"
[[ "${st}" -ne 0 ]] && overall=1

say "done, exit ${overall}; reports in ${C5_REPORT_DIR#"${C5_ROOT_DIR}/"}"
echo "${C5_REPORT_DIR}/step_03_${DS}_score.md"
exit "${overall}"
