#!/usr/bin/env bash
# C8 step 04 (GPU, unattended): NovelHopQA with EVERY book read whole, through Qwen's dual chunk
# attention, so the 6 books beyond 262,144 tokens (skipped by the main run) are read too. Start once:
#   nohup bash experiments/rebuttal/c8_long_context/sh/run_dca_long_books.sh > /dev/null 2>&1 &
# It
#   1. checks step 00 passed (documents, model folder) and that the main run's answers exist;
#   2. makes (once) a separate Python environment with vLLM 0.10.1.1, the last vLLM with the dual
#      chunk attention backend (V0 engine); the main run's environment is not touched;
#   3. uses every free 4-GPU group among 0-3 and 4-7 (TP must divide the model's 28 heads, so 8 GPUs
#      = two servers of 4); waits up to C8_WAIT_MAX_HOURS (default 2) when none is free;
#   4. splits the 16 books into 2 work units (step04_dca.py plan; the 6 long books first in each);
#      each GPU group starts its own vLLM server and runs step 02 on its unit(s), full document only;
#      no new book is started after C8_DCA_DEADLINE_HOURS (default 10);
#   5. scores everything (step04_dca.py score) and pushes after every unit and at the end.
# The main run's answers and reports are never written: this step writes under
#   results/rebuttal/c8_long_context/dca/ and experiments/rebuttal/c8_long_context/reports/dca/.
# Restart the same command to resume (answers are saved per question). Log:
#   experiments/rebuttal/c8_long_context/logs/run_dca_long_books_*.log; vLLM logs under
#   outputs/rebuttal/c8_long_context/vllm_logs/.
set -uo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
# shellcheck disable=SC1091
source "${C8_EXP_DIR}/sh/lib.sh"

DS="novelhopqa"
TS="$(date +%Y%m%d_%H%M%S)"
DCA_REPORTS="${C8_REPORT_DIR}/dca"
DCA_RESULTS="${C8_RESULTS_DIR}/dca"
mkdir -p "${C8_LOG_DIR}" "${DCA_REPORTS}" "${DCA_RESULTS}" "${C8_OUT_DIR}/vllm_logs"
LOG="${C8_LOG_DIR}/run_dca_long_books_${TS}.log"
REPORT="${C8_REPORT_DIR}/step_04_dca_score.md"
PRE_REPORT="${DCA_REPORTS}/step_04_preflight.md"
VLLM_WANT="${C8_DCA_VLLM_VERSION:-0.10.1.1}"
DCA_VENV="${C8_DCA_VENV:-${C8_OUT_DIR}/venv_vllm_${VLLM_WANT//./_}}"
DCA_PY="${C8_DCA_PYTHON:-${DCA_VENV}/bin/python}"
GROUPS_ALL="${C8_DCA_GPU_GROUPS:-0,1,2,3;4,5,6,7}"
BASE_PORT="${C8_DCA_PORT:-18551}"
export VLLM_CACHE_ROOT="${C8_OUT_DIR}/cache/vllm_dca"
export TRITON_CACHE_DIR="${C8_OUT_DIR}/cache/triton_dca"
export TORCHINDUCTOR_CACHE_DIR="${C8_OUT_DIR}/cache/inductor_dca"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
export VLLM_SERVER_DEV_MODE=1
export HF_HUB_OFFLINE=1

say() { echo "[c8-dca] $(date '+%F %T') $*" | tee -a "${LOG}"; }

fail_report() {  # <report.md> <title> <message>
  {
    echo "# C8 step 04 (dual chunk attention) - $2: FAILED"
    echo
    echo "Written $(date '+%F %T') on $(hostname). The main C8 results are unchanged."
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

LOCK="${C8_OUT_DIR}/run_dca_long_books.lock.d"
if ! mkdir "${LOCK}" 2>/dev/null; then
  old="$(cat "${LOCK}/pid" 2>/dev/null)"
  if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
    echo "[c8-dca] another run is going (pid ${old}); stop it first." >&2
    exit 2
  fi
  rm -rf "${LOCK}" && mkdir "${LOCK}"
fi
echo "$$" > "${LOCK}/pid"
CHILDREN=()
cleanup() {
  local p
  for p in "${CHILDREN[@]+"${CHILDREN[@]}"}"; do kill -TERM "${p}" 2>/dev/null; done
  for p in "${CHILDREN[@]+"${CHILDREN[@]}"}"; do wait "${p}" 2>/dev/null; done
  rm -rf "${LOCK}"
}
trap cleanup EXIT
trap 'say "interrupted"; exit 130' INT TERM

say "host $(hostname); GPU groups ${GROUPS_ALL}; vLLM ${VLLM_WANT} in ${DCA_VENV#"${C8_ROOT_DIR}/"}"

# ---------------------------------------------------------------- 1. step 00, main run
s00="$("${PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1])).get('status','?'))" "${C8_REPORT_DIR}/step_00.json" 2>/dev/null || echo missing)"
if [[ "${s00}" != OK* || ! -f "${C8_SETTINGS_FILE}" ]]; then
  fail_report "${PRE_REPORT}" "preflight" "Step 00 has not passed (status ${s00}). Run sh/00_prepare.sh first."
  c8_push "${DS}" 04_dca_preflight "${PRE_REPORT}"; exit 1
fi
# shellcheck disable=SC1090
source "${C8_SETTINGS_FILE}"
MAX_LEN="${C8_DCA_MAX_MODEL_LEN:-${C8_STEP00_MAX_MODEL_LEN}}"
MODEL_DIR="${C8_STEP00_MODEL_PATH:-${C8_MODEL}}"
if [[ ! -f "${MODEL_DIR}/config.json" ]] || ! grep -q dual_chunk_attention_config "${MODEL_DIR}/config.json"; then
  fail_report "${PRE_REPORT}" "preflight" "The model folder ${MODEL_DIR} has no config.json with dual_chunk_attention_config."
  c8_push "${DS}" 04_dca_preflight "${PRE_REPORT}"; exit 1
fi
if [[ ! -s "${C8_RESULTS_DIR}/answers/${DS}/saadi.jsonl" ]]; then
  fail_report "${PRE_REPORT}" "preflight" "The main run's SAADI answers (results/rebuttal/c8_long_context/answers/${DS}/saadi.jsonl) are missing; git pull first."
  c8_push "${DS}" 04_dca_preflight "${PRE_REPORT}"; exit 1
fi
if [[ -f "${C8_OUT_DIR}/run_gpus_4567.lock.d/pid" ]] && kill -0 "$(cat "${C8_OUT_DIR}/run_gpus_4567.lock.d/pid")" 2>/dev/null; then
  fail_report "${PRE_REPORT}" "preflight" "The main C8 GPU run is still going; start this one after it."
  c8_push "${DS}" 04_dca_preflight "${PRE_REPORT}"; exit 1
fi
say "model ${MODEL_DIR}; --max-model-len ${MAX_LEN}"

# ---------------------------------------------------------------- 2. vLLM 0.10.1.1 environment
vllm_ok() { "${DCA_PY}" -c "import sys, vllm; sys.exit(0 if vllm.__version__ == sys.argv[1] else 1)" "${VLLM_WANT}" >/dev/null 2>&1; }
if ! vllm_ok; then
  if [[ -n "${C8_DCA_PYTHON:-}" ]]; then
    fail_report "${PRE_REPORT}" "vLLM environment" "C8_DCA_PYTHON=${C8_DCA_PYTHON} does not have vLLM ${VLLM_WANT}."
    c8_push "${DS}" 04_dca_preflight "${PRE_REPORT}"; exit 1
  fi
  ILOG="${C8_OUT_DIR}/vllm_logs/install_vllm_${VLLM_WANT}_${TS}.log"
  say "installing vLLM ${VLLM_WANT} into ${DCA_VENV} (5-15 min; log ${ILOG#"${C8_ROOT_DIR}/"})"
  rm -rf "${DCA_VENV}"
  made=0
  # vLLM 0.10.1.1 wheels need Python 3.9-3.12.
  for cand in "${C8_DCA_BASE_PYTHON:-}" python3.12 python3.11 python3.10 "${PYTHON}" python3; do
    [[ -z "${cand}" ]] && continue
    command -v "${cand}" >/dev/null 2>&1 || [[ -x "${cand}" ]] || continue
    "${cand}" -c "import sys, venv; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)" 2>/dev/null || continue
    if "${cand}" -m venv "${DCA_VENV}" >> "${ILOG}" 2>&1 && "${DCA_VENV}/bin/python" -m pip --version >> "${ILOG}" 2>&1; then
      made=1; say "environment from $("${cand}" -c 'import sys; print(sys.executable, sys.version.split()[0])')"; break
    fi
    rm -rf "${DCA_VENV}"
  done
  if [[ "${made}" -eq 0 ]] && command -v uv >/dev/null 2>&1; then
    uv venv --seed -p 3.12 "${DCA_VENV}" >> "${ILOG}" 2>&1 && made=1 && say "environment made with uv (Python 3.12)"
  fi
  if [[ "${made}" -eq 0 ]]; then
    fail_report "${PRE_REPORT}" "vLLM environment" "No Python 3.9-3.12 with venv found (and no uv). Set C8_DCA_BASE_PYTHON=/path/to/python3.12 and start again."
    c8_push "${DS}" 04_dca_preflight "${PRE_REPORT}"; exit 1
  fi
  # torch 2.7.1 built for CUDA 12.8, the CUDA vLLM 0.10.1.1 is compiled with.
  if ! { "${DCA_VENV}/bin/python" -m pip install -q --upgrade pip >> "${ILOG}" 2>&1 \
         && "${DCA_VENV}/bin/python" -m pip install "vllm==${VLLM_WANT}" --extra-index-url https://download.pytorch.org/whl/cu128 >> "${ILOG}" 2>&1; } || ! vllm_ok; then
    fail_report "${PRE_REPORT}" "vLLM environment" "pip install vllm==${VLLM_WANT} failed; see ${ILOG#"${C8_ROOT_DIR}/"}: $(tail -n 3 "${ILOG}" | tr '\n' ' ' | cut -c1-300)"
    c8_push "${DS}" 04_dca_preflight "${PRE_REPORT}"; exit 1
  fi
fi
say "vLLM ${VLLM_WANT} ready: $("${DCA_PY}" -c 'import torch, vllm; print("torch", torch.__version__, "cuda", torch.version.cuda)' 2>&1 | tail -n 1)"

# ---------------------------------------------------------------- 3. free GPU groups
group_busy() {  # <gpus> -> prints busy GPUs of the group
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null \
    | awk -F', *' -v lim="${C8_GPU_BUSY_MB}" '$2 > lim {printf "GPU %s: %s MB; ", $1, $2}'
}
IFS=';' read -r -a ALL_GROUPS <<< "${GROUPS_ALL}"
wait_max=$(( ${C8_WAIT_MAX_HOURS:-2} * 3600 )); waited=0
while true; do
  FREE=()
  for g in "${ALL_GROUPS[@]}"; do
    b="$(group_busy "${g}")"
    if [[ -z "${b}" ]]; then FREE+=("${g}"); else [[ $((waited % 600)) -eq 0 ]] && say "group ${g} busy: ${b}"; fi
  done
  [[ "${#FREE[@]}" -gt 0 ]] && break
  if [[ "${waited}" -ge "${wait_max}" ]]; then
    fail_report "${PRE_REPORT}" "waiting for GPUs" "No 4-GPU group free after ${waited} s. Nothing was started."
    c8_push "${DS}" 04_dca_preflight "${PRE_REPORT}"; exit 1
  fi
  sleep 60; waited=$((waited + 60))
done
say "free GPU groups: ${FREE[*]}"

# ---------------------------------------------------------------- 4. plan, servers, step 02
N_SHARDS="${C8_DCA_SHARDS:-2}"
PLAN_OUT="$("${PYTHON}" "${C8_EXP_DIR}/scripts/step04_dca.py" plan --shards "${N_SHARDS}" 2>&1)"
echo "${PLAN_OUT}" >> "${LOG}"
SHARD_LINES=()
while IFS= read -r l; do SHARD_LINES+=("${l}"); done < <(echo "${PLAN_OUT}" | awk -F'\t' 'NF == 2 && $1 ~ /^[0-9]+$/')
if [[ "${#SHARD_LINES[@]}" -ne "${N_SHARDS}" ]]; then
  fail_report "${PRE_REPORT}" "plan" "step04_dca.py plan did not give ${N_SHARDS} work units: $(echo "${PLAN_OUT}" | tail -n 5 | tr '\n' ' ' | cut -c1-400)"
  c8_push "${DS}" 04_dca_preflight "${PRE_REPORT}"; exit 1
fi
rm -f "${PRE_REPORT}" "${PRE_REPORT%.md}.summary.txt"
export C8_DEADLINE_EPOCH=$(( $(date +%s) + ${C8_DCA_DEADLINE_HOURS:-10} * 3600 ))
export C8_SERVED_MODE="dca" C8_SERVED_MAX_MODEL_LEN="${MAX_LEN}"

HELP="$("${DCA_PY}" -m vllm.entrypoints.openai.api_server --help=all 2>&1)"
grep -q -- "--max-model-len" <<< "${HELP}" || HELP="$("${DCA_PY}" -m vllm.entrypoints.openai.api_server --help 2>&1)"
EXTRA=(--enforce-eager --enable-prefix-caching --enable-chunked-prefill)
grep -q -- "--enable-prompt-tokens-details" <<< "${HELP}" && EXTRA+=(--enable-prompt-tokens-details)
grep -q -- "--disable-custom-all-reduce" <<< "${HELP}" && EXTRA+=(--disable-custom-all-reduce)

run_group() {  # <group index> <gpus> <port> <shard ids...>: one server, its units one after the other
  local gi="$1" gpus="$2" port="$3"
  shift 3
  # Not local: the EXIT/TERM traps of this subshell need them after the function returns.
  SERVER_PID="" slog="" tmp=""
  tmp="${C8_TMP_BASE:-/tmp}/c8dca_${USER:-user}_${gi}"
  mkdir -p "${tmp}" && chmod 700 "${tmp}" 2>/dev/null
  export TMPDIR="${tmp}" VLLM_RPC_BASE_PATH="${tmp}" CUDA_VISIBLE_DEVICES="${gpus}"
  ready() { "${PYTHON}" -c "import json,sys,urllib.request
try:
    ids=[m['id'] for m in json.loads(urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/v1/models', timeout=5).read())['data']]
    sys.exit(0 if sys.argv[2] in ids else 1)
except Exception: sys.exit(1)" "${port}" "${C8_MODEL}"; }
  stop() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
      say "[group ${gi}] stopping vLLM (pid ${SERVER_PID})"
      kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null
      local i; for i in $(seq 1 60); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
      kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null
      wait "${SERVER_PID}" 2>/dev/null
    fi
    SERVER_PID=""
  }
  trap stop EXIT
  trap 'stop; exit 143' TERM INT
  start() {
    slog="${C8_OUT_DIR}/vllm_logs/server_dca_g${gi}_${TS}_$(date +%H%M%S).log"
    export C8_SERVER_LOG="${slog}"
    local launcher=(); command -v setsid >/dev/null 2>&1 && launcher=(setsid)
    say "[group ${gi}] starting vLLM ${VLLM_WANT} with dual chunk attention on GPUs ${gpus}, port ${port}; log ${slog#"${C8_ROOT_DIR}/"}"
    (
      export VLLM_ATTENTION_BACKEND=DUAL_CHUNK_FLASH_ATTN VLLM_USE_V1=0
      # shellcheck disable=SC2086
      exec ${launcher[@]+"${launcher[@]}"} "${DCA_PY}" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_DIR}" --served-model-name "${C8_MODEL}" --host 127.0.0.1 --port "${port}" \
        --tensor-parallel-size 4 --dtype bfloat16 --max-model-len "${MAX_LEN}" \
        --gpu-memory-utilization "${C8_GPU_MEMORY_UTILIZATION}" --max-num-seqs "${C8_MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${C8_MAX_NUM_BATCHED_TOKENS}" --seed "${C8_SEED}" "${EXTRA[@]}" ${C8_VLLM_EXTRA_ARGS}
    ) > "${slog}" 2>&1 &
    SERVER_PID=$!
    local w=0
    while ! ready; do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        say "[group ${gi}] vLLM exited during start-up; last lines:"; tail -n 40 "${slog}" | tee -a "${LOG}"; SERVER_PID=""; return 1
      fi
      if [[ "${w}" -ge "${C8_VLLM_STARTUP_TIMEOUT}" ]]; then
        say "[group ${gi}] vLLM not ready after ${w} s"; tail -n 40 "${slog}" | tee -a "${LOG}"; stop; return 1
      fi
      sleep 10; w=$((w + 10))
    done
    say "[group ${gi}] vLLM ready after ${w} s"
  }
  local sid docs line rc rd res
  for sid in "$@"; do
    line="${SHARD_LINES[${sid}]}"; docs="${line#*$'\t'}"
    rd="${DCA_REPORTS}/shard_${sid}"; res="${DCA_RESULTS}/shard_${sid}"
    mkdir -p "${rd}" "${res}"
    local r2="${rd}/step_02_${DS}_reader.md"
    if [[ -z "${SERVER_PID}" ]] || ! ready; then
      stop
      if ! start; then
        fail_report "${r2}" "unit ${sid}, vLLM start" "The vLLM ${VLLM_WANT} server with dual chunk attention did not start on GPUs ${gpus} (log ${slog#"${C8_ROOT_DIR}/"})."
        c8_push "${DS}" "04_dca_unit${sid}" "${r2}"; return 1
      fi
    fi
    grep -a -i -m 3 -E "dual.?chunk|DUAL_CHUNK|attention backend" "${slog}" | cut -c1-240 > "${rd}/server_check.txt"
    [[ -s "${rd}/server_check.txt" ]] || echo "no attention-backend line found in the vLLM log" > "${rd}/server_check.txt"
    say "[group ${gi}] unit ${sid}: books ${docs}"
    rc=0
    C8_RESULTS_DIR="${res}" C8_REPORT_DIR="${rd}" c8_run_step 02 "unit ${sid} reader" "${r2}" "${LOG}" \
      "${PYTHON}" "${C8_EXP_DIR}/scripts/step02_reader.py" --dataset "${DS}" --port "${port}" --conditions full_doc --docs "${docs}" || rc=$?
    if [[ "${rc}" -ne 0 ]] && ! ready; then
      say "[group ${gi}] server down during unit ${sid}; restarting it once"
      stop
      if start; then
        C8_RESULTS_DIR="${res}" C8_REPORT_DIR="${rd}" c8_run_step 02 "unit ${sid} reader" "${r2}" "${LOG}" \
          "${PYTHON}" "${C8_EXP_DIR}/scripts/step02_reader.py" --dataset "${DS}" --port "${port}" --conditions full_doc --docs "${docs}" || rc=$?
      fi
    fi
    say "[group ${gi}] unit ${sid} finished (exit ${rc}): $(head -n 1 "${r2%.md}.summary.txt" 2>/dev/null)"
    c8_push "${DS}" "04_dca_unit${sid}" "${r2}"
  done
  stop
  return 0
}

for gi in "${!FREE[@]}"; do
  mine=()
  for ((s = gi; s < N_SHARDS; s += ${#FREE[@]})); do mine+=("${s}"); done
  [[ "${#mine[@]}" -eq 0 ]] && continue
  ( run_group "${gi}" "${FREE[${gi}]}" "$((BASE_PORT + gi))" "${mine[@]}" ) &
  CHILDREN+=("$!")
  say "group ${gi} (GPUs ${FREE[${gi}]}, port $((BASE_PORT + gi))): units ${mine[*]}"
  sleep 30  # the two servers do not load the weights at the same moment
done
for p in "${CHILDREN[@]}"; do wait "${p}"; done
CHILDREN=()

unit_status() { "${PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1])).get('status','?'))" "${DCA_REPORTS}/shard_$1/step_02_${DS}_reader.json" 2>/dev/null || echo missing; }
# A unit whose server did not start or failed: once more, on the first group that is free now.
RETRY=()
for ((s = 0; s < N_SHARDS; s++)); do
  st="$(unit_status "${s}")"
  [[ "${st}" == OK || "${st}" == PARTIAL* ]] || RETRY+=("${s}")
done
if [[ "${#RETRY[@]}" -gt 0 && "$(date +%s)" -lt "${C8_DEADLINE_EPOCH}" ]]; then
  for gi in "${!FREE[@]}"; do
    [[ -n "$(group_busy "${FREE[${gi}]}")" ]] && continue
    say "retrying units ${RETRY[*]} on GPUs ${FREE[${gi}]}"
    ( run_group "${gi}" "${FREE[${gi}]}" "$((BASE_PORT + gi))" "${RETRY[@]}" ) &
    CHILDREN+=("$!")
    wait "$!"
    CHILDREN=()
    break
  done
fi
overall=0
for ((s = 0; s < N_SHARDS; s++)); do
  st="$(unit_status "${s}")"
  say "unit ${s}: ${st}"
  [[ "${st}" == OK ]] || overall=1
done

# ---------------------------------------------------------------- 5. scores
CUDA_VISIBLE_DEVICES="" c8_run_step 04 "step 04 scores" "${REPORT}" "${LOG}" "${PYTHON}" "${C8_EXP_DIR}/scripts/step04_dca.py" score || overall=1
c8_push "${DS}" 04_dca_score "${REPORT}"
say "done (exit ${overall}): $(head -n 1 "${REPORT%.md}.summary.txt" 2>/dev/null)"
exit "${overall}"
