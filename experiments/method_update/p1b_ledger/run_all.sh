#!/usr/bin/env bash
# Run 1b (profile p1b_ledger) end to end, unattended:
#   01_gate (CPU, reports only, never blocks) -> 02_build (GPUs 0-3) -> 03_memory_check
#   -> 04_retrieve -> 05_report.
# Each stage writes reports/step_0N.md (+ .json) and, when it finishes, a done
# marker in outputs/method_update/p1b_ledger/markers/. A finished stage is skipped
# on a rerun unless --force. A failing stage writes its report with the traceback
# and the last 50 log lines; later stages do not run (stage 01 excepted) and
# reports/FINAL.md says where it stopped. The build is resumed once
# automatically after a failure (it continues from the construction cache).
#
# Usage (from anywhere; nohup-safe, never asks for input):
#   nohup bash experiments/method_update/p1b_ledger/run_all.sh > experiments/method_update/p1b_ledger/logs/run_all_nohup.out 2>&1 &
#   bash experiments/method_update/p1b_ledger/run_all.sh --from 03_memory_check
#   bash experiments/method_update/p1b_ledger/run_all.sh --from 02_build --force
# The LAST line printed is the path of reports/FINAL.md.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${EXP_DIR}/env.sh"

FROM=1
FORCE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      [[ $# -ge 2 ]] || { echo "--from needs a stage (01_gate, 02_build, 03_memory_check, 04_retrieve, 05_report)"; exit 2; }
      case "$2" in
        1|01|01_gate|gate) FROM=1 ;;
        2|02|02_build|build) FROM=2 ;;
        3|03|03_memory_check|memory_check) FROM=3 ;;
        4|04|04_retrieve|retrieve) FROM=4 ;;
        5|05|05_report|report) FROM=5 ;;
        *) echo "Unknown stage: $2"; exit 2 ;;
      esac
      shift 2 ;;
    --force) FORCE="--force"; shift ;;
    *) echo "Unknown argument: $1 (use --from <stage> and --force)"; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
export P1_LOG_FILE="${P1_LOG_DIR}/run_all_${TS}.log"
# The stages write their reports where scripts/common_p1b.py says (P1B_OVERRIDE_REPORT_DIR only in local dry runs).
REPORT_DIR="${P1B_OVERRIDE_REPORT_DIR:-${P1_REPORT_DIR}}"
FINAL="${REPORT_DIR}/FINAL.md"
exec > >(tee -a "${P1_LOG_FILE}") 2>&1

echo "== run 1b (p1b_ledger) started $(date '+%Y-%m-%d %H:%M:%S'); from stage ${FROM}; force='${FORCE}'"
echo "== commit $(git -C "${P1B_ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown); log ${P1_LOG_FILE}"
echo "== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} VLLM_PORT=${VLLM_PORT} SADI_OUTPUT_DIR=${SADI_OUTPUT_DIR}"

# gliner: importable from run 1's pydeps (read only) or the venv; otherwise install
# it into run 1b's own pydeps without dependencies. The venv is never modified.
if ! "${PYTHON}" -c "import gliner" >/dev/null 2>&1; then
  echo "== gliner not importable; installing into ${P1_PYDEPS} (--no-deps)"
  mkdir -p "${P1_PYDEPS}"
  "${PYTHON}" -m pip install --quiet --no-deps --target "${P1_PYDEPS}" "gliner==0.2.30" || echo "== pip install of gliner failed; stages 01 and 02 will report it"
fi
# Offline whenever the candidate net is already in the cache (run 1 downloaded it), so
# the same weights are used; online only if it is missing.
if [[ -d "${HF_HUB_CACHE}/models--urchade--gliner_large-v2.1" ]]; then
  export HF_HUB_OFFLINE=1
else
  echo "== the candidate net is not in ${HF_HUB_CACHE}; stage 01 may download it"
  export HF_HUB_OFFLINE=0
fi

STAGE_NAMES=(_ 01_gate 02_build 03_memory_check 04_retrieve 05_report)
STAGE_SCRIPTS=(_ stage01_gate.py stage02_build.py stage03_memory_check.py stage04_retrieve.py stage05_report.py)

write_failure_final() {
  local stage="$1" code="$2"
  {
    echo "# Run 1b (profile p1b_ledger): stopped at stage ${stage}"
    echo
    echo "Stage **${stage}** exited with code ${code} at $(date '+%Y-%m-%d %H:%M:%S'); later stages did not run."
    echo "Its report: \`${REPORT_DIR}/step_${stage:0:2}.md\` (traceback and the last 50 log lines are in it)."
    echo "Resume after fixing: \`bash experiments/method_update/p1b_ledger/run_all.sh --from ${stage}\`"
    echo
    echo "## Stage status"
    echo
    for i in 1 2 3 4 5; do
      local rep="${REPORT_DIR}/step_0${i}.md"
      if [[ -f "${rep}" ]]; then
        echo "- ${STAGE_NAMES[$i]}: $(grep -m1 '^Status:' "${rep}" || echo 'no status line')"
      else
        echo "- ${STAGE_NAMES[$i]}: not run"
      fi
    done
    echo
    echo "## Last 50 lines of the run log"
    echo
    echo '```'
    tail -n 50 "${P1_LOG_FILE}" || true
    echo '```'
  } > "${FINAL}"
}

run_stage() {
  local i="$1"
  local stage_force=""
  [[ "${i}" -ge "${FROM}" ]] && stage_force="${FORCE}"
  echo
  echo "== stage ${STAGE_NAMES[$i]} $(date '+%H:%M:%S')"
  set +e
  "${PYTHON}" "${EXP_DIR}/scripts/${STAGE_SCRIPTS[$i]}" ${stage_force}
  local code=$?
  set -e
  echo "== stage ${STAGE_NAMES[$i]} exit ${code} $(date '+%H:%M:%S')"
  return "${code}"
}

if [[ "${FROM}" -le 1 ]]; then
  if ! run_stage 1; then
    echo "== stage 01_gate failed; it never blocks the build, continuing (see reports/step_01.md)"
  fi
fi

for i in 2 3 4 5; do
  [[ "${i}" -lt "${FROM}" ]] && continue
  if ! run_stage "${i}"; then
    if [[ "${i}" -eq 2 ]]; then
      echo "== the build failed; resuming it once from the construction cache"
      if run_stage 2; then
        continue
      fi
    fi
    code=1
    write_failure_final "${STAGE_NAMES[$i]}" "${code}"
    echo "== run 1b stopped at ${STAGE_NAMES[$i]} $(date '+%Y-%m-%d %H:%M:%S')"
    echo "${FINAL}"
    exit "${code}"
  fi
done

echo "== run 1b finished $(date '+%Y-%m-%d %H:%M:%S')"
echo "${FINAL}"
