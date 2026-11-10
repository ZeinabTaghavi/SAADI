#!/usr/bin/env bash
# C1 Contextual Retrieval: finish everything that is missing, on all eight GPUs. ONE command, from the repo root:
#
#   bash experiments/rebuttal/C1_contextual_retrieval/run_all_gpus.sh
#
# It stops any C1 run still going (it resumes here; finished contexts and jobs are kept), starts the
# work in the background (safe to close the terminal) and shows the log. Ctrl-C only closes the log view.
# Safe to run again at any time. Progress: bash experiments/rebuttal/C1_contextual_retrieval/status.sh
#
# What runs: contexts with two vLLM engines (GPUs 0-3 and 4-7) sharing the remaining segments, then every
# missing retrieval job over the eight GPUs (Qwen jobs split over GPU pairs), then all reports.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"

if [[ "${1:-}" == "--worker" ]]; then
  shift
  exec "${PYTHON}" scripts/rebuttal/c1_contextual_retrieval.py --all-gpus "$@"
fi

# 1. Stop C1 processes that are still running (group scripts, engines, jobs). SIGINT first so vLLM
#    releases its GPUs cleanly; whatever is left after two minutes is killed.
if pgrep -f "c1_contextual_retrieval.py" > /dev/null; then
  echo "Stopping the C1 run that is still going (it resumes in the new run) ..."
  pkill -INT -f "c1_contextual_retrieval.py"
  for _ in $(seq 1 60); do pgrep -f "c1_contextual_retrieval.py" > /dev/null || break; sleep 2; done
  pkill -KILL -f "c1_contextual_retrieval.py" 2> /dev/null
  sleep 10
fi

# 2. Start the run in the background and follow its log.
TAG=""
for arg in "$@"; do [[ "${arg}" == --dry-run* ]] && TAG="_dry_run"; done
LOG="${C1_EXP_DIR}/logs/all_gpus${TAG}_$(date +%Y%m%d_%H%M%S).log"
nohup bash "${HERE}/run_all_gpus.sh" --worker "$@" > "${LOG}" 2>&1 &
echo "C1 started on all GPUs (pid $!). Log: ${LOG}"
echo "Ctrl-C closes this log view only; the run keeps going."
[[ -n "${C1_NO_TAIL:-}" ]] && exit 0
sleep 2
tail -f "${LOG}"
