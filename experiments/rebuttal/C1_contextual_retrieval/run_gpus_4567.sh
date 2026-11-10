#!/usr/bin/env bash
# C1 Contextual Retrieval, GPUs 4-7: MuSiQue-32K, then QASPER-64K. From the repository root:
#
#   bash experiments/rebuttal/C1_contextual_retrieval/run_gpus_4567.sh
#
# Starts in the background (closing the terminal is fine) and shows the log; Ctrl-C only closes the log.
# If this group is already running, it only shows that run's log. Safe to run again: it resumes and does
# only what is missing. The other GPUs are run_gpus_0123.sh. Progress: status.sh in this folder.

GROUP=B
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"

if [[ "${1:-}" == "--worker" ]]; then
  shift
  exec "${PYTHON}" scripts/rebuttal/c1_contextual_retrieval.py --group "${GROUP}" "$@"
fi

TAG=""
for arg in "$@"; do [[ "${arg}" == --dry-run* ]] && TAG="_dry_run"; done
if pgrep -f "c1_contextual_retrieval.py --group ${GROUP}" > /dev/null; then
  LOG="$(ls -t "${C1_EXP_DIR}"/logs/group_${GROUP}${TAG}_*.log 2> /dev/null | head -1)"
  echo "Group ${GROUP} is already running; showing its log ${LOG} (Ctrl-C closes the view only)."
else
  LOG="${C1_EXP_DIR}/logs/group_${GROUP}${TAG}_$(date +%Y%m%d_%H%M%S).log"
  nohup bash "${BASH_SOURCE[0]}" --worker "$@" > "${LOG}" 2>&1 &
  echo "Group ${GROUP} started (pid $!). Log: ${LOG} (Ctrl-C closes the view only; the run keeps going)."
  sleep 2
fi
[[ -n "${C1_NO_TAIL:-}" || -z "${LOG}" ]] && exit 0
tail -f "${LOG}"
