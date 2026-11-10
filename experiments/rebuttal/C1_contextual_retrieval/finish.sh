#!/usr/bin/env bash
# C1: finish everything that is missing, on all four datasets.
#
# For each GPU group (A: GPUs 0-3, LooGLE + NovelHopQA; B: GPUs 4-7, MuSiQue-32K + QASPER-64K):
#   - everything done          -> nothing to start
#   - a run is still going     -> left alone (never a second copy on the same GPUs)
#   - something missing, idle  -> its run script is started in the background; it resumes and does only
#                                 what is missing (contexts, retrieval jobs such as LooGLE/Qwen, reports)
# The final report is written by the group that finishes last. Check progress any time with status.sh.
#
# Start from the repository root so it survives a closed terminal:
#   bash experiments/rebuttal/C1_contextual_retrieval/finish.sh
# Options are passed on (e.g. --dry-run for the CPU test).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"

"${PYTHON}" scripts/rebuttal/c1_contextual_retrieval.py --status "$@"
rc=$?
if [[ ${rc} -eq 0 ]]; then
  echo "C1: everything is done; nothing to start."
  exit 0
fi

for group in A B; do
  "${PYTHON}" scripts/rebuttal/c1_contextual_retrieval.py --status --group "${group}" "$@" > /dev/null
  grc=$?
  if [[ ${group} == A ]]; then script="${HERE}/run_gpus_0123.sh"; else script="${HERE}/run_gpus_4567.sh"; fi
  case ${grc} in
    0) echo "group ${group}: done" ;;
    4) echo "group ${group}: still running; left alone" ;;
    *)
      nohup bash "${script}" "$@" > /dev/null 2>&1 &
      echo "group ${group}: started $(basename "${script}") in the background (pid $!); log in ${C1_EXP_DIR}/logs/"
      ;;
  esac
done
# If both groups were already done but the final report is not, rebuild it.
if [[ ${rc} -ne 4 ]] && "${PYTHON}" scripts/rebuttal/c1_contextual_retrieval.py --status --group A "$@" > /dev/null \
   && "${PYTHON}" scripts/rebuttal/c1_contextual_retrieval.py --status --group B "$@" > /dev/null; then
  "${PYTHON}" scripts/rebuttal/c1_contextual_retrieval.py --report-only "$@"
fi
echo "Check progress with: bash experiments/rebuttal/C1_contextual_retrieval/status.sh"
