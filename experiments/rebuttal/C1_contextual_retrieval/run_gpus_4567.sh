#!/usr/bin/env bash
# C1 Contextual Retrieval, group B: GPUs 4-7 (never 0-3). MuSiQue-32K, then QASPER-64K.
#
# Start from the repository root so it survives a closed terminal:
#   nohup bash experiments/rebuttal/C1_contextual_retrieval/run_gpus_4567.sh > /dev/null 2>&1 &
# or inside tmux:
#   bash experiments/rebuttal/C1_contextual_retrieval/run_gpus_4567.sh
# Options (passed on to scripts/rebuttal/c1_contextual_retrieval.py):
#   --dry-run        CPU test: 2 documents per dataset, BM25 only, stub contexts
#   --no-c2          recompute the memory and summary channels instead of reusing C2's streams
# Rerunning the same command resumes: finished contexts, jobs and datasets are skipped.
# The log goes to experiments/rebuttal/C1_contextual_retrieval/logs/ (this script tees it there);
# per-step logs are next to it; reports in experiments/rebuttal/C1_contextual_retrieval/reports/.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"

TAG=""
for arg in "$@"; do [[ "${arg}" == "--dry-run" ]] && TAG="_dry_run"; done
LOG="${C1_EXP_DIR}/logs/group_B${TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "C1 group B: log ${LOG}"
"${PYTHON}" scripts/rebuttal/c1_contextual_retrieval.py --group B "$@" 2>&1 | tee -a "${LOG}"
exit "${PIPESTATUS[0]}"
