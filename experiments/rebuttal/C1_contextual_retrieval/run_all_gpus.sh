#!/usr/bin/env bash
# C1 Contextual Retrieval on all eight GPUs at once: everything still missing on every dataset.
#
#   1. contexts: two vLLM engines (GPUs 0-3 and 4-7, TP 4 each, same settings as before) share the
#      remaining segments of all datasets, longest first to the less loaded engine; one file per engine
#   2. retrieval: every missing job over the eight GPUs; Qwen3-Embedding-8B jobs split by document
#      over two GPU pairs, then merged (gate, rankings); BM25 on CPU
#   3. per-dataset reports and the final report
#
# Needs all eight GPUs free and no group script running (it takes both group locks; exit code 3 if
# one is held, 2 if a GPU is busy). Rerunning resumes. Start from the repository root:
#   nohup bash experiments/rebuttal/C1_contextual_retrieval/run_all_gpus.sh > /dev/null 2>&1 &
# Progress: bash experiments/rebuttal/C1_contextual_retrieval/status.sh
# Report of this run: experiments/rebuttal/C1_contextual_retrieval/reports/group_all_gpus.md

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"

TAG=""
for arg in "$@"; do [[ "${arg}" == "--dry-run" ]] && TAG="_dry_run"; done
LOG="${C1_EXP_DIR}/logs/all_gpus${TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "C1 all GPUs: log ${LOG}"
"${PYTHON}" scripts/rebuttal/c1_contextual_retrieval.py --all-gpus "$@" 2>&1 | tee -a "${LOG}"
exit "${PIPESTATUS[0]}"
