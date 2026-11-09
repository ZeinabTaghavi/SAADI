#!/usr/bin/env bash
# C2 held-out fusion weights, group B: GPUs 4-7 (never 0-3).
#
# Start from the repository root so it survives a closed terminal:
#   nohup bash experiments/rebuttal/C2_heldout_fusion/run_gpus_4567.sh > /dev/null 2>&1 &
# or inside tmux:
#   bash experiments/rebuttal/C2_heldout_fusion/run_gpus_4567.sh
# Options (passed on to scripts/rebuttal/c2_heldout_fusion.py):
#   --analysis-only   rerun the CPU analysis only (needs all 20 DONE markers)
#   --dry-run         CPU test: BM25 only, 2 documents per dataset, reduced grid
# The log goes to experiments/rebuttal/C2_heldout_fusion/logs/ (this script tees it there itself);
# per-job logs are next to it; the report is reports/group_B.md and .json.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"

TAG=""
for arg in "$@"; do [[ "${arg}" == "--dry-run" ]] && TAG="_dry_run"; done
LOG="${C2_EXP_DIR}/logs/group_B${TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "C2 group B: log ${LOG}"
"${PYTHON}" scripts/rebuttal/c2_heldout_fusion.py --group B "$@" 2>&1 | tee -a "${LOG}"
exit "${PIPESTATUS[0]}"
