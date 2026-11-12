#!/usr/bin/env bash
# Table 1 with SAADI's held-out fusion weights (follow-up to C2). CPU only, about a minute.
# Reads C2's saved grid arrays (outputs/rebuttal/C2_heldout_fusion/grid/) and the paper's Table 1;
# writes results/rebuttal/c2_table1/{REPORT.md,results.json,table1_published.csv,table1_heldout.csv}.
# Run from anywhere on the server:   bash experiments/rebuttal/c2_table1/run.sh
# Rerun it after C2's analysis covers Jina and Qwen; they are then included automatically.
# Exit code 0 = done; 1 = stopped (published weights do not reproduce Table 1, or no input).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source scripts/env.sh # PYTHONPATH=src; sets -euo pipefail
set +eu
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
PYTHON="${PYTHON:-python3}"

mkdir -p "${HERE}/logs"
LOG="${HERE}/logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "c2_table1: log ${LOG}"
"${PYTHON}" experiments/rebuttal/c2_table1/scripts/c2_table1.py "$@" 2>&1 | tee -a "${LOG}"
code="${PIPESTATUS[0]}"
echo "c2_table1: exit code ${code}; report results/rebuttal/c2_table1/REPORT.md" | tee -a "${LOG}"
exit "${code}"
