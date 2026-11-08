#!/usr/bin/env bash
# p1_ledger step 01: gate. Run from anywhere; executes at the repository root.
# Flags pass through to scripts/step01_gate.py (--force reruns a finished step; see the
# script's --help for its own flags). Refuses to run while the previous step's
# done marker is missing; never asks for input; nohup-safe.
# The LAST line printed is the path of this step's report.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/experiments/method_update/p1_ledger/env.sh"

TS="$(date +%Y%m%d_%H%M%S)"
export P1_LOG_FILE="${P1_LOG_DIR}/01_gate_${TS}.log"
REPORT="${P1_REPORT_DIR}/step_01.md"
# Keep the previous report aside so the path below always shows THIS run.
if [[ -f "${REPORT}" ]]; then
  mv "${REPORT}" "${REPORT%.md}.prev.md"
fi

# The candidate net needs the `gliner` package. The study venv is shared with
# the parallel build and is never modified: gliner goes into $P1_PYDEPS with
# --no-deps (torch, transformers, sentencepiece come from the venv) and env.sh
# prepends that folder to PYTHONPATH. Idempotent; skipped when importable.
if ! "${PYTHON}" -c "import gliner" >/dev/null 2>&1; then
  if [[ "$*" != *"--candidate-net regex"* ]]; then
    echo "Installing gliner into ${P1_PYDEPS} (no dependencies; the venv is left untouched)"
    mkdir -p "${P1_PYDEPS}"
    "${PYTHON}" -m pip install --quiet --no-deps --target "${P1_PYDEPS}" "gliner>=0.2.20" || echo "pip install of gliner failed; the step will report it"
  fi
fi
# The gate downloads the candidate net (and its DeBERTa tokenizer) into HF_HOME;
# the build runs offline afterwards.
export HF_HUB_OFFLINE="${P1_GATE_OFFLINE:-0}"

set +e
"${PYTHON}" "${P1_EXP_DIR}/scripts/step01_gate.py" "$@" 2>&1 | tee "${P1_LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

# A step that died before reporting still leaves a report with the log tail.
if [[ ! -f "${REPORT}" ]]; then
  {
    echo "# p1_ledger - step 01: FAILED before writing its report"
    echo
    echo "Exit code: ${status}. Last 50 log lines:"
    echo
    echo '```'
    tail -n 50 "${P1_LOG_FILE}" || true
    echo '```'
  } > "${REPORT}"
fi

echo "${REPORT}"
exit "${status}"
