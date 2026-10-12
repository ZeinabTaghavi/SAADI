#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
exec "${PYTHON}" "${SADI_ROOT}/scripts/run_study.py" --suite memory_conditioning "$@"
