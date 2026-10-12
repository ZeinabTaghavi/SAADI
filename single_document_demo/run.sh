#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GLOBAL_VISIBLE_DEVICES="${GLOBAL_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -n "${SAADI_HF_CACHE_ROOT:-}" ]]; then
  export HF_HOME="${HF_HOME:-${SAADI_HF_CACHE_ROOT}}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
fi

exec "${PYTHON_BIN:-python}" single_document_demo/run.py "$@"
