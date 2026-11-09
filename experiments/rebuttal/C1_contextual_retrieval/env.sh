#!/usr/bin/env bash
# Environment of C1 (Contextual Retrieval on SAADI's four benchmarks). Sourced by
# run_gpus_0123.sh (group A) and run_gpus_4567.sh (group B). Writes only to
# experiments/rebuttal/C1_contextual_retrieval/{reports,logs}, outputs/rebuttal/C1_contextual_retrieval/
# and results/rebuttal/C1_contextual_retrieval/.

C1_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${C1_ROOT_DIR}"
# The main study's environment (PYTHONPATH=src, data defaults). It sets -euo pipefail;
# the run scripts handle exit codes themselves.
source scripts/env.sh
set +e

# Hugging Face cache: the server's shared cache, as for SAADI's runs and C2 (weights must be there).
export HF_HOME="${C1_HF_HOME:-/mnt/cache/taghavi}"
export SAADI_HF_CACHE_ROOT="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE="${C1_HF_HUB_OFFLINE:-1}"   # weights are checked in preflight; never download mid-run
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

# The study venv (python 3.12, torch + vLLM + transformers).
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHON="${PYTHON:-python3}"

# vLLM as in B2's server run (same model, same runtime).
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Optional overrides (defaults are SAADI's saved runtime): C1_MAX_NUM_SEQS (16), C1_GPU_MEMORY_UTILIZATION (0.8).

export C1_EXP_DIR="${C1_ROOT_DIR}/experiments/rebuttal/C1_contextual_retrieval"
mkdir -p "${C1_EXP_DIR}/logs" "${C1_EXP_DIR}/reports" outputs/rebuttal/C1_contextual_retrieval
