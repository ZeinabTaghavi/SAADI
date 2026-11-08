#!/usr/bin/env bash
# Environment of the p1_ledger run (method update tracker, run 1). Sourced by
# every sh/NN_*.sh after `set -euo pipefail`. Nothing here touches GPUs 4-7 or
# any folder of the parallel build.
#
# Layout (all paths relative to the repository root):
#   experiments/method_update/p1_ledger/   this folder: env.sh, sh/, scripts/, reports/, logs/
#   outputs/method_update/p1_ledger/       construction cache, ledger event logs, markers, gate data
#   results/method_update/p1_ledger/       README.md, table.md, table.csv, REPORT_FOR_GDOC.md

P1_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${P1_ROOT_DIR}"
# The main study's environment (PYTHONPATH=src, data/output defaults, HF cache under .cache).
source scripts/env.sh

export P1_PROFILE="p1_ledger"
export P1_EXP_DIR="${P1_ROOT_DIR}/experiments/method_update/${P1_PROFILE}"
export P1_OUT_DIR="${P1_ROOT_DIR}/outputs/method_update/${P1_PROFILE}"
export P1_RESULTS_DIR="${P1_ROOT_DIR}/results/method_update/${P1_PROFILE}"
export P1_MARKER_DIR="${P1_OUT_DIR}/markers"
export P1_LOG_DIR="${P1_EXP_DIR}/logs"
export P1_REPORT_DIR="${P1_EXP_DIR}/reports"
# The study runner writes under ${SADI_OUTPUT_DIR}/novelhopqa_probe; the ledger
# event logs go to ${SADI_OUTPUT_DIR}/ledger (configs/profiles/p1_ledger.yaml).
export SADI_OUTPUT_DIR="${P1_OUT_DIR}"
export P1_LEDGER_DIR="${P1_OUT_DIR}/ledger"

# Seeds and fixed settings (nothing is tuned on the probe).
export P1_SEED=42
export P1_N_BOOKS=10
export P1_BOOTSTRAP_RESAMPLES=10000

# GPUs: group A only. The parallel build uses 4-7 and its own port.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GLOBAL_VISIBLE_DEVICES="${GLOBAL_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
export P1_TENSOR_PARALLEL=4
# vLLM's internal distributed-init port for this tensor-parallel group. The
# other build must use a different VLLM_PORT (it defaults to a random free port
# when unset; set it explicitly there too so the two never collide).
export VLLM_PORT="${VLLM_PORT:-29517}"
# As in the probe's step 5: no CUDA toolkit on the server, PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Models (the construction LLM is fixed; the candidate net is the profile default).
export P1_LLM_MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"
export P1_CANDIDATE_MODEL="${P1_CANDIDATE_MODEL:-urchade/gliner_large-v2.1}"

# Hugging Face cache: the server's shared cache, as in the probe's step 5
# (PROBE_HF_HOME overrides). The gate (step 1) downloads the candidate net into
# it; the build (step 2) runs offline.
export HF_HOME="${PROBE_HF_HOME:-/mnt/cache/taghavi}"
export SAADI_HF_CACHE_ROOT="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Dataset: the frozen probe spec, unchanged (step 2 refuses to start if the hash differs).
export P1_PROBE_DIR="${P1_ROOT_DIR}/experiments/probe_bridge"
export P1_RUN0_OUT_DIR="${P1_ROOT_DIR}/outputs/probe_bridge"   # run 0 (read-only here)
export NOVELHOPQA_REPORT_DIR="${P1_LOG_DIR}/dataset_reports"
export NOVELHOPQA_SUBSET_MODE="${NOVELHOPQA_SUBSET_MODE:-1}"

# Extra Python packages that the study venv lacks (the gate installs `gliner`
# here with --no-deps; torch and transformers come from the venv). Prepended to
# PYTHONPATH so the shared venv is never modified.
export P1_PYDEPS="${P1_OUT_DIR}/pydeps"
export PYTHONPATH="${P1_PYDEPS}:${P1_ROOT_DIR}/src:${P1_ROOT_DIR}/experiments/probe_bridge${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

# The same virtual environment as the main study, when present.
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHON="${PYTHON:-python3}"

mkdir -p "${P1_OUT_DIR}" "${P1_MARKER_DIR}" "${P1_LOG_DIR}" "${P1_REPORT_DIR}" "${P1_RESULTS_DIR}"
