#!/usr/bin/env bash
# Environment of the p1b_ledger run (method update tracker, run 1b). Sourced by
# run_all.sh after `set -euo pipefail`. Nothing here touches GPUs 4-7, run 1's
# folders (experiments|outputs|results/method_update/p1_ledger, read only) or
# run 0 (outputs/probe_bridge, read only).
#
# Layout (relative to the repository root):
#   experiments/method_update/p1b_ledger/   env.sh, run_all.sh, scripts/, reports/, logs/
#   outputs/method_update/p1b_ledger/       construction cache, ledger event logs, markers, gate data
#   results/method_update/p1b_ledger/       FINAL tables, REPORT_FOR_GDOC.md, tracker row

P1B_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${P1B_ROOT_DIR}"
# The main study's environment (PYTHONPATH=src, data/output defaults).
source scripts/env.sh

# The measurement code of run 1 (experiments/method_update/p1_ledger/scripts) is
# imported, not copied; it reads its paths from these P1_* variables, so they
# point at run 1b's own folders here.
export P1_PROFILE="p1b_ledger"
export P1_EXP_DIR="${P1B_ROOT_DIR}/experiments/method_update/${P1_PROFILE}"
export P1_OUT_DIR="${P1B_ROOT_DIR}/outputs/method_update/${P1_PROFILE}"
export P1_RESULTS_DIR="${P1B_ROOT_DIR}/results/method_update/${P1_PROFILE}"
export P1_MARKER_DIR="${P1_OUT_DIR}/markers"
export P1_LOG_DIR="${P1_EXP_DIR}/logs"
export P1_REPORT_DIR="${P1_EXP_DIR}/reports"
export SADI_OUTPUT_DIR="${P1_OUT_DIR}"
export P1_LEDGER_DIR="${P1_OUT_DIR}/ledger"
# Run 1 (read only) and run 0 (read only), for the comparison columns.
export P1B_RUN1_OUT_DIR="${P1B_RUN1_OUT_DIR:-${P1B_ROOT_DIR}/outputs/method_update/p1_ledger}"
export P1B_RUN1_REPORT_DIR="${P1B_ROOT_DIR}/experiments/method_update/p1_ledger/reports"
export P1_RUN0_OUT_DIR="${P1B_ROOT_DIR}/outputs/probe_bridge"

# Seeds and fixed settings (nothing is tuned on the probe).
export P1_SEED=42
export P1_N_BOOKS=10
export P1_BOOTSTRAP_RESAMPLES=10000

# GPUs: group A only, tensor parallel 4. The parallel build uses GPUs 4-7.
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export GLOBAL_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export P1_TENSOR_PARALLEL=4
# vLLM's distributed-init port for this tensor-parallel group: not run 1's 29517.
# Override with P1B_VLLM_PORT if the build on GPUs 4-7 uses 29537.
export VLLM_PORT="${P1B_VLLM_PORT:-29537}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Models (fixed).
export P1_LLM_MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"
export P1_CANDIDATE_MODEL="urchade/gliner_large-v2.1"

# Hugging Face cache: the server's shared cache, as in runs 0 and 1.
export HF_HOME="${PROBE_HF_HOME:-/mnt/cache/taghavi}"
export SAADI_HF_CACHE_ROOT="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Dataset: the frozen probe spec, unchanged (stages 01 and 02 refuse to start if its hash differs).
export P1_PROBE_DIR="${P1B_ROOT_DIR}/experiments/probe_bridge"
export NOVELHOPQA_REPORT_DIR="${P1_LOG_DIR}/dataset_reports"
export NOVELHOPQA_SUBSET_MODE="${NOVELHOPQA_SUBSET_MODE:-1}"

# `gliner` is not in the study venv. Run 1 installed it (with --no-deps) into its
# own pydeps folder; that folder is put on PYTHONPATH read only. If it is absent,
# run_all.sh installs gliner into run 1b's own pydeps. The venv is never modified.
export P1_PYDEPS="${P1_OUT_DIR}/pydeps"
export P1B_RUN1_PYDEPS="${P1B_RUN1_OUT_DIR}/pydeps"
export PYTHONPATH="${P1_PYDEPS}:${P1B_RUN1_PYDEPS}:${P1B_ROOT_DIR}/src:${P1B_ROOT_DIR}/experiments/probe_bridge${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

# The same virtual environment as the main study, when present.
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHON="${PYTHON:-python3}"

mkdir -p "${P1_OUT_DIR}" "${P1_MARKER_DIR}" "${P1_LOG_DIR}" "${P1_REPORT_DIR}" "${P1_RESULTS_DIR}"
