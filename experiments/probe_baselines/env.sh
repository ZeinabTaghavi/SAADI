#!/usr/bin/env bash
# Environment of the probe baselines (B0-B4 on the persona probe). Sourced by run_all.sh.
# GPUs 4-7 only; SAADI's runs use GPUs 0-3. Writes only to experiments/probe_baselines/
# {reports,logs,third_party}, outputs/probe_baselines/ and results/probe_baselines/.

PBL_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PBL_ROOT_DIR}"
# The main study's environment (PYTHONPATH=src, data defaults).
source scripts/env.sh

export PBL_EXP_DIR="${PBL_ROOT_DIR}/experiments/probe_baselines"
export PBL_OUT_DIR="${PBL_ROOT_DIR}/outputs/probe_baselines"
export PBL_RESULTS_DIR="${PBL_ROOT_DIR}/results/probe_baselines"
export PBL_REPORT_DIR="${PBL_EXP_DIR}/reports"
export PBL_LOG_DIR="${PBL_EXP_DIR}/logs"
export PBL_TP_DIR="${PBL_EXP_DIR}/third_party"

# GPUs 4-7, tensor parallel 4. Never GPUs 0-3.
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PBL_TP=4
# The OpenAI-compatible HTTP port of this run's vLLM server, and vLLM's distributed-init
# port (SAADI run 1 used 29517, run 1b uses 29537).
export PBL_HTTP_PORT="${PBL_HTTP_PORT:-8047}"
export VLLM_PORT="${PBL_VLLM_PORT:-29547}"
export PBL_VLLM_URL="http://127.0.0.1:${PBL_HTTP_PORT}/v1"

# The construction LLM of SAADI's builds, for B2 (contexts) and B3 (OpenIE, triple filter).
export PBL_MODEL_ID="Qwen/Qwen3-30B-A3B-Instruct-2507"
export PBL_MODEL="${PBL_MODEL_ID}"
# Server settings: SAADI's saved runtime (bf16, TP 4, memory utilization 0.8, max_num_seqs 16,
# max_num_batched_tokens 16384, custom all-reduce off), except the window: 262144 tokens,
# Qwen3-2507's native window (decided with the user for B2; SAADI's builds use 32768). The 20%
# left free on each GPU holds NV-Embed-v2 for B3 (HippoRAG loads it with device_map="auto",
# about 4 GB per GPU). Temperature 0 in every request; --generation-config vllm so that the
# model's own sampling defaults (temperature 0.7, top_k 20) never apply.
export PBL_MAX_MODEL_LEN=262144
export PBL_GPU_UTIL="${PBL_GPU_UTIL:-0.8}"
export PBL_SEED=42
export PBL_B2_CONCURRENCY="${PBL_B2_CONCURRENCY:-16}"

# Hugging Face cache: the server's shared cache, as for SAADI's runs.
export HF_HOME="${PROBE_HF_HOME:-/mnt/cache/taghavi}"
export SAADI_HF_CACHE_ROOT="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Third-party code: own virtual environments (their pins clash with the study venv) and
# the LQCA clone, all under third_party/ (created by run_all.sh, not committed).
export PBL_HIPPORAG_PY="${PBL_TP_DIR}/venv_hipporag/bin/python"
export PBL_LQCA_PY="${PBL_TP_DIR}/venv_lqca/bin/python"
export PBL_LQCA_DIR="${PBL_TP_DIR}/LQCA"
export PBL_LQCA_URL="https://github.com/OceannTwT/LQCA.git"
export PBL_LQCA_COMMIT="2d05899479d5ca08635cb544711dde32eea57b79"
# B3 installs HippoRAG the way the user's SAADI_Hipporag repository does (its tested
# dependency set; hipporag 2.0.0a4's own metadata pins openai==1.91.1, which does not exist
# on PyPI) and uses that repository's OpenIE adapter (saadi_hypporag/openie_compat.py).
export PBL_SHR_DIR="${PBL_TP_DIR}/SAADI_Hipporag"
export PBL_SHR_URL="https://github.com/ZeinabTaghavi/SAADI_Hipporag.git"
export PBL_SHR_COMMIT="2fbdf51ccc1654d345ca7a33363eac661b23ebc6"
export TIKTOKEN_CACHE_DIR="${PBL_TP_DIR}/tiktoken_cache"
export NLTK_DATA="${PBL_TP_DIR}/nltk_data"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

# The study venv (vLLM 0.23 for the server; B0, B1, B2, scoring).
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHON="${PYTHON:-python3}"
# Interpreter the third-party venvs are created from (3.10 or newer).
export PBL_BASE_PYTHON="${PBL_BASE_PYTHON:-${PYTHON}}"

mkdir -p "${PBL_OUT_DIR}/markers" "${PBL_RESULTS_DIR}" "${PBL_REPORT_DIR}" "${PBL_LOG_DIR}" "${PBL_TP_DIR}" "${TIKTOKEN_CACHE_DIR}" "${NLTK_DATA}"
