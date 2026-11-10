#!/usr/bin/env bash
# Environment of C8 (long-context reader baseline, ICLR rebuttal, reviewer iVU8 Q1).
# Sourced by every sh/*.sh. Paths, the model, the GPUs, the port and the vLLM settings are set
# here; each one can be overridden from the shell, e.g.
#   C8_VLLM_PYTHON=/path/to/venv/bin/python bash experiments/rebuttal/c8_long_context/sh/run_gpus_4567.sh
#
# C8 reads C5's outputs (contexts, labels, answers) and never writes to them. It writes only to
#   experiments/rebuttal/c8_long_context/   (code, reports/, logs/)           -> pushed to GitHub
#   results/rebuttal/c8_long_context/       (tables, answers, REPORT.md)      -> pushed to GitHub
#   outputs/rebuttal/c8_long_context/       (full documents, prompts index, vLLM logs; git-ignored)

C8_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export C8_ROOT_DIR

# C5's environment, read only: its paths (contexts, results), the Hugging Face cache, the reader
# settings C8 must match (max output tokens, temperature, seed) and its choice of Python.
# shellcheck disable=SC1091
source "${C8_ROOT_DIR}/experiments/rebuttal/c5_end_to_end/env.sh"
set +e
cd "${C8_ROOT_DIR}"
# C5 sets TRANSFORMERS_CACHE=${HF_HOME}/transformers; transformers then looks for models there and not
# in ${HF_HOME}/hub, where huggingface_hub downloads them. C8 uses the hub cache only (the model is
# also loaded from its snapshot folder, see step 00).
unset TRANSFORMERS_CACHE

# ---------------------------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------------------------
export C8_EXP_DIR="${C8_ROOT_DIR}/experiments/rebuttal/c8_long_context"
export C8_RESULTS_DIR="${C8_RESULTS_DIR:-${C8_ROOT_DIR}/results/rebuttal/c8_long_context}"
export C8_REPORT_DIR="${C8_REPORT_DIR:-${C8_EXP_DIR}/reports}"
export C8_LOG_DIR="${C8_LOG_DIR:-${C8_EXP_DIR}/logs}"
export C8_OUT_DIR="${C8_OUT_DIR:-${C8_ROOT_DIR}/outputs/rebuttal/c8_long_context}"
# Full document texts and per-query prompt token counts (step 00); the documents are large, so
# they stay here (git-ignored) and the reader builds each prompt from them.
export C8_PREPARED_DIR="${C8_PREPARED_DIR:-${C8_OUT_DIR}/prepared}"
# vLLM settings chosen by step 00 from the measured document lengths (sourced by the GPU script).
export C8_SETTINGS_FILE="${C8_SETTINGS_FILE:-${C8_PREPARED_DIR}/vllm_settings.env}"

# C5 inputs (read only). C5_CONTEXTS_DIR and C5_RESULTS_DIR come from C5's env.sh.
export C8_C5_CONTEXTS_DIR="${C8_C5_CONTEXTS_DIR:-${C5_CONTEXTS_DIR}}"
export C8_C5_RESULTS_DIR="${C8_C5_RESULTS_DIR:-${C5_RESULTS_DIR}}"

# ---------------------------------------------------------------------------------------------
# Queries and conditions (the same queries as C5)
# ---------------------------------------------------------------------------------------------
export C8_DATASETS="${C8_DATASETS:-musique_32k,novelhopqa}"
# full_doc = the whole document then the question; the other three reuse C5's context files.
export C8_CONDITIONS="full_doc,saadi,vanilla,closed_book"
# Timing run before the full run: this many queries per dataset (the first ones of a seed-42
# permutation of the C5 query list; the same permutation defines any subset).
export C8_TIMING_QUERIES="${C8_TIMING_QUERIES:-20}"
# Stop before the full run of a dataset when its estimated total exceeds this many GPU hours
# (wall-clock hours x 4 GPUs).
export C8_MAX_GPU_HOURS="${C8_MAX_GPU_HOURS:-6}"
# Subset (only if you decide so after the timing report): number of queries, empty = all. The
# subset is the first N queries of the seed-42 permutation, the same for all four conditions.
export C8_SUBSET_MUSIQUE_32K="${C8_SUBSET_MUSIQUE_32K:-}"
export C8_SUBSET_NOVELHOPQA="${C8_SUBSET_NOVELHOPQA:-}"

# ---------------------------------------------------------------------------------------------
# Model and decoding (same decoding as C5's reader)
# ---------------------------------------------------------------------------------------------
export C8_MODEL="${C8_MODEL:-Qwen/Qwen2.5-7B-Instruct-1M}"
# Tokenizer used for every count; empty = the model's own. (Only for a test without the model.)
export C8_TOKENIZER="${C8_TOKENIZER:-}"
export C8_MAX_OUTPUT_TOKENS="${C5_MAX_OUTPUT_TOKENS:-64}"
export C8_TEMPERATURE=0
export C8_SEED=42

# ---------------------------------------------------------------------------------------------
# GPUs and the vLLM server (GPUs 4-7 only; a port none of C5's servers uses)
# ---------------------------------------------------------------------------------------------
export C8_GPUS="4,5,6,7"
export C8_TENSOR_PARALLEL=4
export C8_PORT="${C8_PORT:-18541}"
# Python that runs the vLLM server; empty = the same Python as the CPU steps. Books longer than the
# model's native window (262,144 tokens) need vLLM's dual chunk attention backend, which exists
# only in vLLM's V0 engine (vLLM 0.8.5 to 0.10.x). Step 00 checks the installed vLLM.
export C8_VLLM_PYTHON="${C8_VLLM_PYTHON:-}"
# Qwen's recommended serving setup for the 1M model (model card): dual chunk attention with its
# sparse prefill, V0 engine, chunked prefill of 131,072 tokens, no CUDA graphs.
export C8_ATTENTION_BACKEND="${C8_ATTENTION_BACKEND:-DUAL_CHUNK_FLASH_ATTN}"
export C8_VLLM_USE_V1="${C8_VLLM_USE_V1:-0}"
export C8_MAX_NUM_BATCHED_TOKENS="${C8_MAX_NUM_BATCHED_TOKENS:-131072}"
export C8_ENFORCE_EAGER="${C8_ENFORCE_EAGER:-1}"
export C8_ENABLE_PREFIX_CACHING="${C8_ENABLE_PREFIX_CACHING:-1}"
export C8_GPU_MEMORY_UTILIZATION="${C8_GPU_MEMORY_UTILIZATION:-0.90}"
# Sequences in flight. The reader never sends two uncached documents at once (per document: one
# request alone to fill the prefix cache, then the other questions together), so several
# sequences share one cached document.
export C8_MAX_NUM_SEQS="${C8_MAX_NUM_SEQS:-8}"
export C8_READER_CONCURRENCY="${C8_READER_CONCURRENCY:-8}"
# Context window: empty = step 00's choice (longest prompt + output tokens + margin), from
# ${C8_SETTINGS_FILE}. The margin is the larger of these two, rounded up to a multiple of 8192.
export C8_MAX_MODEL_LEN="${C8_MAX_MODEL_LEN:-}"
export C8_CONTEXT_MARGIN_TOKENS="${C8_CONTEXT_MARGIN_TOKENS:-8192}"
export C8_CONTEXT_MARGIN_PCT="${C8_CONTEXT_MARGIN_PCT:-2}"
# GPU memory used by step 00's fit check when nvidia-smi cannot be read (RTX A6000: 49,140 MiB).
export C8_GPU_MEM_MIB="${C8_GPU_MEM_MIB:-49140}"
export C8_VLLM_STARTUP_TIMEOUT="${C8_VLLM_STARTUP_TIMEOUT:-2400}"
# Per-request timeout (s); an uncached 760K-token book takes minutes.
export C8_REQUEST_TIMEOUT="${C8_REQUEST_TIMEOUT:-5400}"
export C8_VLLM_EXTRA_ARGS="${C8_VLLM_EXTRA_ARGS:-}"
# A GPU script refuses to start when one of its GPUs already uses more than this (MB).
export C8_GPU_BUSY_MB="${C8_GPU_BUSY_MB:-2000}"

# ---------------------------------------------------------------------------------------------
# Statistics (fixed; nothing is tuned)
# ---------------------------------------------------------------------------------------------
export C8_BOOTSTRAP_RESAMPLES=10000

# ---------------------------------------------------------------------------------------------
# GitHub: commit and push the C8 folders after every step (never "git add -A")
# ---------------------------------------------------------------------------------------------
export C8_GIT_PUSH="${C8_GIT_PUSH:-1}"
export C8_GIT_REMOTE="${C8_GIT_REMOTE:-origin}"
export C8_GIT_BRANCH="${C8_GIT_BRANCH:-fix/notebook-key-and-install}"
export C8_GIT_MAX_FILE_MB=5

# Python: C5's choice (PYTHON, from C5's env.sh) for the CPU steps.
export C8_VLLM_PY="${C8_VLLM_PYTHON:-${PYTHON}}"
