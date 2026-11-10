#!/usr/bin/env bash
# Environment of C5 (end-to-end QA, ICLR rebuttal, reviewers mokb Q2 and iVU8 Q1).
# Sourced by every sh/*.sh. Every path, the reader model, the GPUs and the ports are set here;
# each one can be overridden from the shell, e.g.
#   C5_RAPTOR_DIR=/somewhere/SAADI_Raptor bash experiments/rebuttal/c5_end_to_end/sh/00_prepare.sh
#
# C5 writes only to
#   experiments/rebuttal/c5_end_to_end/   (code, reports/, logs/)          -> pushed to GitHub
#   results/rebuttal/c5_end_to_end/       (tables, answers, REPORT.md)     -> pushed to GitHub
#   outputs/rebuttal/c5_end_to_end/       (contexts with chunk text, vLLM logs, re-runs; git-ignored)
# Everything else (the paper's saved rankings and caches, C1/C2 outputs, the method repos) is read only.

C5_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export C5_ROOT_DIR
cd "${C5_ROOT_DIR}"
# The main study's environment (PYTHONPATH=src). It sets -euo pipefail; the sh/ scripts handle
# exit codes themselves.
# shellcheck disable=SC1091
source scripts/env.sh
set +e

# First existing directory of a list (used for the method repositories, which live next to
# SAADI on the server and under outputs/other_methods_method/ on the Mac).
c5_first_dir() {
  local d
  for d in "$@"; do
    if [[ -n "${d}" && -d "${d}" ]]; then
      echo "${d}"
      return 0
    fi
  done
  echo "${1}"
}

# First existing executable of a list; empty when none (the caller then uses ${PYTHON}).
c5_first_dir_file() {
  local f
  for f in "$@"; do
    if [[ -n "${f}" && -x "${f}" ]]; then
      echo "${f}"
      return 0
    fi
  done
  echo ""
}

export C5_EXP_DIR="${C5_ROOT_DIR}/experiments/rebuttal/c5_end_to_end"
export C5_RESULTS_DIR="${C5_ROOT_DIR}/results/rebuttal/c5_end_to_end"
export C5_REPORT_DIR="${C5_EXP_DIR}/reports"
export C5_LOG_DIR="${C5_EXP_DIR}/logs"
# Large files (contexts with chunk text, re-run method outputs, vLLM server logs): git-ignored.
export C5_OUT_DIR="${C5_OUT_DIR:-${C5_ROOT_DIR}/outputs/rebuttal/c5_end_to_end}"
export C5_CONTEXTS_DIR="${C5_CONTEXTS_DIR:-${C5_OUT_DIR}/contexts}"

# ---------------------------------------------------------------------------------------------
# Datasets and conditions (all queries; nothing is sampled).
# ---------------------------------------------------------------------------------------------
export C5_DATASETS="${C5_DATASETS:-musique_32k,novelhopqa}"
# One base retriever: BM25, the retriever of the paper's agent comparison (Table 2, matched
# exposure: outputs/saadi/reference retrieval_depth sources are all bm25).
export C5_RETRIEVER="bm25"
export C5_TOP_K=10

# ---------------------------------------------------------------------------------------------
# SAADI and Vanilla (read only)
# ---------------------------------------------------------------------------------------------
# The paper's saved per-query rankings and expected hit counts.
export C5_REFERENCE_DIR="${C5_REFERENCE_DIR:-${C5_ROOT_DIR}/outputs/saadi/reference}"
# Published fusion and field weights are read from this config (cached_retrieval block).
export C5_CONFIG="${C5_CONFIG:-${C5_ROOT_DIR}/configs/methods/sadi.yaml}"
# Table 1 Vanilla (reported summary) for the MuSiQue-32K reproduction check.
export C5_VANILLA_TABLE="${C5_VANILLA_TABLE:-${C5_ROOT_DIR}/outputs/other_methods/vanilla/retrieval.csv}"
# Full per-chunk BM25 channel scores in C2's stream format (<dataset>__bm25.parquet + .queries.json).
# Searched in this order (colon separated); computed on CPU with C2's run_stream_job when absent.
export C5_STREAM_DIRS="${C5_STREAM_DIRS:-${C5_ROOT_DIR}/outputs/rebuttal/C2_heldout_fusion/streams:${C5_ROOT_DIR}/experiments/rebuttal/c4_subdoc_control/work/streams:${C5_ROOT_DIR}/outputs/rebuttal/c6_agent_exposure/per_query/c2_streams/streams:${C5_OUT_DIR}/c2_streams/streams}"
export C5_COMPUTE_STREAMS="${C5_COMPUTE_STREAMS:-1}"
# Gold answers. MuSiQue-32K: the prepared qa.json of the paper's run. NovelHopQA: the Hugging
# Face dataset (abhaygupta1266/novelhopqa, through the repo's loader, offline cache), with the
# RAPTOR run's qa_entries.json as fallback; "auto" tries them in that order.
export C5_MUSIQUE_QA="${C5_MUSIQUE_QA:-${C5_ROOT_DIR}/outputs/musique_ablation/32K_expansion/prepared/qa.json}"
export C5_NOVELHOPQA_QA_SOURCE="${C5_NOVELHOPQA_QA_SOURCE:-auto}"

# ---------------------------------------------------------------------------------------------
# Comparison methods (read only; each lives in its own repository on the server)
# ---------------------------------------------------------------------------------------------
# Contextual Retrieval: C1's saved rankings (<dataset>/retrieval/bm25/rankings.pkl.gz, all systems).
export C5_C1_DIR="${C5_C1_DIR:-${C5_ROOT_DIR}/outputs/rebuttal/C1_contextual_retrieval}"
export C5_C1_TABLE="${C5_C1_TABLE:-${C5_ROOT_DIR}/results/rebuttal/C1_contextual_retrieval/tables/per_retriever.csv}"
# RAPTOR: the SAADI_Raptor repository (raptor_10_runs/ and raptor_10_evaluations/ inside it).
export C5_RAPTOR_DIR="${C5_RAPTOR_DIR:-$(c5_first_dir /mnt/data/zeinab/SAADI_Raptor "${HOME}/SAADI_Raptor" "${HOME}/data/SAADI_Raptor" "${C5_ROOT_DIR}/../SAADI_Raptor" "${C5_ROOT_DIR}/outputs/other_methods_method/SAADI_Raptor")}"
# Every copy of the RAPTOR repo that may hold saved runs (searched in this order after C5_RAPTOR_DIR;
# on the server the NovelHopQA run was made in /mnt/data/zeinab/SAADI_Raptor and the MuSiQue/QASPER
# runner lives in /home/iataghav/SAADI_Raptor). The step 00 report names the copy it used.
export C5_RAPTOR_SEARCH="${C5_RAPTOR_SEARCH:-/mnt/data/zeinab/SAADI_Raptor:${HOME}/SAADI_Raptor:${HOME}/data/SAADI_Raptor:${C5_ROOT_DIR}/../SAADI_Raptor:${C5_ROOT_DIR}/outputs/other_methods_method/SAADI_Raptor}"
# Code used for the MuSiQue-32K re-run: empty = the first copy above that has RAPTOR's
# prepared_expanded loader and run-status script.
export C5_RAPTOR_CODE_DIR="${C5_RAPTOR_CODE_DIR:-}"
export C5_RAPTOR_RUNS="${C5_RAPTOR_RUNS:-raptor_10_runs}"
export C5_RAPTOR_EVALS="${C5_RAPTOR_EVALS:-raptor_10_evaluations}"
# A RAPTOR re-run made by C5 (only if a dataset is missing and you decide to run it) goes here
# and is preferred over the saved run.
export C5_RAPTOR_RERUN_DIR="${C5_RAPTOR_RERUN_DIR:-${C5_OUT_DIR}/raptor_rerun}"
# auto = re-run RAPTOR inside run_gpus_0123.sh when step 00 found no RAPTOR contexts for
# MuSiQue-32K (decided with the user: re-run on SAADI's 45 groups); 0 = never.
export C5_RAPTOR_RERUN="${C5_RAPTOR_RERUN:-auto}"
# Python of RAPTOR's environment (umap-learn, scikit-learn, tiktoken, vLLM). Empty = the code
# copy's .venv/bin/python when it exists, else this study's Python. Step 01 checks the imports first.
export C5_RAPTOR_PYTHON="${C5_RAPTOR_PYTHON:-}"
# Agent: A-RAG (see the step 00 report for why). Its repository and the two saved runs that
# reproduce Table 2 exactly (C6 per-query match).
export C5_AGENT="a_rag"
export C5_ARAG_DIR="${C5_ARAG_DIR:-$(c5_first_dir /mnt/data/zeinab/SAADI_A_RAG "${HOME}/SAADI_A_RAG" "${HOME}/data/SAADI_A_RAG" "${C5_ROOT_DIR}/../SAADI_A_RAG" "${C5_ROOT_DIR}/outputs/other_methods_method/SAADI_A_RAG")}"
export C5_ARAG_SEARCH="${C5_ARAG_SEARCH:-/mnt/data/zeinab/SAADI_A_RAG:${HOME}/SAADI_A_RAG:${HOME}/data/SAADI_A_RAG:${C5_ROOT_DIR}/../SAADI_A_RAG:${C5_ROOT_DIR}/outputs/other_methods_method/SAADI_A_RAG}"
export C5_ARAG_RUN_MUSIQUE_32K="${C5_ARAG_RUN_MUSIQUE_32K:-musique_32k_a_rag_raw_f06714eda6267987}"
export C5_ARAG_RUN_NOVELHOPQA="${C5_ARAG_RUN_NOVELHOPQA:-novelhopqa_a_rag_raw_479355014fce5066}"
# Paper summaries the comparison methods must reproduce (Table 2 / fixed-vs-agentic table).
export C5_OTHER_METHODS_DIR="${C5_OTHER_METHODS_DIR:-${C5_ROOT_DIR}/outputs/other_methods}"

# ---------------------------------------------------------------------------------------------
# Reader (one model, one prompt, greedy) and the two vLLM servers
# ---------------------------------------------------------------------------------------------
export C5_READER_MODEL="${C5_READER_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
export C5_MAX_OUTPUT_TOKENS="${C5_MAX_OUTPUT_TOKENS:-64}"
export C5_TEMPERATURE=0
export C5_SEED=42
# Longest prompt must fit; step 00 reports the longest prompt of every condition and fails if
# one does not fit in C5_MAX_MODEL_LEN - C5_MAX_OUTPUT_TOKENS.
export C5_MAX_MODEL_LEN="${C5_MAX_MODEL_LEN:-32768}"
export C5_TENSOR_PARALLEL=4
export C5_GPU_MEMORY_UTILIZATION="${C5_GPU_MEMORY_UTILIZATION:-0.85}"
export C5_MAX_NUM_SEQS="${C5_MAX_NUM_SEQS:-64}"
# Group A: MuSiQue-32K on GPUs 0-3; group B: NovelHopQA on GPUs 4-7. Different ports, different
# files; the two GPU scripts can run at the same time.
export C5_GPUS_MUSIQUE_32K="${C5_GPUS_MUSIQUE_32K:-0,1,2,3}"
export C5_GPUS_NOVELHOPQA="${C5_GPUS_NOVELHOPQA:-4,5,6,7}"
export C5_PORT_MUSIQUE_32K="${C5_PORT_MUSIQUE_32K:-18531}"
export C5_PORT_NOVELHOPQA="${C5_PORT_NOVELHOPQA:-18532}"
export C5_VLLM_STARTUP_TIMEOUT="${C5_VLLM_STARTUP_TIMEOUT:-1800}"
# Extra vLLM server flags, if your vLLM version needs any (none by default; every sampling
# setting is sent with each request, so the model's generation_config.json does not matter).
export C5_VLLM_EXTRA_ARGS="${C5_VLLM_EXTRA_ARGS:-}"
# Requests in flight from the reader client.
export C5_READER_CONCURRENCY="${C5_READER_CONCURRENCY:-48}"
# A GPU script refuses to start when one of its GPUs already uses more than this (MB).
export C5_GPU_BUSY_MB="${C5_GPU_BUSY_MB:-2000}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Hugging Face cache: the server's shared cache (as for SAADI's runs, C1 and C2), offline.
if [[ -n "${C5_HF_HOME:-}" ]]; then
  export HF_HOME="${C5_HF_HOME}"
elif [[ -d /mnt/cache/taghavi ]]; then
  export HF_HOME=/mnt/cache/taghavi
fi
export SAADI_HF_CACHE_ROOT="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE="${C5_HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${C5_HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${C5_HF_HUB_OFFLINE:-1}"

# ---------------------------------------------------------------------------------------------
# Statistics (fixed; nothing is tuned)
# ---------------------------------------------------------------------------------------------
export C5_BOOTSTRAP_RESAMPLES=10000

# ---------------------------------------------------------------------------------------------
# GitHub: commit and push the C5 folders after every step (never "git add -A")
# ---------------------------------------------------------------------------------------------
export C5_GIT_PUSH="${C5_GIT_PUSH:-1}"
export C5_GIT_REMOTE="${C5_GIT_REMOTE:-origin}"
export C5_GIT_BRANCH="${C5_GIT_BRANCH:-fix/notebook-key-and-install}"
export C5_GIT_MAX_FILE_MB=5

export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
# The study venv, when present (python 3.12, torch + vLLM + transformers).
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${C5_EXP_DIR}/scripts:${PYTHONPATH}"

mkdir -p "${C5_REPORT_DIR}" "${C5_LOG_DIR}" "${C5_RESULTS_DIR}" "${C5_OUT_DIR}" "${C5_CONTEXTS_DIR}"
