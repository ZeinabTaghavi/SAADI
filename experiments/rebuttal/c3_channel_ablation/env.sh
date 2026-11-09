#!/usr/bin/env bash
# Environment of C3 (channel ablation by re-scoring, ICLR rebuttal). Sourced by every
# sh/NN_*.sh. CPU only: no GPU, no LLM, no network. Writes only to
#   experiments/rebuttal/c3_channel_ablation/   (this folder: reports/, logs/, work/)
#   results/rebuttal/c3_channel_ablation/       (tables and the final report)
# Everything else (the paper's saved rankings, construction caches, C2's channel scores)
# is read only.

C3_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export C3_ROOT_DIR
cd "${C3_ROOT_DIR}"
# The main study's environment (PYTHONPATH=src, SADI_* defaults the configs expand).
# It sets -euo pipefail; the sh/ scripts handle exit codes themselves.
# shellcheck disable=SC1091
source scripts/env.sh
set +e

export C3_EXP_DIR="${C3_ROOT_DIR}/experiments/rebuttal/c3_channel_ablation"
export C3_RESULTS_DIR="${C3_ROOT_DIR}/results/rebuttal/c3_channel_ablation"
export C3_REPORT_DIR="${C3_EXP_DIR}/reports"
export C3_LOG_DIR="${C3_EXP_DIR}/logs"
# Large intermediate files (BM25 channel scores, per-query hit ranks); git-ignored.
export C3_WORK_DIR="${C3_EXP_DIR}/work"

# Input: per-channel scores of the dense retrievers, in C2's stream format
# (<dataset>__<retriever>.parquet + .queries.json). A parameter, so C3 reads a later
# recomputation with no code change: C3_STREAMS_DIR=/path/to/streams bash sh/01_check.sh
export C3_STREAMS_DIR="${C3_STREAMS_DIR:-${C3_ROOT_DIR}/outputs/rebuttal/C2_heldout_fusion/streams}"
# C2's job markers (DONE / FAILED), read only to report why a pair has no streams.
export C3_STREAMS_MARKERS_DIR="${C3_STREAMS_MARKERS_DIR:-$(dirname "${C3_STREAMS_DIR}")/markers}"

# Published fusion and field weights are read from this config (cached_retrieval block).
export C3_CONFIG="${C3_CONFIG:-${C3_ROOT_DIR}/configs/methods/sadi.yaml}"

# Fixed settings: nothing is tuned.
export C3_SEED=42
export C3_BOOTSTRAP_RESAMPLES=10000
export C3_DATASETS="${C3_DATASETS:-loogle,musique_32k,qasper_64k,novelhopqa}"
export C3_RETRIEVERS="${C3_RETRIEVERS:-bm25,contriever,bge_m3,jina,qwen}"

# CPU only.
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

# The study venv, when present.
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${C3_EXP_DIR}/scripts:${PYTHONPATH}"

mkdir -p "${C3_REPORT_DIR}" "${C3_LOG_DIR}" "${C3_WORK_DIR}" "${C3_RESULTS_DIR}"
