#!/usr/bin/env bash
# Environment of C4 (sub-document control, ICLR rebuttal, reviewer iVU8 Q2). Sourced by
# every sh/NN_*.sh. CPU only: no GPU, no LLM, no network, no index rebuilt. Writes only to
#   experiments/rebuttal/c4_subdoc_control/   (this folder: reports/, logs/, work/)
#   results/rebuttal/c4_subdoc_control/       (sub-document maps, tables, the final report)
# Everything else (prepared concatenated documents, construction caches, the paper's saved
# rankings, C2's channel scores) is read only.

C4_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export C4_ROOT_DIR
cd "${C4_ROOT_DIR}"
# The main study's environment (PYTHONPATH=src). It sets -euo pipefail; the sh/ scripts
# handle exit codes themselves.
# shellcheck disable=SC1091
source scripts/env.sh
set +e

export C4_EXP_DIR="${C4_ROOT_DIR}/experiments/rebuttal/c4_subdoc_control"
export C4_RESULTS_DIR="${C4_ROOT_DIR}/results/rebuttal/c4_subdoc_control"
export C4_REPORT_DIR="${C4_EXP_DIR}/reports"
export C4_LOG_DIR="${C4_EXP_DIR}/logs"
# Large intermediate files (BM25 channel scores, per-query ranks); git-ignored.
export C4_WORK_DIR="${C4_EXP_DIR}/work"
export C4_MAP_DIR="${C4_RESULTS_DIR}/subdoc_map"

# The paper's saved per-query rankings and expected hit counts (read only).
export C4_REFERENCE_DIR="${C4_REFERENCE_DIR:-${C4_ROOT_DIR}/outputs/saadi/reference}"
export C4_COMPARISON_DIR="${C4_COMPARISON_DIR:-${C4_ROOT_DIR}/outputs/saadi/comparison_inputs}"

# Input: full per-chunk channel scores of the dense retrievers, in C2's stream format
# (<setting>__<retriever>.parquet + .queries.json, e.g. musique_32k__contriever.parquet).
# A parameter, so C4 reads a later recomputation with no code change:
#   C4_DENSE_STREAMS_DIR=/path/to/streams bash sh/04_dense.sh
export C4_DENSE_STREAMS_DIR="${C4_DENSE_STREAMS_DIR:-${C4_ROOT_DIR}/outputs/rebuttal/C2_heldout_fusion/streams}"

# Published fusion and field weights are read from this config (cached_retrieval block).
export C4_CONFIG="${C4_CONFIG:-${C4_ROOT_DIR}/configs/methods/sadi.yaml}"

# Fixed settings: nothing is tuned.
export C4_SEED=42
export C4_BOOTSTRAP_RESAMPLES=10000
# Every concatenated setting of the paper (MuSiQue and QASPER, 32K to 256K tokens).
export C4_SETTINGS="${C4_SETTINGS:-musique_32k,musique_64k,musique_128k,musique_256k,qasper_32k,qasper_64k,qasper_128k,qasper_256k}"
export C4_RETRIEVERS="${C4_RETRIEVERS:-bm25,contriever,bge_m3,jina,qwen}"
# Main query set of the inside-sub-document measures: R (labels re-derived inside the gold
# sub-document, every query) or S (strict: only queries whose paper evidence is all inside).
# Both are always computed and reported; this only picks which one leads.
export C4_MAIN_SET="${C4_MAIN_SET:-R}"

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
export PYTHONPATH="${C4_EXP_DIR}/scripts:${PYTHONPATH}"

mkdir -p "${C4_REPORT_DIR}" "${C4_LOG_DIR}" "${C4_WORK_DIR}" "${C4_RESULTS_DIR}" "${C4_MAP_DIR}"
