#!/usr/bin/env bash
set -euo pipefail

SADI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SADI_CACHE_DIR="${SADI_CACHE_DIR:-${SADI_ROOT}/.cache}"
export SADI_DATA_DIR="${SADI_DATA_DIR:-${SADI_ROOT}/data}"
export SADI_OUTPUT_DIR="${SADI_OUTPUT_DIR:-${SADI_ROOT}/outputs}"
export PYTHONPATH="${SADI_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export SAADI_CACHE_ROOT="${SADI_CACHE_DIR}"
export SAADI_HF_CACHE_ROOT="${SADI_CACHE_DIR}/huggingface"
export HF_HOME="${SAADI_HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export NOVELHOPQA_BOOKS_ROOT="${NOVELHOPQA_BOOKS_ROOT:-${SADI_DATA_DIR}/novelhopqa/book-corpus-root}"
export NOVELHOPQA_REPORT_DIR="${NOVELHOPQA_REPORT_DIR:-${SADI_OUTPUT_DIR}/novelhopqa/dataset_reports}"
export TOKENIZERS_PARALLELISM=false
PYTHON="${PYTHON:-python3}"
cd "${SADI_ROOT}"
