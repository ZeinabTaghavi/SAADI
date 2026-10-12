#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
exec "${PYTHON}" -m saadi.cli.main eval-static-statistics-chunk-level-changes --config "configs/evaluation/updated_chunks/2_1_Chunk_Level_Changes/${DATASET:-qasper}_default.yaml" "$@"
