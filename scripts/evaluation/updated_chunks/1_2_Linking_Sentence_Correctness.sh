#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
exec "${PYTHON}" -m saadi.cli.main eval-faithfulness-linking-sentence --config "configs/evaluation/updated_chunks/1_2_Linking_Sentence_Correctness/${DATASET:-qasper}_default.yaml" "$@"
