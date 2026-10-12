#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
exec "${PYTHON}" -m saadi.cli.main eval-faithfulness-entity-disambiguation --config "configs/evaluation/updated_chunks/1_1_Entity_Disambiguation_Correctness/${DATASET:-qasper}_default.yaml" "$@"
