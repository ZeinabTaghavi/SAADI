#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
exec "${PYTHON}" -m saadi.cli.main eval-faithfulness-active-entity --config "configs/evaluation/updated_chunks/1_3_Active_Entity_Faithfulness/${DATASET:-qasper}_default.yaml" "$@"
