#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
exec "${PYTHON}" -m saadi.cli.main eval-static-statistics-hierarchical-summary-statistics --config "configs/evaluation/updated_chunks/2_4_Hierarchical_Summary_Statistics/${DATASET:-qasper}_default.yaml" "$@"
