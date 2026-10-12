#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
exec "${PYTHON}" -m saadi.cli.main eval-cumulative-growth-aggregated-curves --config "configs/evaluation/updated_chunks/3_2_Dataset_Level_Aggregated_Cumulative_Curves/${DATASET:-qasper}_default.yaml" "$@"
