#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
exec "${PYTHON}" -m saadi.cli.main eval-cumulative-growth-comparative-plotting --config "configs/evaluation/updated_chunks/3_3_Comparative_Plotting_Across_Datasets/default.yaml" "$@"
