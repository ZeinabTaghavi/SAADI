#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
exec "${PYTHON}" -m saadi.cli.main eval-cumulative-growth-reporting-summaries --config "configs/evaluation/updated_chunks/3_4_Methodology_Summary_and_Reporting_Outputs/default.yaml" "$@"
