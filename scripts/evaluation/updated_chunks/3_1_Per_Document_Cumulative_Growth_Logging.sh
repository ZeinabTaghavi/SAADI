#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
exec "${PYTHON}" -m saadi.cli.main eval-cumulative-growth-per-document-logging --config "configs/evaluation/updated_chunks/3_1_Per_Document_Cumulative_Growth_Logging/${DATASET:-qasper}_default.yaml" "$@"
