#!/usr/bin/env bash
# C5 group A: MuSiQue-32K on GPUs 0-3, its own vLLM server (tensor parallel 4, port
# C5_PORT_MUSIQUE_32K). Runs the RAPTOR re-run on SAADI's 45 groups first (when step 00 found no
# RAPTOR contexts), then the reader for every condition, then the scores; pushes after each stage.
# Start it next to run_gpus_4567.sh (they share no GPU, port or file):
#   nohup bash experiments/rebuttal/c5_end_to_end/sh/run_gpus_0123.sh > /dev/null 2>&1 &
# Restart the same command to resume. Logs: experiments/rebuttal/c5_end_to_end/logs/run_musique_32k_*.log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/../env.sh"
exec bash "${HERE}/run_group.sh" musique_32k "${C5_GPUS_MUSIQUE_32K}" "${C5_PORT_MUSIQUE_32K}"
