#!/usr/bin/env bash
# C5 group B: NovelHopQA on GPUs 4-7, its own vLLM server (tensor parallel 4, port
# C5_PORT_NOVELHOPQA). Every comparison output is saved for NovelHopQA, so it runs the reader for
# every condition, then the scores; pushes after each stage.
# Start it next to run_gpus_0123.sh (they share no GPU, port or file):
#   nohup bash experiments/rebuttal/c5_end_to_end/sh/run_gpus_4567.sh > /dev/null 2>&1 &
# Restart the same command to resume. Logs: experiments/rebuttal/c5_end_to_end/logs/run_novelhopqa_*.log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/../env.sh"
exec bash "${HERE}/run_group.sh" novelhopqa "${C5_GPUS_NOVELHOPQA}" "${C5_PORT_NOVELHOPQA}"
