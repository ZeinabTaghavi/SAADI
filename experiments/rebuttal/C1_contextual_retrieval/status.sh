#!/usr/bin/env bash
# C1: what is done and what is missing, per dataset and retriever, and which C1 processes are running.
# Run from anywhere:  bash experiments/rebuttal/C1_contextual_retrieval/status.sh
# Exit code: 0 everything done (final report complete), 1 something missing and nothing running, 4 still running.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"
"${PYTHON}" scripts/rebuttal/c1_contextual_retrieval.py --status "$@"
