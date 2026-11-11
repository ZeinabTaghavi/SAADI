#!/usr/bin/env bash
# C6 step 1: read-only inventory of saved comparison-method outputs.
#
# Usage (all arguments go to inventory.py; see: ./inventory.sh --help):
#   ./inventory.sh --machine local   --root outputs/other_methods_method
#   ./inventory.sh --machine server1 --root ./SAADI_UMA --out ~/c6_reports
#
# Without --out, reports go to <this folder>/reports. The scan never writes inside
# the scanned roots; the script refuses an --out that lies inside one.
# Needs only python3 (>= 3.6, standard library) and, optionally, git.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
if [[ -z "$PY" ]]; then
  echo "inventory.sh: python3 not found (set PYTHON=/path/to/python3)" >&2
  exit 127
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)'; then
  echo "inventory.sh: need Python >= 3.6, got: $("$PY" --version 2>&1)" >&2
  exit 127
fi

has_out=0
for a in "$@"; do
  case "$a" in
    --out|--out=*) has_out=1 ;;
  esac
done

args=("$@")
if [[ $has_out -eq 0 ]]; then
  args+=(--out "$SCRIPT_DIR/reports")
fi

# -I: isolated mode, so no local file or PYTHON* variable can shadow the standard library.
exec "$PY" -I "$SCRIPT_DIR/inventory.py" "${args[@]}"
