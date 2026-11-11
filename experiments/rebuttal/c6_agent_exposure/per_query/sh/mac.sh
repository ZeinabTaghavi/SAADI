#!/usr/bin/env bash
# C6 step 2, Mac part: SAADI's full rankings, unpack the servers' exposure archives, write the report.
#
#   bash experiments/rebuttal/c6_agent_exposure/per_query/sh/mac.sh
#
# Archives from extract_server.sh go in outputs/rebuttal/c6_agent_exposure/incoming/*.tar.gz.
# SAADI's rankings are built once (CPU, about a minute) and then reused.
# Report: results/rebuttal/c6_agent_exposure/per_query/REPORT.md
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
cd "$ROOT"
SCRIPTS="$HERE/../scripts"
OUT="$ROOT/outputs/rebuttal/c6_agent_exposure/per_query"
INCOMING="$ROOT/outputs/rebuttal/c6_agent_exposure/incoming"
RES="$ROOT/results/rebuttal/c6_agent_exposure/per_query"
LOG_DIR="$HERE/../logs"
mkdir -p "$OUT" "$INCOMING" "$LOG_DIR"
LOG="$LOG_DIR/mac_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
PY="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""

status=0
if [[ -f "$OUT/saadi_rankings/summary.json" ]] && python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if len(r)==4 and all(x.get('passed') for x in r) else 1)" "$OUT/saadi_rankings/summary.json"; then
  echo "[c6] SAADI full rankings already built and checked"
else
  echo "[c6] building SAADI full rankings"
  "$PY" "$SCRIPTS/c6_saadi_rankings.py" --out "$OUT" --compute-missing || status=1
fi

for a in "$INCOMING"/*.tar.gz; do
  [[ -f "$a" ]] || continue
  echo "[c6] unpacking $(basename "$a")"
  tar -xzf "$a" -C "$OUT" || status=1
done

python3 -I "$SCRIPTS/c6_match.py" --work "$OUT" --results "$RES" || status=1
echo "[c6] report: $RES/REPORT.md (status $status)"
exit $status
