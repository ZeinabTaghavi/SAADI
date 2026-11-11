#!/usr/bin/env bash
# C6 step 2, server part: read each agent's saved per-query outputs (read only) and pack them.
#
# Copy this file and ../scripts/c6_extract.py into one folder on the server (e.g. ~/c6_tools), then:
#   bash ~/c6_tools/extract_server.sh ~/data      # gpu03: SAADI_A_RAG, SAADI_RAG_R1, SAADI_Agent_R, SAADI_GRIP
#   bash ~/c6_tools/extract_server.sh ~           # gpu01: SAADI_UMA
#
# It scans whichever of the five agent repos exist in the given folder and writes one archive:
#   ~/c6_exposure_<host>.tar.gz
# Needs only python3 (standard library). No GPU. Nothing inside the agent repos is changed.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/c6_extract.py"
[[ -f "$SCRIPT" ]] || SCRIPT="$HERE/../scripts/c6_extract.py"
BASE="${1:-$HOME/data}"
HOST="$(hostname -s 2>/dev/null || hostname)"
OUT="$HOME/c6_exposure_$HOST"
ARCHIVE="$HOME/c6_exposure_$HOST.tar.gz"
mkdir -p "$OUT"
echo "[c6] looking for agent repos in $BASE; writing $OUT"

found=0 status=0
for pair in a_rag:SAADI_A_RAG rag_r1:SAADI_RAG_R1 agentic_r:SAADI_Agent_R grip:SAADI_GRIP uma:SAADI_UMA; do
  m="${pair%%:*}"; repo="$BASE/${pair#*:}"
  [[ -d "$repo/outputs" ]] || continue
  found=$((found + 1))
  echo "[c6] $m: $repo"
  python3 -I "$SCRIPT" --method "$m" --repo "$repo" --out "$OUT/exposure" || status=1
done
if [[ $found -eq 0 ]]; then
  echo "[c6] no agent repo (SAADI_A_RAG, SAADI_RAG_R1, SAADI_Agent_R, SAADI_GRIP, SAADI_UMA) with outputs/ in $BASE" >&2
  exit 1
fi
tar -czf "$ARCHIVE" -C "$OUT" exposure && echo "[c6] archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
echo "[c6] done: $found repo(s), status $status (0 = ok)"
exit $status
