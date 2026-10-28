#!/usr/bin/env bash
# Pinned ModelScope model fetch, same shape as fetch_hf.sh: modelscope_complete.py diffs remote
# vs local and writes .hfd/missing.locked.txt, mt_download.py fetches it with per-file
# multi-connection splits, repeated rounds until nothing is missing. Every file is admitted only
# when its own sha256 (served by the listing API) reproduces, so this is resumable and verified.
# usage: fetch_modelscope.sh <model_id> <dest> [revision] [connections_per_file]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export DL_LOG=${DL_LOG:-$(dirname "$HERE")/dl_logs/fetch_modelscope-${1//\//_}.log}
source "$HERE/_daemonize.sh"
ID=$1 DEST=$2 REV=${3:-master} CONN=${4:-8}
PY=$HOME/venvs/dl/bin/python
JOBS=${SPEECHRL_HFD_JOBS:-8}
export MS_COMPLETE_CONNECTIONS_PER_FILE=$CONN

mkdir -p "$DEST/.hfd"
MAN="$DEST/.hfd/missing.locked.txt"
for round in 1 2 3 4 5 6 7 8; do
  "$PY" -I "$HERE/modelscope_complete.py" "$ID" "$DEST" "$MAN" "$REV" || { sleep 30; continue; }
  missing=$(grep -c '^http' "$MAN" || true)
  if [ "$missing" = 0 ]; then
    echo "$REV" > "$DEST/.hfd/speechrl-complete-revision"
    echo "MODELSCOPE complete and verified after round $((round - 1)): $ID -> $DEST"
    exit 0
  fi
  echo "MODELSCOPE round $round: $missing files missing"
  "$PY" -I "$HERE/mt_download.py" -i "$MAN" -d "$DEST" -j "$JOBS" --max-tries 10 --timeout 120 || true
done
echo "MODELSCOPE still incomplete after 8 rounds: $ID (re-run to resume)" >&2
exit 1
