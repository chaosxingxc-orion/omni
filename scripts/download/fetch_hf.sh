#!/usr/bin/env bash
# Pinned HF repo fetch, mirroring exploring-l4-intelligence asset_lock.py fetch_hf():
# hf_complete.py diffs remote vs local and writes .hfd/missing.locked.txt (aria2 format),
# then mt_download.py (aria2c -j stand-in) fetches it; repeated rounds until nothing is missing,
# then verify_hf.py checks sha256 against the LFS oids before the completion marker is written.
# Resumable: re-running only fetches what is missing; partial files continue from .part*.
# Runs detached by default (see _daemonize.sh); run_queue.sh runs it inline.
# usage: fetch_hf.sh <repo_id> <dest> <revision> <model|dataset> [connections_per_file]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export DL_LOG=${DL_LOG:-$(dirname "$HERE")/dl_logs/fetch_hf-${1//\//_}.log}
source "$HERE/_daemonize.sh"
REPO=$1 DEST=$2 REV=$3 TYPE=$4 CONN=${5:-${SPEECHRL_HFD_CONNECTIONS_PER_FILE:-1}}
DIFFER=/home/xingchao/workspace/omni/exploring-l4-intelligence/scripts/data/hf_complete.py
PY=$HOME/venvs/dl/bin/python
# Official huggingface.co, not hf-mirror (owner 2026-10-27). Measured through the corporate proxy the same
# evening: 14.5-15.1 MB/s per connection, 108 MB/s over 8 ranged connections.
export HF_ENDPOINT=${SPEECHRL_HF_ENDPOINT:-https://huggingface.co}
export HF_COMPLETE_REVISION=$REV HF_COMPLETE_CONNECTIONS_PER_FILE=$CONN
JOBS=${SPEECHRL_HFD_JOBS:-16}

mkdir -p "$DEST/.hfd"
MAN="$DEST/.hfd/missing.locked.txt"
for round in 1 2 3 4 5 6; do
  "$PY" -I "$DIFFER" "$REPO" "$DEST" "$MAN" "$TYPE" || { sleep 30; continue; }
  missing=$(grep -c '^http' "$MAN" || true)
  if [ "$missing" = 0 ]; then
    if "$PY" -I "$HERE/verify_hf.py" "$REPO" "$DEST" "$REV" "$TYPE"; then
      echo "$REV" > "$DEST/.hfd/speechrl-complete-revision"
      echo "HF complete and verified after round $((round - 1)): $REPO -> $DEST"
      exit 0
    fi
    continue
  fi
  echo "HF round $round: $REPO missing_files=$missing"
  "$PY" -I "$HERE/mt_download.py" -i "$MAN" -d "$DEST" -j "$JOBS" --max-tries 10 --timeout 120 || true
done
echo "HF still incomplete after 6 rounds: $REPO (re-run to resume)" >&2
exit 1
