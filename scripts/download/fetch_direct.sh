#!/usr/bin/env bash
# Fetch one content-addressed URL and verify it against its pinned sha256, for lock entries whose
# source.kind is "direct" (no Git revision exists -- the checkpoint IS its hash).
# usage: fetch_direct.sh <url> <dest_dir> <filename> <sha256> [expected_bytes]
# Resumable: mt_download.py continues from the .part file; a verified file short-circuits on re-run.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export DL_LOG=${DL_LOG:-$(dirname "$HERE")/dl_logs/fetch_direct-$3.log}
source "$HERE/_daemonize.sh"
URL=$1 DEST=$2 NAME=$3 SHA=$4 BYTES=${5:-}
PY=$HOME/venvs/dl/bin/python
mkdir -p "$DEST/.hfd"
MARK="$DEST/.hfd/speechrl-complete-revision"
[ -f "$MARK" ] && { echo "already complete: $DEST/$NAME"; exit 0; }

MAN="$DEST/.hfd/direct.locked.txt"
printf '%s\n  dir=%s\n  out=%s\n  split=%s\n  max-connection-per-server=%s\n' \
  "$URL" "$DEST" "$NAME" "${SPEECHRL_HFD_CONNECTIONS_PER_FILE:-8}" \
  "${SPEECHRL_HFD_CONNECTIONS_PER_FILE:-8}" > "$MAN"

for round in 1 2 3 4 5 6; do
  if [ -f "$DEST/$NAME" ]; then
    got=$(sha256sum "$DEST/$NAME" | cut -d' ' -f1)
    size=$(stat -c %s "$DEST/$NAME")
    if [ "$got" = "$SHA" ] && { [ -z "$BYTES" ] || [ "$size" = "$BYTES" ]; }; then
      echo "$SHA" > "$MARK"
      echo "DIRECT verified $NAME sha256=$got bytes=$size (round $((round - 1)))"
      exit 0
    fi
    echo "DIRECT round $round: $NAME sha256=$got want=$SHA bytes=$size want=${BYTES:-any}; refetching"
    rm -f "$DEST/$NAME"
  fi
  echo "DIRECT round $round: fetching $NAME"
  "$PY" -I "$HERE/mt_download.py" -i "$MAN" -d "$DEST" -j 1 --max-tries 10 --timeout 120 || true
done
echo "DIRECT still unverified after 6 rounds: $NAME (re-run to resume)" >&2
exit 1
