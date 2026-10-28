#!/usr/bin/env bash
# Clone the study repo (speech-aware-evidence-acquisition) through the flaky proxy: a tree-only
# partial clone (tiny, survives), then the blobs in small parallel batches (_gitbatch.sh), then
# checkout. Plain `git clone` failed 6 times because one ~215 MB stream always dropped midway.
# Resumable; runs detached by default (_daemonize.sh).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/_daemonize.sh"
source "$HERE/_gitbatch.sh"
DEST=$(dirname "$HERE")/speech-aware-evidence-acquisition
URL=https://github.com/chaosxingxc-orion/speech-aware-evidence-acquisition.git

retry() { local n; for n in 1 2 3 4 5 6 7 8 9 10; do "$@" && return 0; echo "retry $n: $*"; sleep $((n * 5)); done; return 1; }

if [ -e "$DEST" ] && [ ! -d "$DEST/.git" ]; then mv "$DEST" "$DEST.broken.$(date +%s)"; fi
if [ ! -d "$DEST/.git" ]; then
  retry git clone --filter=blob:none --no-checkout --depth 1 "$URL" "$DEST" || exit 1
fi
REV=$(git -C "$DEST" rev-parse origin/HEAD 2>/dev/null || git -C "$DEST" rev-parse FETCH_HEAD)
echo "target commit $REV"
git_fetch_missing_blobs "$DEST" "$REV" || exit 1
git -C "$DEST" checkout -q "$REV" -- . && git -C "$DEST" checkout -q -B master "$REV"
echo "HEAD=$(git -C "$DEST" rev-parse HEAD) files=$(git -C "$DEST" ls-files | wc -l)"
echo SAEA_CLONE_DONE
