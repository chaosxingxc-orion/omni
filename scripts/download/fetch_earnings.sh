#!/usr/bin/env bash
# Earnings21 + Earnings22: sparse checkout of revdotcom/speech-datasets at the pinned rev with
# Git LFS, mirroring asset_lock.py fetch_git() for the revdotcom-earnings21-22 fetch group
# ($SPEECHRL_DATA_DIR/datasets/earnings21-22). Resumable: every step is idempotent and git-lfs /
# the pointer materializer skip objects that are already present. Runs detached by default.
#
# The partial clone's missing blobs are completed over plain HTTPS (fix_earnings_blobs.py), NOT
# by `git fetch --stdin`: on this host the git pack protocol to github.com hangs indefinitely
# (a single-blob fetch sat 5+ minutes moving zero bytes, and 10 batched passes stalled at a
# fixed 80 blobs), while raw.githubusercontent.com and api.github.com both respond normally.
# Each blob is still admitted only when `git hash-object` reproduces the requested oid.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/_daemonize.sh"
export PATH="$HOME/.local/bin:$PATH"
DR=${SPEECHRL_DATA_DIR:-/mnt/cephfs/xingchao/omni}
DEST="$DR/datasets/earnings21-22"
REV=c05ab6fd8b4b627d123c922a22a39e993dd37635
MATERIALIZER=/home/xingchao/workspace/omni/exploring-l4-intelligence/scripts/data/materialize_lfs_pointers.py

retry() { local n; for n in 1 2 3 4 5 6 7 8 9 10; do "$@" && return 0; echo "retry $n: $*"; sleep $((n * 5)); done; return 1; }

git lfs install --skip-repo >/dev/null
if [ ! -d "$DEST/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 retry git clone --filter=blob:none --no-checkout \
    https://github.com/revdotcom/speech-datasets.git "$DEST"
fi
# timeout, because a git-protocol fetch here does not fail, it hangs
git -C "$DEST" cat-file -e "$REV^{commit}" 2>/dev/null \
  || retry timeout 600 git -C "$DEST" fetch -q origin "$REV"
git -C "$DEST" sparse-checkout init --cone
git -C "$DEST" sparse-checkout set earnings21 earnings22
retry python3 -I "$HERE/fix_earnings_blobs.py" "$DEST" --rev "$REV" \
  --paths earnings21 earnings22 --jobs 6 --passes 5
GIT_LFS_SKIP_SMUDGE=1 git -C "$DEST" checkout -q --detach "$REV"

# upstream omits the LFS attribute for earnings22 mp3s; add it locally (as the lock records)
ATTR="$DEST/.git/info/attributes"
LINE='earnings22/media/*.mp3 filter=lfs diff=lfs merge=lfs -text'
grep -qxF "$LINE" "$ATTR" 2>/dev/null || echo "$LINE" >> "$ATTR"

# `|| true`: with pipefail, grep finding NO pointers (= done) fails the pipe and set -e would
# abort right there -- which is what kept earnings "FAILED" with every file present.
pointers() { { grep -rlI -m1 '^version https://git-lfs.github.com/spec/v1' "$DEST/earnings21" "$DEST/earnings22" 2>/dev/null || true; } | wc -l; }
# LFS objects: 16 parallel transfers, each retried; already-present objects are skipped.
# Only while pointers remain: once the materializer has replaced them all, `git lfs pull` has
# nothing to do and still fails from this host (2026-10-27: queue marked earnings FAILED with
# every file present; 125/125 LFS files matched their pointer oid+size).
if [ "$(pointers)" != 0 ]; then
  retry git -C "$DEST" -c lfs.concurrenttransfers=16 -c lfs.transfer.maxretries=10 \
    lfs pull --include "earnings21/**,earnings22/**" || true
  retry python3 -I "$MATERIALIZER" "$DEST" earnings21 earnings22 --jobs 8
fi

left=$(pointers)
echo "HEAD=$(git -C "$DEST" rev-parse HEAD) remaining_lfs_pointers=$left"
du -sh "$DEST/earnings21" "$DEST/earnings22"
[ "$left" = 0 ] && echo EARNINGS_DONE
