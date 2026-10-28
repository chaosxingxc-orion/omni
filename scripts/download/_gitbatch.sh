#!/usr/bin/env bash
# Sourced helper: fetch the blobs a partial (--filter=blob:none) clone is missing, in small
# parallel batches, through the flaky proxy. One lazy fetch is a single slow stream that restarts
# from zero whenever the proxy drops it; a landed batch persists, so re-runs only fetch the rest.
# git_fetch_missing_blobs <repo> <rev> [path...]   (env: GIT_BATCH=20 GIT_JOBS=8 GIT_PASSES=30)
git_fetch_missing_blobs() {
  local repo=$1 rev=$2; shift 2
  local batch=${GIT_BATCH:-20} jobs=${GIT_JOBS:-8} passes=${GIT_PASSES:-30}
  local bdir="$repo/.git/blob-batches" pass n
  for ((pass = 1; pass <= passes; pass++)); do
    rm -rf "$bdir"; mkdir -p "$bdir"
    git -C "$repo" rev-list --objects --missing=print "$rev" -- "$@" \
      | sed -n 's/^?//p' > "$bdir/missing.txt"
    n=$(wc -l < "$bdir/missing.txt")
    echo "$(date +%T) pass $pass: missing blobs: $n"
    [ "$n" = 0 ] && { rm -rf "$bdir"; return 0; }
    split -l "$batch" "$bdir/missing.txt" "$bdir/batch."
    ls "$bdir"/batch.* | xargs -P "$jobs" -I{} sh -c \
      'timeout 900 git -C "$0" -c fetch.negotiationAlgorithm=noop -c http.lowSpeedLimit=1000 \
         -c http.lowSpeedTime=120 fetch -q origin --no-tags --no-write-fetch-head \
         --recurse-submodules=no --filter=blob:none --stdin < "$1" >/dev/null 2>&1 \
         || echo "  batch $(basename "$1") failed; retried next pass"' "$repo" {} || true
  done
  echo "still missing blobs after $passes passes" >&2
  return 1
}
