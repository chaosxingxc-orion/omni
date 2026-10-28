#!/usr/bin/env bash
# Move everything this work generated under /mnt/cephfs/xingchao/ into /mnt/cephfs/xingchao/omni/ (owner instruction 2026-10-28).
# Same CephFS mount, so every move is a rename (instant, no copy). Stays where it is (owner, 2026-10-28): docker/ (including
# docker/saea-base). Not ours and never touched: experiments/.
#
# usage: bash relocate_to_omni.sh [--apply]      (default is a dry run that only prints what it would do)
# Refuses while any process on this host, npu or npu2 still names an old path, or when a target already exists.
# Historical records (docs/readiness registrations, coordination runtime configs, receipts, serving_view.json) keep the
# absolute paths they recorded; only live scripts are rewritten. The tracked model fragment is re-pinned by a commit.
set -euo pipefail
OLD=/mnt/cephfs/xingchao
NEW=$OLD/omni
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
DIRS=(attempts coordination datasets knowledge locks manifests models preparation-runs prepared repos src tmp vectors work)
REPO_REL=work/umbrella/studies/speech-aware-evidence-acquisition
WORKTREES=(work/dev/umbrella work/dev/wt-port work/dev/wt-sess work/dev/wt-lock work/dev/wt-deploy work/dev/wt-umb)
say() { echo "[$(date +%FT%T)] $*"; }
do_() { if [ $APPLY = 1 ]; then "$@"; else echo "DRY: $*"; fi; }

# Old-path users: any argv naming $OLD/<one of DIRS> (the omni/ path itself does not match).
pattern="$OLD/($(IFS='|'; echo "${DIRS[*]}"))(/|\$| )"
busy=0
for host in local npu npu2; do
  if [ $host = local ]; then out=$(ps -eo pid,args --no-headers); else out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 $host ps -eo pid,args --no-headers); fi
  hits=$(grep -E "$pattern" <<< "$out" | grep -v -e relocate_to_omni -e 'grep -E' || true)
  if [ -n "$hits" ]; then busy=1; say "BUSY on $host:"; cut -c1-160 <<< "$hits" | head -5; fi
done
if [ $busy = 1 ] && [ $APPLY = 1 ]; then say "refused: old paths still in use"; exit 1; fi

for d in "${DIRS[@]}"; do
  [ -e "$OLD/$d" ] || { say "skip $d (absent)"; continue; }
  [ -e "$NEW/$d" ] && { say "refused: $NEW/$d already exists"; exit 1; }
done
do_ mkdir -p "$NEW"
for d in "${DIRS[@]}"; do
  [ -e "$OLD/$d" ] && do_ mv "$OLD/$d" "$NEW/$d"
done

# Git: the pristine clone (holds the git dir) and every linked worktree moved together; re-link both directions.
repaired=()
for w in "${WORKTREES[@]}"; do repaired+=("$NEW/$w/studies/speech-aware-evidence-acquisition"); done
do_ git -c safe.directory='*' -C "$NEW/$REPO_REL" worktree repair "${repaired[@]}"

# Live scripts: rewrite $OLD -> $NEW except where $OLD is followed by /omni, /docker (stays put) or /experiments.
targets=$( { ls "$NEW"/work/scripts/*.sh "$NEW"/work/scripts/*.py 2>/dev/null || ls "$OLD"/work/scripts/*.sh "$OLD"/work/scripts/*.py; \
             find "${NEW}/work/scratch" "${OLD}/work/scratch" -name '*.sh' 2>/dev/null; \
             ls /home/xingchao/workspace/omni/dl_scripts/*.sh 2>/dev/null; } | sort -u)
for f in $targets; do
  case $f in */relocate_to_omni.sh) continue ;; esac
  grep -q "$OLD" "$f" || continue
  if [ $APPLY = 1 ]; then perl -pi -e "s#\Q$OLD\E(?!/omni|/docker|/experiments)(?=[/:\"' \\n}]|\$)#$NEW#g" "$f"; else echo "DRY: rewrite $(grep -c "$OLD" "$f") line(s) in $f"; fi
done

if [ $APPLY = 1 ]; then
  for w in "${repaired[@]}"; do git -c safe.directory='*' -C "$w" rev-parse --abbrev-ref HEAD >/dev/null && say "ok $w"; done
  say "left at $OLD: $(ls -A "$OLD" | tr '\n' ' ')"
fi
