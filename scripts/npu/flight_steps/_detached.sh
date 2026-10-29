#!/usr/bin/env bash
# Detached runner for the long steps (started by _common.sh under `nohup setsid`; never run by hand).
# usage: _detached.sh <state-dir> <step> <commit> <command-string>
# Writes <step>.running (pid) while alive, appends to <step>.log, then <step>.done (rc 0) or <step>.failed (rc != 0).
# Optional env FS_RELEASE_LEASE=<fifo>: after the command, send `release` (rc 0) or `preserve` (rc != 0) to the lease holder.
STATE=$1 STEP=$2 COMMIT=$3 SCRIPT=$4
LOG=$STATE/$STEP.log
echo $$ > "$STATE/$STEP.running"
rm -f "$STATE/$STEP.failed"
{
  echo "# $STEP start $(date -u +%FT%TZ) commit $COMMIT"
  eval "$SCRIPT"
  RC=$?
  echo "# $STEP exit $RC $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1
RC=${RC:-1}
if [ -n "${FS_RELEASE_LEASE:-}" ]; then
  if [ "$RC" = 0 ]; then WORD=release; else WORD=preserve; fi
  timeout 5 bash -c "echo $WORD > '$FS_RELEASE_LEASE'" >/dev/null 2>&1
  if [ "$WORD" = release ]; then
    for _ in $(seq 1 60); do [ -e "${FS_LEASE_TOKEN:-/nonexistent}" ] || break; sleep 1; done
  fi
  if [ "$WORD" = release ] && [ ! -e "${FS_LEASE_TOKEN:-/nonexistent}" ]; then
    date -u +%FT%TZ > "$STATE/lease.released"
    echo "# lease released $(date -u +%FT%TZ)" >> "$LOG"
  else
    echo "# lease NOT released (word $WORD, token ${FS_LEASE_TOKEN:-none}); reconcile by hand before any new lease" >> "$LOG"
  fi
fi
rm -f "$STATE/$STEP.running"
if [ "$RC" = 0 ]; then
  printf '{"step":"%s","rc":0,"finished_utc":"%s","commit":"%s","log":"%s"}\n' "$STEP" "$(date -u +%FT%TZ)" "$COMMIT" "$LOG" > "$STATE/$STEP.done"
else
  printf '{"step":"%s","rc":%s,"finished_utc":"%s","log":"%s"}\n' "$STEP" "$RC" "$(date -u +%FT%TZ)" "$LOG" > "$STATE/$STEP.failed"
fi
exit "$RC"
