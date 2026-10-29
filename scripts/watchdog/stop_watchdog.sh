#!/usr/bin/env bash
# usage: stop_watchdog.sh <host-alias>   (run ON that host)
H=${1:?host alias}; D=$(cd "$(dirname "$0")" && pwd); PIDF=$D/run/watchdog-$H.pid
[ -f "$PIDF" ] || { echo "no pid file for $H"; exit 0; }
P=$(cat "$PIDF")
if kill -0 "$P" 2>/dev/null && grep -q "watchdog.py" "/proc/$P/cmdline" 2>/dev/null; then
  kill -TERM "$P"
  for _ in $(seq 1 15); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
  kill -0 "$P" 2>/dev/null && kill -KILL "$P"
  echo "stopped watchdog $H (pid $P)"
else
  echo "watchdog $H not running (stale pid file)"
fi
rm -f "$PIDF"
