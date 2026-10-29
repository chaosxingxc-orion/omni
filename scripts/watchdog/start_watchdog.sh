#!/usr/bin/env bash
# usage: start_watchdog.sh <host-alias> [interval-seconds]   (run ON that host: dev | npu | npu2)
# Detached (nohup setsid), pid file, refuses a second instance, survives ssh drops.
set -u
H=${1:?host alias (dev|npu|npu2)}; INTERVAL=${2:-60}
D=$(cd "$(dirname "$0")" && pwd); mkdir -p "$D/run" "$D/bundles"
PIDF=$D/run/watchdog-$H.pid
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && grep -q "watchdog.py" "/proc/$(cat "$PIDF")/cmdline" 2>/dev/null; then
  echo "watchdog $H already running (pid $(cat "$PIDF"))"; exit 1
fi
nohup setsid python3 "$D/watchdog.py" --host "$H" --interval "$INTERVAL" >> "$D/run/watchdog-$H.out" 2>&1 < /dev/null &
echo $! > "$PIDF"
sleep 1
if kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "started watchdog $H pid $(cat "$PIDF")"
else echo "failed to start; see $D/run/watchdog-$H.out"; exit 1; fi
