#!/usr/bin/env bash
# Sourced by every download driver. Unless DL_FOREGROUND=1 is already set (run_queue.sh sets it
# for the steps it runs inline), re-launch the calling script detached from the terminal under
# nohup + setsid, append its output to a log, print pid + log path and return to the shell.
# The detached copy survives SSH disconnects and terminal closes.
# Override the log with DL_LOG=<path>; default <workspace>/dl_logs/<script-name>.log.
if [ -z "${DL_FOREGROUND:-}" ]; then
  _dl_self=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
  _dl_logs=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dl_logs
  mkdir -p "$_dl_logs"
  _dl_log=${DL_LOG:-$_dl_logs/$(basename "$0" .sh).log}
  DL_FOREGROUND=1 nohup setsid bash "$_dl_self" "$@" >> "$_dl_log" 2>&1 < /dev/null &
  echo "[$(date '+%F %T')] $(basename "$0") started in background: pid $!  log $_dl_log"
  exit 0
fi
export DL_FOREGROUND=1
