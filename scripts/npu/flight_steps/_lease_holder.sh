#!/usr/bin/env bash
# Hold a lease_cli process alive on a FIFO (started by step_lease.sh under `nohup setsid`; never run by hand).
# usage: _lease_holder.sh <fifo> <log> <exit-file> <command...>
# The FIFO is opened read-write so lease_cli never sees EOF; it reads one line from it: `release` (clean release) or
# anything else / EOF (preserve the lease metadata). When lease_cli exits, its exit status goes to <exit-file>.
FIFO=$1 LOG=$2 EXITF=$3; shift 3
exec 3<>"$FIFO"
"$@" <&3 >> "$LOG" 2>&1
echo $? > "$EXITF"
