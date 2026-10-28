#!/usr/bin/env bash
# Launch ONE registered port encoder preparation (CPU float32 pool, cores 128-191) detached on this host (npu).
# usage: bash port_encoder_prep.sh <fold-b|keys>    log: /mnt/cephfs/xingchao/omni/work/logs/port-encoder-<run>-<stamp>.log
# The runtime config is written create-only and bound to the clean HEAD. The runtime itself admits the plan
# (execution authority, exposure row, lock, protocol, inventory pins) before any contact.
set -euo pipefail
RUN=$1
TREE=${TREE:-/mnt/cephfs/xingchao/omni/work/dev/umbrella}
S=$TREE/studies/speech-aware-evidence-acquisition
D=/mnt/cephfs/xingchao/omni
NAME=saea-port-encoder-$RUN
LOG=$D/work/logs/port-encoder-$RUN-$(date +%Y%m%dT%H%M%S).log
mkdir -p "$D/work/logs"
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "refusing: $NAME is already running" >&2; exit 1; fi
COMMON=(--user 1008:1008 -e HOME=/tmp -e USER=saea -e LOGNAME=saea -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --cpuset-cpus 128-191 -v "$D:$D" -w "$S"
  -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR=$D -e SAEA_DATA_ROOT=$D)
read -r CONFIG SHA < <(docker run --rm "${COMMON[@]}" --entrypoint bash saea-base:20261027 -c \
  "git config --global --add safe.directory '*' && /opt/saea-venv/bin/python $D/work/scripts/port_write_runtime_config.py '$S' '$RUN' $D")
[ -n "${SHA:-}" ] || { echo "refusing: no runtime config was written" >&2; exit 1; }
echo "config $CONFIG sha256 $SHA" | tee "$LOG"
nohup docker run --rm --name "$NAME" "${COMMON[@]}" --entrypoint bash saea-base:20261027 -c \
  "git config --global --add safe.directory '*' && exec /opt/saea-venv/bin/python -m speech_aware_evidence_acquisition.core.speech_preparation_runtime \
   --config '$CONFIG' --config-sha256 $SHA --data-root $D --source-root '$S'" >> "$LOG" 2>&1 &
echo "launched $NAME (pid $!), log $LOG"
