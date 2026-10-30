#!/usr/bin/env bash
# Launch ONE registered port encoder preparation (CPU float32 pool) detached on this host (npu2).
# usage: bash port_encoder_prep.sh <fold-b|fold-a|keys> [CPUSET] [--dry-run]
#   CPUSET     the cores of the encoder container (docker --cpuset-cpus syntax); default $PORT_ENCODER_CPUSET or 128-191 (the historical Fold B cores).
#              The pinned encoder inventory fixes the pool slice (first_core 128, 4 x 16 threads = cores 128-191): a cpuset that does not cover it is refused.
#   --dry-run  print the chain (config writer command, encoder command, log path, pool-core preflight) and exit; writes nothing, starts no container.
# log: $OMNI_ROOT/work/logs/port-encoder-<run>-<stamp>.log   (OMNI_ROOT defaults to /mnt/cephfs/xingchao/omni)
# The runtime config is written create-only and bound to the clean HEAD. The runtime itself admits the plan
# (execution authority, exposure row, lock, protocol, inventory pins) before any contact.
set -euo pipefail
RUN=""; CPUSET=${PORT_ENCODER_CPUSET:-128-191}; DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) if [ -z "$RUN" ]; then RUN=$arg; else CPUSET=$arg; fi ;;
  esac
done
[ -n "$RUN" ] || { echo "usage: bash port_encoder_prep.sh <fold-b|fold-a|keys> [CPUSET] [--dry-run]" >&2; exit 2; }
TREE=${TREE:-/mnt/cephfs/xingchao/omni/work/dev/umbrella}
S=$TREE/studies/speech-aware-evidence-acquisition
D=${OMNI_ROOT:-/mnt/cephfs/xingchao/omni}
NAME=saea-port-encoder-$RUN
LOG=$D/work/logs/port-encoder-$RUN-$(date +%Y%m%dT%H%M%S).log
IMAGE=saea-base:20261027
PYBIN=/opt/saea-venv/bin/python
COMMON=(--user 1008:1008 -e HOME=/tmp -e USER=saea -e LOGNAME=saea -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --cpuset-cpus "$CPUSET" -v "$D:$D" -w "$S"
  -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR=$D -e SAEA_DATA_ROOT=$D)
WRITE_CMD="git config --global --add safe.directory '*' && $PYBIN $D/work/scripts/port_write_runtime_config.py '$S' '$RUN' $D --cpuset '$CPUSET'"
if [ "$DRY_RUN" = 1 ]; then
  echo "DRY RUN: nothing is written and no container is started"
  echo "run $RUN  container $NAME  cpuset $CPUSET  log $LOG"
  echo "preflight: ${DRY_RUN_PYTHON:-python3} $D/work/scripts/port_write_runtime_config.py '$S' '$RUN' $D --cpuset '$CPUSET' --check-only"
  PRE=0; "${DRY_RUN_PYTHON:-python3}" "$(dirname "$(readlink -f "$0")")/port_write_runtime_config.py" "$S" "$RUN" "$D" --cpuset "$CPUSET" --check-only || PRE=$?
  echo "step 0: refuse if a container named $NAME is already running (docker ps)"
  echo "step 1 (config, create-only, clean HEAD): docker run --rm ${COMMON[*]} --entrypoint bash $IMAGE -c \"$WRITE_CMD\""
  echo "step 2 (encoder, detached): nohup docker run --rm --name $NAME ${COMMON[*]} --entrypoint bash $IMAGE -c \"git config --global --add safe.directory '*' && exec $PYBIN -m speech_aware_evidence_acquisition.core.speech_preparation_runtime --config <CONFIG from step 1> --config-sha256 <SHA from step 1> --data-root $D --source-root '$S'\" >> $LOG 2>&1 &"
  [ "$PRE" = 0 ] && echo "preflight verdict: OK" || echo "preflight verdict: a real launch would be REFUSED (see the pool line above)"
  exit 0
fi
mkdir -p "$D/work/logs"
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "refusing: $NAME is already running" >&2; exit 1; fi
read -r CONFIG SHA < <(docker run --rm "${COMMON[@]}" --entrypoint bash $IMAGE -c "$WRITE_CMD")
[ -n "${SHA:-}" ] || { echo "refusing: no runtime config was written" >&2; exit 1; }
echo "config $CONFIG sha256 $SHA" | tee "$LOG"
nohup docker run --rm --name "$NAME" "${COMMON[@]}" --entrypoint bash $IMAGE -c \
  "git config --global --add safe.directory '*' && exec $PYBIN -m speech_aware_evidence_acquisition.core.speech_preparation_runtime \
   --config '$CONFIG' --config-sha256 $SHA --data-root $D --source-root '$S'" >> "$LOG" 2>&1 &
echo "launched $NAME (pid $!), log $LOG"
