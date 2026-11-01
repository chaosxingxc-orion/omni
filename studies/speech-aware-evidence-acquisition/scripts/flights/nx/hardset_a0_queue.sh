#!/usr/bin/env bash
# Detached formal queue. Restart observes existing custody before any dispatch.
set -euo pipefail
: "${TREE:?Set TREE to the committed umbrella runtime tree}"
D=${SPEECHRL_DATA_DIR:-/mnt/cephfs/xingchao/omni}
S=$TREE/studies/speech-aware-evidence-acquisition
P=$D/work/codex/hardset-a0-execution-v1/queue
export TREE CPUSET=${CPUSET:-192-207} PORT_CONTAINER_USER=0:0
export PORT_IMAGE=sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026
export SAEA_DEPLOYMENT_PROFILE=qwen3-omni-real500-2xtp4-s128-b2048-v1
mkdir -p "$P"
case ${1:-} in
  launch)
    # Kernel queue lock in the formal coordinator prevents duplicate launches;
    # launch intent survives host/container death and refuses ambiguous replay.
    nohup setsid bash "$0" dispatch >>"$P/coordinator.log" 2>&1 </dev/null &
    printf 'Detached queue host PID %s; custody: %s\n' "$!" "$P"
    ;;
  dispatch)
    export PORT_DOCKER_FLAGS='--pid host --network host --name saea-hardset-a0-queue-v1'
    exec bash "$D/work/scripts/port_py.sh" scripts/analysis/a0_recovery_queue.py supervise --repo "$S" --data "$D" --candidate slurp-tfrl-hardset-a0-observation-r1-v1 --out "$P"
    ;;
  *) printf 'Usage: %s launch\n' "$0" >&2; exit 2;;
esac
