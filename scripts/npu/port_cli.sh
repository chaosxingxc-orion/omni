#!/usr/bin/env bash
# Run one flight CLI mode for one experiment cell in saea-base (offline modes only).
# usage: bash port_cli.sh <mode> <experiment> <cell> [extra cli args...]   (TREE env overrides the umbrella tree)
set -uo pipefail
MODE=$1 EXP=$2 CELL=$3; shift 3
TREE=${TREE:-/mnt/cephfs/xingchao/omni/work/dev/umbrella}
S=$TREE/studies/speech-aware-evidence-acquisition
exec docker run --rm --cpuset-cpus 128-191 -v /mnt/cephfs/xingchao/omni:/mnt/cephfs/xingchao/omni -w "$S" \
  -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR=/mnt/cephfs/xingchao/omni \
  --entrypoint bash saea-base:20261027 -c "
    git config --global --add safe.directory '*' &&
    /opt/saea-venv/bin/python -m speech_aware_evidence_acquisition.core.flight.cli $MODE \
      --repo-root '$S' --data-root /mnt/cephfs/xingchao/omni --runtime-root /mnt/cephfs/xingchao/omni/work/scratch/runtime-root \
      --lock-path '$TREE/docs/datasets.lock.json' --experiment '$EXP' --cell '$CELL' $*"
