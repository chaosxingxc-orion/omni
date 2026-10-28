#!/usr/bin/env bash
# Run the diagnostic port preflight probe in saea-base. usage: bash port_probe.sh <experiment> <cell> [TREE]
set -uo pipefail
EXP=$1 CELL=$2 TREE=${3:-/mnt/cephfs/xingchao/work/dev/umbrella}
S=$TREE/studies/speech-aware-evidence-acquisition
exec docker run --rm --cpuset-cpus 128-191 -v /mnt/cephfs/xingchao:/mnt/cephfs/xingchao -w "$S" \
  -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR=/mnt/cephfs/xingchao \
  --entrypoint bash saea-base:20261027 -c "
    git config --global --add safe.directory '*' &&
    /opt/saea-venv/bin/python -P /mnt/cephfs/xingchao/work/scratch/probe/port_preflight_probe.py '$TREE' '$EXP' '$CELL'" 
