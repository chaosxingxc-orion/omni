#!/usr/bin/env bash
# Run a study command (python args) in saea-base against our data root. usage: bash port_py.sh <python args...>
set -uo pipefail
TREE=${TREE:-/mnt/cephfs/xingchao/work/dev/umbrella}
S=$TREE/studies/speech-aware-evidence-acquisition
CPUSET=${CPUSET:-128-191}
exec docker run --rm --user 1008:1008 -e HOME=/tmp --cpuset-cpus "$CPUSET" -v /mnt/cephfs/xingchao:/mnt/cephfs/xingchao -w "$S" \
  -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR=/mnt/cephfs/xingchao \
  --entrypoint bash saea-base:20261027 -c "git config --global --add safe.directory '*' && /opt/saea-venv/bin/python $*"
