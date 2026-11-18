#!/usr/bin/env bash
# Run the diagnostic port preflight probe in the pinned image.
# usage: bash port_probe.sh <experiment> <cell> [TREE]
#
# The data root, umbrella tree and image come from the local deployment profile
# (scripts/lib/local_profile.sh); TREE and PORT_IMAGE still override per call.
set -uo pipefail
EXP=$1 CELL=$2
PORT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$PORT_DIR/../lib/local_profile.sh" || exit 1
lp_default OMNI_ROOT storage.shared_root || exit 1
lp_default PORT_IMAGE image.reference || exit 1
TREE=${3:-${TREE:-}}; [ -n "$TREE" ] || TREE=$(lp_get roots.source) || exit 1
S=$TREE/studies/speech-aware-evidence-acquisition
CPUSET=${CPUSET:-128-191}  # policy constant (the free NUMA node), not local information
PROBE=${PORT_PROBE_SCRIPT:-$OMNI_ROOT/work/scratch/probe/port_preflight_probe.py}
exec docker run --rm --cpuset-cpus "$CPUSET" -v "$OMNI_ROOT:$OMNI_ROOT" -w "$S" \
  -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR="$OMNI_ROOT" \
  --entrypoint bash "$PORT_IMAGE" -c '
    git config --global --add safe.directory "*" &&
    exec /opt/saea-venv/bin/python -P "$@"' port-probe \
  "$PROBE" "$TREE" "$EXP" "$CELL"