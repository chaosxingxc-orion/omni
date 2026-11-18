#!/usr/bin/env bash
# Read-only flight `check` of one experiment cell in the pinned image.
# usage: bash port_check.sh <experiment> <cell> [TREE]
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
RUNTIME_ROOT=${PORT_RUNTIME_ROOT:-$OMNI_ROOT/work/scratch/runtime-root}
exec docker run --rm --cpuset-cpus "$CPUSET" -v "$OMNI_ROOT:$OMNI_ROOT" -w "$S" \
  -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR="$OMNI_ROOT" \
  --entrypoint bash "$PORT_IMAGE" -c '
    git config --global --add safe.directory "*" &&
    exec /opt/saea-venv/bin/python -m speech_aware_evidence_acquisition.core.flight.cli check --read-only \
      --repo-root "$@"' port-check \
  "$S" --data-root "$OMNI_ROOT" --runtime-root "$RUNTIME_ROOT" \
  --lock-path "$TREE/docs/datasets.lock.json" --experiment "$EXP" --cell "$CELL"