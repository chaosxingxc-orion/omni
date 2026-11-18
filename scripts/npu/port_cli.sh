#!/usr/bin/env bash
# Run one flight CLI mode for one experiment cell in the pinned image (offline modes only).
# usage: bash port_cli.sh <mode> <experiment> <cell> [extra cli args...]
#
# The data root, umbrella tree and image come from the local deployment profile
# (scripts/lib/local_profile.sh). TREE, PORT_IMAGE and PORT_RUNTIME_ROOT still override per call.
set -uo pipefail
MODE=$1 EXP=$2 CELL=$3; shift 3
EXTRA=("$@")
PORT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$PORT_DIR/../lib/local_profile.sh" || exit 1
lp_default OMNI_ROOT storage.shared_root || exit 1
lp_default TREE roots.source || exit 1
lp_default PORT_IMAGE image.reference || exit 1
S=$TREE/studies/speech-aware-evidence-acquisition
CPUSET=${CPUSET:-128-191}  # policy constant (the free NUMA node), not local information
RUNTIME_ROOT=${PORT_RUNTIME_ROOT:-$OMNI_ROOT/work/scratch/runtime-root}
exec docker run --rm --cpuset-cpus "$CPUSET" -v "$OMNI_ROOT:$OMNI_ROOT" -w "$S" \
  -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR="$OMNI_ROOT" \
  --entrypoint bash "$PORT_IMAGE" -c '
    git config --global --add safe.directory "*" &&
    exec /opt/saea-venv/bin/python -m speech_aware_evidence_acquisition.core.flight.cli "$@"' port-cli \
  "$MODE" --repo-root "$S" --data-root "$OMNI_ROOT" --runtime-root "$RUNTIME_ROOT" \
  --lock-path "$TREE/docs/datasets.lock.json" --experiment "$EXP" --cell "$CELL" "${EXTRA[@]}"