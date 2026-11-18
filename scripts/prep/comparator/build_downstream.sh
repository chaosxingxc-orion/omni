#!/usr/bin/env bash
# Model-free: centroid keys (--pins) then the agreement table from the comparator cell.  Runs on the
# deployment's rollout host, inside the frozen image.
#
# Every machine fact -- the port worktree, the shared root, the image, the uid/gid to run as -- comes
# from the local deployment profile.  The cpuset is a policy constant (this deployment reserves cores
# 64-127 for model-free CPU preparation), not a machine fact, so it stays here as an explicit,
# reviewable constant.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../../lib/local_profile.sh
. "$HERE/../../lib/local_profile.sh" || { echo "cannot load the deployment profile bridge" >&2; exit 6; }
DATA=$(lp_get storage.shared_root) || exit 6
TREE=$(lp_get paths.port_worktree) || exit 6
IMAGE=$(lp_get image.reference) || exit 6
UID_GID="$(lp_get identity.uid):$(lp_get identity.gid)" || exit 6
CPUSET=64-127
S=$TREE/studies/speech-aware-evidence-acquisition
PINS=docs/readiness/2026-10-28-slurp-intent-boundary-centroid-keys-npu-port-v1/pins-v1.json
run() {
  docker run --rm --user "$UID_GID" -e HOME=/tmp --cpuset-cpus "$CPUSET" \
    -v "$DATA:$DATA" -w "$S" \
    -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR="$DATA" \
    --entrypoint bash "$IMAGE" -c "git config --global --add safe.directory '*' && /opt/saea-venv/bin/python $*"
}
echo "[$(date +%FT%T)] centroid keys"
run scripts/build_slurp_intent_boundary_centroid_keys.py --repo-root "$S" --data-root "$DATA" --pins "$PINS"
echo "[$(date +%FT%T)] agreement table"
run scripts/build_slurp_intent_exemplar_agreement.py \
  --repo-root "$S" --data-root "$DATA" \
  --runtime-root "$DATA/work/scratch/runtime-root" --lock-path "$TREE/docs/datasets.lock.json" \
  --experiment slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1 \
  --cell slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1 \
  --vote-store train-exemplars --vote-phase solver \
  --target-store intent-boundaries --target-facet-template "intent-boundary:{key}" \
  --output prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1 \
  --artifact-id SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1 --workers 16
echo "[$(date +%FT%T)] DOWNSTREAM DONE"