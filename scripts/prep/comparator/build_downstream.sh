#!/usr/bin/env bash
# Model-free: centroid keys (--pins) then the agreement table from the comparator cell. Runs on host npu.
set -euo pipefail
TREE=/mnt/cephfs/xingchao/omni/work/dev/wt-port
S=$TREE/studies/speech-aware-evidence-acquisition
PINS=docs/readiness/2026-10-28-slurp-intent-boundary-centroid-keys-npu-port-v1/pins-v1.json
run() {
  docker run --rm --user 1008:1008 -e HOME=/tmp --cpuset-cpus 64-127 \
    -v /mnt/cephfs/xingchao/omni:/mnt/cephfs/xingchao/omni -w "$S" \
    -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR=/mnt/cephfs/xingchao/omni \
    --entrypoint bash saea-base:20261027 -c "git config --global --add safe.directory '*' && /opt/saea-venv/bin/python $*"
}
echo "[$(date +%FT%T)] centroid keys"
run scripts/build_slurp_intent_boundary_centroid_keys.py --repo-root "$S" --data-root /mnt/cephfs/xingchao/omni --pins "$PINS"
echo "[$(date +%FT%T)] agreement table"
run scripts/build_slurp_intent_exemplar_agreement.py \
  --repo-root "$S" --data-root /mnt/cephfs/xingchao/omni \
  --runtime-root /mnt/cephfs/xingchao/omni/work/scratch/runtime-root --lock-path "$TREE/docs/datasets.lock.json" \
  --experiment slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1 \
  --cell slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1 \
  --vote-store train-exemplars --vote-phase solver \
  --target-store intent-boundaries --target-facet-template "intent-boundary:{key}" \
  --output prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1 \
  --artifact-id SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1 --workers 16
echo "[$(date +%FT%T)] DOWNSTREAM DONE"
