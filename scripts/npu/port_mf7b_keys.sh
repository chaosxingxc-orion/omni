#!/usr/bin/env bash
# Launch the registered port MF-7b KEYS pass (SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-NPU-PORT-V1) detached on this host
# (npu): CPU float32, 4 pinned workers x 16 threads on cores 128-191; when it seals, the model-free kus3 conversion
# (build_slurp_kus3_vectors.py --facets-only) writes vectors/slurp-kus3/facet-vectors.jsonl.
# usage: bash port_mf7b_keys.sh       log: /mnt/cephfs/xingchao/omni/work/logs/port-mf7b-keys-<stamp>.log
set -euo pipefail
TREE=${TREE:-/mnt/cephfs/xingchao/omni/work/dev/umbrella}
S=$TREE/studies/speech-aware-evidence-acquisition
D=/mnt/cephfs/xingchao/omni
NAME=saea-port-mf7b-keys
ROW=SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-NPU-PORT-V1
LOG=$D/work/logs/port-mf7b-keys-$(date +%Y%m%dT%H%M%S).log
mkdir -p "$D/work/logs"
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then echo "refusing: $NAME is already running" >&2; exit 1; fi
if [ -n "$(git -c safe.directory='*' -C "$S" status --porcelain)" ]; then echo "refusing: the study checkout is not clean" >&2; exit 1; fi
want=$(grep -oE 'Input vectors/slurp-mf7/train-facets.jsonl SHA256 [0-9a-f]{64}' "$S/docs/exposure-ledger.md" | awk '{print $4}')
have=$(sha256sum "$D/vectors/slurp-mf7/train-facets.jsonl" | awk '{print $1}')
if [ "$want" != "$have" ]; then echo "refusing: train-facets.jsonl is $have, the row pins $want" >&2; exit 1; fi
HEAD=$(git -c safe.directory='*' -C "$S" rev-parse HEAD)
echo "head $HEAD input $have" | tee "$LOG"
PY=/usr/local/python3.12.13/bin/python3
nohup docker run --rm --name "$NAME" --user 1008:1008 -e HOME=/tmp -e USER=saea \
  -e LOGNAME=saea -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e HF_HUB_OFFLINE=1 \
  --cpuset-cpus 128-191 -v "$D:$D" -w "$S" -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR=$D \
  --entrypoint bash saea-base:20261027 -c \
  "git config --global --add safe.directory '*' && \
   $PY scripts/slurp_mf7b_nemotron_embedding_pass.py --sets keys --runtime cpu-float32 --workers 4 --cores 128-191 --attention-geometry causal \
     --exposure-row $ROW && \
   $PY scripts/build_slurp_kus3_vectors.py --facets-only --mf7b-row $ROW \
     --receipt-dir $D/vectors/slurp-kus3/receipts-npu-port-v1" >> "$LOG" 2>&1 &
echo "launched $NAME (pid $!), log $LOG"
