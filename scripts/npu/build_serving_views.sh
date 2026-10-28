#!/usr/bin/env bash
# Build the vLLM serving views of the three Thinker embedding checkpoints, one after another.
set -euo pipefail
S=/mnt/cephfs/xingchao/work/dev/umbrella/studies/speech-aware-evidence-acquisition/src
M=/mnt/cephfs/xingchao/models
V=$M/serving-views
mkdir -p "$V"
for pair in omni-embed-nemotron-3b:omni-embed-nemotron-3b-causal-v1 lco-embedding-omni-3b-2605:lco-embedding-omni-3b-2605-v1 lco-embedding-omni-7b:lco-embedding-omni-7b-v1; do
  src=${pair%%:*}; id=${pair##*:}
  if [ -e "$V/$id" ]; then echo "$(date +%T) exists: $id"; continue; fi
  echo "$(date +%T) build: $id"
  PYTHONPATH="$S" python3 -m speech_aware_evidence_acquisition.core.vllm_serving_view build "$M/$src" "$V/$id" --view-id "$id"
  PYTHONPATH="$S" python3 -m speech_aware_evidence_acquisition.core.vllm_serving_view verify "$V/$id"
done
echo "$(date +%T) VIEWS_DONE"
