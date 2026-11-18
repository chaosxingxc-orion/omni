#!/usr/bin/env bash
# Build the vLLM serving views of the three Thinker embedding checkpoints, one after another.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../lib/local_profile.sh
. "$HERE/../lib/local_profile.sh" || { echo "cannot load the deployment profile bridge" >&2; exit 6; }
S=$(lp_get roots.source) || exit 6
S=$S/studies/speech-aware-evidence-acquisition/src
M=$(lp_get roots.model) || exit 6
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
