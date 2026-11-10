#!/usr/bin/env bash
# Main executes this locally on the host with the already cached fixed image.
set -euo pipefail
workspace=$(cd "$(dirname "$0")/../../.." && pwd -P)
owner=delivery/s-terminal-handoff-repair-v1
image=sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026
attempt=${1:?usage: run_fixed_cpu.sh fresh_attempt_name}
[[ "$attempt" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
evidence="$workspace/$owner/evidence"
[[ ! -e "$evidence/$attempt" ]] || { echo 'choose a fresh attempt; preserve prior evidence' >&2; exit 2; }
actual_image=$(docker image inspect --format '{{.Id}}' "$image")
[[ "$actual_image" == "$image" ]] || { echo 'fixed image mismatch' >&2; exit 2; }
exec docker run --rm --pull never --network none --cpus 3 --memory 12g \
  --security-opt no-new-privileges --cap-drop ALL \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e OMP_NUM_THREADS=1 -e TOKENIZERS_PARALLELISM=false -e PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$workspace,dst=$workspace,readonly" \
  --mount "type=bind,src=$evidence,dst=$evidence" \
  --entrypoint /opt/saea-venv/bin/python "$image" -B \
  "$workspace/$owner/code/fixed_cpu_suite.py" --output "$evidence/$attempt" \
  --image-id "$image" --expected-python /opt/saea-venv/bin/python
