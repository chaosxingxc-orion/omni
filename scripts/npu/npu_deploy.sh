#!/usr/bin/env bash
# Resident multi-instance vLLM deployment from a study deployment profile (configs/deployment/<id>.json).
#
# usage (on the profile's host, e.g. `ssh npu2 bash /mnt/cephfs/xingchao/omni/work/scripts/npu_deploy.sh up <id>`):
#   bash npu_deploy.sh plan   <profile>  # print the exact per-instance docker commands; runs nothing on a card
#   bash npu_deploy.sh up     <profile>  # refuse busy cards, start every instance detached + resident, wait for health
#   bash npu_deploy.sh wait   <profile>  # wait for /health and /v1/models on every instance port
#   bash npu_deploy.sh status <profile>  # container state, /health, /v1/models, HBM per card
#   bash npu_deploy.sh down   <profile>  # stop and remove ONLY this profile's labelled containers
# <profile> is a profile id under $SAEA_SRC/../configs/deployment or a path to a profile JSON.
#
# The profile, the card check and every docker argv come from the study's own module
# (speech_aware_evidence_acquisition.core.deployment), run in a throwaway saea-base container with no devices and no
# network; this script only executes the NUL-delimited argv it emits (no eval). Nothing here uses llama.cpp slots.
# Env (every path derives from ONE root):
#   OMNI_ROOT            the data root, mounted at the same path in every container; the weights are
#                        $OMNI_ROOT/<profile model_relpath>. Default /mnt/cephfs/xingchao/omni.
#   SAEA_SRC             the study's src dir; default $OMNI_ROOT/work/dev/umbrella/studies/speech-aware-evidence-acquisition/src
#   DEPLOY_WAIT_SECONDS  readiness timeout for up/wait (default 2400)
set -euo pipefail
CMD=${1:?usage: npu_deploy.sh plan|up|wait|status|down <profile>}
PROFILE=${2:?usage: npu_deploy.sh plan|up|wait|status|down <profile>}
OMNI_ROOT=${OMNI_ROOT:-/mnt/cephfs/xingchao/omni}
SAEA_SRC=${SAEA_SRC:-$OMNI_ROOT/work/dev/umbrella/studies/speech-aware-evidence-acquisition/src}
REPO=$(dirname "$SAEA_SRC")
IMAGE=saea-base:20261027
WAIT_SECONDS=${DEPLOY_WAIT_SECONDS:-2400}
LABEL=saea.deployment.profile

helper() {
  # The study's deployment CLI in a device-free, network-free container; extra stdin is passed through.
  # The profile's data_mount is overridden by OMNI_ROOT here, so the root lives in exactly one variable.
  docker run --rm -i --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$REPO:$REPO:ro" -e PYTHONPATH="$SAEA_SRC" -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint /opt/saea-venv/bin/python "$IMAGE" \
    -m speech_aware_evidence_acquisition.core.deployment "$@" --repo-root "$REPO" --data-root "$OMNI_ROOT"
}

npu_table() { command -v npu-smi >/dev/null 2>&1 && npu-smi info; }

instances() { helper instances "$PROFILE" < /dev/null; }  # name<TAB>port<TAB>cards<TAB>base_url

profile_field() {  # one top-level field of the validated record
  helper record "$PROFILE" < /dev/null | python3 -I -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

require_host() {
  local want; want=$(profile_field host)
  if [[ "$(hostname)" != "$want" ]]; then
    echo "refused: profile $PROFILE is for host $want, this is $(hostname)" >&2; exit 1
  fi
}

served_name() { profile_field served_model_name; }

wait_ready() {
  local served deadline name port cards url state pending
  served=$(served_name)
  deadline=$(( $(date +%s) + WAIT_SECONDS ))
  while :; do
    pending=0
    while IFS=$'\t' read -r name port cards url; do
      state=$(docker inspect -f '{{.State.Status}} {{.RestartCount}}' "$name" 2>/dev/null) || state="missing 0"
      case "$state" in
        running*) ;;
        *) echo "FAILED: $name is ${state}; last log lines:" >&2; docker logs --tail 40 "$name" >&2 2>&1 || true; return 1 ;;
      esac
      if [[ "${state#* }" != 0 ]]; then
        echo "FAILED: $name restarted ${state#* } time(s); last log lines:" >&2
        docker logs --tail 40 "$name" >&2 2>&1 || true; return 1
      fi
      if curl -sf -m 5 "$url/health" >/dev/null && curl -sf -m 5 "$url/v1/models" | grep -q "\"id\": *\"$served\""; then
        :
      else
        pending=$((pending + 1))
      fi
    done < <(instances)
    if (( pending == 0 )); then echo "ready: every instance of $PROFILE answers /health and /v1/models ($served)"; return 0; fi
    if (( $(date +%s) > deadline )); then echo "FAILED: $pending instance(s) not ready after ${WAIT_SECONDS}s" >&2; return 1; fi
    echo "[$(date +%FT%T)] waiting: $pending instance(s) not ready"
    sleep 20
  done
}

case $CMD in
  plan)
    # Read-only: includes the card verdict when this host has npu-smi, never starts anything.
    if TABLE=$(npu_table); then helper plan "$PROFILE" --npu-smi-file - <<< "$TABLE"
    else helper plan "$PROFILE" < /dev/null; fi
    ;;
  up)
    require_host
    WEIGHTS=$(profile_field model_path)
    [[ -f "$WEIGHTS/config.json" ]] || { echo "refused: no weights at $WEIGHTS (OMNI_ROOT=$OMNI_ROOT)" >&2; exit 1; }
    while IFS=$'\t' read -r name port cards url; do
      if docker inspect "$name" >/dev/null 2>&1; then
        echo "refused: container $name already exists (run status, or down first)" >&2; exit 1
      fi
      if [[ -n "$(ss -Hltn "sport = :$port" 2>/dev/null)" ]]; then
        echo "refused: port $port is already listening" >&2; exit 1
      fi
    done < <(instances)
    TABLE=$(npu-smi info)
    EMITTED=$(mktemp)
    trap 'rm -f "$EMITTED"' EXIT
    # emit exits non-zero (and prints the per-card refusal) unless every card of the profile is free.
    helper emit "$PROFILE" --npu-smi-file - <<< "$TABLE" > "$EMITTED"
    exec 3< "$EMITTED"
    while IFS= read -r -d '' name <&3 && IFS= read -r -d '' argc <&3; do
      run=()
      for ((k = 0; k < argc; k++)); do IFS= read -r -d '' token <&3; run+=("$token"); done
      [[ "${run[0]} ${run[1]}" == "docker run" ]] || { echo "refused: emitted argv for $name is not docker run" >&2; exit 1; }
      id=$("${run[@]}")
      echo "started $name (${id:0:12})"
    done
    exec 3<&-
    wait_ready
    ;;
  wait)
    wait_ready
    ;;
  status)
    served=$(served_name)
    while IFS=$'\t' read -r name port cards url; do
      state=$(docker inspect -f '{{.State.Status}} restarts={{.RestartCount}} started={{.State.StartedAt}}' "$name" 2>/dev/null) || state=missing
      health=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$url/health" || true)
      models=$(curl -sf -m 5 "$url/v1/models" | grep -q "\"id\": *\"$served\"" && echo "$served" || echo none)
      echo "$name cards=$cards port=$port container=[$state] health=$health models=$models"
    done < <(instances)
    if TABLE=$(npu_table); then
      python3 -I -c '
import re, sys
for line in sys.stdin.read().splitlines():
    m = re.match(r"^\|\s*(\d+)\s+\S+\s*\|\s*(\S+)\s*\|", line)
    if m: head = m
    c = re.match(r"^\|\s*\d+\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*([\d.]+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)", line)
    if c: print("NPU %s %s aicore=%s%% hbm=%s/%s MiB" % (head.group(1), head.group(2), c.group(1), c.group(2), c.group(3)))
' <<< "$TABLE"
    fi
    ;;
  down)
    require_host
    ID=$(profile_field profile_id)
    while IFS=$'\t' read -r name port cards url; do
      owner=$(docker inspect -f "{{index .Config.Labels \"$LABEL\"}}" "$name" 2>/dev/null) || owner="<missing>"
      if [[ "$owner" == "<missing>" ]]; then echo "skip $name: no such container"; continue; fi
      if [[ "$owner" != "$ID" ]]; then
        echo "refused: $name carries $LABEL='$owner', not this profile; left untouched" >&2; continue
      fi
      docker stop -t 60 "$name" >/dev/null && docker rm "$name" >/dev/null && echo "stopped and removed $name"
    done < <(instances)
    ;;
  *) echo "unknown command $CMD (plan|up|wait|status|down)" >&2; exit 1 ;;
esac
