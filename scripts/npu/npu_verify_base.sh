#!/usr/bin/env bash
# Acceptance test of the locked engineering base, run on an NPU host against the RESIDENT deployment.
#
# usage (on the profile's host, e.g. `ssh npu2 bash /mnt/cephfs/xingchao/omni/work/scripts/npu_verify_base.sh ...`):
#   bash npu_verify_base.sh <profile id|path> <base-lock.json> [verifier args...]      # verify; exit 0 only on PASS
#   bash npu_verify_base.sh <profile id|path> -  --print-observed [--receipt P] [...]   # write a base-lock from a live run
# Verifier args (after the lock): --only image,kv,...  --rehash-weights  --receipt P  --encoder NAME=SUBDIR  (see
# scripts/serving/verify_engineering_base.py --help). Env: OUT (report path; default
# $OMNI_ROOT/work/logs/base-verification-<profile>-<UTC stamp>.json), CPUSET (default 32-63,96-127 = NUMA nodes 1 and 3,
# which the 181 profiles leave free), OMNI_ROOT, SAEA_SRC.
#
# What it does, in two read-only steps. Nothing here starts, stops, restarts or execs into a container, and no NPU is
# opened (no --device is passed):
#  1. On the HOST it collects facts with read-only commands into a throw-away directory under $OMNI_ROOT/work/scratch:
#     hostname, core and card counts, the global-scope NIC names (`ip`), `docker image inspect` of the profile image and
#     the base image, and per instance `docker inspect`, the startup lines of `docker logs`, and the raw
#     /proc/<pid>/cmdline of the container's main process (same shape the runtime receipt pins).
#  2. In a throw-away `saea-base` container (--network host for the loopback endpoints; NO docker.sock, NO devices) it
#     runs the verifier with $OMNI_ROOT mounted read-only (only the report's directory is writable), the host's
#     /usr/local/Ascend and /etc/ascend_install.info mounted read-only under /host, and the facts under /facts.
# Run as a user in the docker group; the container runs as the caller's uid/gid. Do not pass `-I` to Python.
set -uo pipefail
PROFILE_ARG=${1:?usage: npu_verify_base.sh <profile id|path> <base-lock.json|-> [verifier args...]}
LOCK=${2:?usage: npu_verify_base.sh <profile id|path> <base-lock.json|-> [verifier args...]}
shift 2
OMNI_ROOT=${OMNI_ROOT:-/mnt/cephfs/xingchao/omni}
SAEA_SRC=${SAEA_SRC:-$OMNI_ROOT/work/dev/umbrella/studies/speech-aware-evidence-acquisition/src}
REPO=$(dirname "$SAEA_SRC")
CPUSET=${CPUSET:-32-63,96-127}
DEFAULT_BASE_REF=quay.io/ascend/vllm-ascend:v0.30.0rc2

if [[ -f "$PROFILE_ARG" ]]; then PROFILE=$PROFILE_ARG; else PROFILE=$REPO/configs/deployment/$PROFILE_ARG.json; fi
[[ -f "$PROFILE" ]] || { echo "refused: no deployment profile $PROFILE_ARG" >&2; exit 2; }
[[ "$LOCK" == "-" || -f "$LOCK" ]] || { echo "refused: no base-lock $LOCK" >&2; exit 2; }

# Read-only parse of the profile with the host python (the verifier re-validates it with the study's own loader).
mapfile -t PROFILE_FIELDS < <(python3 -I -c '
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d["profile_id"]); print(d["host"]); print(d["image"])
for i in d["instances"]: print(i["name"])
' "$PROFILE") || { echo "refused: $PROFILE is not a readable profile" >&2; exit 2; }
PROFILE_ID=${PROFILE_FIELDS[0]} PROFILE_HOST=${PROFILE_FIELDS[1]} IMAGE=${PROFILE_FIELDS[2]}
NAMES=("${PROFILE_FIELDS[@]:3}")
if [[ "$(hostname)" != "$PROFILE_HOST" ]]; then
  echo "refused: profile $PROFILE_ID is for host $PROFILE_HOST, this is $(hostname)" >&2; exit 2
fi

BASE_REF=$DEFAULT_BASE_REF
if [[ "$LOCK" != "-" ]]; then
  BASE_REF=$(python3 -I -c '
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d.get("image", {}).get("base", {}).get("ref") or sys.argv[2])
' "$LOCK" "$DEFAULT_BASE_REF" 2>/dev/null) || BASE_REF=$DEFAULT_BASE_REF
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PRINTING=0; for arg in "$@"; do [[ "$arg" == "--print-observed" ]] && PRINTING=1; done
if (( PRINTING )); then KIND=base-lock-observed; else KIND=base-verification; fi
OUT=${OUT:-$OMNI_ROOT/work/logs/$KIND-$PROFILE_ID-$STAMP.json}
OUT_DIR=$(dirname "$OUT")
mkdir -p "$OUT_DIR" "$OMNI_ROOT/work/scratch"
FACTS=$(mktemp -d "$OMNI_ROOT/work/scratch/base-verify-facts.XXXXXX")
trap 'rm -rf "$FACTS"' EXIT

# ---- 1. host facts (read-only commands) --------------------------------------------------------------------------
NICS=$(ip -o -4 addr show scope global 2>/dev/null | awk '{print $2}' | awk '!seen[$0]++' | paste -sd, -)
CARDS=$(ls /dev/davinci[0-9]* 2>/dev/null | wc -l)
python3 -I -c '
import json, sys
host, cpus, cards, nics = sys.argv[1:5]
print(json.dumps({"hostname": host, "cpu_count": int(cpus), "card_count": int(cards),
                  "global_nics": [n for n in nics.split(",") if n]}, sort_keys=True))
' "$(hostname)" "$(nproc --all)" "$CARDS" "$NICS" > "$FACTS/host.json"

ref_file() { printf 'image-%s.json' "$(printf %s "$1" | sha256sum | cut -c1-16)"; }
docker image inspect "$IMAGE" > "$FACTS/$(ref_file "$IMAGE")" 2>/dev/null || rm -f "$FACTS/$(ref_file "$IMAGE")"
docker image inspect "$BASE_REF" > "$FACTS/$(ref_file "$BASE_REF")" 2>/dev/null || rm -f "$FACTS/$(ref_file "$BASE_REF")"
for name in "${NAMES[@]}"; do
  # A missing container leaves no file; the verifier then fails the checks that need it (closed).
  docker inspect "$name" > "$FACTS/container-$name.json" 2>/dev/null || rm -f "$FACTS/container-$name.json"
  docker logs "$name" 2>&1 | grep -E 'KV cache size|Available KV cache memory|Initializing a V1 LLM engine|Block size is set' \
    > "$FACTS/logs-$name.txt" || true
  pid=$(docker inspect -f '{{.State.Pid}}' "$name" 2>/dev/null || echo 0)
  if [[ "$pid" =~ ^[1-9][0-9]*$ ]]; then cat "/proc/$pid/cmdline" > "$FACTS/cmdline-$name.bin" 2>/dev/null || rm -f "$FACTS/cmdline-$name.bin"; fi
done

# ---- 2. the verifier, in a throw-away saea-base container (no docker.sock, no devices) -------------------------------
MOUNTS=(-v "$OMNI_ROOT:$OMNI_ROOT:ro" -v "$OUT_DIR:$OUT_DIR" -v "$FACTS:/facts:ro" -v /usr/local/Ascend:/host/usr/local/Ascend:ro)
[[ -f /etc/ascend_install.info ]] && MOUNTS+=(-v /etc/ascend_install.info:/host/etc/ascend_install.info:ro)
LOCK_ARGS=(); [[ "$LOCK" != "-" ]] && LOCK_ARGS=(--base-lock "$LOCK")
docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp --cpuset-cpus "$CPUSET" "${MOUNTS[@]}" \
  -w "$REPO" -e PYTHONPATH="$SAEA_SRC" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR="$OMNI_ROOT" \
  --entrypoint /opt/saea-venv/bin/python "$IMAGE" "$REPO/scripts/serving/verify_engineering_base.py" \
  --profile "$PROFILE" "${LOCK_ARGS[@]}" --out "$OUT" --repo-root "$REPO" --data-root "$OMNI_ROOT" \
  --facts-dir /facts --host-root /host "$@"
exit $?
