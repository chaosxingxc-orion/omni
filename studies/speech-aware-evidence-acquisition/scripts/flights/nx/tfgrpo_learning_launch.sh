#!/usr/bin/env bash
# Parent commits between register/admit/launch; detached queue reads compact events.
set -euo pipefail
: "${TREE:?Set committed execution umbrella tree}"
D=${SPEECHRL_DATA_DIR:-/mnt/cephfs/xingchao/omni}
S=$TREE/studies/speech-aware-evidence-acquisition
P=${TFGRPO_LAUNCH_ROOT:-$D/work/codex/tfgrpo-learning-execution-v1}
JOB=${TFGRPO_JOB:-$D/work/codex/tfgrpo-learning-job-v1/prepared-v2}
AFTER=${TFGRPO_AFTER:-$D/work/codex/hardset-a0-execution-v1/run/generation-released.json}
export TREE CPUSET=${CPUSET:-232-239} PORT_CONTAINER_USER=0:0
export PORT_IMAGE=sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026
export SAEA_DEPLOYMENT_PROFILE=qwen3-omni-real500-2xtp4-s128-b2048-v1
mkdir -p "$P"
case ${1:-} in
  verify|register)
    export PORT_DOCKER_FLAGS="--network none --name saea-tfgrpo-learning-${1}-v1"
    exec bash "$D/work/scripts/port_py.sh" scripts/analysis/tfgrpo_learning_job.py "$1" --repo "$S" --data "$D" --job "$JOB"
    ;;
  admit)
    export PORT_DOCKER_FLAGS='--pid host --network host --name saea-tfgrpo-learning-admit-v1'
    exec bash "$D/work/scripts/port_py.sh" scripts/analysis/tfgrpo_learning_job.py admit --repo "$S" --data "$D" --job "$JOB"
    ;;
  launch)
    mkdir "$P/dispatch-claim"
    nohup setsid bash "$0" dispatch > "$P/dispatch.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$P/dispatch-claim/pid"
    ;;
  dispatch)
    test -d "$P/dispatch-claim"
    exec 9>"$P/dispatch.lock"
    flock -n 9
    test ! -e "$P/dispatch.exit"
    trap 'code=$?; printf "%s\n" "$code" > "$P/dispatch.exit"' EXIT
    while ! test -f "$AFTER"; do
      printf '%s waiting-for-A0-generation-release\n' "$(date -u +%FT%TZ)" > "$P/status.txt"
      sleep 600
    done
    export PORT_DOCKER_FLAGS='--pid host --network host --name saea-tfgrpo-learning-generation-v1'
    bash "$D/work/scripts/port_py.sh" scripts/analysis/tfgrpo_learning_job.py run --repo "$S" --data "$D" --job "$JOB" --after "$AFTER" --out "$P/run"
    ;;
  *) printf 'Usage: %s verify|register|admit|launch\n' "$0" >&2; exit 2;;
esac
