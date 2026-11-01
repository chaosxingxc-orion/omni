#!/usr/bin/env bash
# Execute one reviewed step. Parent commits between stage/register/admit.
# Generation detaches itself; existing state is never interpreted as retry authority.
set -euo pipefail
: "${TREE:?Set TREE to the committed umbrella runtime tree}"
D=${SPEECHRL_DATA_DIR:-/mnt/cephfs/xingchao/omni}
S=$TREE/studies/speech-aware-evidence-acquisition
NAME=slurp-tfrl-hardset-a0-observation-r1-v1
P=$D/work/codex/hardset-a0-execution-v1
export TREE CPUSET=${CPUSET:-192-207} PORT_CONTAINER_USER=0:0
export PORT_IMAGE=sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026
export SAEA_DEPLOYMENT_PROFILE=qwen3-omni-real500-2xtp4-s128-b2048-v1
mkdir -p "$P"
case ${1:-} in
  stage)
    export PORT_DOCKER_FLAGS='--network none --name saea-hardset-a0-stage-v1'
    exec bash "$D/work/scripts/port_py.sh" scripts/analysis/tfrl_hardset_a0_registration.py stage --repo "$S" --data "$D" --prepared "$D/work/codex/hardset-a0-workload-v1/prepared-v2/complete.json" --derivation "$D/work/codex/hardset-a0-workload-v1/staged-v2/complete.json" --out "$P/stage"
    ;;
  register)
    export PORT_DOCKER_FLAGS='--network none --name saea-hardset-a0-register-v1'
    exec bash "$D/work/scripts/port_py.sh" scripts/analysis/tfrl_hardset_a0_registration.py register --repo "$S" --data "$D" --stage-record "$P/stage/complete.json"
    ;;
  admit)
    export PORT_DOCKER_FLAGS='--pid host --network host --name saea-hardset-a0-admit-v1'
    exec bash "$D/work/scripts/port_py.sh" scripts/analysis/admit_real_engine_candidate.py --repo "$S" --data "$D" --candidate "$NAME"
    ;;
  launch)
    # mkdir is atomic across launchers. A stale claim requires inspection, not restart.
    mkdir "$P/dispatch-claim"
    nohup setsid bash "$0" dispatch > "$P/dispatch.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$P/dispatch-claim/pid"
    printf 'Detached host PID %s; status: %s\n' "$!" "$P/dispatch.exit"
    ;;
  dispatch)
    test -d "$P/dispatch-claim"
    exec 9>"$P/dispatch.lock"
    flock -n 9
    test ! -e "$P/dispatch.exit"
    export PORT_DOCKER_FLAGS='--pid host --network host --name saea-hardset-a0-generation-v1'
    trap 'code=$?; printf "%s\n" "$code" > "$P/dispatch.exit"' EXIT
    bash "$D/work/scripts/port_py.sh" scripts/analysis/run_real_engine_candidate.py --repo "$S" --data "$D" --candidate "$NAME" --out "$P/run"
    ;;
  status)
    # Call on completion events or roughly every600s, never a busy loop.
    if test -f "$P/dispatch.exit"; then cat "$P/dispatch.exit"; fi
    if test -f "$P/dispatch-claim/pid"; then ps -p "$(cat "$P/dispatch-claim/pid")" -o pid=,stat=,etime=,args= || true; fi
    ;;
  *) printf 'Usage: %s stage|register|admit|launch|status\n' "$0" >&2; exit 2;;
esac
