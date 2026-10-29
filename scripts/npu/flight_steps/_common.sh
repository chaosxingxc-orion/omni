#!/usr/bin/env bash
# Shared helpers for the operator flight steps. Sourced by every step_*.sh; never run on its own.
#
# One step = one command a person runs. Nothing here advances to the next step. Without --confirm a step only prints
# the exact study command(s) it would run and reports its gates; with --confirm it enforces the gates and executes.
#
# Exit codes: 0 ok (or dry-run), 2 usage, 3 previous step's done marker absent, 4 preflight failed, 5 step already
# done or running, 6 other refusal (wrong host, missing input), 7 flight commit moved, 8 the step's own command failed.
#
# Test/override environment (all optional):
#   OMNI_ROOT, FLIGHT_STEPS_STUDY, FLIGHT_STEPS_STATE, FLIGHT_STEPS_RUNNER (port_py.sh), FLIGHT_STEPS_PREFLIGHT,
#   FLIGHT_STEPS_LEASE_RUNNER, FLIGHT_STEPS_PYTHON (host python3), CPUSET (default 160-191, the free NUMA node).

FS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OMNI_ROOT=${OMNI_ROOT:-/mnt/cephfs/xingchao/omni}
STUDY=${FLIGHT_STEPS_STUDY:-$OMNI_ROOT/work/dev/umbrella/studies/speech-aware-evidence-acquisition}
STATE=${FLIGHT_STEPS_STATE:-$OMNI_ROOT/work/flight_state/fold-b-npu-port}
RUNNER=${FLIGHT_STEPS_RUNNER:-$OMNI_ROOT/work/scripts/port_py.sh}
HOST_PY=${FLIGHT_STEPS_PYTHON:-python3}
PREFLIGHT_CMD=${FLIGHT_STEPS_PREFLIGHT:-"$HOST_PY $FS_DIR/preflight.py"}
CPUSET=${CPUSET:-160-191}
RUNTIME_ROOT=$OMNI_ROOT/work/scratch/runtime-root
IMAGE=saea-base:20261027

FS_CONFIRM=0
FS_ARGS=()
FS_CMD_LABEL=()
FS_CMD_STR=()

fs_usage_common() {
  cat <<EOF
common flags:
  --confirm   execute (without it the step is a dry-run: it prints the command and reports its gates)
  -h|--help   this text
state directory: $STATE   (markers <step>.done / .failed / .running, logs <step>.log)
EOF
}

# fs_parse "$@": sets FS_CONFIRM and FS_ARGS (the step's own flags, in order).
fs_parse() {
  FS_CONFIRM=0; FS_ARGS=()
  local arg
  for arg in "$@"; do
    case "$arg" in
      --confirm) FS_CONFIRM=1 ;;
      -h|--help) fs_usage; exit 0 ;;
      *) FS_ARGS+=("$arg") ;;
    esac
  done
}

# fs_flag <name> <default>: value of "--name VALUE" or "--name=VALUE" from FS_ARGS, else the default.
fs_flag() {
  local name=$1 default=${2:-} i
  for ((i = 0; i < ${#FS_ARGS[@]}; i++)); do
    case "${FS_ARGS[$i]}" in
      "--$name") printf '%s' "${FS_ARGS[$((i + 1))]:-$default}"; return 0 ;;
      "--$name="*) printf '%s' "${FS_ARGS[$i]#--$name=}"; return 0 ;;
    esac
  done
  printf '%s' "$default"
}

fs_unknown_flags() {  # fs_unknown_flags <flag names...>: refuse an unknown flag; a name written "name:bool" takes no value
  local arg name allowed skip=0 known kind
  for arg in "${FS_ARGS[@]}"; do
    if [ $skip = 1 ]; then skip=0; continue; fi
    case "$arg" in
      --*=*) name=${arg%%=*}; name=${name#--} ;;
      --*) name=${arg#--} ;;
      *) echo "unexpected argument: $arg" >&2; exit 2 ;;
    esac
    known=0; kind=value
    for allowed in "$@"; do
      if [ "$allowed" = "$name" ]; then known=1; fi
      if [ "$allowed" = "$name:bool" ]; then known=1; kind=bool; fi
    done
    [ $known = 1 ] || { echo "unknown flag --$name" >&2; exit 2; }
    case "$arg" in --*=*) ;; *) [ $kind = value ] && skip=1 ;; esac
  done
}

# fs_load_cell smoke|flight: exports CI_* from the committed fragments.
fs_load_cell() {
  local out
  out=$($HOST_PY "$FS_DIR/cellinfo.py" "$1" --study "$STUDY" --omni-root "$OMNI_ROOT") || { echo "cannot read the $1 cell identities" >&2; exit 6; }
  eval "$out"
}

# fs_add_cmd <label> <word>...: register one command (stored quoted, run later with eval).
fs_add_cmd() {
  local label=$1; shift
  FS_CMD_LABEL+=("$label")
  FS_CMD_STR+=("$(printf '%q ' "$@")")
}

# fs_study_py <python args...>: the argv of a study command run inside saea-base through port_py.sh.
fs_study_py_argv() {
  printf '%s\0' env "CPUSET=$CPUSET" "SPEECHRL_DATA_DIR=$OMNI_ROOT" bash "$RUNNER" "$@"
}

fs_add_study() {  # fs_add_study <label> <python args...>
  local label=$1; shift
  local argv=()
  while IFS= read -r -d '' word; do argv+=("$word"); done < <(fs_study_py_argv "$@")
  fs_add_cmd "$label" "${argv[@]}"
}

fs_cli_common() {  # the five flags every core.flight.cli mode takes
  printf '%s\n' --repo-root "$STUDY" --data-root "$OMNI_ROOT" --runtime-root "$RUNTIME_ROOT" --lock-path "$CI_LOCK_PATH"
}

fs_head() { git -C "$STUDY" rev-parse HEAD 2>/dev/null; }

fs_marker() { printf '%s/%s' "$STATE" "$1"; }

# fs_gate <prev-step-or-empty> <preflight-stage>: prints a refusal reason and returns its exit code, or returns 0.
FS_GATE_REASON=""
fs_gate() {
  local prev=$1 stage=$2 pin head out
  FS_GATE_REASON=""
  if [ -e "$(fs_marker "$STEP.done")" ]; then FS_GATE_REASON="step $STEP is already done ($(fs_marker "$STEP.done")); remove that marker by hand to re-run"; return 5; fi
  if [ -e "$(fs_marker "$STEP.running")" ]; then
    local pid; pid=$(cat "$(fs_marker "$STEP.running")" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then FS_GATE_REASON="step $STEP is still running (pid $pid)"; return 5; fi
  fi
  if [ -n "$prev" ] && [ ! -e "$(fs_marker "$prev.done")" ]; then
    FS_GATE_REASON="previous step '$prev' has no done marker ($(fs_marker "$prev.done"))"; return 3
  fi
  pin=$(cat "$(fs_marker commit.txt)" 2>/dev/null); head=$(fs_head)
  if [ -n "$pin" ] && [ -n "$head" ] && [ "$pin" != "$head" ]; then
    FS_GATE_REASON="the study HEAD moved: pinned ${pin:0:12}, now ${head:0:12} (all steps fly from one commit)"; return 7
  fi
  if ! out=$($PREFLIGHT_CMD --stage "$stage" 2>&1); then
    FS_GATE_REASON="preflight (--stage $stage) failed"; FS_GATE_DETAIL=$out; return 4
  fi
  FS_GATE_DETAIL=$out
  # step-specific inputs (a step script may define fs_extra_check; it prints its own reason and returns a code)
  if declare -F fs_extra_check >/dev/null; then
    local reason; reason=$(fs_extra_check 2>&1) || { FS_GATE_REASON=$reason; return 6; }
  fi
  return 0
}

fs_show_cmds() {
  local i
  echo "step $STEP  state dir $STATE"
  for ((i = 0; i < ${#FS_CMD_STR[@]}; i++)); do
    echo "command $((i + 1))/${#FS_CMD_STR[@]} [${FS_CMD_LABEL[$i]}]:"
    echo "  \$ ${FS_CMD_STR[$i]}"
  done
}

# fs_main <prev-step-or-empty> <preflight-stage> <foreground|detached>
# The step script has filled FS_CMD_* before calling this.
fs_main() {
  local prev=$1 stage=$2 mode=$3 rc
  fs_show_cmds
  FS_GATE_DETAIL=""
  fs_gate "$prev" "$stage"; rc=$?
  if [ "$FS_CONFIRM" != 1 ]; then
    if [ $rc = 0 ]; then echo "gates: all pass (previous marker ${prev:-none}, preflight --stage $stage)"
    else echo "gates: WOULD REFUSE (exit $rc): $FS_GATE_REASON"; [ $rc = 4 ] && printf '%s\n' "$FS_GATE_DETAIL" | sed 's/^/  | /'; fi
    echo "DRY-RUN: nothing was executed. Re-run with --confirm to execute."
    exit 0
  fi
  if [ $rc != 0 ]; then
    echo "REFUSED (exit $rc): $FS_GATE_REASON" >&2
    [ $rc = 4 ] && printf '%s\n' "$FS_GATE_DETAIL" | sed 's/^/  | /' >&2
    exit $rc
  fi
  mkdir -p "$STATE" || exit 6
  [ -s "$(fs_marker commit.txt)" ] || { [ "$STEP" = lease ] && fs_head > "$(fs_marker commit.txt)"; }
  local commit; commit=$(cat "$(fs_marker commit.txt)" 2>/dev/null)
  if [ "$mode" = custom ]; then return 0; fi   # the step script carries on by itself (lease only)
  if [ "$mode" = detached ]; then
    local joined=""
    # Chain the registered commands in one detached shell; stop on the first failure.
    local i script=""
    for ((i = 0; i < ${#FS_CMD_STR[@]}; i++)); do script+="${FS_CMD_STR[$i]} && "; done
    script+="true"
    nohup setsid bash "$FS_DIR/_detached.sh" "$STATE" "$STEP" "${commit:-unpinned}" "$script" >/dev/null 2>&1 </dev/null &
    sleep 1
    echo "started detached: step $STEP (pid file $(fs_marker "$STEP.running"), log $(fs_marker "$STEP.log"))"
    echo "returns now; the done marker $(fs_marker "$STEP.done") appears on success, $(fs_marker "$STEP.failed") on failure. Check the watchdog and the log; nothing advances by itself."
    exit 0
  fi
  local script i
  : > "$(fs_marker "$STEP.log")"
  for ((i = 0; i < ${#FS_CMD_STR[@]}; i++)); do
    echo "--- [${FS_CMD_LABEL[$i]}] $(date -u +%FT%TZ)" | tee -a "$(fs_marker "$STEP.log")"
    eval "${FS_CMD_STR[$i]}" 2>&1 | tee -a "$(fs_marker "$STEP.log")"
    rc=${PIPESTATUS[0]}
    if [ $rc != 0 ]; then
      printf '{"step":"%s","rc":%s,"finished_utc":"%s","log":"%s"}\n' "$STEP" "$rc" "$(date -u +%FT%TZ)" "$(fs_marker "$STEP.log")" > "$(fs_marker "$STEP.failed")"
      echo "FAILED: command $((i + 1)) [${FS_CMD_LABEL[$i]}] exit $rc; marker $(fs_marker "$STEP.failed")" >&2
      exit 8
    fi
  done
  rm -f "$(fs_marker "$STEP.failed")"
  printf '{"step":"%s","rc":0,"finished_utc":"%s","commit":"%s","log":"%s"}\n' "$STEP" "$(date -u +%FT%TZ)" "${commit:-unpinned}" "$(fs_marker "$STEP.log")" > "$(fs_marker "$STEP.done")"
  echo "DONE: step $STEP; marker $(fs_marker "$STEP.done")"
  exit 0
}
