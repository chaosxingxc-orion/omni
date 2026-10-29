#!/usr/bin/env bash
# Step 1 (smoke chain): take the worker lease for the smoke-4 cell and hold it for the smoke worker.
# Runs core.flight.lease_cli --stage worker inside saea-base (docker run -i) detached; it waits for the token file and
# returns. The lease stays held until step_smoke_worker.sh releases it (or you reconcile it by hand after a failure).
# The owner steps (countersign, receipt mint, readiness flip) happen BEFORE this step and are not scripted.
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
STEP=lease

fs_usage() {
  cat <<EOF
usage: step_lease.sh [--confirm]
  Takes the generation lease for the smoke-4 cell (lease_cli --stage worker) and keeps it alive on a FIFO.
  Requires: preflight --stage smoke passes. Writes: $STATE/lease.done (and lease.fifo, lease.log, commit.txt).
EOF
  fs_usage_common
}
fs_parse "$@"
fs_unknown_flags
fs_load_cell smoke

TOKEN_REL=coordination/$CI_CELL.worker-lease.json
TOKEN=$OMNI_ROOT/$TOKEN_REL
FIFO=$(fs_marker lease.fifo)
LEASE_ARGS=(-m speech_aware_evidence_acquisition.core.flight.lease_cli --stage worker
  --repo-root "$STUDY" --data-root "$OMNI_ROOT" --runtime-root "$RUNTIME_ROOT" --lock-path "$CI_LOCK_PATH"
  --experiment "$CI_EXPERIMENT" --cell "$CI_CELL" --owner "$CI_OWNER" --expected-session-id "$CI_SESSION_ID"
  --token-output "$TOKEN_REL")

# lease_cli reads its release word from stdin, so port_py.sh runs it with `docker run -i` and a name
# (PORT_DOCKER_FLAGS, an optional hook of port_py.sh).
LEASE_CMD=(env CPUSET="$CPUSET" PORT_DOCKER_FLAGS="-i --name saea-flight-lease" bash "$RUNNER" "${LEASE_ARGS[@]}")
fs_add_cmd "lease holder (detached, stdin on $FIFO)" "${LEASE_CMD[@]}"

fs_main "" smoke custom   # dry-run exits inside; with --confirm it returns once every gate passed

[ ! -e "$TOKEN" ] || { echo "REFUSED: a lease token already exists at $TOKEN; reconcile it by hand" >&2; exit 6; }
rm -f "$FIFO" "$(fs_marker lease.exit)"
mkfifo "$FIFO" || exit 6
nohup setsid bash "$FS_DIR/_lease_holder.sh" "$FIFO" "$(fs_marker lease.log)" "$(fs_marker lease.exit)" "${LEASE_CMD[@]}" >/dev/null 2>&1 </dev/null &
HOLDER=$!
echo "$HOLDER" > "$(fs_marker lease.holder.pid)"
WAIT=${FLIGHT_STEPS_LEASE_WAIT:-90}
for ((i = 0; i < WAIT; i++)); do
  [ -f "$TOKEN" ] && break
  [ -f "$(fs_marker lease.exit)" ] && break
  sleep 1
done
if [ -f "$TOKEN" ]; then
  printf '{"step":"lease","rc":0,"finished_utc":"%s","commit":"%s","token":"%s","fifo":"%s","holder_pid":%s}\n' \
    "$(date -u +%FT%TZ)" "$(cat "$(fs_marker commit.txt)")" "$TOKEN" "$FIFO" "$HOLDER" > "$(fs_marker lease.done)"
  echo "DONE: lease held (token $TOKEN, holder pid $HOLDER). Next: step_smoke_worker.sh --confirm"
  exit 0
fi
printf '{"step":"lease","rc":8,"finished_utc":"%s","log":"%s"}\n' "$(date -u +%FT%TZ)" "$(fs_marker lease.log)" > "$(fs_marker lease.failed)"
echo "FAILED: no lease token within ${WAIT}s; see $(fs_marker lease.log). The holder was left alone: if lease_cli has exited, its lease metadata is preserved and must be reconciled by hand before another lease." >&2
exit 8
