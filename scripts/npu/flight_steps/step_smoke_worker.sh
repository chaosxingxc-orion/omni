#!/usr/bin/env bash
# Step 2 (smoke chain): run the smoke-4 worker (120 generation calls; the native contact FIT runs inline) under the lease
# taken by step_lease.sh. Detached: returns at once; on a clean exit the lease is released, on a failure it is preserved.
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
STEP=smoke_worker

fs_usage() {
  cat <<EOF
usage: step_smoke_worker.sh [--confirm]
  core.flight.cli worker for the smoke-4 cell inside saea-base, detached (nohup setsid), log $STATE/smoke_worker.log,
  done marker $STATE/smoke_worker.done. Requires lease.done and preflight --stage smoke.
EOF
  fs_usage_common
}
fs_parse "$@"
fs_unknown_flags
fs_load_cell smoke

TOKEN=$OMNI_ROOT/coordination/$CI_CELL.worker-lease.json
mapfile -t COMMON < <(fs_cli_common)
fs_add_study "worker (smoke-4)" -m speech_aware_evidence_acquisition.core.flight.cli worker "${COMMON[@]}" \
  --experiment "$CI_EXPERIMENT" --cell "$CI_CELL" --expected-session-id "$CI_SESSION_ID" --lease-token-file "$TOKEN"

export FS_RELEASE_LEASE=$(fs_marker lease.fifo) FS_LEASE_TOKEN=$TOKEN
fs_extra_check() {
  [ -f "$TOKEN" ] || { echo "the lease token $TOKEN is absent; run step_lease.sh first"; return 1; }
  [ -p "$FS_RELEASE_LEASE" ] || { echo "the lease FIFO $FS_RELEASE_LEASE is absent"; return 1; }
}
fs_main lease smoke detached
