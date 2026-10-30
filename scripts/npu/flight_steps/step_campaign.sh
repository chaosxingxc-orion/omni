#!/usr/bin/env bash
# Step 5 (flight chain): the Fold B port flight (1,080 rows; 32,400 generation calls) as one coordinator campaign.
# Detached: returns at once; log $STATE/campaign.log, done marker on a clean exit. The campaign takes its own generation
# custody (no lease step), seals the run and releases custody itself. Requires signoff.done (the smoke sign-off).
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
STEP=campaign

fs_usage() {
  cat <<EOF
usage: step_campaign.sh [--confirm]
  core.flight.cli campaign for the flight cell inside saea-base, detached (nohup setsid).
  Requires signoff.done and preflight --stage flight. Watch it with the watchdog, not by tailing the log.
EOF
  fs_usage_common
}
fs_parse "$@"
fs_unknown_flags
fs_load_cell flight

fs_trace "$CI_TRACE_READ_ID"   # flight trace file: analysis/<port READ id>/rollout-trace.jsonl
mapfile -t COMMON < <(fs_cli_common)
fs_add_study "campaign (flight)" -m speech_aware_evidence_acquisition.core.flight.cli campaign "${COMMON[@]}" \
  --experiment "$CI_EXPERIMENT" --owner "$CI_OWNER" --expected-session-id "$CI_SESSION_ID"
fs_extra_check() {
  [ -s "$(fs_marker signoff.marker)" ] || { echo "no sign-off marker $(fs_marker signoff.marker)"; return 1; }
}
fs_main signoff flight detached
