#!/usr/bin/env bash
# Step 9 (flight chain): close the exposure-ledger rows of the flown run ids (ledger_mark_consumed.py).
# Dry-run of the closer first, then --apply. It edits docs/exposure-ledger.md in the study tree and COMMITS NOTHING:
# the main thread reviews the diff and commits. Idempotent (only rows still at consumed = no with sealed evidence change).
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
STEP=ledger_close

fs_usage() {
  cat <<EOF
usage: step_ledger_close.sh [--today YYYY-MM-DD] [--confirm]
  ledger_mark_consumed.py for the four run ids (smoke, smoke FIT, flight, flight FIT): first a dry-run, then --apply.
  Requires read.done. Writes docs/exposure-ledger.md (uncommitted) and $STATE/ledger_close.done.
EOF
  fs_usage_common
}
fs_parse "$@"
fs_unknown_flags today
SMOKE_ENV=$(python3 "$FS_DIR/cellinfo.py" smoke --study "$STUDY" --omni-root "$OMNI_ROOT") || exit 6
eval "$(printf '%s\n' "$SMOKE_ENV" | sed 's/^CI_/SM_/')"
fs_load_cell flight

TODAY=$(fs_flag today "$(date -u +%F)")
IDS=()
for id in "$SM_RUN_ID" "$SM_FIT_RUN_ID" "$CI_RUN_ID" "$CI_FIT_RUN_ID"; do IDS+=(--run-id "$id"); done
CLOSE=(scripts/flights/batch/ledger_mark_consumed.py --repo-root "$STUDY" --data-root "$OMNI_ROOT" --today "$TODAY" "${IDS[@]}")
fs_add_study "ledger close, dry-run" "${CLOSE[@]}"
fs_add_study "ledger close, APPLY (edits docs/exposure-ledger.md, uncommitted)" "${CLOSE[@]}" --apply
fs_main read post foreground
