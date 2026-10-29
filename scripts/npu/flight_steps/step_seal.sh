#!/usr/bin/env bash
# Step 6 (flight chain): seal acceptance of the flight, read-only.
# The campaign seals the run itself (runtime.seal); there is no separate seal command for a non-resume cell
# (core.flight.cli seal-only refuses a cell without resume_from). This step runs the driver's structural acceptance of the
# sealed arm (batch_inspect.py arm): attempt FINALIZED, manifest COMPLETED at the flight commit, 1,080 rows, ceilings,
# tool surface, one score job.
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
STEP=seal

fs_usage() {
  cat <<EOF
usage: step_seal.sh [--confirm]
  batch_inspect.py <data root> <run> <attempt> arm <rows> <ceiling per row> 1 <flight commit> --result-kind task-answer
  Requires campaign.done. Read-only; writes only $STATE/seal.done.
EOF
  fs_usage_common
}
fs_parse "$@"
fs_unknown_flags
fs_load_cell flight

COMMIT=$(cat "$(fs_marker commit.txt)" 2>/dev/null); COMMIT=${COMMIT:-$(fs_head)}
fs_add_study "flight structural acceptance" scripts/flights/batch/batch_inspect.py "$OMNI_ROOT" "$CI_RUN_ID" "$CI_ATTEMPT_ID" \
  arm "$CI_ROWS" "$CI_CEILING_PER_ROW" 1 "$COMMIT" --result-kind task-answer
fs_main campaign post foreground
