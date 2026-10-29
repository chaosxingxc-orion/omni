#!/usr/bin/env bash
# Step 3 (smoke chain): FIT acceptance for the smoke-4 cell, read-only.
# The native contact FIT of these cells is an INLINE gate of the worker/campaign under the same lease: the cells permit
# only check / prepare-supply / worker / campaign (no prompt-fit stage), so there is no separate FIT model run to start.
# This step verifies the evidence the worker left (FIT directory, flight receipts) and runs the driver's structural
# acceptance of the sealed smoke (batch_inspect.py smoke).
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
STEP=fit

fs_usage() {
  cat <<EOF
usage: step_fit.sh [--confirm]
  1. fit_evidence.py: prepared/<smoke FIT run>/<FIT attempt>/ non-empty and runs/<run>/<attempt>/flight-receipts.json present
  2. batch_inspect.py smoke: attempt/manifest structure, 4 rows, ceiling per row, tool surface, study commit
  Requires smoke_worker.done. Read-only; writes only $STATE/fit.done.
EOF
  fs_usage_common
}
fs_parse "$@"
fs_unknown_flags
fs_load_cell smoke

COMMIT=$(cat "$(fs_marker commit.txt)" 2>/dev/null); COMMIT=${COMMIT:-$(fs_head)}
fs_add_cmd "FIT evidence (host python, read-only)" "$HOST_PY" "$FS_DIR/fit_evidence.py" --data-root "$OMNI_ROOT" \
  --run "$CI_RUN_ID" --attempt "$CI_ATTEMPT_ID" --fit-run "$CI_FIT_RUN_ID" --fit-attempt "$CI_FIT_ATTEMPT_ID" --contact-cap "$CI_FIT_CONTACT_CAP"
fs_add_study "smoke structural acceptance" scripts/flights/batch/batch_inspect.py "$OMNI_ROOT" "$CI_RUN_ID" "$CI_ATTEMPT_ID" \
  smoke "$CI_ROWS" "$CI_CEILING_PER_ROW" 1 "$COMMIT" --result-kind task-answer
fs_main smoke_worker post foreground
