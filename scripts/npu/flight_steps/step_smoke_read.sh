#!/usr/bin/env bash
# Step 3b (smoke chain): the port smoke reader writes the smoke decode-cost table that the machine sign-off reads (model-free, CPU).
# Runs scripts/analysis/slurp_l338_port_column_read.py --stage smoke over the sealed smoke-4 run into
# <data root>/analysis/<port READ id>/smoke-decode-cost.json (the path signoff-conditions-v1.json names). Write-once: the reader refuses an
# existing table, and so does this step (remove the table by hand to re-run). Touches no run data; nothing is scored.
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
STEP=smoke_read

fs_usage() {
  cat <<EOF2
usage: step_smoke_read.sh [--read-id ID] [--confirm]
  slurp_l338_port_column_read.py --stage smoke --data-root --repo-root --analysis-root <data root>/analysis/<READ id>
  <READ id> defaults to the port recipe's READ run id. Requires fit.done. Writes the decode-cost table (write-once) and $STATE/smoke_read.done;
  step_signoff.sh requires smoke_read.done.
EOF2
  fs_usage_common
}
fs_parse "$@"
fs_unknown_flags read-id
fs_load_cell smoke

READ_ID=$(fs_flag read-id "$CI_READ_RUN_ID_DEFAULT")
ANALYSIS=$OMNI_ROOT/analysis/$READ_ID
fs_add_study "port smoke reader (decode-cost table)" scripts/analysis/slurp_l338_port_column_read.py --stage smoke \
  --data-root "$OMNI_ROOT" --repo-root "$STUDY" --analysis-root "$ANALYSIS"
fs_extra_check() {
  [ -n "$READ_ID" ] || { echo "no READ run id"; return 1; }
  [ ! -e "$ANALYSIS/smoke-decode-cost.json" ] || { echo "the decode-cost table $ANALYSIS/smoke-decode-cost.json exists (write-once); remove it by hand to re-run"; return 1; }
}
fs_main fit post foreground
