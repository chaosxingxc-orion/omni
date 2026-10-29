#!/usr/bin/env bash
# Step 7 (flight chain): official scoring of the sealed flight, one complete scoring per arm column
# (stage3_rollout_score.py --mode perception-branches). Model-free (CPU scorer). Detached: returns at once.
# The tool has no per-arm selector: one invocation renders and scores every arm column of the run once each (the same
# call the historical l338_read.sh `columns` stage makes).
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
STEP=score

fs_usage() {
  cat <<EOF
usage: step_score.sh [--read-id ID] [--confirm]
  Scores into <data root>/analysis/<READ id>/columns (default READ id $(python3 "$FS_DIR/cellinfo.py" flight --study "$STUDY" --omni-root "$OMNI_ROOT" 2>/dev/null | sed -n 's/^CI_READ_RUN_ID_DEFAULT=//p')).
  Requires seal.done. Detached, log $STATE/score.log.
EOF
  fs_usage_common
}
fs_parse "$@"
fs_unknown_flags read-id
fs_load_cell flight

READ_ID=$(fs_flag read-id "$CI_READ_RUN_ID_DEFAULT")
ANALYSIS=$OMNI_ROOT/analysis/$READ_ID/columns
fs_add_study "official scoring, every arm column" scripts/analysis/stage3_rollout_score.py --mode perception-branches --stage all \
  --run "$CI_RUN_ID" --attempt "$CI_ATTEMPT_ID" --roster "$CI_ROSTER" --analysis-root "$ANALYSIS" --scorer-id "$CI_SCORER_ID" \
  --repo-root "$STUDY" --data-root "$OMNI_ROOT"
[ "$FS_CONFIRM" = 1 ] && mkdir -p "$ANALYSIS"
fs_main seal post detached
