#!/usr/bin/env bash
# Step 8 (flight chain): the preregistered READ by an analysis recipe (model-free).
# The recipe path is a parameter (the port recipe: docs/readiness/<port package>/analysis-recipe-v1.json). The default reader is the PORT reader
# (scripts/analysis/slurp_l338_port_column_read.py: port cell ids, port analysis root, port-verdicts.json). The step refuses unless the recipe's
# READ run id has a root-countersigned exposure-ledger row (read_gate.py), then runs the reader.
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
STEP=read

fs_usage() {
  cat <<EOF
usage: step_read.sh --recipe REPO_RELATIVE_PATH [--reader REPO_RELATIVE_SCRIPT] [--pass-recipe] [--confirm]
  --recipe       analysis recipe JSON in the study tree (required), e.g. docs/readiness/<port package>/analysis-recipe-v1.json
  --reader       reader script (default scripts/analysis/slurp_l338_port_column_read.py); invoked as
                 <reader> --stage read --data-root <root> --repo-root <study> --analysis-root <root>/analysis/<READ run id>
                 (<READ run id> comes from the recipe; these flags are inferred, see the README)
  --pass-recipe  also append "--recipe <recipe>" for a reader that takes one (always done for the port reader, which validates the recipe's port ids)
  Writes read.json, README.md and port-verdicts.json under <root>/analysis/<READ id> (write-once).
  Requires score.done and a countersigned READ row.
EOF
  fs_usage_common
}
fs_parse "$@"
fs_unknown_flags recipe reader pass-recipe:bool
fs_load_cell flight

RECIPE=$(fs_flag recipe "")
READER=$(fs_flag reader scripts/analysis/slurp_l338_port_column_read.py)
PASS_RECIPE=0
case "$READER" in *slurp_l338_port_column_read.py) PASS_RECIPE=1 ;; esac
for a in "${FS_ARGS[@]}"; do [ "$a" = --pass-recipe ] && PASS_RECIPE=1; done
if [ -z "$RECIPE" ]; then
  if [ "$FS_CONFIRM" = 1 ]; then echo "REFUSED: --recipe is required" >&2; exit 2; fi
  RECIPE="<REQUIRED: --recipe PATH>"
fi
READ_ID="<READ run id from the recipe>"
if [ -f "$STUDY/$RECIPE" ]; then
  READ_ID=$($HOST_PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('read_run_id') or (d.get('run_ids') or {}).get('read') or '')" "$STUDY/$RECIPE") || READ_ID=""
fi
READ_ARGS=("$READER" --stage read --data-root "$OMNI_ROOT" --repo-root "$STUDY" --analysis-root "$OMNI_ROOT/analysis/$READ_ID")
[ $PASS_RECIPE = 1 ] && READ_ARGS+=(--recipe "$RECIPE")
fs_add_study "READ gate (countersigned READ row)" "$FS_DIR/read_gate.py" --recipe "$RECIPE" --repo-root "$STUDY"
fs_add_study "preregistered read" "${READ_ARGS[@]}"
fs_extra_check() {
  [ -f "$STUDY/$RECIPE" ] || { echo "the recipe $STUDY/$RECIPE does not exist"; return 1; }
  [ -n "$READ_ID" ] || { echo "the recipe declares no READ run id"; return 1; }
}
fs_main score post foreground
