#!/usr/bin/env bash
# Step 4 (smoke chain): machine sign-off of the smoke against the registered conditions (smoke_signoff.py).
# Reads the smoke decode-cost table and decides PASS or FAIL against signoff-conditions-v1.json. It touches no run data;
# the only file it writes is the sign-off marker (PASS) or <marker>.refused (FAIL) under the state directory.
# The decode-cost table is written by the smoke reader (slurp_l338_column_read.py --stage smoke, or its port successor, open
# item F); this step does not produce it and refuses when it is missing.
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
STEP=signoff

fs_usage() {
  cat <<EOF
usage: step_signoff.sh [--table PATH] [--confirm]
  smoke_signoff.py --conditions <package>/signoff-conditions-v1.json --data-root --marker $STATE/signoff.marker
  --table defaults to <data root>/<conditions.table_relpath>. Exit 3 of smoke_signoff (FAIL) leaves $STATE/signoff.failed.
  Requires fit.done.
EOF
  fs_usage_common
}
fs_parse "$@"
fs_unknown_flags table
fs_load_cell smoke

TABLE=$(fs_flag table "")
ARGS=(scripts/flights/batch/smoke_signoff.py --conditions "$CI_SIGNOFF_CONDITIONS" --data-root "$OMNI_ROOT" --marker "$(fs_marker signoff.marker)")
[ -z "$TABLE" ] || ARGS+=(--table "$TABLE")
fs_add_study "smoke sign-off (machine, read-only)" "${ARGS[@]}"
fs_extra_check() {
  local table=$TABLE rel
  if [ -z "$table" ]; then
    rel=$($HOST_PY -c "import json,sys;print(json.load(open(sys.argv[1]))['table_relpath'])" "$STUDY/$CI_SIGNOFF_CONDITIONS") || { echo "cannot read the sign-off conditions"; return 1; }
    table=$OMNI_ROOT/$rel
  fi
  [ -s "$table" ] || { echo "the smoke decode-cost table $table does not exist; write it with the smoke reader first"; return 1; }
}
fs_main fit post foreground
