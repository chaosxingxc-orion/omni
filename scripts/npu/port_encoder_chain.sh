#!/usr/bin/env bash
# Run the three registered port encoder runs one after another (they share cores 128-191), detached on the rollout host.
# usage: nohup bash port_encoder_chain.sh > "$OMNI_ROOT"/work/logs/port-encoder-chain-<stamp>.log 2>&1 &
# Order: intent-boundary keys (60 passages), Fold B speech queries (1,080), MF-7b keys pass (11,514) + kus3 conversion.
# Stops at the first failing step. Each step's own log is named by its launcher.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../lib/local_profile.sh
. "$HERE/../lib/local_profile.sh" || { echo "cannot load the deployment profile bridge" >&2; exit 6; }
lp_default TREE roots.source || exit 6
lp_default OMNI_ROOT storage.shared_root || exit 6
S=$TREE/studies/speech-aware-evidence-acquisition
SCRIPTS=$OMNI_ROOT/work/scripts
say() { echo "[$(date +%FT%T)] $*"; }

run_step() {
  local name=$1; shift
  say "launch $name: $*"
  "$@" || { say "FAILED to launch $name"; return 1; }
  for _ in $(seq 1 60); do docker ps --format '{{.Names}}' | grep -qx "$name" && break; sleep 1; done
  if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then say "FAILED: $name never appeared"; return 1; fi
  local code
  code=$(docker wait "$name" 2>/dev/null) || code=unknown
  say "$name exited with code $code"
  [ "$code" = "0" ]
}

say "head $(git -c safe.directory='*' -C "$S" rev-parse HEAD)"
run_step saea-port-encoder-keys bash "$SCRIPTS/port_encoder_prep.sh" keys || exit 1
run_step saea-port-encoder-fold-b bash "$SCRIPTS/port_encoder_prep.sh" fold-b || exit 1
run_step saea-port-mf7b-keys bash "$SCRIPTS/port_mf7b_keys.sh" || exit 1
say "ALL THREE RUNS SEALED"
