#!/usr/bin/env bash
# Read machine-specific values from the local deployment profile.
#
# Sourced by host entry points (port_py.sh, npu_deploy.sh, flight_steps, ...).
# The profile is the only place paths, hosts, ports and image identity live;
# this file is the only shell bridge to it.  There are no defaults: a missing
# or malformed profile stops the caller instead of falling back to someone's
# home directory.
#
# usage:
#   . "$REPO/scripts/lib/local_profile.sh"
#   lp_default OMNI_ROOT storage.shared_root || exit $?   # keep a caller override
#   IMAGE=$(lp_get image.reference) || exit $?
#
# The profile is read once per shell and cached, so a script that needs eight
# fields starts one interpreter, not eight.  OMNI_LOCAL_CONFIG overrides the
# profile path (tests and alternate deployments use it); the default is
# configs/local/deployment.json under OMNI_BASE_ROOT.

LP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LP_REPO=$(cd "$LP_DIR/../.." && pwd)
LP_GET="$LP_DIR/local_get.py"
# The bridge may be deployed away from the repository (work/scripts/lib beside
# work/configs).  OMNI_BASE_ROOT names the tree holding tools/ and configs/; the
# default is the repository this file was copied from.
OMNI_BASE_ROOT=${OMNI_BASE_ROOT:-$LP_REPO}
export OMNI_BASE_ROOT
OMNI_LOCAL_CONFIG=${OMNI_LOCAL_CONFIG:-$OMNI_BASE_ROOT/configs/local/deployment.json}
export OMNI_LOCAL_CONFIG

LP_LOADED=0
declare -gA LP_VALUES=()

lp_load() {  # read the whole profile once; fail loudly, never fall back
  [ "$LP_LOADED" = 1 ] && return 0
  local out line
  out=$(python3 "$LP_GET" --config "$OMNI_LOCAL_CONFIG" --all 2>&1) || {
    printf '%s\n' "$out" >&2
    return 2
  }
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    LP_VALUES["${line%%=*}"]=${line#*=}
  done <<< "$out"
  LP_LOADED=1
}

lp_get() {  # lp_get <dotted.key> [<dotted.key> ...] -> one value per line
  local key
  lp_load || return $?
  for key in "$@"; do
    if [ -z "${LP_VALUES[$key]+set}" ]; then
      printf 'local profile %s: no value for %s; add it in the deployment profile\n' \
        "$OMNI_LOCAL_CONFIG" "$key" >&2
      return 2
    fi
    printf '%s\n' "${LP_VALUES[$key]}"
  done
}

lp_default() {  # lp_default <VARNAME> <dotted.key>: fill VARNAME from the profile unless already set
  local name=$1 key=$2 value
  if [ -n "${!name:-}" ]; then
    return 0
  fi
  value=$(lp_get "$key") || return $?
  printf -v "$name" '%s' "$value"
}

lp_opt() {  # lp_opt <dotted.key>: the value, or an empty line when the profile omits it
  local key=$1
  lp_load || return $?
  printf '%s\n' "${LP_VALUES[$key]:-}"
}

lp_all() {  # every declared field as KEY=VALUE, for diagnostics
  python3 "$LP_GET" --config "$OMNI_LOCAL_CONFIG" --all
}

lp_config() {  # the resolved profile path, for error messages
  printf '%s\n' "$OMNI_LOCAL_CONFIG"
}
