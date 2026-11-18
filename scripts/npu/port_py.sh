#!/usr/bin/env bash
# Run a study command (python args) in the pinned image against our data root. usage: bash port_py.sh <python args...>
# PORT_DOCKER_FLAGS (optional, default empty): extra docker run flags, e.g. "-i --name X" for a process that reads stdin.
#
# Paths and image identity come from the local deployment profile
# (scripts/lib/local_profile.sh, default configs/local/deployment.json):
#   storage.shared_root  the data root, mounted at the same path inside the container
#   roots.source         the umbrella tree that holds studies/
#   image.reference      the runtime image tag
# Every one of them can still be overridden per call (SPEECHRL_DATA_DIR, TREE, PORT_IMAGE).
set -uo pipefail
PORT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$PORT_DIR/../lib/local_profile.sh" || exit 1
lp_default OMNI_ROOT storage.shared_root || exit 1
lp_default TREE roots.source || exit 1
lp_default PORT_IMAGE image.reference || exit 1
S=$TREE/studies/speech-aware-evidence-acquisition
CPUSET=${CPUSET:-160-191}
# Preserve argv across the container boundary. The optional flags remain a whitespace-separated
# list for compatibility; never evaluate them as shell code or expand filename patterns.
read -r -a EXTRA_FLAGS <<< "${PORT_DOCKER_FLAGS:-}"
# Isolate the framework's relative log directory per invocation. Existing study
# logs may belong to a previous container user; never chmod or overwrite them.
PORT_LOG_ROOT=${PORT_LOG_ROOT:-$OMNI_ROOT/work/logs/port-invocations}
mkdir -p "$PORT_LOG_ROOT" || exit 1
PORT_LOG_DIR=$(mktemp -d "$PORT_LOG_ROOT/run-XXXXXXXX") || exit 1
DEPLOYMENT_ENV=()
# The pinned SLURP evaluator has its own Python 3.9 dependencies.
# Inference clients remain in the image's study venv.
SCORER_PYTHON=${SAEA_SLURP_OFFICIAL_PYTHON:-$OMNI_ROOT/work/build/slurp-official-env-v1/python39/bin/python}
if [[ -x "$SCORER_PYTHON" ]]; then
  DEPLOYMENT_ENV+=(-e "SAEA_SLURP_OFFICIAL_PYTHON=$SCORER_PYTHON")
fi
if [[ -n ${SAEA_DEPLOYMENT_PROFILE:-} ]]; then
  DEPLOYMENT_ENV+=(-e "SAEA_DEPLOYMENT_PROFILE=$SAEA_DEPLOYMENT_PROFILE")
fi
if [[ -n ${SAEA_ADMITTED_LEVEL+x} ]]; then
  DEPLOYMENT_ENV+=(-e "SAEA_ADMITTED_LEVEL=$SAEA_ADMITTED_LEVEL")
fi
# Live cross-container process verification needs the serving process's user.
# CPU/model-free invocations keep the caller's UID/GID unless explicitly selected.
PORT_CONTAINER_USER=${PORT_CONTAINER_USER:-$(id -u):$(id -g)}
exec docker run --rm "${EXTRA_FLAGS[@]}" --user "$PORT_CONTAINER_USER" -e HOME=/tmp --cpuset-cpus "$CPUSET" -v "$OMNI_ROOT:$OMNI_ROOT" -w "$S" \
  -v "$PORT_LOG_DIR:$S/logs" "${DEPLOYMENT_ENV[@]}" \
  -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR="$OMNI_ROOT" \
  --entrypoint bash "$PORT_IMAGE" -c 'git config --global --add safe.directory "$PWD" && exec /opt/saea-venv/bin/python "$@"' port-python "$@"