#!/usr/bin/env bash
# Run a study command (python args) in saea-base against our data root. usage: bash port_py.sh <python args...>
# PORT_DOCKER_FLAGS (optional, default empty): extra docker run flags, e.g. "-i --name X" for a process that reads stdin.
set -uo pipefail
# Explicit runtime image pin for successor flights; retain the historical default.
PORT_IMAGE=${PORT_IMAGE:-saea-base:20261027}
TREE=${TREE:-/mnt/cephfs/xingchao/omni/work/dev/umbrella}
S=$TREE/studies/speech-aware-evidence-acquisition
CPUSET=${CPUSET:-160-191}
# Preserve argv across the container boundary. The optional flags remain a whitespace-separated
# list for compatibility; never evaluate them as shell code or expand filename patterns.
read -r -a EXTRA_FLAGS <<< "${PORT_DOCKER_FLAGS:-}"
# Isolate the framework's relative log directory per invocation. Existing study
# logs may belong to a previous container user; never chmod or overwrite them.
PORT_LOG_ROOT=${PORT_LOG_ROOT:-/mnt/cephfs/xingchao/omni/work/logs/port-invocations}
mkdir -p "$PORT_LOG_ROOT" || exit 1
PORT_LOG_DIR=$(mktemp -d "$PORT_LOG_ROOT/run-XXXXXXXX") || exit 1
DEPLOYMENT_ENV=()
# The pinned SLURP evaluator has its own Python 3.9 dependencies.
# Inference clients remain in the image's study venv.
SCORER_PYTHON=${SAEA_SLURP_OFFICIAL_PYTHON:-/mnt/cephfs/xingchao/omni/work/build/slurp-official-env-v1/python39/bin/python}
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
exec docker run --rm "${EXTRA_FLAGS[@]}" --user "$PORT_CONTAINER_USER" -e HOME=/tmp --cpuset-cpus "$CPUSET" -v /mnt/cephfs/xingchao/omni:/mnt/cephfs/xingchao/omni -w "$S" \
  -v "$PORT_LOG_DIR:$S/logs" "${DEPLOYMENT_ENV[@]}" \
  -e PYTHONPATH="$S/src" -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR=/mnt/cephfs/xingchao/omni \
  --entrypoint bash "$PORT_IMAGE" -c 'git config --global --add safe.directory "$PWD" && exec /opt/saea-venv/bin/python "$@"' port-python "$@"
