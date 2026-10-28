#!/usr/bin/env bash
# Offline wheelhouse for the SAEA base image: every wheel the aarch64 / cp312 container needs,
# downloaded ON THE DEV HOST (the NPU box has no route to the internet) into the Docker build
# context, so the image builds with `pip install --no-index --find-links`.
#
# Resumable and re-runnable: pip reuses whatever is already in the wheelhouse, so a dropped
# proxy connection costs only the file in flight. --resume-retries is the flag that actually
# matters here: this proxy interrupts large transfers routinely (the first run died on
# burner_redis at 1.0/1.3 MB after pip's default 5 resume attempts).
#
# Why these packages:
#   openjiuwen 0.1.16.post2  the executor the registered SAEA run uses (baseline bare-core-ojw,
#                            policy obs-agent-loop-ojw); pinned exactly as the ledger records.
#   numpy / soundfile / jiwer  the study's declared TEST-TIME trio (pyproject [dependency-groups]
#                            dev); jiwer>=4.0 is tight on purpose -- jiwer.wer IS a measurement.
#   more-itertools, regex    reached at IMPORT time by reproduction/whisper_normalizer, which
#                            core/__init__ pulls in, so the package will not import without them.
#   mlflow-skinny            core/tracking.py uses only set_tracking_uri / set_experiment /
#                            start_run / log_params / log_metrics / log_artifact against a local
#                            file store, all of which skinny provides without the server stack.
#   pytest                   runbook precondition 1 is `pytest tests/contract/test_real_receipts.py`.
# Runs detached by default.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export DL_LOG=${DL_LOG:-$(dirname "$HERE")/dl_logs/wheelhouse.log}
source "$HERE/_daemonize.sh"

WH=${WH:-/mnt/cephfs/xingchao/docker/saea-base/wheelhouse}
PIP=$HOME/venvs/dl/bin/pip
INDEX=${PIP_INDEX:-https://mirrors.huaweicloud.com/repository/pypi/simple}
mkdir -p "$WH"

PKGS=(
  "openjiuwen==0.1.16.post2"
  "numpy>=2.0,!=2.4.0"
  "soundfile>=0.12"
  "jiwer>=4.0"
  "more-itertools>=10"
  "regex>=2024.0.0"
  "mlflow-skinny>=2.14"
  "pytest>=8"
)

for attempt in 1 2 3 4 5 6 7 8; do
  echo "=== attempt $attempt ($(date '+%F %T')), wheelhouse has $(ls "$WH" | wc -l) files ==="
  "$PIP" download --dest "$WH" \
    --index-url "$INDEX" \
    --retries 20 --timeout 120 --resume-retries 200 \
    --python-version 3.12 --implementation cp --abi cp312 \
    --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 --platform any \
    --only-binary=:all: \
    "${PKGS[@]}" && { echo "WHEELHOUSE_DONE attempt=$attempt files=$(ls "$WH" | wc -l)"; exit 0; }
  echo "--- attempt $attempt failed; wheels already in $WH are reused on the next pass ---"
  sleep 20
done
echo "WHEELHOUSE_FAILED after 8 attempts" >&2
exit 1
