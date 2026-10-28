#!/usr/bin/env bash
# Build the SAEA base experiment image. RUN THIS ON AN NPU HOST (ssh aliases npu or npu2):
#   ssh npu  'bash /mnt/cephfs/xingchao/docker/saea-base/build.sh'
# It must run there because those hosts already hold quay.io/ascend/vllm-ascend:v0.30.0rc2
# (24 GB) and have no internet to pull it, while this build context on the shared disk carries
# everything else the image needs. Runs detached and logs beside itself, so an ssh drop cannot
# kill it. Both hosts see the same /mnt/cephfs, but a Docker image is local to the host that
# built it -- build on each host that has to run it, or `docker save`/`load` through the share.
set -uo pipefail
C=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-saea-base:20261027}
LOG=$C/build-$(date +%Y%m%dT%H%M%S).log

if [ -z "${BUILD_FOREGROUND:-}" ]; then
  BUILD_FOREGROUND=1 nohup setsid bash "$0" "$@" >> "$LOG" 2>&1 < /dev/null &
  echo "build started in background: pid $!  log $LOG"
  exit 0
fi

fail() { echo "PRECHECK FAILED: $*" >&2; exit 1; }
echo "=== prechecks ($(date '+%F %T')) ==="
docker image inspect quay.io/ascend/vllm-ascend:v0.30.0rc2 >/dev/null 2>&1 \
  || fail "base image quay.io/ascend/vllm-ascend:v0.30.0rc2 is not present on this host"
n=$(find "$C/wheelhouse" -name '*.whl' 2>/dev/null | wc -l)
[ "$n" -gt 0 ] || fail "wheelhouse is empty -- stage it from the proxy host first"
echo "  wheelhouse: $n wheels"
[ -f "$C/adapters/pyproject.toml" ] || fail "adapter layer is not staged at adapters/"
n=$(find "$C/adapters/tests" -name 'test_*.py' | wc -l)
[ "$n" -gt 0 ] || fail "adapter tests are missing -- the attention gate must be test-covered"
echo "  adapters: $(find "$C/adapters" -name '*.py' | wc -l) modules, $n test files"
S=$C/umbrella/studies/speech-aware-evidence-acquisition
[ -f "$C/umbrella/docs/datasets.lock.json" ] \
  || fail "umbrella docs/datasets.lock.json missing -- the registered layout is required"
[ -f "$S/pyproject.toml" ] || fail "study not staged at studies/speech-aware-evidence-acquisition"
echo "  umbrella: $(du -sh "$C/umbrella" | cut -f1)"
# Guard the layout the package itself asserts at import time, before the build.
[ -f "$S/../../docs/datasets.lock.json" ] || fail "study is not two levels below the umbrella lock"

echo "=== docker build -> $TAG ($(date '+%F %T')) ==="
docker build -t "$TAG" "$C" || { echo "BUILD FAILED" >&2; exit 1; }

echo "=== smoke ($(date '+%F %T')) ==="
docker run --rm "$TAG" bash -lc '
  set -e
  P=/opt/saea-venv/bin/python
  "$P" -c "import openjiuwen; print(\"openjiuwen\", getattr(openjiuwen, \"__version__\", \"(no __version__)\"))"
  # versions via importlib.metadata: jiwer 4.x has no __version__ attribute
  "$P" -c "import jiwer, numpy, soundfile, more_itertools, regex, mlflow; from importlib.metadata import version as v; print(\"deps ok numpy\", v(\"numpy\"), \"jiwer\", v(\"jiwer\"), \"mlflow-skinny\", v(\"mlflow-skinny\"))"
  "$P" -c "from speech_aware_evidence_acquisition.core.config import compose; print(\"saea core imports ok\")"
  "$P" -c "import speech_aware_evidence_acquisition.data.lock as L; print(\"umbrella lock\", L.umbrella_lock_path())"
  /opt/saea-venv/bin/saea-driver --help > /dev/null && echo "saea-driver ok"
  # the serving side must be untouched by the venv
  python3 -c "import vllm; print(\"vllm\", vllm.__version__)"
  python3 -c "
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
name = \"Qwen3OmniMoeForConditionalGeneration\"
print(name, \"registered:\", name in archs)
"
  # the encoder-preparation side: in-process Omni-Embed-Nemotron-3B needs the image torch
  python3 -c "
import torch, transformers, soundfile
print(\"prep stack: torch\", torch.__version__, \"transformers\", transformers.__version__, \"soundfile\", soundfile.__version__)
from speech_aware_evidence_acquisition.core.speech_embedding import load_stack, encode, SAMPLE_RATE_HZ
from speech_aware_evidence_acquisition.core.speech_preparation_runtime import NemotronSpeechEncoder, NemotronPassageEncoder, verify_encoder
print(\"encoder prep imports ok, target rate\", SAMPLE_RATE_HZ)
"
  # the adapter layer, and the gate that has to be able to FAIL -- a gate whose own
  # tests cannot run is not a gate, so its suite is part of the build, not optional
  python3 -c "
from saea_adapters import device, encoder, attention_gate, backends
print(\"adapters ok | resolved device:\", device.resolve())
print(\"contracts: nemotron geometry\", backends.HISTORICAL_CONTRACT[\"attention_geometry\"],
      \"| lco-3b dim\", backends.LCO_3B_CONTRACT[\"embedding_dimension\"],
      \"| lco-7b dim\", backends.LCO_7B_CONTRACT[\"embedding_dimension\"])
"
  python3 -m pytest /opt/adapters/tests -q -p no:cacheprovider
  "$P" -c "from saea_adapters import backends; print(\"adapters reachable from the harness venv too\")"
' || { echo "SMOKE FAILED" >&2; exit 1; }
echo "IMAGE_BUILD_DONE $TAG"
