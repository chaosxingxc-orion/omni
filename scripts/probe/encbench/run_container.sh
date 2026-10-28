#!/usr/bin/env bash
# Run `encbench.py <args>` inside saea-base:20261027 on THIS host (meant for host npu).
#   CPU (default): cores 128-191 only, no accelerator device nodes mounted (no NPU contact possible).
#   NPU (later):   ENCBENCH_CARDS=0[,1,...] mounts exactly those davinci cards + driver, refusing busy cards.
# usage: NAME=encbench-x bash run_container.sh run --encoder nemotron --device cpu ...
set -euo pipefail
HERE=/mnt/cephfs/xingchao/omni/work/scratch/encbench
SRC=/mnt/cephfs/xingchao/omni/work/dev/umbrella/studies/speech-aware-evidence-acquisition/src
IMAGE=saea-base:20261027
CPUSET=128-191
NAME=${NAME:-encbench-$$}
DEV=()
if [[ -n "${ENCBENCH_CARDS:-}" ]]; then
  # same busy-card refusal as work/scripts/npu_serve.sh (the study's npu-smi parser)
  npu-smi info | python3 -I -c '
import importlib.util, sys
spec = importlib.util.spec_from_file_location("telemetry", sys.argv[1])
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
rows = {r["npu_id"]: r for r in t.parse_npu_smi_table(sys.stdin.read())}
bad = [c for c in map(int, sys.argv[2].split(",")) if c not in rows or rows[c]["memory_used_mib"] > 4096]
sys.exit("refused: NPU(s) %s busy or unreadable" % bad if bad else 0)
' "$SRC/speech_aware_evidence_acquisition/core/session/telemetry.py" "$ENCBENCH_CARDS"
  DEV=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro)
  for c in ${ENCBENCH_CARDS//,/ }; do DEV+=(--device "/dev/davinci$c"); done
fi
exec docker run --rm --name "$NAME" --user 1008:1008 -e HOME=/tmp --cpuset-cpus "$CPUSET" \
  "${DEV[@]}" -v /mnt/cephfs/xingchao/omni:/mnt/cephfs/xingchao/omni -w /tmp \
  -e PYTHONPATH="$SRC" -e PYTHONDONTWRITEBYTECODE=1 -e USER=encbench -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint /usr/local/python3.12.13/bin/python3 "$IMAGE" "$HERE/encbench.py" "$@"
