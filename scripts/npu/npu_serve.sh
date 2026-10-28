#!/usr/bin/env bash
# Serve one model for the port's NPU smoke, on explicitly named free NPU cards only.
# NOT for use while the GLM job holds the cards: check `npu-smi info` first; this script refuses busy cards.
#
# usage (on an NPU host):
#   bash npu_serve.sh thinking <cards e.g. 0,1> <port> [tp]
#   bash npu_serve.sh embed <view-id> <card> <port>
# The container mounts ONLY the named cards, so the harness telemetry (read_gpu -> npu-smi) sees exactly them.
set -euo pipefail
KIND=$1; shift
M=/mnt/cephfs/xingchao/models
LOGS=/mnt/cephfs/xingchao/work/logs
IMAGE=saea-base:20261027
BUSY_MIB=4096
SRC=/mnt/cephfs/xingchao/work/dev/umbrella/studies/speech-aware-evidence-acquisition/src

devices() {
  local args=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
  local card
  for card in ${1//,/ }; do args+=(--device "/dev/davinci$card"); done
  echo "${args[@]}"
}

refuse_busy() {
  # The study's own npu-smi table parser (core/session/telemetry.py), loaded as a standalone file.
  npu-smi info | python3 -I -c '
import importlib.util, sys
spec = importlib.util.spec_from_file_location("telemetry", sys.argv[1])
telemetry = importlib.util.module_from_spec(spec); spec.loader.exec_module(telemetry)
rows = {row["npu_id"]: row for row in telemetry.parse_npu_smi_table(sys.stdin.read())}
bad = [c for c in map(int, sys.argv[2].split(",")) if c not in rows or rows[c]["memory_used_mib"] > int(sys.argv[3])]
if bad:
    sys.exit("refused: NPU(s) %s busy or unreadable: %s" % (bad, {c: rows.get(c, {}).get("memory_used_mib") for c in bad}))
' "$SRC/speech_aware_evidence_acquisition/core/session/telemetry.py" "$1" "$BUSY_MIB"
}

run() {
  local name=$1 cards=$2; shift 2
  local log=$LOGS/$name-$(date +%Y%m%dT%H%M%S).log
  # shellcheck disable=SC2046
  nohup docker run --rm --name "$name" --network host --shm-size 16g --cpuset-cpus 128-191 \
    $(devices "$cards") \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v /mnt/cephfs/xingchao:/mnt/cephfs/xingchao \
    --entrypoint vllm "$IMAGE" serve "$@" > "$log" 2>&1 < /dev/null &
  echo "started $name; log $log"
}

case $KIND in
  thinking)
    CARDS=$1 PORT=$2 TP=${3:-1}
    refuse_busy "$CARDS"
    MESSAGE="Considering the limited time by the user, I have to give the solution based on the thinking directly now."
    # The forced string's whitespace around the message is the section-4 ASSUMPTION the smoke verifies.
    REASONING=$(python3 -c 'import json,sys; print(json.dumps({"reasoning_end_str": sys.argv[1] + "\n</think>\n\n"}))' "$MESSAGE")
    run saea-serve-thinking "$CARDS" "$M/qwen3-omni-30b-a3b-thinking" \
      --served-model-name qwen3-omni-30b-a3b-thinking --host 127.0.0.1 --port "$PORT" \
      --dtype bfloat16 --tensor-parallel-size "$TP" \
      --max-model-len 24576 --max-num-seqs 4 \
      --reasoning-parser qwen3 --reasoning-config "$REASONING" \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --limit-mm-per-prompt '{"audio": 1, "image": 0, "video": 0}' \
      --seed 0
    ;;
  embed)
    VIEW=$1 CARD=$2 PORT=$3
    refuse_busy "$CARD"
    ARGS=$(PYTHONPATH=$SRC \
      docker run --rm -v /mnt/cephfs/xingchao:/mnt/cephfs/xingchao -e PYTHONPATH \
      --entrypoint /opt/saea-venv/bin/python "$IMAGE" -c '
import json, sys
from pathlib import Path
from speech_aware_evidence_acquisition.core import vllm_serving_view as sv
view = Path(sys.argv[1])
receipt = sv.verify_serving_view(view)
argv = sv.vllm_embedding_serve_args(view, receipt, served_model_name=sys.argv[2], max_model_len=8192,
                                    host="127.0.0.1", port=int(sys.argv[3]))
assert not any("\n" in a for a in argv)
print("\n".join(argv[2:]))' "$M/serving-views/$VIEW" "$VIEW" "$PORT")
    mapfile -t SERVE <<< "$ARGS"
    run "saea-serve-embed-$VIEW" "$CARD" "${SERVE[@]}"
    ;;
  *) echo "unknown kind $KIND" >&2; exit 1 ;;
esac
