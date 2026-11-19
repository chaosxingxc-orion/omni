#!/usr/bin/env bash
# Run on the inference host. Arguments are resolved deployment paths and image ID.
set -euo pipefail
input=$(realpath "${1:?input directory}")
model=$(realpath "${2:?read-only model directory}")
image=${3:?inference image digest}
modelinfo=$(realpath "${4:?modelinfos directory of the same sealed rollout}")
test -d "$modelinfo"
hot="$input/hot"
case $(findmnt -n -o FSTYPE -T "$input") in ext4|xfs|btrfs) ;; *) exit 2;; esac
df -kP "$input" | awk 'NR==2 {exit !($4 >= 20*1024*1024 && $5+0 <= 90)}'
df -iP "$input" | awk 'NR==2 {exit !($5+0 <= 95)}'
mkdir -p "$hot"/{tmp,cache,hf,home,inductor,triton,ascend,logs}
mkdir -p "$hot/cache/vllm/modelinfos"
exec docker run -d --network none --read-only --cpus 4 --memory 16g --user "$(id -u):$(id -g)" \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro --entrypoint python -w /hot \
    -v "$input:/e:ro" -v "$hot:/hot" -v "$modelinfo:/hot/cache/vllm/modelinfos:ro" -v "$model:/model:ro" -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -e TMPDIR=/hot/tmp -e HOME=/hot/home -e XDG_CACHE_HOME=/hot/cache -e HF_HOME=/hot/hf -e HF_HUB_OFFLINE=1 \
    -e TRITON_CACHE_DIR=/hot/triton -e TORCHINDUCTOR_CACHE_DIR=/hot/inductor -e ASCEND_WORK_PATH=/hot/ascend \
    -e ASCEND_CACHE_PATH=/hot/ascend -e ASCEND_PROCESS_LOG_PATH=/hot/logs -e PYTHONDONTWRITEBYTECODE=1 \
    -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 "$image" /e/replay_processor_boundary.py --config /e/INPUT.json --model /model
