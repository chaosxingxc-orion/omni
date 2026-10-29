#!/usr/bin/env bash
# Two-card HCCL all_reduce smoke inside saea-base (host 181). usage: hccl_smoke.sh <cards e.g. 0,1> [extra docker args...]
set -uo pipefail
CARDS=$1; shift
DEV=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
for c in ${CARDS//,/ }; do DEV+=(--device "/dev/davinci$c"); done
docker run --rm --network host --shm-size 8g --cpuset-cpus 128-191 "${DEV[@]}" "$@" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  --entrypoint bash saea-base:20261027 -c '
cat > /tmp/ar.py <<PY
import os, torch, torch_npu, torch.distributed as dist
r = int(os.environ["RANK"]); torch.npu.set_device(r)
dist.init_process_group("hccl", rank=r, world_size=2)
x = torch.ones(8, device=f"npu:{r}") * (r + 1)
dist.all_reduce(x)
print("rank", r, "allreduce", x[0].item(), flush=True)
PY
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29555 WORLD_SIZE=2
RANK=0 /usr/local/python3.12.13/bin/python3 /tmp/ar.py 2>&1 | grep -E "allreduce|hccl|HCCL|Error" | head -5 &
RANK=1 /usr/local/python3.12.13/bin/python3 /tmp/ar.py 2>&1 | grep -E "allreduce|hccl|HCCL|Error" | head -5
wait'
