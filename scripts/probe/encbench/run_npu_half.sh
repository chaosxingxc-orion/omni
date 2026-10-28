#!/usr/bin/env bash
# NPU half of the encoder benchmark: same protocol and JSON schema as the CPU half, then cosine agreement
# against the CPU float32 vectors. NOT RUN YET. Run it on host npu only once the cards are free; run_container.sh
# refuses any named card that has more than 4 GiB in use.
#
# usage (on host npu):  CARDS=0,1,2,3 nohup bash run_npu_half.sh > logs/npu-chain.log 2>&1 &
#   The first card in CARDS does the single-process runs; all of CARDS form the pool (one worker per card).
#   Inside the container the mounted cards are numbered npu:0..N-1.
set -uo pipefail
HERE=/mnt/cephfs/xingchao/omni/work/scratch/encbench
R=$HERE/results
L=$HERE/logs
CARDS=${CARDS:?set CARDS, e.g. CARDS=0,1,2,3}
N=$(echo "$CARDS" | tr ',' '\n' | wc -l)
POOL=$(seq -s, -f 'npu:%g' 0 $((N - 1)))
for enc in nemotron lco3b lco7b; do
  for dtype in bfloat16 float32; do
    tag=$enc-npu-$dtype
    echo "[$(date +%F' '%T)] start $tag"
    ENCBENCH_CARDS=$CARDS NAME=encbench-$tag timeout --signal=TERM --kill-after=120 3600 \
      bash "$HERE/run_container.sh" run --encoder "$enc" --device npu:0 --dtype "$dtype" --cores 128-191 \
      --threads 16 --ref-threads 16 --warmup 3 --calls 20 --rounds 3 \
      --pool-devices "$POOL" --pool-threads 16 --out "$R/$tag.json" > "$L/$tag.log" 2>&1 < /dev/null
    rc=$?
    echo "[$(date +%F' '%T)] end $tag exit $rc"
    # cosine agreement vs the CPU float32 reference (CPU-only container step, no cards)
    NAME=encbench-cmp-$tag bash "$HERE/run_container.sh" compare --ref "$R/$enc-cpu-float32.npz" \
      --cand "$R/$tag.npz" --out "$R/compare-$tag-vs-cpu-float32.json" > "$L/compare-$tag.log" 2>&1
  done
done
bash -c "python3 $HERE/encbench.py summarize $R/*-cpu-*.json $R/*-npu-*.json --out $HERE/SUMMARY-tables.md" || true
echo "[$(date +%F' '%T)] npu half done"
