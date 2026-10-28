# Retrieval-encoder benchmark: CPU half (2026-10-28/29, host npu, saea-base:20261027, cores 128-191)

Synthetic inputs only: seeded noise and tone audio at 16 kHz (2/3/5/10 s, plus 16 clips of 2-5 s) and seeded
sentences from a neutral 70-word vocabulary (10 and 30 words, plus 16 sentences). No dataset was read. Loader and
encoder: `saea_adapters` (unmodified). nemotron is forced causal as in the port. Every arm passes the `causal`
gate on its final dtype. Prompts: nemotron `query: ` / `passage: `. LCO uses an explicit empty prompt (`''`) on both sides; its
persona default prompt is declined and recorded. CPU fp32 = full protocol (20 calls after 2 warm-ups; batch 3 rounds,
lco7b 2). CPU bf16 = reduced (64 threads only; nemotron 20 calls, LCO 5 calls).
"Operational" load = what a loader that reads the final dtype directly pays: it excludes the adapter's load-time
bf16 gate, which costs 8-70 s on this CPU. Host load average was 22-80 during all runs, because the other tenants'
vLLM job is not pinned, so 64-thread numbers are the most sensitive to co-tenancy.

## Load and memory

| encoder | run | status | load s (first, as measured) | load s (operational) | pool worker load s (warm) | peak RSS GB (1 process) | pool worker RSS GB | device mem GB |
|---|---|---|---|---|---|---|---|---|
| nemotron | cpu-bfloat16 | ok | 21.4 | 13.7 | - | 7.9 | - | - |
| nemotron | cpu-float32 | ok | 22.1 | 13.2 | 36/37/34/38 | 22.9 | 22.9 | - |
| lco3b | cpu-bfloat16 | ok-with-failures | 11.5 | 5.1 | - | 7.8 | - | - |
| lco3b | cpu-float32 | ok-with-failures | 15.1 | 7.4 | 29/31/29/31 | 25.9 | 23.0 | - |
| lco7b | cpu-bfloat16 | ok-with-failures | 106.6 | 37.1 | - | 15.2 | - | - |
| lco7b | cpu-float32 | ok-with-failures | 53.5 | 18.7 | 56/73/69/102 | 41.9 | 37.1 | - |

## Batch-1 latency, seconds p50 / p95 (end-to-end encode: processor + forward + pooling)

| encoder | run | threads | a2s | a3s | a5s | a10s | t10w | t30w |
|---|---|---|---|---|---|---|---|---|
| nemotron | cpu-bfloat16 | 64 (best) | - | 17.27 / 22.74 | - | 29.75 / 31.26 | 4.66 / 8.51 | 6.37 / 11.84 |
| nemotron | cpu-float32 | 8 | - | 8.33 / 8.46 | - | 16.77 / 16.90 | 3.64 / 3.73 | 4.21 / 4.24 |
| nemotron | cpu-float32 | 16 | 4.80 / 4.85 | 5.43 / 5.49 | 6.81 / 6.85 | 10.37 / 10.45 | 2.46 / 2.51 | 2.80 / 2.82 |
| nemotron | cpu-float32 | 32 | - | 3.79 / 3.83 | - | 7.08 / 7.20 | 1.80 / 1.82 | 2.02 / 2.04 |
| nemotron | cpu-float32 | 64 (best) | 1.44 / 1.67 | 1.65 / 1.81 | 2.03 / 2.24 | 3.61 / 3.84 | 0.82 / 0.90 | 0.86 / 0.92 |
| lco3b | cpu-bfloat16 | 64 (best) | - | 33.72 / 104.00 | - | - | 5.02 / 14.08 | - |
| lco3b | cpu-float32 | 8 | - | 8.61 / 8.65 | - | 17.00 / 17.08 | 3.87 / 3.90 | 4.44 / 4.47 |
| lco3b | cpu-float32 | 16 | 4.91 / 4.99 | 5.54 / 5.56 | 6.92 / 6.98 | 10.47 / 10.54 | 2.60 / 2.62 | 2.93 / 3.02 |
| lco3b | cpu-float32 | 32 | - | 3.86 / 3.87 | - | 7.12 / 7.15 | 1.87 / 1.93 | 2.09 / 2.10 |
| lco3b | cpu-float32 | 64 (best) | 1.49 / 1.63 | 1.67 / 1.86 | 2.00 / 2.15 | 3.77 / 3.98 | 0.80 / 0.94 | 0.90 / 1.02 |
| lco7b | cpu-bfloat16 | 64 (best) | - | 45.85 / 149.69 | - | - | 10.33 / 13.41 | - |
| lco7b | cpu-float32 | 8 | - | 13.47 / 13.67 | - | 28.40 / 28.47 | 5.78 / 5.84 | 8.03 / 8.10 |
| lco7b | cpu-float32 | 16 | 7.10 / 7.13 | 8.30 / 8.43 | 10.69 / 10.76 | 16.28 / 16.44 | 3.75 / 3.85 | 4.96 / 5.09 |
| lco7b | cpu-float32 | 32 | - | 5.58 / 5.91 | - | 11.32 / 12.81 | 2.71 / 2.72 | 3.39 / 3.43 |
| lco7b | cpu-float32 | 64 (best) | 2.80 / 3.00 | 3.54 / 3.81 | 3.84 / 4.19 | 5.86 / 6.12 | 1.75 / 1.82 | 2.08 / 2.16 |

## Batched throughput (16 items; items/s, median of rounds)

| encoder | run | 1x16t seq b1 text | 1x16t b16 text | 4x16 pool b1 text | 4x16 pool b4 text | 1x16t seq b1 audio | 1x16t b16 audio | 4x16 pool b1 audio | 4x16 pool b4 audio | b16 vs b1 cos min text / audio | thread-count vector max abs diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nemotron | cpu-float32 | 0.39 | 1.03 | 1.66 | 3.01 | 0.17 | 0.54 | 0.69 | 1.94 | 1.000000 / 0.957835 | 1.4e-06 |
| lco3b | cpu-float32 | 0.37 | 0.93 | 2.03 | 3.22 | 0.17 | 0.28 | 0.83 | 0.98 | nan / nan | 1.8e-06 |
| lco7b | cpu-float32 | 0.24 | 0.47 | 0.43 | 1.05 | 0.11 | 0.15 | 0.28 | 0.42 | nan / nan | 1.2e-06 |

## Failures

- lco3b cpu-bfloat16: stage adapter_load_stack: ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_omni.configuration_qwen2_5_omni.Qwen2_5OmniThinkerConfig'> for this kind of AutoModel: Au... (handled by: saea_adapters.encoder.load_stack with transformers.AutoModel scoped to config.json architectures[0] (encbench workaround for the adapter bug))
- lco3b cpu-float32: stage adapter_load_stack: ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_omni.configuration_qwen2_5_omni.Qwen2_5OmniThinkerConfig'> for this kind of AutoModel: Au... (handled by: saea_adapters.encoder.load_stack with transformers.AutoModel scoped to config.json architectures[0] (encbench workaround for the adapter bug))
- lco7b cpu-bfloat16: stage adapter_load_stack: ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_omni.configuration_qwen2_5_omni.Qwen2_5OmniThinkerConfig'> for this kind of AutoModel: Au... (handled by: saea_adapters.encoder.load_stack with transformers.AutoModel scoped to config.json architectures[0] (encbench workaround for the adapter bug))
- lco7b cpu-bfloat16: stage attention_gate: text tower is bidirectional, the encoder requires causal. max|delta| over 5 shared prefix tokens = 8.750e-01 against tolerance 1.0e-03. For omni-embed-nemotron-3b this is... (handled by: not handled)
- lco7b cpu-float32: stage adapter_load_stack: ValueError: Unrecognized configuration class <class 'transformers.models.qwen2_5_omni.configuration_qwen2_5_omni.Qwen2_5OmniThinkerConfig'> for this kind of AutoModel: Au... (handled by: saea_adapters.encoder.load_stack with transformers.AutoModel scoped to config.json architectures[0] (encbench workaround for the adapter bug))

## CPU bf16 vs fp32 vector agreement (batch-1, same synthetic inputs)
nemotron cos min 0.99973, lco3b 0.99968, lco7b 0.99942. The NPU bf16 run should land in the same range.

## Findings
1. Live use (one speech query, batch 1). The best setting is 64 threads on all 128-191 cores. A 3 s query takes
   1.65 s p50 on nemotron, 1.67 s on lco3b and 3.54 s on lco7b. A 10 s query takes 3.6 / 3.8 / 5.9 s. A text key
   takes 0.8-0.9 s (3B) and 1.8-2.1 s (7B). At 16 threads, the pool's per-worker slice, a 3 s query takes 5.4 s
   (3B) and 8.3 s (7B).
2. Offline (text keys). The fastest vector-correct setup is the 4x16 pool. Text: nemotron 1.66 items/s at b1 and
   3.01 at b4 (b4 is bit-equivalent to b1, cos 1.0). lco3b 2.03 at b1. lco7b 0.43 at b1. LCO batched rows are
   wrong (finding 4), so only LCO b1 vectors are valid.
3. bf16 on this CPU (Kunpeng-920, no bf16 instructions) is 6-20x slower than fp32, with very large p95 spikes.
   Do not use it.
4. ADAPTER BUG (silent wrong vectors): the Qwen2.5-Omni processor pads on the LEFT for all three checkpoints,
   even though the LCO tokenizers report `padding_side=right` (probe: probe_padding.py).
   `saea_adapters.encoder._pool("lasttoken")` assumes right padding, so in any padded batch it pools position 0,
   which is a masked pad, and returns a ZERO vector. lco3b and lco7b: 27/32 batched vectors are zero (every row
   except the longest in its batch); see count_zero_vectors.py. Batch-1 is unaffected, and so is the port's
   batch-1 pool. The LCO b16/b4 throughput numbers are still a valid measure of compute cost.
5. ADAPTER BUG (load): `load_stack` calls `AutoModel.from_pretrained`, which raises
   `ValueError: Unrecognized configuration class Qwen2_5OmniThinkerConfig for this kind of AutoModel` on
   transformers 5.14.1. Neither LCO arm can load as shipped (traceback: failed/lco3b-cpu-float32.adapter-load-failure.log).
   The bench re-runs the SAME adapter call with `transformers.AutoModel` swapped, for that call only, for a
   shim that loads `config.json` `architectures[0]`. This is recorded in each JSON as `encoder.loader` and in
   `failures`.
6. nemotron batched AUDIO is not vector-equivalent to batch 1 (b16 cos min 0.958, mean 0.983). Batched text is
   (cos 1.0). The cause was not investigated. Query encoding is batch 1, so live use is unaffected.
7. The attention gate's absolute tolerance (1e-3) produced a false "bidirectional" on lco7b CPU bf16
   (delta 0.875). fp32 gives 2.2e-4 (causal). Expect the same risk with NPU bf16.
8. Vectors are deterministic: repeated calls at the same thread count are bit-identical. Changing the thread
   count changes vectors by at most 1.8e-6.

## NPU half (later, when cards are free; on host npu)
    CARDS=0,1,2,3 nohup bash /mnt/cephfs/xingchao/work/scratch/encbench/run_npu_half.sh \
      > /mnt/cephfs/xingchao/work/scratch/encbench/logs/npu-chain.log 2>&1 &
This runs nemotron/lco3b/lco7b x bf16/fp32 on npu:0. The pool is one worker per card. Then it runs `compare`
against the `*-cpu-float32.npz` vectors. run_container.sh refuses any card with more than 4 GiB in use.
A single run:
    ENCBENCH_CARDS=0 NAME=encbench-x bash run_container.sh run --encoder lco3b --device npu:0 --dtype bfloat16 \
      --cores 128-191 --threads 16 --pool-devices npu:0 --out results/lco3b-npu-bfloat16.json
