# NPU port: preliminary arm A synthetic sweep on host 181 (2026-10-29)

Status: PRELIMINARY engine measurement. It is NOT the recorded speed baseline of the locked engineering base: the
servers were launched by the host script `work/scripts/npu_serve.sh`, not by the deployment launcher, and the
receipt/launcher fixes and base verification of plan
`docs/superpowers/plans/2026-10-29-lock-engineering-base-and-efficiency-plan.md` were not yet in place. It grants no
contact authority, no readiness flip and no exposure entry. Load was synthetic only (`scripts/serving/synthetic_sweep.py`,
seed 20261028); no dataset row was read.

## Setup

- Host 181 (`npu2`), 8 x 910B4-1. Image `saea-base:20261027` (vLLM 0.30.0, vllm-ascend 0.30.0rc2, torch_npu
  2.10.0.post7). Model Qwen3-Omni-30B-A3B-Thinking, BF16, `--max-model-len 24576`, every other protocol flag as in the
  port model fragment.
- Arm A: four instances x tensor-parallel 2, cards 0,1 / 2,3 / 4,5 / 6,7, ports 8200-8203, `--max-num-seqs 64`, each
  pinned to its cards' NUMA node (cards 0,1 node 6; 2,3 node 4; 4,5 node 0; 6,7 node 2; 32 cores each), distinct
  `HCCL_IF_BASE_PORT` per instance, HCCL/GLOO/TP socket interface pinned to the host NIC. Sweep client pinned to node 7.
- Levels are per instance (the host carried 4 x the level). Default requests per level: 3 contacts per worker.
- Per instance at startup: KV cache 25.83 GiB per card, 564,224 tokens, 22.96 x the 24,576-token window (identical to
  the earlier `--max-num-seqs 4` run); engine init 85-88 s; four instances ready about 7 min after a parallel launch.

## Results

Correction (2026-10-29, after the planner's review): an earlier version of this table reported the sweep tool's
`output_tok_s` as per-instance and multiplied it by four. The tool sums completion tokens over all ok requests of
the level, across all four instances, and divides by the level's wall time (`scripts/serving/synthetic_sweep.py`,
`summarize_level`), so `output_tok_s` IS the host aggregate. The table below is corrected; per instance is host / 4.
Cross-check at 48: 13.1 contacts/min per instance x about 2.0k completion tokens = about 437 tok/s per instance.

| level per instance | output tok/s per instance (host / 4) | host output tok/s | per-sequence tok/s p50 / p10 | latency s p50 / p95 / max | TTFT s p50 / p95 | preemptions | min HBM free per card | KV peak |
|---|---|---|---|---|---|---|---|---|
| 8 | 226.2 | 904.9 | 39.6 / 32.0 | 64 / 81 / 81 | 0.74 / 2.90 | 0 | 4.0 GiB | 15 % |
| 16 | 274.5 | 1,098.1 | 25.2 / 20.4 | 104 / 129 / 131 | 0.93 / 5.11 | 0 | 4.0 GiB | 30 % |
| 24 | 314.4 | 1,257.4 | 20.3 / 15.5 | 99 / 161 / 170 | 0.92 / 9.09 | 0 | 4.0 GiB | 43 % |
| 32 | 355.7 | 1,422.6 | 18.3 / 14.3 | 121 / 183 / 244 | 0.90 / 11.57 | 0 | 4.0 GiB | 53 % |
| 48 | 448.8 | 1,795.3 | 14.3 / 11.4 | 170 / 229 / 249 | 1.08 / 13.75 | 0 | 4.0 GiB | 80 % |

Zero errors and zero timeouts at every level (96 to 576 requests per level). Mean prompt about 8.5k tokens, mean
completion about 2.0k tokens (most requests reached the 3,112-token thinking cutoff), prefix-cache hit rate 51-54 %.
At 48 per instance: 13.1 contacts per minute per instance.

Thresholds (plan section 3.4): per-sequence p10 >= 10 tok/s, max latency < 420 s, zero preemptions, >= 2 GiB HBM free
per card. Every level passed; the stopping rule never fired.

## Reading

- The knee is NOT bracketed: throughput still rose 26 % from 32 to 48. By the plan's selection rule the best measured
  level is 48 per instance (about 1.8k output tok/s for the host, about 450 per instance), with the p10 margin the tightest (11.4 vs 10).
- Plan predictions refuted: level 48 did not preempt, and the knee is not at or below 32. Prefix-cache reuse across
  the three contacts of a trajectory was not in the prediction.
- HBM was sampled every 30 s, so a short peak could be missed; AI Core means rest on 8-22 samples per level.
- Next steps (levels 56 and 64, repeat runs at 3 x requests, a `--min-tokens 4096` worst case, arm B 2 x TP4) are
  decided by the planner in the plan file.

## Evidence

Reports in `/mnt/cephfs/xingchao/omni/work/logs/` (not committed; they carry host paths), prefix
`sweep-armA-prelim-20261029T162735`:

| file | sha256 |
|---|---|
| `.json` (merged summary) | `3ff89290e895d87f41ab4ffc5330094d182f7d8254c9250680158f89c8c7a807` |
| `-L8.json` | `1c70ddb0d0d3f8bcee776f9d40b9060454024ece37091dc3c628e5c023a38cb7` |
| `-L16.json` | `f7a194c0ed5de13000031fb9e0e2c9738d4d6cc9be2a8bac20921d0da9e069da` |
| `-L24.json` | `8a534c9526ba3300ca9a6c2a7865718ee98b716c9a486e4540f5ac342bf47dee` |
| `-L32.json` | `43278f26cbb66f6ca1c0ec56c65c5f9cf46ffa5450a258c34f52dfc39d7a7736` |
| `-L48.json` | `2548ae735a36d84c4e31c2823540d15321572a989b1d896bbca034384739731e` |
| `-npu-smi.json` | `2e92be06459e545c54a34b0abf8695fd7eaf70839f0d30f10e48ba4744d683ba` |
