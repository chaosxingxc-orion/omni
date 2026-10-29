# NPU port: synthetic speed baseline and hour bound (S8, SKELETON v1, 2026-10-29)

Status: SKELETON. The recorded table (section 1) is empty and is filled only from the R-* runs on the locked base (plan
`docs/superpowers/plans/2026-10-29-lock-engineering-base-and-efficiency-plan.md` 3.7 (5), 3.10 item 2). Every number
outside section 1 is PRELIMINARY: launched by the host script `work/scripts/npu_serve.sh`, not by the deployment
launcher, before the receipt and the base verification existed. Preliminary rows are context only. They never enter
the hour bound (plan 3.7 (5)), and the bound in section 3 is a placeholder computation to be replaced by R-A2.
Synthetic load only (`scripts/serving/synthetic_sweep.py`, seed 20261028); no dataset row was read. It grants no contact
authority, no readiness flip and no exposure entry, and no gate reads it as permission. Raw reports live in
`/mnt/cephfs/xingchao/omni/work/logs/` (abbreviated `LOGS/`) and are not committed. Each is cited by sha256.

## 1. Recorded table (R-* runs on the locked base; EMPTY until S7 runs)

Columns per plan 3.4. "verdict" applies the 3.4 thresholds (p10 e2e >= 10 tok/s; max latency < 420 s; zero timeouts and
errors; preemption delta exactly 0 with a readable counter; >= 2 GiB HBM free per card). Worst-case rows also apply 3.8
rule (a) (zero preemptions on EVERY instance and KV peak <= 90 %). Rows at a level equal to the ceiling also apply the
`waiting` = 0 check (3.10 item 2; see open point P1 in section 9). The greedy probe columns are recorded, not gated (Q-D).

Branch (ii)/(iii)/(iv) of plan 3.10 (arm A resident; profile `qwen3-omni-thinking-bf16-181-4xtp2-v2`, or `-v3` if E3 passes):

| run | level / inst | host conc. | requests | report (LOGS/) | sha256 | host out tok/s | per-inst tok/s | contacts/min/inst | e2e p50 / p10 | latency p50 / p95 / max s | TTFT p95 s | preemptions per inst | KV peak per inst | min HBM free MiB | waiting peak | probe before / after (sequential agreement) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R-A1 | 8 | 32 | 1 pass | TODO | TODO | | | | | | | | | | | | |
| R-A1 | 16 | 64 | 1 pass | TODO | TODO | | | | | | | | | | | | |
| R-A2 | 24 | 96 | 3x | TODO | TODO | | | | | | | | | | | | |
| R-A2 | 32 | 128 | 3x | TODO | TODO | | | | | | | | | | | | |
| R-A3 worst case (`--min-tokens 4096`) | 32 | 128 | 1 pass | TODO | TODO | | | | | | | | | | | | |
| R-D (`--max-num-batched-tokens 8192`), only if R-A2 at 32 has p10 within 5 % of 10 tok/s | 32 | 128 | 1 pass | TODO or "not run: condition not met (p10 = ...)" | | | | | | | | | | | | | |

Branch (i) only (arm B beats 1,422.6 host tok/s by more than 5 % in P3; provisional until the residency check):

| run | level / inst | host conc. | requests | report | sha256 | host out tok/s | contacts/min/inst | p10 | latency max | preemptions per inst | KV peak per inst | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R-A2 (A on the locked base) | 32 | 128 | 3x | TODO | | | | | | | | |
| R-B1 | 16 / 32 / 48 | 32 / 64 / 96 | 1 pass each | TODO | | | | | | | | |
| R-B2 | B admitted (expected 96) and next lower measured (64) | 192 / 128 | 3x | TODO | | | | | | | | |
| R-B3 worst case | B admitted | | 1 pass | TODO | | | | | | | | |
| R-D on B (conditional) | B admitted | | 1 pass | TODO | | | | | | | | |
| residency check | R-B2 (admitted) vs R-A2 (32): B stays only if > 5 % | | | | | | | | | | | TODO |

Recorded-run manifest facts to copy in: profile id and sha256, receipt path and `protocol_sha256`, verification report
sha256 (11/11), client seed 20261028, server `--seed 0`, the per-level probe agreement rates, and `npu-smi` before/after each arm.

## 2. Selection rule, applied explicitly

Rule (plan 3.4): among PASSING levels, the highest host output tok/s, with ties within 5 % going to the lower level.
Between arms, the higher host throughput at the knee wins, with ties within 5 % going to A (Q4). The tool's
`recommended_concurrency_per_instance` is not the answer. Admission (plan 3.8 rule (a), owner Q-A yes): a level is
admissible for recording and for flights only if its one-pass worst case shows zero preemptions on every instance with
a readable counter AND a KV peak <= 90 %. Rule (b): no recorded run and no flight exceeds the admitted level.

Applied to the PRELIMINARY rows (section 6):
1. Normal mix: 8, 16, 24, 32, 48 pass and 56 fails (p10 9.13, 1 preemption), so 64 is skipped by the stopping rule.
   The highest-throughput passing level is 48 (1,795.3 host tok/s; 56 is excluded as failing). The throughput knee is 48.
2. Admission: worst case at 48 FAILS (P2: two engines exited, survivors preempted 11 and 14 times); at 40 NOT
   admissible (9 preemptions, KV peak 96.7-100 %); at 32 ADMISSIBLE (0/0/0/0 preemptions, KV peak 80.1-88.1 % <= 90 %).
3. **Admitted knee: 32 per instance (host 128) by the 3.8 rule**, below the throughput knee. Cost: 1,422.6 vs 1,795.3
   host tok/s = -20.8 %.
4. Recorded candidates (plan 3.9 decision 1): 32 and 24 at 3x (R-A2). The recorded knee is R-A2 at 32 if R-A3 at 32 is
   admissible; otherwise the admitted level drops one measured step to 24 (R-A3 repeats there).
5. Arm B: TODO (P3/P3b). Compared at equal HOST concurrency, each arm at its own admitted level.

## 3. Hour bound in the registered form (PRELIMINARY substitution; to be replaced by R-A2)

Registered formula, verbatim, from `docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/registered-facts-v1.json`
(sha256 `1ce6af21b20d9fb85a87bacc76c1ff5c7ef6b9f01eba49681569c2aaab2ffe6b`), key `accounting.accelerator_bound.formula.arm`:

> PROVISIONAL NPU card-hours = ceil_0.1(1.3 x 1,080 rows x 7.52 lock-row equivalents x T / r / 60), T = tensor-parallel
> size (cards per instance), r = the measured lock-only rows per minute of ONE instance at its chosen sequences per
> instance; the instance count cancels (wall-clock = card-hours / (instances x T)); T and r from PORT_DEPLOYMENT_PROFILE
> (pending; filled from the measured speed sweep)

Inputs in the same file (`accounting.accelerator_bound.inputs`): margin 1.3, rows 1,080, arm mix 7.52 (= (7 x 59.00 + 8 x
64.07) / (59.00 + 64.07) = 7.5206), round 0.1. r derivation (plan 3.5 item 2): r = completed synthetic contacts per minute
per instance at the knee / 2.

Preliminary substitution, arm A at the admitted knee 32 (scan level report `LOGS/sweep-armA-prelim-20261029T162735-L32.json`,
sha256 `43278f26cbb66f6ca1c0ec56c65c5f9cf46ffa5450a258c34f52dfc39d7a7736`; values read from the merged summary
`LOGS/sweep-armA-prelim-20261029T162735.json`, sha256 `3ff89290e895d87f41ab4ffc5330094d182f7d8254c9250680158f89c8c7a807`):

- 384 ok contacts in 536.52 s over 4 instances, giving 42.94 contacts/min host = 10.736 per instance, so **r = 5.368**; T = 2.
- card-hours = ceil_0.1(1.3 x 1,080 x 7.52 x 2 / 5.368 / 60) = ceil_0.1(65.56) = **65.6 card-hours** (wall 65.6 / (4 x 2)
  = 8.2 h). PRELIMINARY: to be replaced by R-A2 at the recorded knee.
- Fallback candidate 24 (same scan, `-L24.json` sha256 `8a534c95...bf47dee`): r = 4.904, giving 71.8 card-hours.
- For reference only, the inadmissible 48: r = 6.547, giving 53.8 card-hours. The admitted knee costs +21.9 % card-hours against 48.

Cross-check (plan 3.5 item 2): 16,200 contacts x mean completion tokens / host output tok/s. At 32 the mean completion is
1,422.59 tok/s x 536.52 s / 384 = 1,987.6 tokens, so 16,200 x 1,987.6 / 1,422.59 = 22,634 s = 6.287 h wall x 8 cards = 50.3
card-hours, or 65.4 with the 1.3 margin. Registered form 65.6 vs cross-check 65.4 (with margin): difference 0.3 %, inside 20 %.
The agreement is close to built in: both use the same level's wall time, and 1,080 x 7.52 x 2 = 16,243 contact-equivalents
is close to 16,200 contacts. The cross-check is not an independent measurement.

Correction against the plan's own numbers: plan 3.9/3.10 item 4 quotes r ~5.2 and 67.7 card-hours, interpolated as 6.5 x
1,422.6 / 1,795.3. The measured contacts/min at 32 give r = 5.368 and 65.6. The "+26 %" bound penalty quoted in 3.8/3.9 is
+21.9 % on the measured r (6.547 / 5.368 = 1.2196). Plan 3.10's cross-check used ~2,050 tokens per contact; the measured
mean at 32 is 1,987.6 (2,050.7 is the worst-case HOST tok/s at 32, not a token count).

Branch (i): T = 4, r from R-B2 only; TODO.

## 4. Flight usability caveat

The knee is not usable by a flight until the flight-side multi-endpoint transport exists (plan 3.3 note, 3.10 item 4,
section 7 follow-up: route a trajectory to one instance by profile, strip `id_slot`, which `VLLM_REFUSED_REQUEST_KEYS` refuses).
Until then a flight may use ONE instance at its admitted level: 1/4 of the host figure on A, 1/2 on B. The flight mix must
stay inside the synthetic one (prompt mean 8,500 in [3,000, 20,000], `max_tokens` 4096, budget 3,072, one clip per
trajectory, 3 contacts), otherwise R-x3 is re-run first. The sweep tool's per-trajectory instance affinity is not the flight transport.

## 5. Determinism (recorded, not gated)

As recorded by D3 (`LOGS/prelim-D3-greedy-20261029T130527Z.json`, sha256 `ca0af18c728e1462abef48e150b1edb1aa6399ea631a8ba3d9f5c5f07b9a4836`;
idle `saea-arm-a-0`, 16 fixed requests, temperature 0, seed 777 per plan 3.10, max_tokens 64): sequential S1-S2 9/16, S1-S3
9/16, S2-S3 16/16; burst B1-B2 8/16, B1-B3 15/16, B2-B3 8/16; sequential vs burst 6-7/16 (S1-B2 7, the rest 6). First
divergence at decode positions 13-58, never position 0. Top-2 logprob gap values at the divergence are 0.0, 0.125, 0.25 and 0.375 (0.375 appears only in pairs S1-B1 and S1-B2).
No preemptions. The tool's classification string is "kernel-or-cache nondeterminism: sequential repeats disagree". The
planner's reading (plan 3.10) is cache-state plus batch-composition nondeterminism with warm sequential greedy repeatable
(16/16, n = 1 pair). Earlier E1/E2 equality probe (`LOGS/prelim-E2-eq-compare-shm-20261029T111900Z.json`, sha256
`843562157e27e54290908970492c2afde4c4a61fd246298465482905919f40ce`): E1 vs E2 16/16 (two cold first runs, per the D2 note),
E1 self 9/16, max |delta logprob| 1.97 (a post-divergence value per D2). Outputs on this stack are reproducible
statistically, not bit-wise (owner Q-D yes). No gate uses byte identity. The S7 sequential probes before and after
each recorded level go in the section 1 probe column.

## 6. PRELIMINARY CONTEXT (all rows preliminary; none enters the bound)

Launch: four `npu_serve.sh` containers `saea-arm-a-0..3`, 4 x TP2 on cards 0,1 / 2,3 / 4,5 / 6,7, ports 8200-8203, NUMA
cpusets 192-223 / 128-159 / 0-31 / 64-95, `--max-num-seqs 64`, engine-default utilisation 0.92, NIC `enp61s0f2`
(capture `LOGS/prelim-chain-20261029T094933Z/arm-a-launch.json`, sha256 `a481373e2f9204985ad8dd1edb04fe50ead5983839cf7abeb3ad90aba8a08f3a`;
byte-identical copies in the P2b and E chain directories).

Deployment-layer differences between these preliminary runs and the recorded runs (plan 3.10 base-port note; all are
deployment facts outside `protocol_sha256`, the protocol tokens are identical):

| fact | preliminary (`npu_serve.sh`, launch capture) | recorded (`npu_deploy.sh up`, profile -v2/-v3, `core.deployment`) |
|---|---|---|
| `HCCL_IF_BASE_PORT` | 60000 / 60100 / 60200 / 60300 (arm B preliminary: 60400 / 60500 per plan 3.10) | 61000 / 61200 / 61400 / 61600 |
| ceiling `--max-num-seqs` | 64 | 32 |
| `--gpu-memory-utilization` | not passed (engine default 0.92) | passed explicitly, 0.92 |
| launch form | `--entrypoint vllm` + `serve ...` | `/usr/local/python3.12.13/bin/python3 -m vllm.entrypoints.cli.main serve ...` |
| restart policy | `no` | `unless-stopped` (as emitted by `docker_run_argv`; see the S5 record, decision D4) |
| container names | `saea-arm-a-0..3` | `saea-omni-i0..i3` |
| mm processor cache | engine default | default (-v2) or `shm` (-v3, only if E3 passes) |

### 6.1 Normal mix, arm A, one pass per level (scan 8-48 and P1 56)

Scan merged summary sha256 `3ff89290...c7a807`. P1 merged summary `LOGS/sweep-armA-prelim-P1-20261029T094933Z.json`
sha256 `7804bd95f81567b05f01fbae18c180589047f08f27d4b9ca46a0187f5f42d805`. P1 chain status
`LOGS/prelim-chain-20261029T094933Z/chain-status.json` sha256 `e1537e4b634ee8e0e08b3b234a6e069c1d19ea49760ec81362a560f128f0b3c0`.
Host tok/s is the tool's level total over all four instances. Per instance = host / 4. contacts/min = ok / wall x 60 / 4.

| level / inst | host conc. | requests ok | wall s | host out tok/s | per-inst tok/s | contacts/min/inst | r (lock rows/min/inst) | e2e tok/s p50 / p10 | latency s p50 / p95 / max | TTFT s p50 / p95 | preempt. | KV peak | min HBM free MiB/card | waiting peak | AI Core mean % | 3.4 verdict | level report sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 32 | 96/96 | 226.2 | 904.9 | 226.2 | 6.37 | 3.18 | 39.62 / 31.97 | 64 / 81 / 81.4 | 0.74 / 2.90 | 0 | 15.3 % | 4136 | 0 | 67.2 | PASS | `1c70ddb0d0d3f8bcee776f9d40b9060454024ece37091dc3c628e5c023a38cb7` |
| 16 | 64 | 192/192 | 350.1 | 1,098.1 | 274.5 | 8.23 | 4.11 | 25.23 / 20.42 | 104 / 129 / 131.0 | 0.93 / 5.11 | 0 | 30.5 % | 4122 | 2 | 78.1 | PASS | `f7a194c0ed5de13000031fb9e0e2c9738d4d6cc9be2a8bac20921d0da9e069da` |
| 24 | 96 | 288/288 | 440.5 | 1,257.4 | 314.3 | 9.81 | 4.90 | 20.28 / 15.50 | 99 / 161 / 170.1 | 0.92 / 9.09 | 0 | 43.0 % | 4120 | 8 | 76.0 | PASS | `8a534c9526ba3300ca9a6c2a7865718ee98b716c9a486e4540f5ac342bf47dee` |
| 32 | 128 | 384/384 | 536.5 | 1,422.6 | 355.6 | 10.74 | 5.37 | 18.26 / 14.31 | 121 / 183 / 243.6 | 0.90 / 11.57 | 0 | 52.5 % | 4116 | 5 | 77.5 | PASS | `43278f26cbb66f6ca1c0ec56c65c5f9cf46ffa5450a258c34f52dfc39d7a7736` |
| 48 | 192 | 576/576 | 659.9 | 1,795.3 | 448.8 | 13.09 | 6.55 | 14.33 / 11.35 | 170 / 229 / 248.6 | 1.08 / 13.75 | 0 | 80.0 % | 4110 | 7 | 82.8 | PASS | `2548ae735a36d84c4e31c2823540d15321572a989b1d896bbca034384739731e` |
| 56 | 224 | 672/672 | 719.5 | 1,952.8 | 488.2 | 14.01 | 7.00 | 12.44 / 9.13 | 214 / 258 / 264.7 | 1.30 / 13.40 | 1 | 99.8 % | 4110 | 18 | 86.4 | FAIL: p10 e2e 9.126755075274968 < 10.0; 1 preemption(s) | `38766520120fdca423d3e00f659d994e9151fca85ff9eb108ef5f9429ade36bf` |

Reading: every level 8-48 passed every 3.4 threshold with zero preemptions, zero errors and zero timeouts; throughput
still rose 26.2 % from 32 to 48, so the throughput knee was not bracketed by the scan. 56 FAILS (p10 9.13 below the 10
floor; 1 preemption; KV peak 99.8 %; `waiting` peak 18 at a ceiling of 64, so the level is not ceiling-bound), and the
stopping rule skipped 64 (chain status `decisions.P1.stopped_at_level` 56).

### 6.2 Worst case `--min-tokens 4096`, arm A, one pass per level (P2 48, P2b 40 and 32)

P2b merged summary `LOGS/sweep-armA-prelim-P2b-mintok4096-20261029T105349Z.json` sha256
`dda10d18faceacf3dcd98abd2f1428c438994e4c16d1b55a92713ffbdfc3288b`, `worst_case_admissible_level` 32. P2b chain status
`LOGS/prelim-chain-p2b-20261029T105349Z/chain-status.json` sha256
`94e1cd70c6f7566c99b5c60a98c47825ba79b54762070bb624e34571d4054fb4`. P2 ran on the first arm A (chain
`prelim-chain-20261029T094933Z`, state `failed`); P2b ran on a fresh arm A with argv identical to the capture
(`steps.p2b-fresh*-wait.decision.argv_identical_to_capture` true).

| phase | level / inst | requests ok | timeouts / errors | wall s | host out tok/s (all 4,096-token completions) | e2e tok/s p10 | latency s max | TTFT s p95 | preemptions per instance (i0/i1/i2/i3) | KV peak per instance | min HBM free MiB/card | waiting peak | admissible (3.8 rule (a)) | level report sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P2 | 48 | 476/576 | 9 / 91 | 1,129.3 | 1,726.4 (contaminated) | 11.95 | 412.3 | 16.74 | unreadable (i0, i3 exited); 11 and 14 on the survivors per the bad-case note | 100.0 % (host max) | 4110 | 18 | NO (FAIL: 91 errors, 9 timeouts; ok 476 of 576 requests; preemption counter unreadable) | `c2a5db37c678c5d42b4823ac6977c16a193c500b6e34905028beed4ee8f12991` |
| P2b | 40 | 480/480 | 0 / 0 | 885.6 | 2,220.1 | 13.85 | 306.0 | 11.16 | 2/4/0/3 | 99.9/100.0/96.7/100.0 % | 4126 | 6 | NO | `108a47f871aaa34d7aa7db65d7f3a934c861a19534ff0740915538963822dff5` |
| P2b | 32 | 384/384 | 0 / 0 | 767.0 | 2,050.7 | 15.76 | 265.6 | 10.39 | 0/0/0/0 | 82.5/88.1/80.1/85.9 % | 4118 | 13 | YES | `9409cf427e9319599409a39e0866c511e4873ee1d54cebd7abb1c81b796d339e` |

P2 at 48 is a pass/fail fact only, not a throughput figure: two of the four engines (`saea-arm-a-0` and `-3`) exited on
`AssertionError ... len(mm_embeddings)=0` in `gpu_model_runner.py:3187 _execute_mm_encoder` while recomputing a
preempted audio request from `num_computed_tokens=0`, so 91 errors and 9 timeouts are recorded, the preemption counter
is unreadable on the dead instances (11 and 14 on the survivors), and 1,726.4 host tok/s excludes 100 lost requests and
two instances that stopped producing mid-level. Evidence: Opus bad-case note
`work/scratch/badcase/2026-10-29-arm-a-exit-under-worst-case.md` (sha256
`ec9eb2faf828b82abca8021c98c52f674d2fdcbb79fd3a56d0122ccd431c90a8`); the chain then stopped at `p2-L32` against dead
ports (`LOGS/sweep-armA-prelim-P2-mintok4096-20261029T094933Z-L32.stderr.log`, sha256
`220ad2b5dbe0b4a9df9ea7fee068c43324fa176da5574c0b03e0668f3e87f054`), leaving 32 unmeasured until P2b. The source read
behind the hypothesis is `work/scratch/badcase/2026-10-29-e0-mm-encoder-preemption-source.md` (sha256
`49b972f44f2f4986757c8a377741678769e5196b04adadf988bce76c938d3d89`); the path stays unconfirmed (plan 3.9 decision 2).

The worst-case host tok/s figures (2,220.1 at 40 and 2,050.7 at 32) are higher than the normal-mix figures at the same
level because every completion is 4,096 tokens; they are not mix throughput and are not comparable with 6.1.

### 6.3 E1 / E2 crash trials, one TP2 instance each (preliminary, cards 0,1 and 6,7)

E chain status `LOGS/prelim-E-chain-20261029T111900Z/chain-status.json` sha256
`9189392521f420a775c992d371f8e95c06eebc15c94bd7f9fa0e6a2d94a00845`; E1 result `LOGS/prelim-E1-20261029T111900Z.json`
sha256 `0194d1e2f730a1c77d5f1ed35baea3b14d5fe9d3df017ff6628433798cce43f9`, verdict "not reproduced in isolation"; E2
result `LOGS/prelim-E2-20261029T111900Z.json` sha256 `7f5eea0bbee26b648dd6f6b84ddb23c8dfe1c4cc01ba36bcbf1393d7c1a999bc`,
verdict "accepted". E2 added `--mm-processor-cache-type shm`; both startups report argv identical to the capture plus
the extra flag. Host tok/s is one instance here, not four.

| track | pass | level | seed | requests ok | timeouts | preemptions | KV peak | e2e p10 | latency max s | host out tok/s | log scan: fatal / mm assertion | report sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 base | 1 | 48 | 20261028 | 144/144 | 0 | 13 | 100.0 % | 12.86 | 407.5 | 595.6 | not recorded | `27dfce2733145e71636a0be54a3d94ce0d7ec4f92c519f19e6b8fe9b4da2163f` |
| E1 base | 2 | 56 | 20261029 | 152/168 | 16 | 19 | 100.0 % | 11.24 | 416.7 | 536.6 | False / False | `12ad9116c6d1a2bba5f006bd6078d5566e095e6c95f5e60cfc98be0c7c8d1509` |
| E1 base | 3 | 56 | 20261030 | 151/168 | 17 | 22 | 100.0 % | 11.18 | 419.7 | 543.3 | False / False | `ef07c31d660299d15c612a727fa95876d9e256e44b89363059e14a40f7ac56bf` |
| E2 shm | 1 | 56 | 20261028 | 148/168 | 20 | 25 | 100.0 % | 11.24 | 415.0 | 511.9 | False / False | `b96465fa580bc3a566b2e40357d97dbd9ea4e03ecaf29ea03f817643c15d77ad` |
| E2 shm | 2 | 56 | 20261029 | 141/168 | 27 | 26 | 100.0 % | 11.03 | 411.7 | 488.7 | False / False | `c4583358c6b7f66daac0e75e3bb3ffd25e139b992b9704eb1096648fdc42327c` |
| E2 shm | 3 | 56 | 20261030 | 151/168 | 17 | 22 | 100.0 % | 10.87 | 418.7 | 531.2 | False / False | `e9479b632317c13c893a52da3fc32908afe8e71e54ab09452557fb72ee135707` |


Both tracks passed three passes with 13-26 preemptions each and zero engine exits, so E2's letter-of-the-rule pass does
not discriminate: the baseline did not crash either (plan 3.9 decision 2). E1 pass 1 ran at 48 and the rest at 56.
Timeouts at 420 s are the client's own abort, which frees KV early.

### 6.4 Rows the recorded runs will NOT re-measure

Preliminary 40, 48 and 56 are not re-run on the locked base: they preempt, a recorded crash costs an R0 redo, and rule
(b) caps every recorded run at the admitted level (plan 3.9 decision 1, superseding 3.8 decision 4). Those rows exist
only as the labelled context above.

### 6.5 Arm B (P3 / P3b): TODO

Not run at the time of writing. The watchdog status file `work/watchdog/status-npu2.json` (sha256
`2c048cacf670c5299adb258032f7dd88904c988d7537afc779a0d8bf51b5159f`, read at 2026-10-29T15:54:15Z) carries the job
`sweep-armB-prelim-P3` in state `unknown`, "no files for prefix yet". Expected reports: `LOGS/sweep-armB-prelim-P3*`
(levels 16, 32, 48, 64, 96 per instance = host 32-192, `MAX_NUM_SEQS=128`, one pass, stopping rule) and the P3b worst
case at B's highest passing level under rule (a). Fill these rows and the 3.9 prediction in section 7, then the plan
3.10 branch cell, before section 1 is filled.

| run | level / inst | host conc. | report (LOGS/) | sha256 | host out tok/s | per-inst tok/s | contacts/min/inst | e2e p50 / p10 | latency max s | preemptions | KV peak | waiting peak | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P3 | 16 | 32 | TODO `sweep-armB-prelim-P3*-L16.json` | TODO | | | | | | | | | |
| P3 | 32 | 64 | TODO | TODO | | | | | | | | | |
| P3 | 48 | 96 | TODO | TODO | | | | | | | | | |
| P3 | 64 | 128 | TODO | TODO | | | | | | | | | |
| P3 | 96 | 192 | TODO | TODO | | | | | | | | | |
| P3b worst case | B's highest passing level | TODO | TODO | TODO | | | | | | | | | rule (a): TODO |

Branch decision once P3/P3b is in (plan 3.10 item 1): (i) B host tok/s at its admitted level > 1,493.7 -> B resident,
TP4, 2 instances; (ii) within +-5 % of 1,422.6 -> A (Q4 tie); (iii) worse -> A; (iv) B crashes or has no admissible
level -> A. A preliminary B win does not displace A on its own: it triggers the recorded comparison (plan 3.7 (3)).

## 7. Predictions of plan 3.1, 3.7, 3.8, 3.9 and 3.10, marked

Each line: prediction, verdict, evidence. "REFUTED" and "CONFIRMED" are statements about the PRELIMINARY evidence;
a recorded run can move any of them.

Plan 3.1 (written before any card was used):

| prediction | verdict | evidence |
|---|---|---|
| TP2 KV 24-25.5 GiB per card at `max-num-seqs 64`, below the 25.83 measured at 4 | REFUTED | 25.83 GiB per card and 564,224 tokens unchanged at 64 (preliminary result doc, startup lines) |
| level 32 fits, level 48 exceeds KV, so preemption at 48 on the normal mix | REFUTED | 48 normal mix: 0 preemptions, KV peak 80.0 % (`-L48.json` sha256 `2548ae73...39731e`) |
| p10 >= 10 tok/s has ~2.5 % margin; max latency < 420 s is the binding gate | REFUTED (for the mix) | at 48 p10 is 11.35 (13.5 % margin) and max latency 248.6 s; p10 is what fails first, at 56 (9.13) |
| arm C (0.95) predicted to fail or barely pass the HBM gate | NOT TESTED | arm C not run (plan 3.7 decision 4): 4.1 GiB free at 0.92, knee not KV-bound |
| TP4 KV ~1.7 M tokens per instance, B compute-bound not memory-bound | NOT TESTED | P3 pending |

Plan 3.7 (after the first scan):

| prediction | verdict | evidence |
|---|---|---|
| p10 10.3-10.6 at 56 | REFUTED | 9.13 (`-P1-...-L56.json` sha256 `38766520...de36bf`) |
| KV peak ~93 % at 56 | REFUTED | 99.8 %, with 1 preemption |
| 56 passes narrowly | REFUTED | FAIL on p10 and preemption |
| throughput +8-12 % at 56 over 48 | CONFIRMED | 1,952.8 vs 1,795.3 = +8.8 % |
| 64 fails on p10 and preempts | NOT TESTED | skipped by the stopping rule after 56 failed |
| worst case PASSES at 48 with max latency 370-420 s | REFUTED | all four instances preempted, two exited (P2) |
| knee 48 per instance | REFUTED | admitted knee 32 (P2b) |
| B within +-5 % of A at equal host concurrency | NOT TESTED | P3 pending (plan 3.9 raises doubt) |

Plan 3.8 (after the crash):

| prediction | verdict | evidence |
|---|---|---|
| expected admitted level 40 per instance | REFUTED | P2b at 40: 9 preemptions (2/4/0/3), KV peak 96.7-100 % |
| fallback 32 | CONFIRMED | P2b at 32: 0 preemptions on all four, KV peak 80.1-88.1 % |
| E1 reproduces the assertion in isolation | REFUTED | 3 passes, 13/19/22 preemptions, zero exits, no assertion |
| E2's knob discriminates | REFUTED | shm 25/26/22 preemptions, zero exits; the baseline also had zero exits |
| host ~1,610 tok/s at 40 or 1,423 at 32 (interpolated) | PARTLY CONFIRMED | measured mix at 32 is 1,422.6; 40 was never measured on the mix |
| hour bound +12 % (at 40) or +26 % (at 32) against 48 | REFUTED (the number) | measured +21.9 % at 32 (65.6 vs 53.8 card-hours), not +26 % |

Plan 3.9 / 3.10:

| prediction | verdict | evidence |
|---|---|---|
| admitted knee 32 = host 128, cost -20.8 % against 48 | CONFIRMED | 1,422.6 / 1,795.3 = -20.8 % |
| r ~5.2 lock rows/min/instance at 32 | REFUTED (the number) | measured r = 5.368 from 384 ok in 536.52 s over 4 instances |
| hour bound +26 % | REFUTED | +21.9 % (section 3) |
| B's admitted host level is 192 and B's host throughput there is within +-5 % of A's 1,795, beating A's 1,423 by 20-25 % | TODO | P3/P3b pending |
| a warm sequential greedy run repeats exactly; the cold first run differs | SUPPORTED, n = 1 | S2-S3 16/16 while S1 differs from both (9/16); the tool's own classification string says "sequential repeats disagree" |
| burst adds batch composition | SUPPORTED | burst pairs 8/16, 15/16, 8/16; sequential vs burst 6-7/16 |

## 8. The 4x over-count correction history

The first version of `docs/results/2026-10-29-npu-port-preliminary-arm-a-sweep-v1.md` (commit `d9feb126`) read the sweep
tool's `output_tok_s` as per-instance and multiplied it by four, publishing a "host output tok/s" column of 3,620 /
4,392 / 5,030 / 5,690 / 7,181. `scripts/serving/synthetic_sweep.py` `summarize_level` sums `completion_tokens` over every
ok result of the level, across all instances, and divides by the level wall time, so the field IS the host aggregate.
Commit `32589378` ("Correct the host throughput in the arm A result and add plan revision 3") replaced the column with
905 / 1,098 / 1,257 / 1,423 / 1,795 host tok/s and per-instance = host / 4. Arithmetic check at 48: 13.09 contacts/min
per instance x 2,056.8 mean completion tokens / 60 = 448.7 tok/s per instance = 1,795 / 4. Contacts per minute and r were
never affected, so no decision rested on the wrong column. This document uses the corrected convention everywhere, and
every host figure here is the tool's own level total.

## 9. Open points for the planner

- **P1.** `waiting` peaks are non-zero below the ceiling in the preliminary data (5 at level 32, 7 at 48, 13 in the
  worst case at 32, 18 at 56), because the sweep sends in bursts. With the `-v2` ceiling equal to the admitted level 32,
  plan 3.10's "the sampled `waiting` gauge must read 0 at the knee" will fail the recorded knee row as ceiling-bound on
  this evidence. The check needs a restatement (for example, `waiting` must be 0 while `running` is below the ceiling,
  or the check applies to a steady-state window) before R-A2 runs.
- **P2.** The r-derivation convention. r is taken from `ok / wall x 60 / instances / 2` on the level report. The level
  wall time includes ramp-up and drain of one pass, which understates r at low levels; a 3x run (R-A2) dilutes that.
  Confirm this is the intended convention before the recorded bound is written.
- **P3.** Section 3 shows the registered form and the cross-check agree to 0.3 % here, but they are not independent
  (same wall time, and 1,080 x 7.52 x 2 ~ 16,200 contacts). The 20 % discrepancy rule of plan 3.5 will almost always pass.
  If an independent check is wanted, it has to come from somewhere other than the same level report.
- **P4.** The admitted knee rests on ONE worst-case pass at 32 (384 requests, one seed). Rule (a) is a zero-count on a
  single pass; a second seed at 32 would cost ~0.25 h wall. Not required by the plan; flagged as the evidence base of
  the whole bound.
- **P5.** Branch cells in sections 1, 6.5 and 7 stay TODO until P3/P3b. If the owner withholds P3 (Q-E no), branch (iv)
  applies and section 6.5 records "B not measured (owner)".
