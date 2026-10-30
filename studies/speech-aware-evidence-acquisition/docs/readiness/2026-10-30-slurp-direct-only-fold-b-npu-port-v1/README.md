# D NPU port: direct-only on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B (DRAFT / HELD)

The NPU-port twin of the 266-E direct-only Thinking arm (historical package `docs/readiness/2026-10-11-slurp-direct-only-fold-b-thinking-vs-instruct-v1`), arm D of the three-way comparison (`docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md`, sections 5 and 9 ruling 6), registered before contact on 2026-10-30 at study commit `534b543ff154ca23be8287c1730b040940a3d038` by `scripts/flights/direct/direct_port_write_registration.py` from the cells `scripts/flights/direct/direct_port_derive_cells.py` derives, with the port settings imported from `scripts/flights/nx/l338_port_family.py`. Text draft: `docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only-cell-draft-v1.md`. Authority: owner decisions of 2026-10-28 on the NPU port (design doc section 9.1: the saea-base Docker container is the execution authority; item 4: the agent drafts the exposure preregistration rows; item 5: the admission receipt is authorised and minted by the main thread; item 6: model-free-check rows for the port preparations; the multi-instance deployment input: the deployment is a named placeholder filled from a measured speed sweep). Development numbers only; never a test claim; claims_benefit false.

## Cells (chain order)

| cell | run | FIT run | rows | generation calls | FIT calls | speech-seconds | composed config | protocol |
|---|---|---|---|---|---|---|---|---|
| `slurp-direct-only-fold-b-smoke-4-npu-port-v1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-NPU-PORT-V1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-FIT-NPU-PORT-V1` | 4 | 8 | 4 | 20 | `e73523aff971...` | `601b536bdc9a...` |
| `slurp-direct-only-fold-b-npu-port-v1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-FIT-NPU-PORT-V1` | 1080 | 2160 | 1080 | 6310 | `7d2f51cba7c3...` | `784edd2ca49b...` |
| `slurp-direct-only-fold-b-aa-npu-port-v1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AA-NPU-PORT-V1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AA-FIT-NPU-PORT-V1` | 1080 | 2160 | 1080 | 6310 | `3833b1943eee...` | `c3e1e4fc4dc1...` |
| `slurp-direct-only-fold-b-ab-npu-port-v1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AB-NPU-PORT-V1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AB-FIT-NPU-PORT-V1` | 1080 | 2160 | 1080 | 6310 | `4260ac28ac4d...` | `fb371c718e70...` |

Cell total: 6,488 generation calls, 3,244 FIT calls, 18,950 speech-seconds. One Solver contact per row per pass (program `phase-direct-only-v1`), plus one answer-only fallback contact per row in the plan ceiling; the FIT cap is rows x 1 x 1 counting endpoint, the value `core.preparation_fit.build_direct_only_raw_fit_gate` demands. The three passes `lock`, `aa`, `ab` are request-identical: their baselines differ only in the `live_phase_fit` identity (a FIT run is a one-time spend identity) and the prose.

## The FIT path

core.preparation_fit.build_direct_only_raw_fit_gate (reached from build_live_phase_fit_gate behind the program id phase-direct-only-v1 on the raw-audio arm); cap rule `live_phase_fit.contact_cap == direct_only_fit_contact_cap(rows) == rows x 1 contact x 1 counting endpoint`. It landed in commit 56d3c1c3 (`tests/unit/test_direct_only_fit.py`), so the FIT cap is no longer open.

## Serving: protocol facts and the deployment placeholder

Model fragment `bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1` (transport vLLM, BF16). Protocol facts: the per-sequence window of 24,576 tokens, reasoning budget 3072 / cap 4096, greedy decoding seed 20260913, prompt order `native-raw-speech-query-prefix-v1`. The DEPLOYMENT is the one named placeholder `PORT_DEPLOYMENT_PROFILE` (a later deployment profile chosen from a measured speed sweep of the BF16 / vLLM-Ascend Thinking server (several vLLM instances, each tensor-parallel over a group of Ascend cards; per-instance concurrency chosen from the sweep)); no instance count, sequence count or tensor-parallel size is a fact of this registration. Session identity, the vLLM runtime receipt and the lock entries are open owner decisions 9.2-9.4. If owner item R1 (thinking budget 6,144) is adopted, D is re-pointed to the budget6144 fragment together with A and T and re-written; the calls, FIT and audio caps do not move with the budget, the card-hour bound does.

## Accelerator bound (PROVISIONAL)

- each pass: PROVISIONAL NPU card-hours of ONE pass = ceil_0.1(1.3 x 1,080 rows x 0.52 lock-row equivalents x T / r / 60), T = tensor-parallel size (cards per instance), r = the measured lock-only rows per minute of ONE instance at its chosen sequences per instance; the instance count cancels (wall-clock = card-hours / (instances x T)); T and r from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep); three passes in the cell.
- smoke: PROVISIONAL NPU card-hours = 0.5 h wall-clock allowance x instances x T (every held card), from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep).
- The experiments' numeric planned_gpu_hours (4.5 / 0.5) are the 266-E placeholders the plan field requires; the ledger gpu-hours cells carry the formula, so the exposure gate refuses these rows ("must be a number") until the sweep re-derives the bound.

## Declared differences (design doc section 8, and D1)

- 8.1 BF16 weights instead of GGUF Q4_K_M; BF16 KV instead of q8_0
- 8.2 audio features computed by the server's audio tower from the raw audio instead of llama.cpp mmproj packets
- 8.3 vLLM sampling and kernels: temperature 0 with a fixed seed is deterministic per server build, not bit-identical to llama.cpp
- 8.5 forced-close whitespace (vLLM reasoning_end_str) until verified
- D1 reasoning budget 3072 with forced close and one answer-only fallback instead of the historical 8,192-token uncapped completion (36 of 1,080 266-E rows ran to the cap)

8.4, 8.6 and 8.7 concern retrieval and knowledge, which D does not use. D1: the historical D had an 8,192-token completion cap and no thinking budget; the port D runs under the port budget with forced close and one answer-only fallback, so rows that delivered at 8,192 may now be force-closed (36 of 1,080 266-E rows ran to the cap; delivering rows had p90 1,641 and p99 3,610 completion tokens). Read the truncation rate from the smoke before the flight.

## Read rule

Primary: official SLURP pooled SLU-F1 (micro; scorer slurp-slu-official-train, the pinned official evaluate.py), natural-mixture weighting, with intent accuracy (scenario_action_exact) beside it; the per-row F1 is a repo diagnostic reported beside the pooled F1, never an official SLURP number. Replicates: lock, aa, ab (k = 2 replicates of lock). |SLU-F1(aa) - SLU-F1(lock)| <= 1.2 pp (the port analysis recipe's twin rule). A-to-D is a cross-run, cross-program comparison read at the cross-run bar of 2.5 pp; the expected gap is about 26 pp, so the bar does not bind. A cap-hit row is sealed as a receipted empty answer (saea-empty-completion-v1 and saea-thinking-answer-fallback-v1) and scored as such; never re-asked beyond the one registered answer-only fallback (266-E preregistration section 3; decision 199). The M4 trace logs the truncation flag per contact. The D reader and recipe are an open item (this package carries the rule, not the recipe).

## Files

`protocol-registered-v1.json`, the eight plan files (`<smoke|lock|aa|ab>-<generation|fit>-plan-registered-v1.json`), `registered-facts-v1.json` (stage3_measurement blocks keyed by the port run ids), `signoff-conditions-v1.json` (smoke machine sign-off conditions; the D smoke table writer is an open item), `root-admission-draft-v1.json` (decision `slurp-direct-only-fold-b-npu-port-admission-v1`, relpath `coordination/slurp-direct-only-fold-b-npu-port-v1/root-admission-v1.json`, eligible run ids smoke, smoke FIT, lock, lock FIT, aa, aa FIT, ab, ab FIT), `countersign-row-texts-v1.json`, `draft-ledger-rows.md` (the eight DRAFT rows).

## What the main thread does (in order)

1. Commit this package and the cells. Write D's plans after the R1 decision, so a budget move does not force a second registration.
2. Run the speed sweep; fill `PORT_DEPLOYMENT_PROFILE`; re-derive the card-hour bound and write it into planned_gpu_hours, the ledger gpu-hours cells and the draft receipt's budget; re-run the derive and this writer.
3. Owner countersigns the rows (`countersign-row-texts-v1.json`); the main thread mints the decision receipt.
4. Re-pin and flip with `scripts/flights/repin_cell.py` from a clean tree; the direct-only program supplies no first phase, so no prepared artifact is created (`--allow-output-change` is never passed).
5. Open items: the D smoke reader and read recipe, the port chain script, the session identity, runtime receipt and lock entries (9.2-9.4).
