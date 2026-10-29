# L338 NPU port: the lever column on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B (DRAFT / HELD)

The NPU-port twin of the L338 lever-column flight (historical package `docs/readiness/2026-10-19-slurp-l338-lever-column-fold-b-v1`), registered before contact on 2026-10-28 at study commit `ad67226f` by `scripts/flights/nx/l338_port_write_registration.py` from the cells `scripts/flights/nx/l338_port_derive_cells.py` derives. Design: `docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md`. Authority: owner decisions of 2026-10-28 on the NPU port (design doc section 9.1: the saea-base Docker container is the execution authority; item 4: the agent drafts the exposure preregistration rows; item 5: the admission receipt is authorised and minted by the main thread; item 6: model-free-check rows for the port preparations; the multi-instance deployment input: the deployment is a named placeholder filled from a measured speed sweep). Development numbers only; never a test claim; claims_benefit false.

## Cells (chain order)

| cell | run | FIT run | rows | generation calls | FIT calls | speech-seconds | accelerator bound |
|---|---|---|---|---|---|---|---|
| `slurp-native-stage3-l338-lever-column-fold-b-smoke-4-npu-port-v1` | `SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-SMOKE-4-NPU-PORT-V1` | `SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-SMOKE-4-FIT-NPU-PORT-V1` | 4 | 120 | 240 | 306 | PROVISIONAL: PROVISIONAL NPU card-hours = 0.5 h wall-clock allowance x instances x T (every held card), from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep) |
| `slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1` | `SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1` | `SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-FIT-NPU-PORT-V1` | 1080 | 32400 | 64800 | 94644 | PROVISIONAL: PROVISIONAL NPU card-hours = ceil_0.1(1.3 x 1,080 rows x 7.52 lock-row equivalents x T / r / 60), T = tensor-parallel size (cards per instance), r = the measured lock-only rows per minute of ONE instance at its chosen sequences per instance; the instance count cancels (wall-clock = card-hours / (instances x T)); T and r from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep) |

The calls, FIT and audio caps are the historical registration's (they do not depend on the deployment). The arms, program (`phase-native-module-v1-c15`, 15 contacts per row), roster, templates, thinking budget 3072 / cap 4096, decoding (greedy, seed 20260913) and read rule are the historical cell's.

## Serving: protocol facts and the deployment placeholder

Model fragment `bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1` (transport vLLM, BF16). Protocol facts: the per-sequence context window of 24,576 tokens, the reasoning budget and cap, the decoding, the prompt order `native-raw-speech-query-prefix-v1`. The DEPLOYMENT -- instances, tensor-parallel size, sequences per instance, devices -- is the one named placeholder `PORT_DEPLOYMENT_PROFILE` (a later deployment profile chosen from a measured speed sweep of the BF16 / vLLM-Ascend Thinking server (several vLLM instances, each tensor-parallel over a group of Ascend cards; per-instance concurrency chosen from the sweep)). No sequence count, replica count or tensor-parallel size is a fact of this registration. Service session name `serving-slurp-thinking-module-fold-b-npu-port-pending-deployment-profile-v1` is deployment-neutral; session identity, the vLLM runtime receipt and the lock entries are open owner decisions 9.2-9.4.

Places that depended on the model fragment's pre-sweep `--max-num-seqs 4`, and where each went (the move was made by commit db798a83 and carried through by stream S13):

- moved: configs/model/bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1.json no longer carries server_args --max-num-seqs; the vLLM argv takes it from the selected deployment profile (core.deployment.instance_server_argv) and a vllm fragment that declares it is REFUSED (core.session.serving, rails.adapters.runtime_view)
- moved: configs/model/bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1.json no longer carries profile_slot_count (core.driver.speech_native requires it, with -np / -c, only on its llama-profile CLI paths -- prompt-audit, profile, finalize-from-disk -- which a vllm fragment fails closed on)
- moved: configs/model/bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1.json geometry_basis text now names the 24576-token per-sequence window and puts the sequence count in the deployment profile
- unchanged by owner ruling: configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json serving_slots 4 / serving_slot_tokens 24576 (rails.adapters.runtime_view.configured_geometry now checks only the --max-model-len window of a vllm fragment against serving_slot_tokens)
- unchanged by owner ruling: configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-smoke-4-baseline-raw-port-v1.json serving_slots 4 / serving_slot_tokens 24576 (same)
- moved: the prepared first-phase supply identity (core.preparation.artifact_inputs digests the composed config, model fragment included): moving the fragment makes every phase supply of these cells stale and the re-pin tool mints the next phase-supply-v<N> (output-equality asserted); prepared/<port run>/phase-supply-v1.json exists for the flight run and is re-pinned by repin_cell.py at the flip, not by the writer
- moved: the composed_config_hash / protocol_hash of every plan in this package (the registration writer was re-run after the move)
- open: the historical cost inputs (l338_cost_inputs.json, 4 slots) behind the illustrative 1.95 lock rows per minute; the port bound uses the measured r

## Accelerator bound (PROVISIONAL)

- flight: PROVISIONAL NPU card-hours = ceil_0.1(1.3 x 1,080 rows x 7.52 lock-row equivalents x T / r / 60), T = tensor-parallel size (cards per instance), r = the measured lock-only rows per minute of ONE instance at its chosen sequences per instance; the instance count cancels (wall-clock = card-hours / (instances x T)); T and r from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep).
- smoke: PROVISIONAL NPU card-hours = 0.5 h wall-clock allowance x instances x T (every held card), from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep).
- Illustration only: at the historical lock rate (1.95 rows per minute) the flight bound reads 90.3 x T card-hours.
- The experiments' numeric planned_gpu_hours (90.3 / 0.5) are placeholders the plan field requires; the ledger gpu-hours cells carry the formula, so the exposure gate refuses these rows ("must be a number") until the sweep re-derives the bound.

## Declared differences (design doc section 8)

- 8.1 BF16 weights instead of GGUF Q4_K_M; BF16 KV instead of q8_0
- 8.2 audio features computed by the server's audio tower from the raw audio instead of llama.cpp mmproj packets
- 8.3 vLLM sampling and kernels: temperature 0 with a fixed seed is deterministic per server build, not bit-identical to llama.cpp
- 8.4 causal retrieval encoder (forced and measured); every sealed retrieval vector regenerated
- 8.5 forced-close whitespace (vLLM reasoning_end_str) until verified
- 8.6 re-authored semantic knowledge (intent-boundaries-npu-port-v1, slot-contracts-npu-port-v2; decision 9.5)
- 8.7 CPU float32 retrieval encoder instead of CUDA bfloat16

## Files

`protocol-registered-v1.json`, the four plan files, `registered-facts-v1.json` (stage3_measurement blocks keyed by the port run ids), `signoff-conditions-v1.json`, `root-admission-draft-v1.json` (decision `slurp-l338-lever-column-fold-b-npu-port-admission-v1`, relpath `coordination/slurp-l338-lever-column-fold-b-npu-port-v1/root-admission-v1.json`, eligible run ids smoke, smoke FIT, flight, flight FIT), `countersign-row-texts-v1.json`, `draft-ledger-rows.md` (the four DRAFT rows and the six model-free-check rows of the port preparations).

## What the main thread does (in order)

1. Commit this package, the two port cells and the ledger rows.
2. Run the speed sweep; fill `PORT_DEPLOYMENT_PROFILE` with a deployment profile (`configs/deployment/*.json`; `--max-num-seqs` per instance lives there, not in the model fragment, the baselines or any hash); re-derive the card-hour bound and write it into planned_gpu_hours, the ledger gpu-hours cells and the draft receipt's budget; re-run the derive and this writer.
3. Owner countersigns the rows (`countersign-row-texts-v1.json`); then mint: `SAEA_STUDY_ROOT=<study> SAEA_DATA_ROOT=<data root> python -B scripts/flights/nx/l338_port_mint_receipt.py --dry` then without `--dry`.
4. Re-pin and flip with `scripts/flights/repin_cell.py` from a clean tree (both cells, `--decision coordination/... --decision-id slurp-l338-lever-column-fold-b-npu-port-admission-v1`): it prepares the smoke's phase supply, re-prepares the flight's as phase-supply-v2 (output equality asserted; never `--allow-output-change`) and pins the decision receipt. `SAEA_ALLOW_9P_FLIP=1` is needed because the tree is under `/mnt/` (CephFS, not 9p).
5. The port reader is `scripts/analysis/slurp_l338_port_column_read.py` (stages smoke and read; it writes `analysis/SAEA-SLURP-L338-LEVER-COLUMN-FOLD-B-READ-NPU-PORT-V1/smoke-decode-cost.json` and `analysis/SAEA-SLURP-L338-LEVER-COLUMN-FOLD-B-READ-NPU-PORT-V1/port-verdicts.json`). Open items: the READ row of the port, the port chain script, the session identity, runtime receipt and lock entries (9.2-9.4).
