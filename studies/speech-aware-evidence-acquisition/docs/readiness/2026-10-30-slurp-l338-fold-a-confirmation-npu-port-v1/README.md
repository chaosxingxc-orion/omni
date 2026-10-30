# L338 NPU port, Fold A confirmation: the lever column on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold A (DRAFT / HELD)

The Fold A CONFIRMATION twin of the L338 port flight (package `docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1`), registered before contact on 2026-10-30 at study commit `28bf317600000000000000000000000000000000` by `scripts/flights/nx/l338_port_write_registration.py --family fold-a-confirmation` from the cell `scripts/flights/nx/fold_a_port_derive_cells.py` derives. Design: `docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md`; plan `docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md` (M2, owner answer R2: Fold A confirmation is required for every accepted lever and for the D/A/T comparison). Authority: owner decisions of 2026-10-28 on the NPU port (design doc section 9.1: the saea-base Docker container is the execution authority; item 4: the agent drafts the exposure preregistration rows; item 5: the admission receipt is authorised and minted by the main thread; item 6: model-free-check rows for the port preparations; the multi-instance deployment input: the deployment is a named placeholder filled from a measured speed sweep). Development numbers only; never a test claim; claims_benefit false.

## The cell (one pass, no smoke)

| cell | run | FIT run | rows | generation calls | FIT calls | speech-seconds | accelerator bound |
|---|---|---|---|---|---|---|---|
| `slurp-native-stage3-l338-lever-column-fold-a-npu-port-v1` | `SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-A-NPU-PORT-V1` | `SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-A-FIT-NPU-PORT-V1` | 1080 | 36720 | 18360 | 104874 | PROVISIONAL: PROVISIONAL NPU card-hours = ceil_0.1(1.3 x 1,080 rows x 8.52 lock-row equivalents x T / r / 60), T = tensor-parallel size (cards per instance), r = the measured lock-only rows per minute of ONE instance at its chosen sequences per instance; the instance count cancels (wall-clock = card-hours / (instances x T)); T and r from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep) |

It is the Fold B port flight (program `phase-native-module-v1-c17`, 17 contacts per row, arms l1-support, l2-union, aa, lock-rep2, lock) moved onto TRAIN Fold A (`SLURP-EVO-FOLD-A-V2`, roster `docs/receipts/rosters/slurp-evo-fold-a-v2.json`, dataset fragment `slurp-speech-native-evo-fold-a-1080`) by one declared transform (every Fold B identity token becomes its Fold A token). The calls and FIT caps equal the Fold B flight's (rows are 1,080 in both); the speech-seconds cap is the Fold A roster's: ceil(3084.5284375 x 34) = 104,874. There is no smoke: the programs and harness bytes were smoked and signed off on Fold B (`docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/signoff-conditions-v1.json`).

## What is pending (the Fold A preparations; nothing is run, nothing is invented)

The cell pins these Fold A artefacts by the zero sentinel (no flip accepts it). Run each (CPU, model-free, card-free; the encoder is a CPU float32 tool contact), record its sha256 in `scripts/flights/nx/fold_a_port_pins.json`, re-run the derive and the writer:

- `preparation-runs/SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-NPU-PORT-V1/encode-v1/receipt.json` (SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-NPU-PORT-V1)
- `prepared/SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-NPU-PORT-V1/query-dimension-alias.json` (SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-NPU-PORT-V1)
- `vectors/slurp-stage3-filler-fold-a-npu-port-v1/query-vectors-speech-content-v1.provenance.json` (SAEA-SLURP-FOLD-A-INPUTS-PREPARE-NPU-PORT-V1)
- `prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-A-NPU-PORT-V1/receipt.json` (SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-A-NPU-PORT-V1)
- `prepared/SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-A-NPU-PORT-V1/phase-supply-v1.json` (SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-A-NPU-PORT-PHASE-SUPPLY-V1)

The agreement table (FA4) is built against the never-flown, agreement-free Fold A comparator pair `slurp-native-stage3-l338-lever-column-fold-a-comparator-npu-port-v1` (config only, written by the same derive).

## Confirmation rule

A lever is accepted only if its Fold A widened CI95 lower bound is above zero (M2); the optimism term (Fold B delta minus Fold A delta) is reported for every accepted state. The reader and recipe of the port (`scripts/analysis/slurp_l338_port_column_read.py`, `analysis-recipe-v1.json` of the Fold B package) are bound to the Fold B roster; re-pointing them at Fold A is a separate item. The D and T Fold A passes are separate cells (D: the direct-only writer on the Fold A roster; T: registered after an accepted iteration exists, because its baseline pins the accepted banks).

## Serving: protocol facts and the deployment placeholder

Model fragment `bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1` (transport vLLM, BF16), per-sequence window 24,576 tokens, prompt order `native-raw-speech-query-prefix-v1`; the deployment is the one named placeholder `PORT_DEPLOYMENT_PROFILE` (a later deployment profile chosen from a measured speed sweep of the BF16 / vLLM-Ascend Thinking server (several vLLM instances, each tensor-parallel over a group of Ascend cards; per-instance concurrency chosen from the sweep)), shared with the Fold B cells.

## Declared differences (design doc section 8)

- 8.1 BF16 weights instead of GGUF Q4_K_M; BF16 KV instead of q8_0
- 8.2 audio features computed by the server's audio tower from the raw audio instead of llama.cpp mmproj packets
- 8.3 vLLM sampling and kernels: temperature 0 with a fixed seed is deterministic per server build, not bit-identical to llama.cpp
- 8.4 causal retrieval encoder (forced and measured); every sealed retrieval vector regenerated
- 8.5 forced-close whitespace (vLLM reasoning_end_str) until verified
- 8.6 re-authored semantic knowledge (intent-boundaries-npu-port-v1, slot-contracts-npu-port-v2; decision 9.5)
- 8.7 CPU float32 retrieval encoder instead of CUDA bfloat16

## Files

`protocol-registered-v1.json`, the two plan files, `registered-facts-v1.json`, `signoff-conditions-v1.json` (no smoke), `root-admission-draft-v1.json` (decision `slurp-l338-lever-column-fold-a-npu-port-admission-v1`, relpath `coordination/slurp-l338-lever-column-fold-a-npu-port-v1/root-admission-v1.json`, eligible run ids flight and flight FIT), `countersign-row-texts-v1.json`, `draft-ledger-rows.md` (the two DRAFT cell rows and the five DRAFT TEXT preparation rows).

## What the main thread does (in order)

1. Commit this package, the Fold A cell and comparator configs, and the tests.
2. Run FA1-FA4 (card-free); pin their sha256; re-run the derive and this writer; write the preparation rows into the ledger only after each run.
3. Fill `PORT_DEPLOYMENT_PROFILE` from the measured sweep; re-derive the card-hour bound; owner countersigns (`countersign-row-texts-v1.json`).
4. Mint the admission receipt, re-pin and flip the cell with `scripts/flights/repin_cell.py` (it prepares the cell's own first-phase supply, FA5).
