# T cell: the agent system plus TFRL-v2 on the BF16 / vLLM-Ascend Thinking core, iteration 1 (DRAFT / HELD)

The T cell `SAEA-SLURP-TFRL-V2-FOLD-B-NPU-PORT-V1` of the three-way comparison (plan `docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md`; drafts `docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1`), registered before contact on 2026-10-30 at study commit `534b543ff154ca23be8287c1730b040940a3d038` by `scripts/flights/nx/tfrl_v2_port_write_registration.py` from the cells `scripts/flights/nx/tfrl_v2_port_derive_cells.py` derives. Design: `docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md`. Authority: owner decisions of 2026-10-28 on the NPU port (design doc section 9.1: the saea-base Docker container is the execution authority; item 4: the agent drafts the exposure preregistration rows; item 5: the admission receipt is authorised and minted by the main thread; item 6: model-free-check rows for the port preparations; the multi-instance deployment input: the deployment is a named placeholder filled from a measured speed sweep); the owner direction of 2026-10-30 (three-way comparison D / A / T on Fold B with Fold A confirmation; the end goal is a better training-free RL algorithm; plan sections 8-11 rulings). Development numbers only; never a test claim; claims_benefit false. Every row is unconsumed and DRAFT; nothing here authorises a contact.

## Cells (chain order)

| cell | run | FIT run | rows | program | generation calls | FIT calls | speech-seconds | composed config hash | protocol hash |
|---|---|---|---|---|---|---|---|---|---|
| `slurp-native-stage3-tfrl-v2-construction-r01c-smoke-4-npu-port-v1` | `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-SMOKE-4-NPU-PORT-V1` | `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-SMOKE-4-FIT-NPU-PORT-V1` | 4 | `phase-native-module-v1-c7` | 56 | 28 | 239 | `c656076e3a2ffa162a5f96188b3b4f6f4d6a8ad93cc6c003b9a6b8fa0f36122c` | `6fa9cf6da9020031d460a35cc269feb08c970465e0ae0c85f2d1a15a268cf75c` |
| `slurp-native-stage3-tfrl-v2-construction-r01c-i1-h1-npu-port-v1` | `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H1-NPU-PORT-V1` | `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H1-FIT-NPU-PORT-V1` | 313 | `phase-native-module-v1-c7` | 4,382 | 2,191 | 18,474 | `86125d03d68c4cefb01cf3b1031842509036e80eeeb4f8d694b74ee1cbf40a85` | `36f08eeb02f8f5e020876277b2a72ad8dbbb550d3b0dd0f3ce8c820a2b5b67da` |
| `slurp-native-stage3-tfrl-v2-construction-r01c-i1-h2-npu-port-v1` | `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H2-NPU-PORT-V1` | `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H2-FIT-NPU-PORT-V1` | 312 | `phase-native-module-v1-c7` | 4,368 | 2,184 | 19,703 | `f1a9749d0ec04e665b5e6c5599c2ae803f94a599be531072bb2554050f70fabb` | `16db3d67fba5fdaaca8aa7a214753c3ce41af824515d72f3dacafefd846d5259` |
| `slurp-native-stage3-tfrl-v2-fold-b-smoke-4-npu-port-v1` | `SAEA-SLURP-TFRL-V2-FOLD-B-SMOKE-4-NPU-PORT-V1` | `SAEA-SLURP-TFRL-V2-FOLD-B-SMOKE-4-FIT-NPU-PORT-V1` | 4 | `phase-native-module-v1-c10` | 80 | 40 | 204 | `b6669e39e1248fb98da80d6dc4e82ffc81e0d4bd535dd9293b27f0e5a45c9907` | `3bb9257e69948172a2e60512987d2e8297833fec6b075e8d8e541fee7a108db3` |
| `slurp-native-stage3-tfrl-v2-fold-b-i1-npu-port-v1` | `SAEA-SLURP-TFRL-V2-FOLD-B-I1-NPU-PORT-V1` | `SAEA-SLURP-TFRL-V2-FOLD-B-I1-FIT-NPU-PORT-V1` | 1080 | `phase-native-module-v1-c10` | 21,600 | 10,800 | 63,096 | `9edcbb4f2337fac6468b2ec22d2c0282274268e6d7bc7e2ae35937f7a7a30c18` | `edcffc2564f3a6a2913d2eafca300c9df84928b131ee57707e6dca38aa1b4082` |

Family total of iteration 1 (five cells): 30,486 generation calls, 15,243 FIT calls, 101,716 speech-seconds. The caps do not depend on the deployment. Construction cells run 7 contacts per row, Fold B cells 10.

## The programs

- **Construction (c7).** The Solver decoding column group s1, s2, s3, s4 (temperature 0.7, top_p 0.80 / 0.90 / 0.95 / 1.00, top_k 20, the lock seed), the twin `aa` and `lock` LAST: 1 Perception + 6 Solver contacts per row, compiled by the module compiler. Chosen over the Perception-level sampling column (n = 5, c10) because the sampler's trace and credit tooling key scores by arm label s1..s4; the alternative is implemented and tested, not registered. The plan's seed-only trajectories need a per-arm seed the Solver decoding group refuses, so the arms differ in top_p.
- **Fold B (c10).** One Perception-level sampling column `t-v2` (a Solver vote of n = 3 at 0.6 / 0.95 / 20, commit `saea-commit-plurality-support-v1`, min_support 2), the twin `aa`, the replicate `lock-rep2` and `lock` LAST: 4 Perception + 6 Solver contacts per row. `saea-commit-vote-v1` does not exist; greedy_first makes the vote one greedy plus two drawn.
- **Supply bound.** T's bound is A's bound minus 8 (exemplars replace supplied items) with the 2,000-token exemplar cap and the 400-token rule cap folded into the frame's base prompt tokens. Active budget `default`: bound 47, estimate 15,768 of 17,406 tokens.

## Budget switch outcome

- default (3,072 / 4,096): T bound 47, estimate 15,768 of 17,406: fits.
- budget6144-max7168: T bound 43, estimate 14,828 of 14,334: does NOT fit; the largest fitting bound is 40 items. The writer and derive refuse under that switch (owner item O-switch-bound).

## Acceptance and kill rules (plan section 4 with M1 / M2)

An iteration is ACCEPTED only if every one of:

- SLU-F1(t-v2) - SLU-F1(lock) has point >= 1.49 pp and a CI95 lower bound > 0 after the CI is WIDENED by the replicate variance of lock / aa / lock-rep2 (noise model M1)
- the intent guard holds: intent CI95 lower bound >= -0.65 pp and intent point >= 0
- the per-row mean delta (mean of per-row F1) is >= 0; an arm whose per-row delta is negative cannot be nominated
- twin admissibility: |SLU-F1(aa) - SLU-F1(lock)| <= 1.2 pp
- the construction-column gain is >= +1.0 pp: SLU-F1 of the vote over s1..s4 (scenario_action by majority, entity support >= 2 of 4) minus SLU-F1 of lock on the construction rows of the iteration, computed model-free from the sealed flight scores, never from the loop adapter (plan section 11 addendum ruling 6)

CONFIRMED only if: in the Fold A flight of the accepted state the widened CI95 lower bound of t-v2 - lock is > 0 (M2); the optimism term (Fold B delta - Fold A delta) is reported always.

Kill and stop:

- iteration 1 construction gain < +1 pp: stop the family
- an accepted iteration fails Fold A: revert to the previous accepted state and report the optimism term
- an iteration that accepts nothing: stop
- at most 3 iterations

Expected: +2 to +4 pp SLU over A if consumption (P5) is the bottleneck, else 0 (plan section 4); claims_benefit is false either way.

## Rails status: `rails_declared`

The marker is `rails_declared` only when all three deliverables below exist (checked from disk by the writer, `tfrl_v2_port_rails.rails_status`); it is `rails_pending` otherwise. the rails are declared (label, consumption declaration with its refusals, analysis recipe and reader, bank-release registration); the five T cells keep their closest registered labels and zero-sentinel bank pins until the main thread re-mints them at the flip.

- stage-3 label and consumption rail: `tfrl-v2` (`core.stage3_algorithms`) and `src/speech_aware_evidence_acquisition/rails/tfrl_v2_consumption.py` (the `tfrl_v2_bank` frame the module block accepts, the flight-start refusals, the Solver composition, `used_basis` read from the trace): ok True
- analysis recipe and reader: `analysis-recipe-v1.json` (sha256 `303d238a1fb0979862607b2ccb931ce492c7650c6f5629fd6092aa5fc8cce3ab`, pinned in `read-row-draft-v1.md`) and `scripts/analysis/slurp_tfrl_v2_port_read.py`: ok True
- bank-release registration: `bank-release-v1.json` (sha256 `ac377eb6a51f38f5dbbc39cfb6f0ecac08bf2698a61f8d7556b1f052f45ad711`; per iteration the source halves and split identities, the bank receipt hashes (PENDING), the union hash, the measure, the token counts, the consuming cell ids, and the rule that T0 rows are never bank sources): ok True

## What the rails still lack

- `stage3-algorithm-label`: no 'tfrl-v2' record in core.stage3_algorithms. The cells wear the closest registered labels (shared-rollout for the sampling cells, tfgrpo-semantic-advantage-v1 for the Fold B cells).
- `bank-store-roles`: no store roles 'tfrl-exemplar-bank' and 'tfrl-rule-bank' and no scenario-conditioned supply from them. The bank pins live in this block (zero sentinel), not in knowledge_stores; nothing reads them, so the bank is not consumed until the rail exists.
- `vote-commit-policy`: 'saea-commit-vote-v1' does not exist (entity union-vote rule unfixed). The closest registered policy saea-commit-plurality-support-v1 (min_support 2, similarity 0.8) is used; greedy_first makes n = 3 one greedy plus two drawn.
- `top-rule-per-vote`: no rail puts one of the top-3 rules into each of the three vote contacts. The three votes differ in seed only; the rule text is declared, not delivered.
- `exemplar-block`: no rail renders the exemplar block (2,000-token cap) into the Solver prompt. The cap is folded into the frame's base prompt tokens so the context fit is already honest.
- `construction-variant-port-names`: slurp_module_construction.check_cell accepts the r01c variant only under vectors/slurp-stage3-construction-r01c-v1/ and the historical R01C agreement pin. The cell pins the minus-r01 train-exemplars twins directly (the committed VAR-2 R01C precedent) and does NOT name construction_roster; the adapter needs the port prefix and port agreement pin before the variant can be named.
- `solver-decoding-seed-override`: the Solver decoding group refuses a seed override and two equal overrides. The plan's seed-only iteration-1 trajectories cannot be a Solver decoding group; s1..s4 differ in top_p at temperature 0.7 on the lock seed. The seed-true alternative is the Perception-level sampling column (c10), which the sampler's arm-keyed trace and credit tooling cannot label.
- `rule-variants`: no rail applies a rule-bank text to a Solver arm. The rule-variants mode (iteration 2 and later) cannot run.

## Serving and accelerator bound (PROVISIONAL)

Model fragment `bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1`; the deployment is the placeholder `PORT_DEPLOYMENT_PROFILE` (a later deployment profile chosen from a measured speed sweep of the BF16 / vLLM-Ascend Thinking server (several vLLM instances, each tensor-parallel over a group of Ascend cards; per-instance concurrency chosen from the sweep)). Bounds:

- construction-r01c-smoke-4: PROVISIONAL NPU card-hours = 0.5 h wall-clock allowance x instances x T (every held card), from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep)
- construction-r01c-i1-h1: PROVISIONAL NPU card-hours = ceil_0.1(1.3 x 313 rows x 3.6 lock-row equivalents x T / r / 60), T = tensor-parallel size (cards per instance), r = the measured lock-only rows per minute of ONE instance at its chosen sequences per instance; the instance count cancels (wall-clock = card-hours / (instances x T)); T and r from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep)
- construction-r01c-i1-h2: PROVISIONAL NPU card-hours = ceil_0.1(1.3 x 312 rows x 3.6 lock-row equivalents x T / r / 60), T = tensor-parallel size (cards per instance), r = the measured lock-only rows per minute of ONE instance at its chosen sequences per instance; the instance count cancels (wall-clock = card-hours / (instances x T)); T and r from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep)
- fold-b-smoke-4: PROVISIONAL NPU card-hours = 0.5 h wall-clock allowance x instances x T (every held card), from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep)
- fold-b-i1: PROVISIONAL NPU card-hours = ceil_0.1(1.3 x 1,080 rows x 5.04 lock-row equivalents x T / r / 60), T = tensor-parallel size (cards per instance), r = the measured lock-only rows per minute of ONE instance at its chosen sequences per instance; the instance count cancels (wall-clock = card-hours / (instances x T)); T and r from PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep)

The experiments' numeric planned_gpu_hours are the formula at the historical rate with T = 1 (smokes 0.5), a placeholder the plan field requires; the ledger gpu-hours cells carry the formula, so the exposure gate refuses these rows until the sweep re-derives the bound.

## Owner items

- **O-switch-bound**: Under the thinking-budget switch (6,144 / 7,168) the Solver prompt limit is 14,334 and T's bound of 43 items plus both caps estimates 14,828 tokens: it does NOT fit. The largest fitting T bound is 40 items. The owner / planner chooses the bound (40 items, a smaller cap, or a larger window) before the switch is used for T; the writer refuses until then. The default fragment fits (47 items, 15,768 of 17,406).
- **O-construction-group**: The construction program is the Solver decoding column group (c7): the plan's seed-only iteration-1 trajectories need a per-arm seed that the group refuses, so s1..s4 differ in top_p at temperature 0.7. The seed-true alternative is the Perception-level sampling column (c10), which the arm-keyed trace and credit tooling cannot label. Confirm the c7 form or fund the labelling change.
- **O-commit-policy**: saea-commit-vote-v1 does not exist; the vote uses saea-commit-plurality-support-v1 (min_support 2, similarity 0.8). Fix the entity union-vote rule or accept the plurality-support rule before the first Fold B contact.
- **O-fold-a-timing**: Plan section 10: confirming every accepted iteration on Fold A makes Fold A a second selection fold; the draft recommends one confirmation of the final accepted state. M2 as written confirms each accepted iteration. Decide before iteration 2.
- **O-deployment**: PORT_DEPLOYMENT_PROFILE (instances, tensor-parallel size, sequences per instance) is filled from the measured speed sweep; the card-hour bounds are PROVISIONAL until then.

## Open questions

- Does the runner's knowledge-source guard cover every module store or only the single knowledge snapshot, and can a store role be swapped per arm (the cross-fit from iteration 2 needs both)?
- Every T cell keeps preserve_wire_responses true, as the L338 port cell and the historical R01C column do. Does the M4 trace need the preserved wire responses, or do the sealed per-contact records suffice (the setting only costs disk)?
- Which rail renders the exemplar block and the per-vote rule text (rails_pending), and when is the first-phase supply of the Fold B cells re-prepared with it?
- The R01C encoder, the alias, the content derivative and the agreement chain have no registration documents or chain scripts yet (rows-only here).
- Iterations 2 and 3 (cells, plans, rows) are registered only after the previous read.

## Declared differences

- 8.1 BF16 weights instead of GGUF Q4_K_M; BF16 KV instead of q8_0
- 8.2 audio features computed by the server's audio tower from the raw audio instead of llama.cpp mmproj packets
- 8.3 vLLM sampling and kernels: temperature 0 with a fixed seed is deterministic per server build, not bit-identical to llama.cpp
- 8.4 causal retrieval encoder (forced and measured); every sealed retrieval vector regenerated
- 8.5 forced-close whitespace (vLLM reasoning_end_str) until verified
- 8.6 re-authored semantic knowledge (intent-boundaries-npu-port-v1, slot-contracts-npu-port-v2; decision 9.5)
- 8.7 CPU float32 retrieval encoder instead of CUDA bfloat16
- T1 (plan section 9 ruling 4): in the T arm exemplars REPLACE supplied items, they do not add to them (supply bound = A's bound minus 8 plus an exemplar block of at most 2,000 tokens; the rule bank is Solver instruction text of at most 400 tokens); A keeps its bound and no bank. A declared T-vs-A protocol difference, not a confound.
- T2 (plan section 4 and section 11 ruling 2): iteration-1 trajectories differ in decoding only (no bank exists yet); from iteration 2 the rule-variants mode applies.
- T3 (plan section 9 ruling 3 and section 11 ruling 1): the construction column R01C is sampled over two interleaved halves; the bank a half consumes in iteration t+1 is distilled from the OTHER half's positives; the Fold B flight consumes the union.

## Files

`protocol-registered-v1.json`, ten plan files (generation and FIT of each cell), `registered-facts-v1.json` (stage3_measurement blocks keyed by the T run ids), `signoff-conditions-construction-v1.json`, `signoff-conditions-fold-b-v1.json`, `root-admission-draft-v1.json` (decision `slurp-tfrl-v2-npu-port-admission-v1`, relpath `coordination/slurp-tfrl-v2-npu-port-v1/root-admission-v1.json`), `countersign-row-texts-v1.json` and `draft-ledger-rows.md` (six preparation rows and the loop rows in chain order), and the rails files `analysis-recipe-v1.json`, `read-row-draft-v1.md` and `bank-release-v1.json`. The configs are `configs/{baseline,experiment}/slurp-native-stage3-tfrl-v2-*` and two campaigns.

## What the main thread does (in order)

1. Commit this package, the five cells and the campaigns; do not append the ledger rows before the owner countersigns.
2. Resolve the owner items; run the speed sweep and fill `PORT_DEPLOYMENT_PROFILE`; re-derive the card-hour bounds; re-run the derive and this writer.
3. Run the rows-only items in order (R01C encoding, alias, content derivative, agreement, rule bank i0), re-pin every zero-sentinel pin, then the construction smoke and the two halves.
4. Run credit and distillation, the reflector and the bank release; re-pin the Fold B cells' bank pins; run the Fold B smoke and the flight.
5. Iterations 2 and 3 are registered only after the previous read.
