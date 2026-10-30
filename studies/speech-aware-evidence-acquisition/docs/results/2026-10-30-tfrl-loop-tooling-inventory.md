# TFRL loop tooling inventory and port plan (Sonnet 5.5, task T5, 2026-10-30)

Study tree `/mnt/cephfs/xingchao/omni/work/dev/umbrella/studies/speech-aware-evidence-acquisition` (HEAD `068ee153` plus the T1 working-tree files). Paths below are relative to it. Model-free
inventory; no model call, card, readiness flip or commit. Classification: **AS-IS** (reusable unchanged), **PORT** (needs a port mode, named below), **REPLACED**
(superseded by TFRL-v2 of plan section 4, `docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md`), **HISTORIC** (one-off read, no port role).

## 1. What the 263-A loop is made of

The Windows-side bindings are: the historical datasets lock path on a Windows drive (`DEFAULT_LOCK` in five scripts), the human PowerShell step (`SAEA-Batch-Driver`), flips on a
Windows-mounted checkout (`SAEA_MAIN`, `/home/chao/...` commit files and the 9p refusal), the `llama.cpp` Q4_K_M session and its build commit, the lock v2.2 / v2.3 layers, the
historical ids (`slurp-native-stage3-tfrl-*`, Fold B V2, `serving-slurp-native-candidate-five-train-a-np4-v1`) and Instruct-era prompts.

### Python modules under `src/speech_aware_evidence_acquisition/`

| file | does | bindings | verdict |
|---|---|---|---|
| `analysis/sealed_credit/archive.py` | write-once archive of states (artefact hashes, credit vector, verdict); text-free | none (slot names fixed: `exemplar_index_release`, `slot_store_release`, `key_view`, `supply_fragment`, `delivery_rule_table`, `scope_routing`, `playbook_release`; arms `typed`, `ace`) | AS-IS (state admission). TFRL-v2 maps rule bank to `playbook_release`, exemplar bank to `exemplar_index_release`, vote setting to `supply_fragment`, arm `typed` |
| `analysis/sealed_credit/selection.py` | ruled selection clauses a-e, stop rule 3, curve; floor numbers via `rules_from_floor_table` | none (reads `scripts/checks/stage3_effect_floor_check.py` FLOOR_TABLE; 1.49 pp SLU, 0.65 intent at 1,080 rows) | AS-IS; the wrapper adds the M1/M2 clauses |
| `analysis/sealed_credit/accept.py` | knob nomination from predicted effects (run-1 inner loop) | historical estimate pools | AS-IS code, not used by v2 (its `design_guard_and_floor` is used through `selection`) |
| `analysis/sealed_credit/parent_choice.py`, `meta_summary.py` | parent = accepted state maximising LB - 0.25 x flown children; proposer-side summary | none | AS-IS, optional (v2 is a linear chain; not wired) |
| `analysis/sealed_credit/{credit,entity_states,estimate,pools,policy_space,row_states}.py` | offline per-node credit and predicted-effect estimator of run 1/2 over sealed Instruct pools | `configs/analysis/sealed-credit-pools-v1.json` names historical run ids | REPLACED by per-row trajectory credit (v2 sample/credit rows) |
| `rails/tfrl_typed_proposer.py` (1,782 lines) | typed proposer seam: one JSON candidate per contact at T 0, closed schema, validation | frozen Instruct core text seam; `llama_cpp_build_commit` and model-file hashes in its receipts; `configs/policy/phase-tfrl-typed-proposer-v1.json`, `configs/probes/tfrl/typed-proposer-v1.txt` | REPLACED (v2 samples trajectories, not rule proposals) |
| `rails/tfrl_ace_material.py` (1,497) | ACE reflector material, deterministic curator, `ace-playbook` store releases | lock v2.3 store id, Instruct prompts (`configs/probes/tfrl/ace-reflector-v1.txt`, `configs/policy/phase-tfrl-ace-reflector-v1.json`) | REPLACED for the reflector (contrast-pair rule only); curator and release writer are PORT candidates if T3 delivers the rule bank as a knowledge store |
| `rails/adapters/slurp_module_construction.py` | construction variant of the lock for R01C: moves the `train-exemplars` pool to the minus-r01 twin, pins the R01C agreement table | historical sha256 pins (see section 3) | **PORT** (port pins) |

### Scripts

| file | does | bindings | verdict |
|---|---|---|---|
| `scripts/flights/tfrl/tfrl_loop_driver.py` (1,268) | `open/close/status` driver: tables, proposer/reflector cells, ACE playbook, column cells, selection, archive, curve | `DEFAULT_LOCK` on a Windows drive; HELD DRAFT ledger flow; PowerShell human steps; `tf.*` historical ids; Fold B V2; lock v2.3 | **REPLACED** by `scripts/flights/nx/tfrl_port_loop.py` (this task) |
| `scripts/flights/tfrl/tfrl_loop_lib.py` (741) | write-once receipts, protocol pin, `Runner`, credit adapters, archive helpers, `zero_credit` | `BUSY_PATTERN` names the Windows flip/batch scripts (matches nothing on the port host, harmless); `list_processes` uses `/proc` | AS-IS for receipts, `put_state_idempotent`, `zero_credit` (reused by the port driver); the proposer/ACE adapters are dead under v2 |
| `scripts/flights/tfrl/tfrl_family.py` | shared ids and the supply frame of the historical iteration cells | `LOOP_ID SAEA-SLURP-TFRL-LOOP-V1`, lock v2.3, Fold B V2 roster, serving session, FIT frame (2000/512/24576/235) | PORT only for its hashing helpers (`canonical_sha256`, `sha256_file`, reused); ids are historical |
| `scripts/flights/tfrl/build_tfrl_loop_protocol.py`, `tfrl_cell_stage_inputs.py`, `tfrl_i1_queue.py`, `derive_tfrl_iteration_cells.py`, `write_tfrl_iteration_registration.py`, `tfrl_overlay.py` | the 263-A protocol pin, proposer-output staging, batch queue, iteration cell derivation and registration | historical ids, `llama.cpp` receipts, `DEFAULT_LOCK`, Windows queue | REPLACED (v2 cells are registered by T4's writer) |
| `scripts/flights/tfrl/batch_chain_tfrl_{const,i1}.sh`, `tfrl_{const,i1}_flip.sh`, `tfrl_*_repin_*.sh`, `tfrl_i1_read.sh`, `tfrl_const_read.sh` | queue, flip and read wrappers | `/mnt/<drive>/...`, `/home/<user>/...`, a Windows venv, the PowerShell batch driver, 9p refusal | REPLACED (port flights use `scripts/flights/nx/l338_port_*` style tools and the `work/scripts/port_py.sh` transport) |
| `scripts/flights/tfrl/derive_tfrl_construction_cells.py`, `write_tfrl_construction_registration.py` | derive and register the construction lock column (R01C) | historical ids, `q4km-...-np4-ctx24k-max512-v1` fragment, lock v2.2 + `r01c` variant, FIT cap 5,000 | **PORT** (T4 writer, pattern of `l338_port_derive_cells.py` / `l338_port_write_registration.py`; delta in section 3) |
| `scripts/flights/tfrl/tfrl_mint_receipt.py` | mints the root admission receipt under `$SPEECHRL_DATA_DIR/coordination` | data-root relative | PORT (pattern of `l338_port_mint_receipt.py`; the main thread mints) |
| `scripts/flights/tfrl/tfrl_row_scores.py` | exports per-row official SLU-F1 of `lock` and `ace` columns | `DEFAULT_LOCK` on a Windows drive (must pass `--lock`), requires an `ace` arm | **PORT**: lock-only mode that writes `{"arms": {"lock": {row_id: f1}}}`; the port driver reads that file as `construction-lock-row-scores.json` |
| `scripts/analysis/tfrl_contrast_tables.py`, `tfrl_delivery_fit.py` | failure-cube tables and cross-fitted delivery-threshold fits over the lock column | `DEFAULT_LOCK` (Windows), `SAEA-SLURP-STAGE3-TFRL-*` ids | PORT as diagnostics only (reading the port lock column); not an input to v2 |
| `scripts/analysis/sealed_credit_inner_loop.py`, `sealed_credit_psv1_preread.py` | run-1/2 inner-loop CLI and a P-SV-1 pre-read | `DEFAULT_LOCK` (Windows), historical pools | HISTORIC |
| `scripts/analysis/slurp_tfrl_coverage_preread.py` (1,202) | decision-265 offline coverage pre-read, Fold B | Instruct pools | HISTORIC |
| `scripts/build_ace_playbook_i0_release.py` | builds the EMPTY `ace-playbook` release i0 | lock v2.3 store | PORT only if the rule bank ships as a store; the construction column stays on the plain lock (the variant refuses a layer that declares the store) |
| `scripts/flights/revtfrl/*` | Reviewer-TFRL cells (Sept) | Instruct, Fold A | HISTORIC (Reviewer is retired) |
| `rails/stage3_advantage*.py`, `stage3_gepa*.py`, `group_relative_feedback.py` (Sept generation) | TF-GRPO semantic advantage and GEPA | Instruct, Q4_K_M | HISTORIC (the v2 contrast-pair reflector replaces both) |

Tests of the historical tooling stay as they are (`tests/unit/test_tfrl_*.py`, `test_sealed_credit_*.py`); none was edited.

## 2. The port driver (this task)

`scripts/flights/nx/tfrl_port_loop.py`, tests `tests/unit/test_tfrl_port_loop.py` (22 tests, passing in `saea-base` on `npu2`, cores 184-191). `--dry-run` prints the plan and touches no file.
Reused unchanged: `selection.select` / `rules_from_floor_table`, `archive.build_state` / `Archive`, `tfrl_loop_lib.write_receipt` / `put_state_idempotent` / `zero_credit`, the T1 validator
`contracts/rollout_trace.validate_line` (the T1 file is in the working tree; the driver imports it directly). Added: the acceptance wrapper (adds twin admissibility within 1.2 pp,
a no-negative-per-row clause, the M1 replicate widening as a placeholder marked TODO-T2, and `fold_a_confirms` for M2), the state-admission wrapper, the resumable loop, and the three
call sites below. Every data path is under `$SPEECHRL_DATA_DIR/analysis/SAEA-SLURP-TFRL-V2-NPU-PORT-V1/`.

**Sampler interface (T3 fills it).** `Sampler.sample(SamplerRequest) -> Sequence[Trajectory]`.
- `SamplerRequest`: `iteration`, `column_rows` (625 construction row ids), `k` (4), `temperature` (0.7), `rule_bank` (opaque mapping, `{"rules": [{"rule_id", "kind", "text"}]}`), `exemplar_bank`
  (opaque mapping keyed by scenario), `lock_row_f1` (row id to the lock column's row F1), `seed`.
- `Trajectory`: `trace` (a `tfrl` line exactly as T1 closes it: `ts, run_id, attempt_id, row_id, arm, iteration, trajectory_id, reward, selected, reason`), plus the fields the T1 line does not carry
  and credit needs: `k_index, temperature, row_f1, lock_row_f1, rule_ids, exemplar_ids, truncated`. The driver refuses a result that is not exactly K valid trajectories per row, with
  `trace["reward"] == row_f1 - lock_row_f1` and `lock_row_f1` equal to the construction lock column's score.
- `Distiller.distil(iteration, trajectories, rule_bank, exemplar_bank) -> DistilResult` (new banks, `vote_n`, contrast pairs, refused rules, optional `construction_gain_pp`); `CandidateReader.read`
  returns the Fold B credit vectors (candidate, twin, optional parent, optional replicate sd and per-row delta) from `i<t>/fold-b-read.json` written after the registered flight; absent file
  means the loop halts as `awaiting-read`. Stubs for sampler and distiller fail closed (`TODO-T3`).
- Note for T1/T3: the T1 `tfrl` line has no row F1, so the credit fields travel in the driver's own `trajectories.jsonl` beside `trace.jsonl` (the T1 lines only).

## 3. The construction column

| item | value |
|---|---|
| roster | `SLURP-STAGE3-CONSTRUCTION-R01C`, receipt `docs/receipts/rosters/slurp-stage3-construction-r01c.json`, **625 rows** (`items` length verified), split identity `f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515`; TRAIN rows outside Fold A and Fold B |
| dataset fragment | `slurp-speech-native-stage3-construction-r01c` |
| program | `phase-native-module-v1-c2` (plain lock: Perception 1 + Solver 1), 2 contacts per row, no supply arms, no columns |
| role | the material source: one lock column whose per-row official scores feed the sampler's `lock_row_f1` and the credit; claims no benefit (`measurement_design material-source`) |
| historical cell | `slurp-native-stage3-tfrl-construction-lock-r01c-v1`, run `SAEA-SLURP-STAGE3-TFRL-CONSTRUCTION-LOCK-R01C-V1`; smoke `...-smoke-4-v1` (4 rows) |
| **re-flown on the port?** | **Yes.** The historical column ran on the Q4_K_M llama.cpp core with historical vectors; the lock changed (BF16 vLLM-Ascend core, regenerated Nemotron vectors and pools, re-authored stores), and the historical protocol itself says the column is re-flown if the lock changes. Nothing from the historical column is reused |
| port ids (rule of `l338_port_family`, `-V1`/`-v1` to `-NPU-PORT-V1`/`-npu-port-v1`) | cell `slurp-native-stage3-tfrl-construction-lock-r01c-npu-port-v1`, run `SAEA-SLURP-STAGE3-TFRL-CONSTRUCTION-LOCK-R01C-NPU-PORT-V1`, attempt `attempt-slurp-native-stage3-tfrl-construction-lock-r01c-generation-npu-port-v1`, baseline `...-baseline-raw-port-v1`; smoke and FIT ids by the same rule |

**Registration delta for the writer (T4 / T5 follow-up).**
1. Serving model: fragment `bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1` (core fragment hash `a35b274058d7ac7558d3d0ac32cca8bb3c9321cb13ea89f16b38003fdd18c720`) replaces
   `q4km-mmproj-free-slurp-native-bounded-perception-proof-np4-ctx24k-max512-v1`; session placeholder `serving-slurp-thinking-module-fold-b-npu-port-pending-deployment-profile-v1`; deployment fields
   (instances, TP, sequences) stay out of the protocol hash.
2. Caps (unchanged by the port, from the historical registration): generation 1,250 (625 x 2), smoke 8; FIT endpoint cap 5,000 (4 per contact), smoke 32; audio ceiling 5,454 s, smoke 35 s.
   Card-hours: the formula of `l338_port_family.provisional_card_hours` with arm mix 1 (plain lock): `ceil_0.1(1.3 x 625 x T / r / 60)`, T and r from the deployment profile and the measured lock rate; the
   historical 1.0 GPU-h cap does not carry over as a number.
3. Retrieval and preparation pins that do not exist yet for R01C on the port (the Fold B twins exist under `prepared/` and `vectors/`; the R01C twins do not):
   a. the two sealed R01C speech-query vector files (historical prefix `vectors/slurp-stage3-construction-r01c-v1/`: `query-vectors-speech-content-v1.jsonl`, `query-vectors-speech-segments.jsonl`) regenerated with the port Nemotron encoder, with
      provenance and the speech-query encoding receipt (pattern: `SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-NPU-PORT-V1`);
   b. the R01C exemplar-agreement table against the port minus-r01 pool (pattern: `prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1`; 625 rows x 20 votes), built by `scripts/build_slurp_intent_exemplar_agreement.py`;
   c. the intent-boundary query-dimension alias and text-facet cache for R01C rows (pattern: `SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-B-ALIAS-NPU-PORT-V1`);
   d. the cell's own first-phase supply `prepared/<port run id>/phase-supply-v1.json`, built at the flip by the re-pin tool;
   e. stores: port `intent-boundaries-npu-port-v1`, `slot-contracts-npu-port-v2` and facet vectors `vectors/slurp-kus3/facet-vectors.jsonl` already exist on the data root.
4. `rails/adapters/slurp_module_construction.py` carries historical pins for the pool and the agreement table. Observed on the data root: the port `size-pool-annotated-exemplars/snapshot.json` sha256 is
   `fcb5506c63653abe996dea15fdccd7c81a84a837b6377ca13f5b35d73283d6b2` (hook constant `4c498dc0...`), and the port `size-pool-annotated-exemplars-minus-r01/snapshot.json` is `f549466f...5795ca` (hook constant `6eaa03ae...`);
   the two `release.json` files match their constants. The committed port Fold B baseline still names `4c498dc0...`, so the pin state of the pool is not settled by this inventory: the writer must
   re-pin the pools on the port and the hook needs port constants (or a port mode that reads them from the registration). The agreement pin is the R01C table of item 3b.
5. Construction variant rule stays: the column composes on the plain lock, not on a layer that declares the `ace-playbook` store (the hook refuses it); TFRL-v2 consumes its banks only in the later T cell, not in this column.
6. Scoring: the port scorer reads `runs/<run id>/<attempt>/<run id>.scores.json`; `tfrl_row_scores.py` needs the lock-only mode above to produce `construction-lock-row-scores.json`.

Cost of the later flights: one 625-row lock column (1,250 contacts), then per iteration 2,500 Solver trajectories (625 x K=4; Perception assumed reused from the lock column, TODO-T3 to confirm), 7,500 over three iterations.

## 4. Open items

- T3 fills the sampler and the distiller; T2 replaces `widen_credit`; the Fold A confirmation flight (M2) is a separate step the driver reports but does not run.
- No exposure entry, readiness flip or flight is implied by any file here.

**Successor note (2026-10-30, pool-pin check):** the "pool pins differ on disk" item above is withdrawn. `4c498dc0…` and `6eaa03ae…` are `KnowledgeSnapshot.content_hash()` values (`rails/knowledge_stores.py:403`), not file sha256s; recomputed on the data root they match, and the files are byte-identical to the historical pools (file sha256 `fcb5506c…` / `f549466f…`, 9,354 / 8,554 utterances). No re-pin. Report: `work/scratch/brief/2026-10-30-pool-pin-check.md`.
