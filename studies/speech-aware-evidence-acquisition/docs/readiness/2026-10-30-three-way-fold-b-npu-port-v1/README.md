# Three-way comparison on the NPU port (D / A / T, Fold B selection, Fold A confirmation): registration texts (DRAFT / HELD)

**Status: DRAFT TEXTS. Nothing here is a registration or grants any authority.**
- No plan, ledger row, store release or decision receipt exists for these cells.
- No contact, flip, readiness change or exposure entry is authorised.
- Every number cites its source; the owner countersigns before anything is written into `docs/exposure-ledger.md`.

Written by Opus 5.5 for task T4 of `docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md`
(PLAN), at study commit `c1b0be33` on branch `npu-vllm-port`. The brief named `a3bd6307`, which is not a valid object
here. Development numbers only; never a test claim.

## Files

| file | content |
|---|---|
| `direct-only-cell-draft-v1.md` | D cell `SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1`: derivation from 266-E, port transform, identities, caps, read rule, 8 DRAFT ledger rows |
| `tfrl-v2-cell-draft-v1.md` | T cell `SAEA-SLURP-TFRL-V2-FOLD-B-NPU-PORT-V1`: the construction column (re-flight, not reuse), arms, artefacts and pins, hash movement, acceptance and kill rules, caps, iteration-1 DRAFT rows |
| `fold-a-confirmation-fragment-draft-v1.md` | Fold A dataset fragment (reused unchanged, 1,080 / 1,080 audio pins verified), the port preparations to regenerate, the three confirmation cells, DRAFT rows |

## 1. Registration order relative to the L338 port cell (A)

A is `SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1`, registered 2026-10-28 and unflown
(`docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/`). Steps 0 and 1 are card-free.

0. **Before anything else, fix A's FIT cap.** A's `native_contact_fit.contact_cap` is 64,800
   (`configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json`:134). The smoke's is 240.
   For a raw-arm cell the gate requires rows x max_contacts x 1, i.e. 16,200 and 60
   (`src/speech_aware_evidence_acquisition/core/preparation_fit.py`:1130-1137). Otherwise the FIT step raises before the
   first generation contact. The fix is a re-derive plus a writer re-run, with no other moved leaf. Owner item O3.
1. **Card-free preparations now (CLAUDE.md first principle 3).** Register and run:
   - Fold A FA1-FA4 (`fold-a-...`, section 3): one CPU-encoder row of 1,080 contacts, then model-free rows;
   - R01C P1-P5 (`tfrl-v2-...`, section 2): one CPU-encoder row of 625 contacts, then model-free rows.

   These need neither the deployment profile nor a card.
2. **D on Fold B, in A's campaign.** D's smoke and its three passes are registered together with A's re-derive. A's flight
   is the registered S0/S1 flight; D is 1,080 cheap contacts (PLAN section 5). Both wait for the speed sweep
   (`PORT_DEPLOYMENT_PROFILE`), for M3 (R1), and D also for its FIT path (D draft, E-D1).
   - Recommendation: write D's plans after the R1 decision, so a budget move does not force a second registration.
3. **T, iteration 1.** Register in this order:
   - rule-bank-i0 release;
   - construction smoke and construction i1 (after T1, T2, T3 and P1-P6);
   - after the construction i1 read: credit i1, reflector i1, bank release i1;
   - Fold B smoke and Fold B i1.

   Each row is registered before its own contact.
4. **T, iterations 2-3** are conditional on the stop rules (T draft, section 7). Their rows are written only after the
   previous read, with the R01C-H1 / R01C-H2 halves.
5. **Fold A.**
   - D-FA and the fallback A-FA can be registered with the Fold B cells.
   - T-FA (which also carries A's Fold A arms) is registered after the accepted iteration exists, before its contact.

## 2. Which existing writer can emit these, and with what extension

`scripts/flights/nx/l338_port_write_registration.py` is bound at import time to:
- `l338_port_family` (`pf`), `l338_family` (`fam`) and `l338_port_derive_cells` (`pdc`);
- the historical writer `l338_write_registration` (arm table, word rules, measurement blocks) (:36-39);
- a Fold B-only `PREPARATIONS` table (:56-81).

**T cells (construction, Fold B, Fold A): this writer, extended.**
1. A family switch (`--family l338-port | tfrl-v2-port`). Factor `pf` into a family interface: `CELLS`, `ALL_KEYS`,
   `PACKAGE`, `PREREG`, `DECISION_ID`, `generation_calls`, `fit_calls`, `audio_seconds`, `CARD_HOUR_FORMULA`,
   `ledger_gpu_cell`, `model_text`, `DECLARED_DIFFERENCES`, `eligible_run_ids`, `signoff_conditions`, and the READ ids.
   Add a new `scripts/flights/nx/tfrl_v2_port_family.py`.
2. An arm table for `t-v2 / aa / ab / lock`. `l338_family.validate_arm_table` accepts only the `perception_samples` and
   `solver_samples` axes (L3F:100, 143-165). It needs the t-v2 arm (store swap, rule, vote commit) and, with T2, a second twin.
3. The FIT cap computed by the gate's own rule (raw: rows x contacts x 1), for every family.
4. `PREPARATIONS` keyed by roster (Fold B, R01C and its halves, Fold A pins).
5. Model-free row writers for the new preparation kinds (credit, halves, bank release).
6. Per-iteration package subdirectories, as `tfrl_family.iteration_package_dir` (`scripts/flights/tfrl/tfrl_family.py`:334-335).

**D cells: not this writer as it stands.** It has no non-module program, no arm-free cell and no live/raw direct-only FIT
block. Recommended base: `scripts/flights/direct/direct_fb_write_registration.py` with `direct_fb_family.py` (the 266-E
writer: smoke, arms and in-chain replicates, `REPLICATE_OF`, the same package layout). Add:
- a new `scripts/flights/direct/direct_port_family.py` that imports the port layer from `l338_port_family`
  (`MODEL_FRAGMENT`, `PORT_DEPLOYMENT_PROFILE`, `DEPLOYMENT_REF`, the card-hour formula form, the D subset of
  `DECLARED_DIFFERENCES`, a direct-only `model_text`);
- a port derive in the manner of `l338_port_derive_cells.py`, applying the D draft's section-2 transform and refusing any
  other moved leaf.

Cleaner still: lift the port layer into one shared module used by both writers, so no port fact is spelled twice. All
writer runs follow the same rules:
- `--study-commit <sha> --data-root <root>`;
- no `--append-ledger` before the countersignature;
- every output says DRAFT / HELD.

## 3. Owner countersign items

| id | item | recommendation |
|---|---|---|
| O1 | Register the D, T and Fold A families in the section-1 order. Nothing flies before the sweep fills `PORT_DEPLOYMENT_PROFILE` and the PROVISIONAL card-hour formulas become numbers | yes |
| O2 | D on the port fragment, budget 3,072 with forced close and one answer-only fallback, instead of 266-E's uncapped 8,192 (declared difference D1); re-point with A and T if R1 (6,144) is adopted | yes; register after R1 |
| O3 | FIT caps by the gate's raw rule (rows x contacts x 1) for all port cells. This includes correcting A's registered 64,800 to 16,200 (smoke 240 to 60) before its flip | yes |
| O4 | T construction flight c7 (lock, twin aa, K = 4 trajectories at 0.7), re-flown each iteration. The twin is an addition to PLAN section 4; without it, c6 | yes, c7 |
| O5 | The hand-written seed rule bank (the only hand-authored TFRL-v2 input), and both banks as knowledge stores with release registrations: construction scope R01C, forbidden rosters Fold A, Fold B and DEV-1000. Every iteration moves the T baseline hash | yes |
| O6 | CPU-encoder exposure rows for Fold A (1,080 contacts, 3,085 s) and R01C (625, 2,727 s). No Fold A smoke | yes |
| O7 | How often Fold B and Fold A are read in the T loop. PLAN section 4 reads Fold B up to three times and confirms each accepted iteration on Fold A. Recommendation: construction gain as the per-iteration gate, one Fold A confirmation of the final state (T draft, H3) | owner decides |
| O8 | D id `SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1` (brief) rather than the port id rule's `...-FOLD-B-THINKING-NPU-PORT-V1` | brief's id |
| O9 | M1 for A. A carries only the one twin `aa` (Decision 4 revised). A second replicate `ab` (T2) would make A c17: 36,720 calls, 107,264 s, FIT 18,360 (same formulas) and a re-registration. Otherwise A's replicate variance comes from one twin | owner decides |
| O10 | The window conflict (T draft, H1). Under R1 the Solver prompt limit falls to 14,334 tokens against a measured max of 13,465, leaving about 400-870 tokens for bank and rules. Options: raise the per-sequence window (protocol fact; lower knee), reduce k = 20 (changes A), or keep the bank tiny | Fable analysis first, then owner |
| O11 | The cross-fit halves of R01C at iterations 2-3, forced by the knowledge-source guard (T draft, section 5.3) | yes |
| O12 | Reflector contacts on the frozen core: at most 18 text contacts per iteration (the ACE precedent) | yes |

PLAN section 7 items R1-R4 are prerequisites of this order.

## 4. Findings recorded while drafting (for the planner)

- **The FIT cap defect in the registered L338 port cell** (section 1, step 0). The port package carried the
  feature-packet 4 endpoints per contact (`tfrl_family.FIT_ENDPOINTS_PER_CONTACT`), but a raw arm uses 1.
- **No FIT path for a raw-arm direct-only cell.** The live phase fit refuses raw (`preparation_fit.py`:713-716), and the
  native fit refuses a non-native program (:967-970).
- **The two model fragment hashes `a35b2740...` and `bcb98207...` are the same `fragment_hash` with and without a resolved
  data root.** `model_materialization_root` is the literal data-root path, which `collapse_data_root` replaces.
  - Flights hash `a35b2740...`.
  - The base-lock record's L9 row and its D2 correction (`docs/readiness/2026-10-29-engineering-base-lock-npu-port-v1.md`:53,
    109) each give one value without the reason.
  - An English successor line is recommended; the sealed text is not edited.
- **The construction column must be re-flown; its inputs are reusable.** The historical construction column
  (`SAEA-SLURP-STAGE3-TFRL-CONSTRUCTION-LOCK-R01C-V1`) is Instruct Q4_K_M lock v2.2, and the TFRL-v2 design needs
  trajectories, so it is re-flown. Its dataset fragment, roster and audio pins (625 / 625 verified) are reused.
- **A lock-only sampling column (lock + twins) is illegal.** At least one candidate arm is required
  (`rails/supply_variants.py`:2457), so A's Fold A pass rides in the T Fold A flight or flies as separate c2 runs.
