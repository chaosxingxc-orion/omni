# Fold A confirmation on the port: the dataset fragment, the port preparations and the three confirmation cells (DRAFT TEXT)

**Status: DRAFT TEXT ONLY. This is not a registration and it grants no authority.** No encoder contact, model contact,
plan, receipt, ledger row or flip is authorised. Nothing is countersigned. Written by Opus 5.5 for task T4 at study commit
`c1b0be33`. Development numbers only; `claims_benefit` false. Abbreviations as in the other two drafts.

## 1. Role (PLAN section 2, M2; section 5)

Selection is done on Fold B and confirmation on Fold A. A lever is accepted only if its Fold A widened CI95 lower bound is
above zero. The optimism term is selection minus confirmation. Arms D, A and T fly once each on Fold A.

Fold A is a held-out **discovery** fold, not a test split. It is rollout-exposed by prior SLURP cells, for example:
- VER-1c (LEDGER:8837-8841);
- ROLL-2 and the VER-1 confirmation (`docs/superpowers/specs/2026-10-19-slurp-338-lever-column-design.md`:42-45).

So its confirmation is a second-fold check, never a claim.

## 2. The Fold A dataset fragment: reused unchanged

**`configs/dataset/slurp-speech-native-evo-fold-a-1080.json`**, the historical Fold A 1,080 fragment. It is model-free and
label-free ("MODEL-FREE, LABEL-FREE ENGINEERING INPUT"). The port's Fold B cell reuses its historical twin
`slurp-speech-native-evo-fold-b-1080` byte-for-byte (PEXP:13), and Fold A is reused the same way. No port copy is made.

| fact | value | checked |
|---|---|---|
| file sha256 | `4ba55688e80d8d768ec6cfc3ff4bacc9802fb34e4dab54913934015184497215` | recomputed today |
| split | `SLURP-EVO-FOLD-A-V2`, role discovery, identity `61b0f06254975873bea6afb287ab54823dc71b57e5398697b9bef817e25398b6` | key `split_identity_hash` |
| rows, audio | 1,080 items, `audio_seconds_total` 3,084.5284375 (re-summed from items: equal) | keys `count`, `audio_seconds_total` |
| roster | `docs/receipts/rosters/slurp-evo-fold-a-v2.json`, sha256 `1fe7908c31396774d7535fdf37bd791762215c324999d66ed7beb9e7b2f9033d` | recomputed today |
| d1 supplement | `receipts/artifacts/d1-slurp-evo-fold-a-v2.json`, sha256 `2b3d2299...a8397`; the committed copy `docs/receipts/artifacts/d1-slurp-evo-fold-a-v2.json` matches | recomputed today |
| **audio pins** | every item's `audio_sha256` against `${SAEA_DATA_ROOT}/<media_relpath>` (`datasets/slurp/slurp_real/*.flac`): **1,080 / 1,080 match** | re-hashed today on `/mnt/cephfs/xingchao/omni` (CPU, model-free) |
| store disjointness | Fold A against the port intent-support roster (2,037) and the slot-support roster (3,162): 0 shared. Fold A is a forbidden roster of both port semantic stores (`docs/readiness/2026-10-28-slurp-intent-boundaries-preparation-registration-npu-port-v1.json` and `...slot-contracts-...-npu-port-v2.json`, `forbidden_rosters[0]`). The fold-excluded TRAIN views carry 0 Fold A ids (DFF:89-93) | checked today |

## 3. What must be regenerated on the port side for Fold A

These are the Fold A twins of the six port preparations the Fold B cell pins (PORTPKG `draft-ledger-rows.md` rows 1-6;
`scripts/flights/nx/l338_port_write_registration.py`:56-81). D needs none of them. A and T need all of them.

| id | run | what | historical twin | contact |
|---|---|---|---|---|
| FA1 | `SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-NPU-PORT-V1` | Fold A speech-query vectors: Omni-Embed-Nemotron-3B, CPU float32 pool (4 x 16 threads, cores 128-191), forced causal. Protocol fragment to write: `configs/preparation/slurp-tfrl-fold-a-speech-query-encoding-protocol-npu-port-v1.json`, the Fold A twin of `slurp-tfrl-fold-b-speech-query-encoding-protocol-npu-port-v1.json` | `SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-V1` (`docs/readiness/2026-09-12-slurp-training-query-embedding-registration-v1.json`, plan `fold-a`: 1,080 calls, 3,085 s) | 1,080 encoder contacts, 3,085 s, 2.0 h wall ceiling, 0 NPU |
| FA2 | `SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-NPU-PORT-V1` | alias into the intent-boundary dimension, giving `vectors/slurp-intent-boundaries-train-fold-a-1080-npu-port-v1/query-vectors-speech-segments.jsonl` | `SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-V1` | model-free |
| FA3 | `SAEA-SLURP-FOLD-A-INPUTS-PREPARE-NPU-PORT-V1` | content derivative, giving `vectors/slurp-stage3-filler-fold-a-npu-port-v1/query-vectors-speech-content-v1.jsonl` | the Fold B port row 2 | model-free |
| FA4 | `SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-A-NPU-PORT-V1` | self-excluded agreement table for Fold A. Its source identity is a never-flown, agreement-free Fold A comparator cell `slurp-native-stage3-l338-lever-column-fold-a-comparator-npu-port-v1` (config only; the twin of `configs/experiment/slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1.json`) | `SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-A-V1` | model-free |
| FA5 | `<run>-PHASE-SUPPLY-V1`, one per Fold A module run | first-phase Perception supply under the run's own id (`core.flight.cli prepare-supply`) | the Fold B port row 6 | model-free |
| FA6 | pins file | the Fold A port pins (FA1-FA4 artefact and receipt sha256), the twin of `scripts/flights/nx/l338_port_pins.json`, read by the derive | - | file |

Shared and reused unchanged:
- the re-authored stores `intent-boundaries-npu-port-v1` and `slot-contracts-npu-port-v2`;
- the train-exemplars pool;
- the port kus3 key vectors (`SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-NPU-PORT-V1`);
- the centroid keys (`SAEA-SLURP-INTENT-BOUNDARY-CENTROID-KEYS-NPU-PORT-V1`);
- the minus-r01 store;
- the TFRL-v2 banks of the accepted iteration (their forbidden rosters include Fold A).

Order: FA1 through FA4 are card-free (CPU encoder plus arithmetic). CLAUDE.md first principle 3 puts them first, so they
should run while the Fold B work proceeds and not after it.

## 4. The three confirmation cells (one pass each)

| arm | run(s) | form | contacts / row | generation calls | speech-seconds | FIT (code rule, raw) | FIT (port formula) | card-hours (PROVISIONAL; at r = 1.95) |
|---|---|---|---|---|---|---|---|---|
| D | `SAEA-SLURP-DIRECT-ONLY-FOLD-A-NPU-PORT-V1`, `...-FOLD-A-AA-NPU-PORT-V1`, `...-FOLD-A-AB-NPU-PORT-V1` (+ `-FIT-`) | the D cell of `direct-only-cell-draft-v1.md` on Fold A: 3 request-identical passes. The historical Fold A direct form is `configs/baseline/slurp-direct-max512-train-fold-a-v1.json`, the 266-E baseline's parent (DFF `PARENT_BASELINE`) | 1 per pass (3 total) | 2,160 per pass; 6,480 total | 6,170 per pass; 18,510 total | 1,080 per pass | 4,320 per pass | ceil_0.1(1.3 x 1,080 x 0.52 x T / r / 60) per pass; 6.3 x T |
| A | preferred: the `lock`, `aa`, `ab` arms of the T Fold A flight below (no extra contact). Fallback when T has no accepted state: `SAEA-SLURP-STAGE3-LOCK-FOLD-A-NPU-PORT-V1`, `...-LOCK-FOLD-A-AA-NPU-PORT-V1`, `...-LOCK-FOLD-A-AB-NPU-PORT-V1`, three c2 runs | A's lock harness on Fold A, with lock, twin and second replicate (M1) | fallback: 2 per run (6 total) | fallback: 4,320 per run; 12,960 total | fallback: 12,339 per run; 37,017 total | fallback: 2,160 per run | 8,640 per run | fallback: ceil_0.1(1.3 x 1,080 x 1.00 x T / r / 60) per run; 12.0 x T |
| T | `SAEA-SLURP-TFRL-V2-FOLD-A-NPU-PORT-V1` (+ `-FIT-`) | the accepted iteration's Fold B baseline (both banks, the vote harness) re-pointed to Fold A pins: t-v2, aa, ab, lock; program c10 | 10 | 21,600 | 61,691 | 10,800 | 43,200 | ceil_0.1(1.3 x 1,080 x 5.04 x T / r / 60); 60.5 x T |

Why A is not a lock-only sampling column: a `sampling_variants` block needs at least one candidate arm
(`MIN_SAMPLING_CANDIDATE_ARMS` 1, `rails/supply_variants.py`:2457). So {lock, aa, ab} alone is refused. The 268-B
precedent flew lock and aa as separate cells (`configs/experiment/slurp-native-stage3-thinking-system-fold-b-lock-v1.json`
and `...-aa-v1.json`). The fallback A runs follow that precedent. Separate runs measure cross-run rather than within-run
noise; the reading says so.

Caps use the port package's formulas (L3F:576-585; FIT rule as in the D draft, section 6). Fold A roster audio is
3,084.5284375 s. Totals with the preferred A: D 6,480 + T 21,600 = 28,080 calls; speech 18,510 + 61,691 = 80,201 s.

PLAN section 5 estimates "about 80" card-hours at the knee for the three, so the registered bound comes from the sweep.

**No Fold A smoke.** The Fold A runs reuse the programs and harness bytes that were smoked and signed off on Fold B. The
Fold A inputs are verified model-free (FA1-FA6). This is owner item O6.

Registration timing:
- D-FA and the fallback A-FA can be registered with the Fold B cells.
- T-FA pins the accepted banks, so its baseline hash exists only after acceptance. It is registered after the accepted
  iteration and before its own contact. Its rows cannot be pre-written with final pins.

## 5. Read rule (draft)

- **Confirmation.** In the T Fold A flight, the widened CI95 lower bound of SLU(t-v2) - SLU(lock) must be > 0 (M2). The
  widened CI is the row-paired bootstrap plus the replicate variance from lock / aa / ab.
- **Optimism.** Fold B delta minus Fold A delta, reported for every accepted state.
- **Three-way report on Fold A.** It leads with D, A and T on SLU-F1 and intent with CI95, then the A-to-D gap (cross-run,
  2.5 pp bar) and the T-to-A gap (within-run), then cost (PLAN section 5, reporting rule).
- Twin admissibility as on Fold B: |aa - lock| <= 1.20 pp.

## 6. Exposure-ledger DRAFT row texts

## Fold A confirmation on the NPU port: Fold A preparations and the D / A / T confirmation passes (DRAFT, owner countersignature pending)

DRAFT rows; drafted under owner decision 2026-10-28 items 4 and 6. Package docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/fold-a. No row is countersigned and no row authorises a contact; model-flight gpu-hours cells are PROVISIONAL formulas.

| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-30 | SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-NPU-PORT-V1 | bounded-discovery-probe | Port twin of SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-V1: complete-speech retrieval-query encoding for TRAIN fold A; no answers or scoring | Frozen local Omni-Embed-Nemotron-3B retrieval tool on the CPU float32 worker pool; no answering core | slurp SLURP-EVO-FOLD-A-V2 TRAIN discovery | discovery | 61b0f06254975873bea6afb287ab54823dc71b57e5398697b9bef817e25398b6 | no | 1080 | 3085 | 2.0 | The historical Fold A roster, packet and query-encoding exposures carried forward; this row adds only the port's pinned retrieval-encoder contacts | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/fold-a/fold-a-speech-query-encoding-registration-npu-port-v1.json | DRAFT TEXT. CPU float32 retrieval encoder (declared difference 8.7, design section 6.4): pinned pool of 4 workers x 16 threads on cores 128-191 of the Ascend host, batch-one forwards; planned_gpu_hours is the attempt's wall-clock ceiling and the actual GPU/NPU hours are 0. Attention geometry FORCED CAUSAL on every text layer and measured on the loaded weights at worker start (declared difference 8.4). One encode-v1 attempt; no retry, fallback, scorer, answer read, or cross-fold feedback. Owner countersignature pending; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-NPU-PORT-V1 | model-free-check | Port twin of SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-V1: carry the port's sealed TRAIN Fold A content-dimension speech-query segment vectors into the intent-boundary retrieval dimension, values unchanged, segment identities preserved. Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts. | no model or tool contact; local arithmetic over receipt-pinned files | slurp TRAIN Fold A SLURP-EVO-FOLD-A-V2 1080 | discovery | 61b0f06254975873bea6afb287ab54823dc71b57e5398697b9bef817e25398b6 | no | 0 | 0 | 0 | No new speech exposure: every vector read was sealed by SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-NPU-PORT-V1. Prior exposure inherited in full. | prepared/SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-NPU-PORT-V1/query-dimension-alias.json (sha256 <filled after the run>) | DRAFT TEXT; to be RUN BEFORE THE ROW IS WRITTEN; output vectors/slurp-intent-boundaries-train-fold-a-1080-npu-port-v1/query-vectors-speech-segments.jsonl. It constructs no ExecutionPlan, reserves no contact, releases no store beyond the one named, re-pins no committed config and authorises nothing. |
| 2026-10-30 | SAEA-SLURP-FOLD-A-INPUTS-PREPARE-NPU-PORT-V1 | model-free-check | The port's model-free CONTENT DERIVATIVE of the Fold A speech-query vectors (scripts/build_slurp_query_content_derivative.py), the exemplar channel's query source. Zero model contacts. | no model or tool contact; local arithmetic over receipt-pinned files | slurp TRAIN Fold A SLURP-EVO-FOLD-A-V2 1080 | discovery | 61b0f06254975873bea6afb287ab54823dc71b57e5398697b9bef817e25398b6 | no | 0 | 0 | 0 | No new speech exposure (as the alias row). | vectors/slurp-stage3-filler-fold-a-npu-port-v1/query-vectors-speech-content-v1.provenance.json (sha256 <filled after the run>) | DRAFT TEXT; model-free; authorises nothing. |
| 2026-10-30 | SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-A-NPU-PORT-V1 | model-free-check | Port twin of SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-A-V1: the self-excluded exemplar-intent agreement table for TRAIN Fold A, arithmetic over the pinned exemplar snapshot, the port pool key vectors and the port Fold A speech query vectors, against the committed never-flown agreement-free port Fold A comparator cell slurp-native-stage3-l338-lever-column-fold-a-comparator-npu-port-v1. Zero model contacts. | no model or tool contact; local arithmetic over receipt-pinned files | slurp TRAIN Fold A SLURP-EVO-FOLD-A-V2 1080 | discovery | 61b0f06254975873bea6afb287ab54823dc71b57e5398697b9bef817e25398b6 | no | 0 | 0 | 0 | No new speech exposure. | prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-A-NPU-PORT-V1/agreement.jsonl (sha256 <filled after the run>) | DRAFT TEXT; model-free; authorises nothing. |
| 2026-10-30 | SAEA-SLURP-DIRECT-ONLY-FOLD-A-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- REGISTERED. Fold A confirmation pass of arm D: the D cell of docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only-cell-draft-v1.md (program phase-direct-only-v1, one contact per row, task instruction probes/slurp_direct_closed_list.txt, no knowledge supply) on TRAIN Fold A 1,080, served by the BF16 / vLLM-Ascend Thinking core over the original recordings (native-raw-speech-query-prefix-v1); pass lock (1 of 3: lock, aa, ab); declared differences 8.1, 8.2, 8.3, 8.5 and D1 (as the D cell rows); a capability read, no mechanism, no claim | (model/tool of the D cell rows) | slurp TRAIN Fold A 1,080 | discovery | 61b0f06254975873bea6afb287ab54823dc71b57e5398697b9bef817e25398b6 | no | 2160 | 6170 | PROVISIONAL: ceil_0.1(1.3 x 1080 x 0.52 x T / r / 60) NPU card-hours; T (cards per instance) and r (measured lock rows per minute per instance) from PORT_DEPLOYMENT_PROFILE (pending) | TRAIN Fold A discovery rows already exposed by prior SLURP cells (the VER-1 / VER-1c Fold A cells and the Fold A encoder pass included); this cell opens no split. Prior exposure is inherited in full. | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/fold-a/direct-only-lock-generation-plan-registered-v1.json | DRAFT at study commit <commit>; plan SHA256 <plan sha>; owner countersignature pending; PROVISIONAL gpu-hours; nothing authorised. Its FIT row SAEA-SLURP-DIRECT-ONLY-FOLD-A-FIT-NPU-PORT-V1 carries calls 1080 (raw rule, OPEN as D F1) and audio 6170; the aa and ab passes (SAEA-SLURP-DIRECT-ONLY-FOLD-A-AA-NPU-PORT-V1, ...-AB-...) and their FIT rows repeat these numbers. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-FOLD-A-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- REGISTERED AFTER THE ACCEPTED ITERATION, BEFORE CONTACT. Fold A confirmation of the accepted TFRL-v2 state: the accepted iteration's Fold B baseline (exemplar bank, rule bank, Solver vote n = 3, commit saea-commit-vote-v1) re-pointed to the Fold A port pins, with the twin aa, the second replicate ab and the delivering lock arm (program phase-native-module-v1-c10), TRAIN Fold A 1,080; its lock / aa / ab arms are also the A confirmation; confirmation rule: widened CI95 lower bound of t-v2 minus lock > 0; declared differences 8.1-8.7 | (model/tool of the T Fold B rows) | slurp TRAIN Fold A 1,080 | discovery | 61b0f06254975873bea6afb287ab54823dc71b57e5398697b9bef817e25398b6 | no | 21600 | 61691 | PROVISIONAL: ceil_0.1(1.3 x 1080 x 5.04 x T / r / 60) NPU card-hours; T and r from PORT_DEPLOYMENT_PROFILE (pending) | (as the D Fold A row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/fold-a/tfrl-v2-generation-plan-registered-v1.json | DRAFT; its FIT row SAEA-SLURP-TFRL-V2-FOLD-A-FIT-NPU-PORT-V1 carries calls 10800 (1,080 x 10 x 1) and audio 61691; owner countersignature pending; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-STAGE3-LOCK-FOLD-A-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- FALLBACK ONLY (flown only if T has no accepted state). A's lock harness (the plain review-off module graph on the port lock, program phase-native-module-v1-c2, 2 contacts per row) on TRAIN Fold A 1,080 with the Fold A port pins; pass lock of three (lock, aa, ab as separate request-identical runs, the 268-B lock / aa precedent); declared differences 8.1-8.7 | (model/tool of the L338 port rows with module program phase-native-module-v1-c2 and no sampling axis) | slurp TRAIN Fold A 1,080 | discovery | 61b0f06254975873bea6afb287ab54823dc71b57e5398697b9bef817e25398b6 | no | 4320 | 12339 | PROVISIONAL: ceil_0.1(1.3 x 1080 x 1.00 x T / r / 60) NPU card-hours; T and r from PORT_DEPLOYMENT_PROFILE (pending) | (as the D Fold A row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/fold-a/lock-generation-plan-registered-v1.json | DRAFT; its FIT row carries calls 2160 (1,080 x 2 x 1) and audio 12339; the aa / ab runs repeat these numbers; nothing authorised. |
