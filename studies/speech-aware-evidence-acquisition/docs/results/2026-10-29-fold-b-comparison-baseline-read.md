# Comparison baseline for the L338 Fold B NPU port (read-only extraction, 2026-10-29)

Analyst brief (Opus 5.5). Model-free: no host, card, scorer or commit. Paths are relative to the study tree
`/mnt/cephfs/xingchao/omni/work/dev/umbrella/studies/speech-aware-evidence-acquisition` (branch `npu-vllm-port`).
Abbreviations: PKG = `docs/readiness/2026-10-19-slurp-l338-lever-column-fold-b-v1/`, PORT = `docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/`,
RES = `docs/results/`, LEDGER = `docs/exposure-ledger.md`. Every RES note cited here is byte-identical (sha256) to its copy in the upstream clone.

**Headline: L338 was never flown, so the cell has no result of its own.** The flight, FIT and READ rows at LEDGER:9144-9146 are consumed = `no`. The upstream
registry ends with NX-T R4 in flight and the L338 hook armed behind it (upstream `docs/readiness/2026-09-16-slurp-next-stage-owner-decisions-v1.md`:3777-3780).
No R4 read exists either. L338's delivering arm `lock` is "the 268-B lock form; its requests are the lock's byte for byte"
(PKG `registered-facts-v1.json` `accounting.cost_estimate.arm_table[3]`). **The comparison baseline is therefore the 268-B Thinking lock form,
read 8 times on this roster** (section 3).

## 1. Dataset and split
- Roster `SLURP-EVO-FOLD-B-V2` (`docs/receipts/rosters/slurp-evo-fold-b-v2.json`, sha256 `47b89b9e…a944a8`): 1,080 SLURP **TRAIN**
  utterances, discovery role, split hash `4407794c…3484a`, seed 20260906, 18 scenarios x 60 (`per_stratum`). One recording per utterance:
  527 headset / 553 far-field, 3,154.8 s of audio. Fold A, the exemplars and 739 previously touched utterances are excluded (`selection_rule`).
- Dataset fragment `configs/dataset/slurp-speech-native-evo-fold-b-1080.json` (`count` 1080; each `items[]` entry has `sample_id`, a 16 kHz `slurp_real/*.flac` path and `audio_sha256`).
  It carries no gold labels (`status` "LABEL-FREE").
- **Row** = one utterance recording. **Task** = SLURP SLU: output `{"scenario","action","entities":[{type,filler}]}` from the speech (locked Solver template
  `configs/probes/slurp_native_role_solver_v2.txt`, sha256 `18bff5c5…`, lines 1-4). Evidence acquisition = retrieval: Perception proposes 1-3 intent candidates
  from an 8-entry retrieved boundary block, then the Solver chooses the intent and extracts the entities (`RES/2026-10-19-slurp-v3-gap-to-finetuned-sota-decomposition.md`:36).
- **Contact** = one request to the served Thinking model at one node (Perception or Solver). It carries the speech, a reasoning budget of 3,072, a cap of 4,096
  and at most one answer-only fallback (PKG README:26-27; `fit-plan-registered-v1.json` `thinking_knobs`).
- **`phase-native-module-v1-c15`** = 15 contacts per row = 7 Perception + 8 Solver: lock 1P+1S, aa 1P+1S, l1-support 1P+5S, l2-union 4P+1S (PKG README:21-22, 55).
  Caps (LEDGER:9144-9145): 32,400 calls (= 1,080 x 15 x 2 for the fallback), 64,800 FIT, 94,644 s of audio.

## 2. Metrics (PKG `analysis-recipe-v1.json`)
- **Primary**: official SLURP **SLU-F1** (pooled micro, scorer `slurp-slu-official-train` = the pinned official `evaluate.py`, natural mixture), with **intent
  accuracy** (`scenario_action` exact) beside it (PKG `registered-facts` `stage3_measurement…V1.primary_metric`). Contrast: arm minus `lock`, paired per row,
  utterance-cluster bootstrap CI95, 10,000 draws, seed 20261018 (recipe:50-56). L1 is judged on SLU with an intent guard; L2 on intent with an SLU guard (recipe:57-60, 151-166).
  Words: NOMINATED / HARM / NEUTRAL (PKG README:94-105). The twin `aa` gives the in-flight A/A offset and enters no word.
- **Secondaries** (recipe:111-126): L1 pure-filter delta and the s3 / filter / gate / N1 forms; false fills; L2 merged recall; conversion by gold position;
  forced-close share by list size; support and union tables; tokens; s per contact; empty-gold false fill; per-scenario; stream discordance vs the twin.
- **Effect floor** (CI95 half-width of a paired same-flight contrast): `same-flight-node-column` → **SLU 1.49 pp / intent 0.65 pp at n = 1,080**, scaled by
  sqrt(1080/rows) (`scripts/checks/stage3_effect_floor_check.py`:119-137, 371-382; owner decision 190; `tests/unit/test_stage3_effect_floor_node_column.py`:32-35).
  The cell registers `expected_effect_pp` 0.0, `claims_benefit` false and the claim ceiling "nominated".
- **Cross-flight floor** (the one that applies to port vs baseline): `cross-flight-with-in-chain-replicate` = max(sampling 1.2485, **bar 2.5 pp**) at n = 1,080
  (floor check:141-146, 450-454). A cross-flight read without a replicate is refused (rule 4).
- **Fit plan** (PKG `fit-plan-registered-v1.json`) is not a statistical fit. It is the live phase-fit identity: "request envelopes only, no generation" (`purpose`).
  It proves that every contact fits the 24,576-token slot: prompt limit 17,406; estimate at 55 items 15,248; observed Solver prompt max 13,918 (`context_fit`).
  Seeds: 20260913 at T 0 (lock); sampled contacts use lock seed + offset at T 0.6 / p 0.95 / k 20 (`thinking_knobs`, `sampling_arms`).
- **Lock row** = one delivering-lock row (1P+1S) = 123.07 s at 4 slots → 1.95 rows/min; a lock-only flight = 9.23 GPU-h (NX-T R2b inputs).
  L338 = 7.52 lock-row equivalents per row ((7 x 59.00 + 8 x 64.07) / 123.07). 1,080 x 7.52 = 8,121.6 lock rows / 1.95 per min = 4,165 min = **69.42 GPU-h expected,
  ceiling 90.3** = ceil_0.1(x 1.3). Smoke ceiling 0.5 GPU-h (PKG README:57-64; `registered-facts` `accounting.cost_estimate`).

## 3. Registered baseline numbers (Fold B, n = 1,080, Q4_K_M llama.cpp Thinking; development numbers, claims_benefit false)
| lock-form reading | SLU-F1 % | intent % | source |
| --- | ---: | ---: | --- |
| 268-B lock | **71.61** | **85.00** | RES/2026-10-13-slurp-268b-thinking-system-fold-b-read.md:30-33 (sha256 `1d471b52…`) |
| 268-B twin aa | 72.75 | 84.91 | same; offset +1.15 [-0.09, +2.42] SLU, -0.09 intent |
| NX-T R1 both-on / twin | 72.35 / 72.23 | 85.74 / – | RES/2026-10-17-slurp-nxt-r1-thinking-policy-column-read.md:22; R2b read:157 |
| NX-T R2b budget-3072 | 72.74 | 85.37 | RES/2026-10-18-slurp-nxt-r2b-solver-budget-column-read.md:25 |
| NX-T R2 budget-3072 | 72.21 | 84.91 | RES/2026-10-24-slurp-nxt-r2-thinking-budget-column-read.md:9 |
| NX-T R3 lock / twin | 72.04 / 72.49 | 85.28 / 85.19 | RES/2026-10-25-slurp-nxt-r3-solver-template-column-read.md:36-37 |
- Span across readings: SLU 71.61-72.75 (1.14 pp), intent 84.91-85.74. The recipe pins `external_references.thinking_system_slu` = **71.61** (recipe:26).
  The historical `analysis/…/read.json` files are not on our data root, so these numbers come from the committed notes only.
- Other populations: DEV-2000-ADDED lock 74.69 / 84.50; v3.0 on a 3,010-row TEST sample 73.05 / 86.33. Full-test tiers: LauraGPT 73.45 / 87.87,
  SFT Qwen2-Audio-7B 76.75 / 88.13, UniverSLU-14 79.5 / 90.3 (gap note:15-28).
- L338's pre-registered offline expectations: L1 +1.09 [+0.53, +1.63] SLU; L2 recall 88.15 → 89.81 %, intent 0 to +1.0 (PKG README:21-22, 68).

## 4. Lineage on Fold B (delivering level SLU / intent; what each lever bought)
- Instruct core (Q4_K_M): P-SV-2 61.63 / 82.96 (psv2 read:96) → P-SV-4 62.54 / 82.78, K6p NOT NOMINATED (psv4 read:173) → NX R1 62.13 / 82.22, Solver
  templates NOT NOMINATED (nx-r1 read:67) → NX R6, lock v2.2 system of record, 64.34 / 82.41, Perception templates NOT NOMINATED (r6 read:99).
- TFRL loop (263-A): locks 64.00 / 82.31, 64.51 / 82.96, 64.45 / 82.69. ACE was refused 3x (+0.87, +1.38, +0.27, all under 1.49): no admitted state (tfrl-three read:1, 36-42).
- Core swap: bare Thinking (266-E) 45.39 / 76.76 (+11.90 over bare Instruct). **268-B: the same system on Thinking, +7.15 [+5.08, +9.26] SLU, +2.31 [+0.56, +4.07]
  intent (64.45 → 71.61)**, TRANSFERS-ABOVE-BAR (268b read:1, 30-33). P3.1 replicated it on DEV-2000 (+7.52 / +2.50). This is the only resolved gain.
- After 268-B nothing moved: R1 Solver no-think -7.11; R2b 6,144 budget -0.81; R2 -1.49 / -0.94 UNRESOLVED; R3 templates +0.21..+1.19 NEUTRAL; R4 flying; L338 never flown.
- "Improvement" in this repo is a same-flight paired delta whose CI95 lower bound is >= 1.49 SLU / 0.65 intent. Cross-flight it is >= 2.5 pp with a twin.

## 5. What the port changes (PKG vs PORT `registered-facts-v1.json`, diff of flattened keys)
- Declared differences 8.1-8.7 (PORT README:36-44): BF16 weights and KV (not Q4_K_M / q8_0); the server audio tower on raw audio (not mmproj packets); vLLM kernels;
  a forced-causal encoder with all vectors regenerated; vLLM forced-close whitespace; re-authored knowledge stores; a CPU fp32 encoder.
- Model fragment `q4km-thinking-mmproj-free-…-np4-ctx24k-…-v1` → `bf16-thinking-vllm-…-seq4-len24k-…-port-v1` (fragment hash df95b65b → **a35b2740**
  after db798a83, `tests/unit/test_l338_port_package_staleness.py`:3). Changed: `transport` llama-server→vllm, `model_lock_key`, `prompt_order`
  (`speech-query-prefix-v2`→`native-raw-speech-query-prefix-v1`), `server_args`, `runtime_receipt`; removed: `profile_slot_count` and the speech feature root.
- `identity.arm.arm` `speech-feature-query-native-tools` → `raw-speech-agent-native-tools`; baseline fragment `…-baseline-v1` → `…-baseline-raw-port-v1`.
- Hashes: flight composed `6932eb00…` → `77185307…`, protocol `d9607915…` → `de9937af…`; smoke `2f8a24c3…` → `22f715bf…`, `9729cec8…` → `205ae3fc…` (LEDGER:9171).
- Supply pins, all regenerated: agreement `06990a86`→`db4614a3`; centroids `16c33640`→`f0528755`; boundary queries `83c5da61`→`fe23703a`; content
  queries `573a8a37`→`e1d3230e`; kus3 keys `5dfe7523`→`24f0cdf7`; boundary and slot stores re-released. The packet receipt `ef409ef4` is dropped; phase supply `1141f08c` is added.
- Session → `…-npu-port-pending-deployment-profile-v1`; `think-budget-np4` dropped; 4 slots → `PORT_DEPLOYMENT_PROFILE` (pending); GPU-h → a provisional card-hour formula.
- Unchanged: roster, rows, c15, arms, sampling axes, budget and cap, seed, call caps, stage3_measurement values, floors.

## 6. Authority chain before a model call on a dataset row (PORT README:50-56; LEDGER:9171-9180)
1. Commit the package. 2. The speed sweep fills `PORT_DEPLOYMENT_PROFILE`. Re-derive the card-hour bound and re-run the writer; until then the gate refuses the rows (README:34).
3. **The owner countersigns** the 4 DRAFT rows (`countersign-row-texts-v1.json`); then mint `l338_port_mint_receipt.py`. 4. **Re-pin and readiness flip**
(`scripts/flights/repin_cell.py`): this mints phase-supply-v2 with output equality asserted (v1 is stale, README:25). 5. Close owner items 9.2-9.4;
register the port analysis recipe and READ row; write the chain script. **Smoke-4** (4 rows, 60 contacts, ≤ 120 calls) is a FORM proof that claims nothing:
c15 loads, the thinking fields are carried, no cap hits, the fit passes and the receipt seals. It measures the real arm mix, and the sign-off refuses above 1.3x (PKG README:77, 80).

## 7. Open gaps
- No L338 result and no R4 read exist. Port vs baseline is cross-flight across engines and no recipe is registered for it. PORT has no `analysis-recipe-v1.json` and no READ row.
- The multi-endpoint flight transport is not built (`docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md`:538; plan `…/2026-10-29-lock-engineering-base-and-efficiency-plan.md`:216, 583, 740, 921).
- Phase supply v1 (`1141f08c`) is stale until the flip. The port encoder rows (LEDGER:9152-9154) still say consumed = `no`, yet their outputs exist and match the pins (`fe23703a`, `24f0cdf7`).
- `vectors/slurp-kus3/facet-vectors.jsonl` was regenerated at the historical relpath with new bytes, so the historical pin `5dfe7523` fails on our data root.
  The historical sealed vectors and read.json files are absent.
- LEDGER:9136 says "root countersignature pending", but rows 9142-9145 say "root countersigned 2026-10-24T16:28:14Z" (the READ row is DRAFT).
- Greedy decoding is not bit-reproducible across batch shapes (PKG `protocol-registered-v1.json`:98). Judge the port against the 1.14 pp span and its own twin, not against one number.
