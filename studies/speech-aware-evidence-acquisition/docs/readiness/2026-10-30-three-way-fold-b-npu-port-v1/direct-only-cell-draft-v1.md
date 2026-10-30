# D cell `SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1`: direct-only on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B (DRAFT TEXT)

**Status: DRAFT TEXT ONLY. This is not a registration and it grants no authority.** No plan, receipt, ledger row or flip
exists. No contact is authorised, nothing is countersigned and nothing is ready. The writer run (task T4, SONNET side) turns
these texts into plan files and rows. The owner countersigns before any contact. Written by Opus 5.5 for task T4 of
`docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md` (PLAN below) at study commit
`c1b0be33`. The brief named HEAD `a3bd6307`, but that is not a valid object in this repository; `c1b0be33` contains the
PLAN commit `3600ad6b`. Development numbers only; never a test claim; `claims_benefit` false.

Abbreviations:
- PLAN = `docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md`
- READ = `docs/results/2026-10-30-tfrl-history-and-systematic-problems-read.md`
- PKG266 = `docs/readiness/2026-10-11-slurp-direct-only-fold-b-thinking-vs-instruct-v1/`
- PORTPKG = `docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/`
- LEDGER = `docs/exposure-ledger.md`
- DFF = `scripts/flights/direct/direct_fb_family.py`
- L3F = `scripts/flights/nx/l338_family.py`
- L3PF = `scripts/flights/nx/l338_port_family.py`
- PF = `src/speech_aware_evidence_acquisition/core/preparation_fit.py`

## 1. Derivation: the historical 266-E direct-only cell (Thinking arm)

| link | historical object | cite |
|---|---|---|
| campaign | `configs/campaign/slurp-direct-only-fold-b-thinking-v1.json` (one cell, one experiment) | file |
| experiment | `configs/experiment/slurp-direct-only-fold-b-thinking-v1.json`, run `SAEA-SLURP-DIRECT-ONLY-FOLD-B-THINKING-V1`, attempt `attempt-slurp-direct-only-fold-b-thinking-generation-rec1-v1` | keys `cells[0]` (:12-13), `execution_plan` (:43), `family_budget` (:53), `generation_budget` (:65) |
| baseline | `configs/baseline/slurp-direct-only-fold-b-thinking-baseline-v1.json`: `policy` `phase-direct-only-v1` (:21); `arm` `speech-feature-query` (:2); `evidence_source` none (:4); the fold-excluded knowledge view for the roster guard only (:5, sha `9c64e8ec...`, DFF:94-95); `live_phase_fit` cap 3,241 (:7); phase template `probes/slurp_phase_direct.txt` (:16); task template `probes/slurp_direct_closed_list.txt` (:33); packet receipt (:30); `whole_roster_wave_barrier` true (:36) | lines |
| parent of the baseline | `configs/baseline/slurp-direct-max512-train-fold-a-v1.json` (the sealed Fold A direct form, re-pointed at Fold B) | DFF `PARENT_BASELINE` |
| program | `configs/policy/phase-direct-only-v1.json`: one phase `direct`, `knowledge` none, `max_contacts` 1, `stop_rule` `always-run-full-program` | file |
| model fragment | `q4km-thinking-mmproj-free-slurp-direct-fold-b-np4-ctx24k-max8192-v1` (completion cap 8,192, no forced budget) | DFF:60-62 |
| package / prereg | PKG266 (README, `thinking-*-plan-registered-rec1-v1.json`, `analysis-recipe-v1.json`); `docs/readiness/2026-10-11-slurp-direct-only-fold-b-thinking-vs-instruct-v1-preregistration.md` (sections 1-6, recovery sections from :100) | files |
| ledger | rows `SAEA-SLURP-DIRECT-ONLY-FOLD-B-THINKING-V1` / `-FIT-REC1-V1`, CONSUMED: 1,080 contacts, 3,154.8 s | LEDGER:8937-8938 |
| result | 45.39 SLU-F1 / 76.76 intent; 36 of 1,080 rows hit the 8,192 cap (abstentions by truncation) | `docs/results/2026-10-12-slurp-266e-thinking-vs-instruct-bare-core-read.md`:32-33, 198-205 |
| in-chain replicate precedent | the Instruct arm flew twice in one chain as separate runs (`REPLICATE_OF`); the twin offset was +0.18 SLU | DFF:375; same read :32 |

## 2. What the port D cell is (the port transform)

"One Solver call per row, no knowledge, no program" (PLAN section 5) means the direct-only program: one contact, no
Perception node, no supply, no module graph. The port cell is the 266-E Thinking cell with exactly the following moves.

| field | historical (266-E Thinking) | port D cell | why |
|---|---|---|---|
| model fragment | `q4km-thinking-...-np4-ctx24k-max8192-v1` | `bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1` (the PORT cell's own, `configs/experiment/slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1.json`:14) | brief: the same model fragment as the port |
| arm | `speech-feature-query` (llama.cpp packets) | `raw-speech-agent-native-tools` (the original recordings, re-hashed per send) | vLLM has no packet input (design doc `docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md` section 5.2) |
| packet receipt | `speech_packet_promotion_receipt` (+ sha) | removed | no packets on the port |
| fit block | `live_phase_fit` (1 + 3 x rows) | OPEN, see F1 in section 6 | the live phase fit refuses a raw arm (PF:713-716); the native contact fit refuses a non-native program (PF:967-970) |
| decoding | greedy; completion cap 8,192; no budget | greedy, seed 20260913; max_tokens 4,096 = reasoning budget 3,072 + answer reserve 1,024, forced close, one answer-only fallback re-ask per row | port fragment; `docs/readiness/2026-10-29-engineering-base-lock-npu-port-v1.md` row L10; `core/reasoning_knobs.py`:49-50, 433-440 |
| program, templates, knowledge view, scorer | `phase-direct-only-v1`, `slurp_phase_direct.txt`, `slurp_direct_closed_list.txt`, `9c64e8ec...`, `slurp-slu-official-train` | unchanged | the bare-core form is the comparison object |
| geometry facts | `serving_slots` 4 / `serving_slot_tokens` 24,576 | unchanged (client-side geometry, owner ruling) | L3PF:67-69 |
| deployment | np4 llama.cpp session | `PORT_DEPLOYMENT_PROFILE` (pending; filled from the measured speed sweep); not a fact of this registration | L3PF:61-71 |

Declared differences that apply to D (design doc section 8, L3PF:247-255): 8.1 (BF16 weights and KV), 8.2 (server audio
tower over raw audio), 8.3 (vLLM sampling, deterministic per build, not bit-identical), 8.5 (forced-close whitespace).
8.4, 8.6 and 8.7 concern retrieval and knowledge, which D does not use. **D-specific difference D1:** the historical D had
an 8,192-token completion cap and no thinking budget. The port D runs under the port budget of 3,072 with forced close.
Historically, 1,044 delivering rows had p90 1,641 and p99 3,610 completion tokens, and 36 rows ran away to the cap
(266-E read :202-205). So some rows that delivered at 8,192 will now be force-closed. If owner item R1 (M3, budget 6,144;
PLAN section 7) is adopted, D re-points to the 6,144 fragment together with A and T. The calls, FIT and audio caps below
do not move with the budget. The card-hour bound does.

## 3. Identities (draft)

The brief fixes the flight id `SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1`. Under the port id rule (L3PF:94-103), the
historical id would map to `...-FOLD-B-THINKING-NPU-PORT-V1`. The brief's id drops `THINKING` because there is only one
core on the port. Owner item O8 in the README covers this.

The direct-only program has no arm table, and the rails accept exactly one in-flight twin, and only inside a module
sampling block (`rails/supply_variants.py`:2749-2754). So the M1 replicates (PLAN section 2, M1; T2) of D are separate
request-identical runs in the same chain, as in 266-E (DFF:375). Same-chain replicates also absorb session drift (READ
section E.3), so they measure at least the within-run decoding term.

| key | cell (config) | run | FIT run | generation attempt / FIT attempt | rows | roster |
|---|---|---|---|---|---|---|
| smoke | `slurp-direct-only-fold-b-smoke-4-npu-port-v1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-NPU-PORT-V1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-FIT-NPU-PORT-V1` | `attempt-slurp-direct-only-fold-b-smoke-4-generation-npu-port-v1` / `...-fit-npu-port-v1` | 4 | PKG266 `smoke-roster-receipt-v1.json` (sha `856933c2...`, split `26b306e5...`, 9.9395 s) |
| lock | `slurp-direct-only-fold-b-npu-port-v1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-FIT-NPU-PORT-V1` | `attempt-slurp-direct-only-fold-b-generation-npu-port-v1` / `...-fit-npu-port-v1` | 1,080 | `docs/receipts/rosters/slurp-evo-fold-b-v2.json` |
| aa (twin) | `slurp-direct-only-fold-b-aa-npu-port-v1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AA-NPU-PORT-V1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AA-FIT-NPU-PORT-V1` | `attempt-slurp-direct-only-fold-b-aa-generation-npu-port-v1` / `...-fit-npu-port-v1` | 1,080 | same |
| ab (second replicate, T2) | `slurp-direct-only-fold-b-ab-npu-port-v1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AB-NPU-PORT-V1` | `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AB-FIT-NPU-PORT-V1` | `attempt-slurp-direct-only-fold-b-ab-generation-npu-port-v1` / `...-fit-npu-port-v1` | 1,080 | same |

Config chain to write (SONNET, by a port derive of the 266-E Thinking cell; README section 2):
- `configs/campaign/slurp-direct-only-fold-b-npu-port-v1.json` (cells smoke, lock, aa, ab in chain order)
- `configs/experiment/slurp-direct-only-fold-b{,-smoke-4,-aa,-ab}-npu-port-v1.json`
- `configs/baseline/slurp-direct-only-fold-b{,-smoke-4}-baseline-raw-port-v1.json`

aa and ab use the lock baseline unchanged, so the requests are identical and only the run ids differ.

Package: `docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only/`. Decision receipt (not minted):
`coordination/slurp-direct-only-fold-b-npu-port-v1/root-admission-v1.json`, decision id
`slurp-direct-only-fold-b-npu-port-admission-v1`. Eligible run ids, in chain order: smoke, smoke FIT, lock, lock FIT, aa,
aa FIT, ab, ab FIT.

## 4. Dataset fragment (the port cell's own)

`configs/dataset/slurp-speech-native-evo-fold-b-1080.json`. This is the dataset fragment of the L338 port cell
(`configs/experiment/slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1.json`:13). Its facts:
- file sha256 `1d30c501...322cea`; split `SLURP-EVO-FOLD-B-V2`, identity `4407794c...3484a`; 1,080 items;
  `audio_seconds_total` 3,154.797375;
- `d1_media_supplement` `receipts/artifacts/d1-slurp-evo-fold-b-v2.json`, sha `c543067e...`. It matches the committed
  copy under `docs/receipts/artifacts/`.

Roster `docs/receipts/rosters/slurp-evo-fold-b-v2.json` (sha `47b89b9e...`). The Fold B audio pins were already verified
1,080/1,080 (Sonnet readiness report, `docs/superpowers/plans/2026-10-30-fold-b-comparison-baseline-and-improvement-plan.md`
section 4). D needs no vector, store or supply preparation.

## 5. Model fragment (the port's)

`configs/model/bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1.json` (file sha256
`bceb079d59853ca10ebdd772940720f92a4ef41370736d82e9de432cee9f7fc5`, recomputed here).

**The two fragment hashes are both correct; they differ only by the data root.** `core.config.fragment_hash`
(`core/config.py`:328-333) hashes the canonical JSON after `collapse_data_root` (`core/data_root_placeholder.py`:58-78):
- when `SAEA_DATA_ROOT` resolves to `/mnt/cephfs/xingchao/omni`, the fragment's `model_materialization_root` (that exact
  path) collapses to `${SAEA_DATA_ROOT}` and the hash is `a35b274058d7ac7558d3d0ac32cca8bb3c9321cb13ea89f16b38003fdd18c720`;
- with no data root resolved, the literal path is hashed and the result is
  `bcb98207bd929e0c21b3f6f3dd2835ff07e6e24d11bad9882520766c9b752447`.

Both values were recomputed on the dev host with the same canonicalisation (`sort_keys`, separators `,` `:`, UTF-8). The
hash that composes into a flight's protocol hash is `a35b2740...`, because flights run with the data root set. Records:
- `docs/readiness/2026-10-29-engineering-base-lock-npu-port-v1.md` row L9 carries `bcb98207`, and its D2 correction (:109)
  carries `a35b2740`; neither states the data-root dependence;
- the brief's phrase "true hash `bcb98207`" holds only for a run without a data root.

Recommended: a successor line in the base-lock record (not an edit of the sealed text).

## 6. Caps (the port package's formulas, applied to c = 1)

The formulas are the port package's (L3PF:175-185, which re-uses L3F):
- generation calls = rows x (contacts + fallback contacts per row) (L3F:582-585);
- speech-seconds = ceil(roster audio x (contacts + fallback contacts per row)) (L3F:576-579);
- FIT = rows x contacts x endpoints per contact (L3F:546-549, through `tfrl_family.fit_contact_cap`, 4 endpoints:
  `scripts/flights/tfrl/tfrl_family.py`:106, 328-331).

For the direct-only program, contacts = 1 (DFF:43). Fallback contacts per row = 1:
`FALLBACK_CONTACTS_PER_ROW_DIRECT` (`core/reasoning_knobs.py`:50, 433-440), applicable because the port fragment is a
budgeted thinking request. As a check, the same formulas reproduce the L338 port caps (32,400 / 64,800 / 94,644; L3PF:187).

| run | rows | roster audio s | generation calls | speech-seconds | FIT (code rule, raw arm) | FIT (port-package formula) | card-hours |
|---|---|---|---|---|---|---|---|
| smoke | 4 | 9.9395 | 8 | 20 | 4 | 16 | PROVISIONAL 0.5 x instances x T (L3PF:209-212) |
| lock | 1,080 | 3,154.797375 | 2,160 | 6,310 | 1,080 | 4,320 | PROVISIONAL ceil_0.1(1.3 x 1,080 x 0.52 x T / r / 60) |
| aa | 1,080 | same | 2,160 | 6,310 | 1,080 | 4,320 | same |
| ab | 1,080 | same | 2,160 | 6,310 | 1,080 | 4,320 | same |
| **cell total** | | | **6,488** | **18,950** | **3,244** | **12,976** | |

Contacts per row: 1 per pass, 3 passes (lock, aa, ab), so 3 per Fold B row. The ceiling is 2 per pass with the fallback
re-ask. PLAN section 5 says "2". That figure counts the twin only; T2's second replicate adds the third.

The 0.52 term is one Solver contact in lock-row equivalents: 64.07 / (59.0 + 64.07), rounded half-up
(`scripts/flights/nx/l338_cost_inputs.json` `r2b_measured`; rounding as L3F:291). A direct-only contact carries no
knowledge, so the prefill side is an over-estimate. The Thinking reasoning length is not. The smoke measures the real mix.
At the historical rate r = 1.95 the illustration reads 6.3 x T card-hours per pass. That figure is not registered; as in
PORTPKG, the numeric `planned_gpu_hours` is a placeholder until the sweep fills `PORT_DEPLOYMENT_PROFILE`.

**F1 (OPEN): no FIT path exists today for a raw-arm direct-only cell, and the port-package FIT formula disagrees with the
gate.**
- `build_live_phase_fit_gate` refuses an arm that has neither feature packets nor prepared text (PF:713-716).
- `build_native_contact_fit_gate` requires the `native-role-episode` stop rule (PF:967-970), which `phase-direct-only-v1`
  does not have.
- The runner itself accepts a raw direct-only contact (`rails/runner.py`:8115-8139), so only the fit is missing.
- For raw contacts the native gate demands cap = rows x max_contacts x 1 (one resident multimodal counting endpoint per
  contact, PF:1130-1137). The port package instead carried the packet-path 4 endpoints. This draft therefore registers the
  code-rule column (1,080 per 1,080-row pass) once a raw direct-only fit exists (T-engineering, SONNET). The port-package
  column is shown for traceability.
- The same mismatch affects the registered L338 port cell. Its `native_contact_fit.contact_cap` is 64,800
  (`configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json`:134). The gate will demand
  1,080 x 15 x 1 = 16,200 and raise "native contact fit cap must equal the registered maximum endpoint count" at the FIT
  step. It must be re-derived before that cell's flip (README, owner item O3).

## 7. Read rule (draft, for the T2 recipe)

- **Primary.** Official pooled SLU-F1 and intent (`scenario_action_exact`) of `lock`, with the per-row F1 beside the
  pooled F1 (M1).
- **Replicate variance.** Read from lock / aa / ab (k = 2 replicates of lock; M1). Twin admissibility: |SLU(aa) - SLU(lock)|
  <= 1.20 pp (port recipe `PORTPKG/analysis-recipe-v1.json`:172-206).
- **A-to-D gap.** A cross-run comparison (different flights and programs), read at the cross-run bar of 2.5 pp
  (DFF:52; READ E.1). The expected gap is about 26 pp (PLAN section 1), so the bar does not bind.
- **Truncation.** Cap-hit rows are sealed as receipted empty answers (`saea-empty-completion-v1` and
  `saea-thinking-answer-fallback-v1`) and scored as such, never re-asked beyond the one registered fallback (266-E
  preregistration section 3; decision 199). The M4 trace (T1) logs the truncation flag per contact.
- Bootstrap seed and draws are the 266-E ones (DFF:49-50) unless the T2 recipe fixes others.

## 8. Engineering prerequisites (card-free; SONNET)

| id | what | acceptance |
|---|---|---|
| E-D1 | FIT path for a raw direct-only cell (F1): extend `build_live_phase_fit_gate` to the raw arm with the vLLM multimodal counting endpoint (`core/preparation.py`:785 `measure_vllm_multimodal_request`), or admit `phase-direct-only-v1` to the native contact fit | a contract test fits a raw direct-only request through `VllmRouterTransport`; cap = rows x 1 |
| E-D2 | render check: `native-raw-speech-query-prefix-v1` on a direct-only payload (no evidence, no state) | layout test in `tests/unit/test_vllm_speech_layout_and_fit.py` style; audio first on the wire |
| E-D3 | port derive of the 266-E Thinking cell (port transform of section 2) and the writer extension (README section 2) | derive refuses any moved leaf outside the section-2 list |
| E-D4 | M4 trace (T1) on the direct-only controller | one JSONL per run; truncation flag per contact |

## 9. Exposure-ledger DRAFT row texts

Format: `PORTPKG/draft-ledger-rows.md` (15 columns). The writer fills `<commit>`, `<plan sha>` and the final gpu-hours
formula. Block heading and intro:

## Cell D NPU port: direct-only on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B (the port twin of the 266-E Thinking arm; smoke, then three request-identical 1,080-row passes) (DRAFT, owner countersignature pending)

8 unconsumed DRAFT rows (generation and FIT identity of the smoke and of the lock, aa and ab passes) for arm D of the three-way comparison (docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md section 5), drafted under owner decision 2026-10-28 item 4 (the agent drafts the exposure preregistration rows). Package docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only. No row is countersigned and no row authorises a contact. Each gpu-hours cell is the PROVISIONAL NPU card-hour formula, not a number: the exposure gate refuses these rows until the measured speed sweep fills PORT_DEPLOYMENT_PROFILE. The FIT rows' calls carry the code rule for a raw arm (rows x 1) and stay OPEN until a raw direct-only fit path exists.

| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-30 | SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- REGISTERED. NPU-port twin of SAEA-SLURP-DIRECT-ONLY-FOLD-B-THINKING-SMOKE-4-V1 (cell 266-E). Four-row decode-cost smoke of the sealed direct-only form (program phase-direct-only-v1, one contact per row, task instruction probes/slurp_direct_closed_list.txt, no knowledge supply), TRAIN Fold B, served by the BF16 / vLLM-Ascend Thinking core over the original recordings (native-raw-speech-query-prefix-v1); arm D of the three-way comparison; declared differences (design doc section 8): 8.1 BF16 weights instead of GGUF Q4_K_M; BF16 KV instead of q8_0; 8.2 audio features computed by the server's audio tower from the raw audio instead of llama.cpp mmproj packets; 8.3 vLLM sampling and kernels: temperature 0 with a fixed seed is deterministic per server build, not bit-identical to llama.cpp; 8.5 forced-close whitespace (vLLM reasoning_end_str) until verified; D1 reasoning budget 3,072 with forced close and one answer-only fallback instead of the historical 8,192-token uncapped completion | Frozen Qwen3-Omni-30B-A3B-Thinking BF16 safetensors (qwen3-omni-30b-a3b-thinking-bf16) on vLLM-Ascend, BF16 KV, per-sequence window 24,576 tokens, max_tokens 4096 (reasoning budget 3072 + answer reserve 1024), --reasoning-parser qwen3 with the registered forced-close reasoning_end_str (model fragment bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1); deployment (instances, tensor-parallel size, sequences per instance, devices) = PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep); the server's own audio tower over the original recordings under native-raw-speech-query-prefix-v1; direct-only program phase-direct-only-v1 (greedy, seed 20260913; no sampling axis); no tool authority | slurp TRAIN Fold B 4 | discovery | 26b306e5ccbfbe48ece85c7a3d07dacd947db9265cf5193bcb6e6d6e7f6500fe | no | 8 | 20 | PROVISIONAL: 0.5 x instances x T NPU card-hours; T and instances from PORT_DEPLOYMENT_PROFILE (pending) | TRAIN Fold B discovery rows already exposed by prior SLURP cells (the 266-E smoke and Thinking arm included); this cell opens no split. Prior exposure is inherited in full. | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only/smoke-generation-plan-registered-v1.json | DRAFT at study commit <commit>; plan SHA256 <plan sha>; drafted by the agent on owner decision 2026-10-28 item 4, owner countersignature pending; the decision receipt coordination/slurp-direct-only-fold-b-npu-port-v1/root-admission-v1.json is not minted; the gpu-hours cell is the PROVISIONAL card-hour formula and refuses the exposure gate until the measured sweep fills PORT_DEPLOYMENT_PROFILE; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-FIT-NPU-PORT-V1 | bounded-discovery-probe | (scope of the smoke row) -- FIT spend identity | (model/tool of the smoke row) | slurp TRAIN Fold B 4 | discovery | 26b306e5ccbfbe48ece85c7a3d07dacd947db9265cf5193bcb6e6d6e7f6500fe | no | 4 | 20 | PROVISIONAL: 0.5 x instances x T NPU card-hours; T and instances from PORT_DEPLOYMENT_PROFILE (pending) | (as the smoke row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only/smoke-fit-plan-registered-v1.json | DRAFT at study commit <commit>; plan SHA256 <plan sha>; FIT cap = 4 rows x 1 contact x 1 resident multimodal counting endpoint (core/preparation_fit.py:1130-1137 rule for raw contacts), OPEN until a raw direct-only fit path exists (F1); owner countersignature pending; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- REGISTERED. NPU-port twin of SAEA-SLURP-DIRECT-ONLY-FOLD-B-THINKING-V1 (cell 266-E Thinking arm). The sealed direct-only form (program phase-direct-only-v1, one contact per row, task instruction probes/slurp_direct_closed_list.txt, no knowledge supply) on TRAIN Fold B 1,080, served by the BF16 / vLLM-Ascend Thinking core over the original recordings (native-raw-speech-query-prefix-v1); arm D of the three-way comparison, pass lock (1 of 3 request-identical passes: lock, aa, ab; M1 replicates); declared differences 8.1, 8.2, 8.3, 8.5 (as the smoke row) and D1 reasoning budget 3,072 with forced close and one answer-only fallback instead of the historical 8,192-token uncapped completion; a capability read, no mechanism, no claim | (model/tool of the smoke row) | slurp TRAIN Fold B 1,080 | discovery | 4407794c4a5d2e8883d93fdf242deb33c0987405d62e05a7d8416f5251a3484a | no | 2160 | 6310 | PROVISIONAL: ceil_0.1(1.3 x 1080 x 0.52 x T / r / 60) NPU card-hours; T (cards per instance) and r (measured lock rows per minute per instance) from PORT_DEPLOYMENT_PROFILE (pending) | TRAIN Fold B discovery rows already exposed by prior SLURP cells (the 266-E Thinking arm SAEA-SLURP-DIRECT-ONLY-FOLD-B-THINKING-V1 and the L338 port registration included); this cell opens no split. Prior exposure is inherited in full. | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only/lock-generation-plan-registered-v1.json | DRAFT at study commit <commit>; plan SHA256 <plan sha>; generation cap 1,080 x (1 contact + 1 answer-only fallback); speech cap ceil(3,154.797375 x 2); drafted by the agent on owner decision 2026-10-28 item 4, owner countersignature pending; decision receipt not minted; PROVISIONAL gpu-hours; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-DIRECT-ONLY-FOLD-B-FIT-NPU-PORT-V1 | bounded-discovery-probe | (scope of the lock row) -- FIT spend identity | (model/tool of the smoke row) | slurp TRAIN Fold B 1,080 | discovery | 4407794c4a5d2e8883d93fdf242deb33c0987405d62e05a7d8416f5251a3484a | no | 1080 | 6310 | (as the lock row) | (as the lock row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only/lock-fit-plan-registered-v1.json | DRAFT at study commit <commit>; plan SHA256 <plan sha>; FIT cap 1,080 x 1 x 1 (raw rule), OPEN (F1); owner countersignature pending; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-DIRECT-ONLY-FOLD-B-AA-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- REGISTERED. Request-identical twin pass aa of SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1 (same baseline, model, dataset and decoding; only the run and attempt ids differ), flown in the same chain right after the lock pass; the within-chain A/A null of arm D (M1, 2 of 3 passes); otherwise the scope of the lock row | (model/tool of the smoke row) | slurp TRAIN Fold B 1,080 | discovery | 4407794c4a5d2e8883d93fdf242deb33c0987405d62e05a7d8416f5251a3484a | no | 2160 | 6310 | (as the lock row) | (as the lock row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only/aa-generation-plan-registered-v1.json | (as the lock row) |
| 2026-10-30 | SAEA-SLURP-DIRECT-ONLY-FOLD-B-AA-FIT-NPU-PORT-V1 | bounded-discovery-probe | (scope of the aa row) -- FIT spend identity | (model/tool of the smoke row) | slurp TRAIN Fold B 1,080 | discovery | 4407794c4a5d2e8883d93fdf242deb33c0987405d62e05a7d8416f5251a3484a | no | 1080 | 6310 | (as the lock row) | (as the lock row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only/aa-fit-plan-registered-v1.json | (as the lock FIT row) |
| 2026-10-30 | SAEA-SLURP-DIRECT-ONLY-FOLD-B-AB-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- REGISTERED. Second request-identical replicate pass ab of SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1 (T2; same baseline, model, dataset and decoding; only the run and attempt ids differ), flown in the same chain after aa; with aa it gives k = 2 replicates of lock for the replicate variance term of the CI (M1, 3 of 3 passes); otherwise the scope of the lock row | (model/tool of the smoke row) | slurp TRAIN Fold B 1,080 | discovery | 4407794c4a5d2e8883d93fdf242deb33c0987405d62e05a7d8416f5251a3484a | no | 2160 | 6310 | (as the lock row) | (as the lock row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only/ab-generation-plan-registered-v1.json | (as the lock row) |
| 2026-10-30 | SAEA-SLURP-DIRECT-ONLY-FOLD-B-AB-FIT-NPU-PORT-V1 | bounded-discovery-probe | (scope of the ab row) -- FIT spend identity | (model/tool of the smoke row) | slurp TRAIN Fold B 1,080 | discovery | 4407794c4a5d2e8883d93fdf242deb33c0987405d62e05a7d8416f5251a3484a | no | 1080 | 6310 | (as the lock row) | (as the lock row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/direct-only/ab-fit-plan-registered-v1.json | (as the lock FIT row) |

The writer expands each "(as ...)" cell to the full text. The ledger rows must be self-contained; the shorthand is for
review only.

## 10. Open items specific to D

- F1 (section 6): the raw direct-only FIT path, and the L338 port FIT cap 64,800 versus the gate's 16,200.
- D1 (section 2): the budget-3,072 forced close on a form that historically ran to 8,192. Read the truncation rate from the
  smoke before the flight (the 266-E smoke precedent: one cap hit in four, PKG266 README Finding 3).
- Whether D shares a campaign with the L338 port flight (PLAN section 5: "D and A in one campaign"). One deployment,
  sequential runs. Same-deployment flights make the A-to-D gap a same-session, cross-run reading.
