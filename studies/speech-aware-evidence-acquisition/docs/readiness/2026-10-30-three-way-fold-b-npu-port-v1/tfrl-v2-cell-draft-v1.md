# T cell `SAEA-SLURP-TFRL-V2-FOLD-B-NPU-PORT-V1`: the agent system plus TFRL-v2 on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B (DRAFT TEXT)

**Status: DRAFT TEXT ONLY. This is not a registration and it grants no authority.** No plan, store release, receipt,
ledger row or flip exists. No contact (model, encoder, reflector) is authorised and nothing is countersigned. Written by
Opus 5.5 for task T4 at study commit `c1b0be33`. Development numbers only; `claims_benefit` false.

Abbreviations: those of `direct-only-cell-draft-v1.md`, plus:
- PROT = `docs/readiness/2026-10-11-slurp-tfrl-loop-v1/tfrl-loop-protocol-v1.json`
- CSRC = `docs/readiness/2026-10-11-slurp-tfrl-construction-source-v1/`
- I3 = `docs/results/2026-10-12-slurp-tfrl-loop-three-iterations-read.md`
- PBASE = `configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json`
- PEXP = `configs/experiment/slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1.json`

## 1. What T is (PLAN section 4, made registrable)

T = A's lock harness plus three TFRL-v2 outputs:
1. an **exemplar bank**: scenario-keyed exemplars distilled from positive trajectories, consumed by the Solver as
   scenario-conditioned exemplars;
2. a **rule bank**: consumption rules, each written by the reflector only from a positive/negative contrast pair on the
   same row; a rule without a pair is refused;
3. a **vote harness**: self-consistency n = 3 at the Solver over the top-ranked rules, scenario_action by majority, entities
   by union-vote (PLAN sections 3 and 4).

A's lock harness is the delivering `lock` arm of the L338 port cell:
- the review-off module graph, one Perception and one Solver contact (PEXP `generation_budget.definition`, :77);
- greedy decoding, seed 20260913;
- the port's re-authored knowledge (intent-boundaries-npu-port-v1, slot-contracts-npu-port-v2) and the port's regenerated
  vectors (PBASE `knowledge_stores`, :5-131).

TFRL-v2 samples **trajectories** on the construction column and credits them **per row** against lock (PLAN section 4,
rows "sample" and "credit"). No parameter is trained.

## 2. The construction column: which one, and reuse versus re-flight

**Historical 625-row construction column:**
- cell `slurp-native-stage3-tfrl-construction-lock-r01c-v1`: `configs/experiment/` and `configs/campaign/` of that name;
  baseline `configs/baseline/slurp-native-stage3-tfrl-construction-lock-r01c-baseline-v1.json`; smoke
  `...-smoke-4-v1`;
- run `SAEA-SLURP-STAGE3-TFRL-CONSTRUCTION-LOCK-R01C-V1`, package CSRC;
- protocol: 625 rows, program `phase-native-module-v1-c2`, 2 contacts per row, lock v2.2 plus the construction variant
  `r01c` (PROT:186-199);
- ledger LEDGER:8858-8862: 1,250 calls, 5,454 s, the Instruct Q4_K_M core.

**It cannot be reused; it must be re-flown on the port.** Four independent reasons:
1. Core and precision differ: Instruct Q4_K_M versus Thinking BF16 on vLLM (READ E.7, E.9).
2. The lock differs: re-authored stores (8.6) and regenerated vectors (8.4, 8.7). PROT:199 itself says "re-flown only if
   the lock changes".
3. TFRL-v2 needs K = 4 trajectories per row, while the historical column has one lock pass.
4. Its material was consumed as aggregate tables (I3:79-83), and the historical read files are absent (READ section F).

PLAN T5 says the same.

**Reusable unchanged (model-free, label-free inputs; checked today):**
- dataset fragment `configs/dataset/slurp-speech-native-stage3-construction-r01c.json`: sha256 `36949a68...6a56cba`,
  split `STAGE3-CONSTRUCTION-R01C`, identity `f56bc386...4515`, 625 items, 2,726.8471875 s;
- roster `docs/receipts/rosters/slurp-stage3-construction-r01c.json`, sha256 `e8abc65b...eeddd`;
- d1 supplement `docs/receipts/artifacts/d1-slurp-stage3-construction-r01.json`, sha256 `7cf5f978...` (matches);
- **625 / 625 audio sha256 pins** re-hashed on the data root today;
- smoke roster `docs/receipts/rosters/slurp-stage3-roll-aa-r01c-smoke-4-v1.json` (split `e9dcea53...`, 17.0349375 s);
- the construction-variant mechanism (`rails/adapters/slurp_module_construction.py`:7, 71-99);
- the port's rebuilt minus-r01 exemplar store (`SAEA-SLURP-STAGE3-MINUS-R01-KNOWLEDGE-REBUILD-NPU-PORT-V1`, PORTPKG
  `draft-ledger-rows.md` row 3).

The construction variant is required because the port Fold B lock pins the FULL `size-pool-annotated-exemplars` pool
(PBASE :85-131), which contains every R01C row (CSRC README, resolved blocker `knowledge-source-guard`).

**Overlap checks run today (model-free, roster ids only):**
- R01C against the port intent-support construction roster (2,037 ids): 0 shared;
- R01C against the slot-support roster (3,162 ids): 0 shared;
- R01C sits inside the port outer construction scope (37,407 ids);
- (sources: `prepared/SAEA-SLURP-SEMANTIC-AUTHORING-PREPARE-V1/` on the data root, pinned by
  `docs/readiness/2026-10-28-slurp-intent-boundaries-preparation-registration-npu-port-v1.json` and
  `...slot-contracts-...-npu-port-v2.json`);
- R01C against Fold A: 0 shared.

**Must be regenerated on the port before the first construction flight:**

| id | what | historical twin | kind |
|---|---|---|---|
| P1 | R01C speech-query vectors (Nemotron-3B, CPU float32, forced causal), 625 rows | `SAEA-SLURP-STAGE3-CONSTRUCTION-R01-SPEECH-QUERY-ENCODING-V1` (800-row R01, then the R01C export) | tool contacts |
| P2 | query-dimension alias | `SAEA-SLURP-STAGE3-CONSTRUCTION-R01-ALIAS-V1` | model-free |
| P3 | content derivative (exemplar-channel query source) | `vectors/slurp-stage3-construction-r01c-v1/query-vectors-speech-content-v1.jsonl` | model-free |
| P4 | R01C exemplar-intent agreement table, against the port minus-r01 snapshot | `SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-R01C-V1` (CSRC README, `agreement-table-owed`) | model-free |
| P5 | two frozen halves R01C-H1 / R01C-H2 (scenario-stratified selection, each with its own roster receipt and split identity; section 5.3) | none | model-free |
| P6 | seed rule bank `rule-bank-i0` (section 5.2) | none | model-free (authored) |

Plus one first-phase supply per construction run (`core.flight.cli prepare-supply`, as PORTPKG row 6).

## 3. Identities (draft)

The family id is the brief's `SAEA-SLURP-TFRL-V2-FOLD-B-NPU-PORT-V1`. The protocol hash moves every iteration (section 6),
so every iteration's Fold B flight is its own run with the iteration in its id. Nothing is ever edited in place.

| step | run (FIT run = the same with `-FIT` before `-NPU-PORT-V1`) | rows | program | per iteration |
|---|---|---|---|---|
| construction smoke (i1 only) | `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-SMOKE-4-NPU-PORT-V1` | 4 | `phase-native-module-v1-c7` | once |
| construction flight | `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-NPU-PORT-V1`; for i2 and i3 the halves `...-R01C-I<t>-H1-NPU-PORT-V1` and `...-H2-...` | 625 (i1); about 312 + 313 (i2, i3) | c7 | 1 or 2 runs |
| credit + exemplar distillation | `SAEA-SLURP-TFRL-V2-CREDIT-I<t>-NPU-PORT-V1` | model-free | - | 1 |
| contrast-pair reflector | `SAEA-SLURP-TFRL-V2-REFLECTOR-I<t>-NPU-PORT-V1` | at most 18 contacts (one per scenario with a pair) | text-only | 1 |
| bank release | `SAEA-SLURP-TFRL-V2-BANK-RELEASE-I<t>-NPU-PORT-V1` | model-free | - | 1 |
| Fold B smoke (i1 only) | `SAEA-SLURP-TFRL-V2-FOLD-B-SMOKE-4-NPU-PORT-V1` | 4 (`slurp-stage3-filler-fold-b-c01-smoke-4-v1`, split `bdd025a1...`) | c10 | once |
| Fold B selection flight | `SAEA-SLURP-TFRL-V2-FOLD-B-I<t>-NPU-PORT-V1` | 1,080 | c10 | 1 |
| Fold A confirmation | `fold-a-confirmation-fragment-draft-v1.md` | 1,080 | c10 | per accepted state (section 7) |

Config chain, one set per iteration:
- baseline `configs/baseline/slurp-native-stage3-tfrl-v2-fold-b-i<t>-baseline-raw-port-v1.json`;
- experiment `configs/experiment/slurp-native-stage3-tfrl-v2-fold-b-i<t>-npu-port-v1.json`;
- construction cells `slurp-native-stage3-tfrl-v2-construction-r01c-i<t>[-h1|-h2]-npu-port-v1`;
- campaign `configs/campaign/slurp-native-stage3-tfrl-v2-npu-port-v1.json`.

Dataset fragments: `slurp-speech-native-evo-fold-b-1080` (Fold B, the port cell's, PEXP:13) and
`slurp-speech-native-stage3-construction-r01c` (construction). Model fragment: the port's, as in D (section 5 of the D
draft, hash `a35b2740...` with the data root set). Package: `docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/`.
Decision receipt (not minted): `coordination/slurp-tfrl-v2-npu-port-v1/root-admission-v1.json`.

## 4. Contacts per row

**Construction flight, c7.** Arms at the Solver, in order:
- `s1`..`s4`: K = 4 trajectories at temperature 0.7 (PLAN section 4); each carries rule-bank variant k, a consumption
  instruction plus an evidence ordering;
- `aa`: twin of lock;
- `lock`: delivering arm, LAST, greedy.

The mechanism is the node-template arm group at the Solver: ceiling 2 + (K' - 1) with K' = 6 arms. The family allows
K' = 2..8, "one candidate to six, the delivering arm and a twin" (`rails/controllers.py`:1806-1809). So 1 Perception + 6
Solver = 7 contacts per row.

The `aa` twin is this draft's addition to PLAN section 4. Per-row credit against a single greedy lock is confounded by
lock's own decoding flips: 8.5 % of rows differ within one flight (READ E.1). A positive or negative is counted only on rows
where lock and aa agree. Without aa the program is c6 (625 rows: 7,500 calls, 32,723 s, FIT 3,750); see owner item O4.

**Fold B and Fold A selection flight, c10.** Arms:
- `t-v2`: the candidate. It has one Perception contact and Solver vote n = 3. Each of the three Solver contacts carries the
  exemplar-bank items of the Perception-predicted scenario and one of the top-3 ranked rules, at the registered
  sampling-axis decoding (temperature 0.6, top_p 0.95, top_k 20; L3PF:243-244).
- `aa`: twin of lock.
- `ab`: second replicate of lock (T2).
- `lock`: delivering, LAST.

By the registered formula 2 + 2 (K - 1) + (n - 1) with K = 4 and n = 3 (PEXP:77): 2 + 6 + 2 = 10 contacts per row (4
Perception + 6 Solver). The T arm itself costs 4 contacts per row. PLAN section 5 lists "2", which counts lock + twin
without the vote and T2.

`ab` needs T2: the rails accept exactly one twin today (`rails/supply_variants.py`:2749-2754). Without T2 the program is
c8 (1,080 rows: 17,280 calls, 50,477 s, FIT 8,640). One candidate arm is legal (`MIN_SAMPLING_CANDIDATE_ARMS` 1,
`supply_variants.py`:2457).

**Reflector.** One text-only contact per scenario that has at least one contrast pair, at most 18 per iteration. This
follows the ACE precedent of 18 carriers, one per scenario (I3:91-96). The reflector sees no audio.

## 5. The harness outputs as artefacts (paths, pins)

### 5.1 Per iteration t (all sealed, sha256-pinned, receipts on the data root)

| artefact | path (proposed) | made by |
|---|---|---|
| trajectories | the construction run's sealed outputs (standard per-run layout) plus the M4 trace `analysis/<run>/m4-trace-v1.jsonl` (schema by T1: row id, node, arm, supply item ids and scores, thinking tokens, truncation flag, answer, per-row score, trajectory id, reward, selected) | construction flight |
| per-row credit | `prepared/SAEA-SLURP-TFRL-V2-CREDIT-I<t>-NPU-PORT-V1/credit.jsonl` + `receipt.json`. Reward r = rowF1(traj) - rowF1(lock). Positive: lock wrong on the row, r > 0, lock = aa. Negative: lock right, r < 0, lock = aa | model-free (official per-row scores) |
| contrast pairs | `.../contrast-pairs.jsonl` (same row, one positive, one negative) | model-free |
| exemplar bank | store `knowledge/slurp/tfrl-v2/exemplar-bank-i<t>-npu-port-v1/{snapshot.json,release.json,completion.json}`, store_role `tfrl-exemplar-bank`, keyed by scenario. Each entry is the solved row's evidence-to-answer path (supplied item ids used, answer). Capped per scenario at B_max, fixed in the sampler config before i1 contact | model-free distillation |
| rule bank | store `knowledge/slurp/tfrl-v2/rule-bank-i<t>-npu-port-v1/{snapshot.json,release.json}`, store_role `tfrl-rule-bank`. Each rule records the contrast-pair ids it came from; rules without a pair are refused and listed in `prepared/SAEA-SLURP-TFRL-V2-REFLECTOR-I<t>-NPU-PORT-V1/refused.jsonl`. Ranked by script credit (helpful / harmful counts on the construction rows, as the ACE curator's counters, I3:91-96) | reflector + deterministic curator |
| release registrations | `docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/{exemplar,rule}-bank-i<t>-release-registration.json`, in the form of `docs/readiness/2026-10-11-slurp-tfrl-loop-v1/ace-playbook-i1-release-registration.json`. Construction scope R01C (or one half); forbidden rosters Fold A v2 `1fe7908c...`, Fold B v2 `47b89b9e...`, DEV-1000 `653c7556...` (the port semantic stores' list) | model-free |
| loop state | `docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/loop-state.json` (iteration, accepted state, refusals, optimism term) | main thread |

### 5.2 Seed rule bank (before i1)

`rule-bank-i0` holds at least 4 variants, each a consumption instruction crossed with an evidence ordering (for example:
as retrieved, score-ascending, scenario-grouped). Opus authors it under Fable's guidance. It is released as a store and
pinned by sha256 in the i1 construction baseline. It is the only hand-written TFRL-v2 input.

### 5.3 Why the halves: the knowledge-source guard

The banks' sources are R01C rows. The runner refuses a flight in two cases (`rails/runner.py`:8327-8336):
- its query rows overlap a pinned snapshot's source ids;
- the snapshot's source split identity equals the flight's.

So an i2 construction flight on R01C that consumes the i1 bank is refused. Proposed fix: cross-fit. At t >= 2, the H1 run
consumes the bank built from H2 positives and vice versa. The Fold B and Fold A flights consume the union, whose sources
(R01C) are disjoint from both folds. Two things are unverified:
- whether the guard covers every module store or only the single `knowledge_snapshot`; T3 checks this;
- whether a store role can be swapped per arm. The ACE `release_swap` is the precedent (I3:96).

## 6. How the outputs enter the hashed protocol

- **The exemplar bank is a knowledge store.** It is carried by the T cell's BASELINE fragment
  (`...-tfrl-v2-fold-b-i<t>-baseline-raw-port-v1.json`): `knowledge_stores.tfrl-exemplar-bank`
  `{release, snapshot (relpath, sha256), store_role, phase_supply.solver}`, plus its release receipt in
  `preparation_source_receipts`, the same layout as the three port stores in PBASE :5-131 and :411. The rule bank is a
  second store in the same block. The vote harness is a `native_episode.consumption_module.columns` entry on arm `t-v2`
  (`solver_samples` n 3, commit policy `saea-commit-vote-v1`), as PBASE carries the L338 columns.
- **Therefore the composed config hash and the protocol hash move every iteration.** `core.config.compose` /
  `protocol_hash` digest the composed baseline. So does the first-phase supply identity (`core.preparation.artifact_inputs`
  digests the composed config; L3PF:84-86). Each iteration gets a new baseline file, a new experiment, new run ids and new
  plans.
- The model fragment, the dataset fragment and the deployment profile do not move. The deployment is outside the protocol
  hash (L3PF:61-71).
- The vote policy, the scenario-conditioned supply and the two new store roles are code (T3) and must exist before the i1
  baseline can compose.
- Context fit: the bank adds Solver prompt tokens. The registered limit is 24,576 - 3,072 - 4,096 - 2 = 17,406, and the
  measured Solver prompts on Thinking Fold B are p99 12,486 / max 13,465 (`scripts/flights/nx/l338_cost_inputs.json`
  `fit_observations`). The bank's token cap must hold the max inside the limit. The ACE precedent capped at 15,608
  (I3:91-96). See open item H1.

## 7. Acceptance rule and kill criteria (PLAN section 4 with M1 / M2)

**Iteration t is ACCEPTED** only if every one of the following holds in the Fold B flight `SAEA-SLURP-TFRL-V2-FOLD-B-I<t>-NPU-PORT-V1`:

1. SLU-F1(t-v2) - SLU-F1(lock) has point >= 1.49 pp (L3F:253) and a **widened** CI95 lower bound > 0. The widened CI
   adds the replicate variance from lock / aa / ab to the row-paired cluster bootstrap (M1; recipe T2).
2. The intent guard holds: intent CI-LB >= -0.65 and intent point >= 0 (PROT:264-270; L3F:254).
3. The per-row mean delta is >= 0. An arm whose per-row delta is negative cannot be nominated (M1; closes P4).
4. Twin admissibility: |SLU(aa) - SLU(lock)| <= 1.20 pp (PORTPKG `analysis-recipe-v1.json`:172-206).
5. The construction-column gain is >= +1 pp. It is computed model-free as SLU(vote over s1..s4, majority scenario_action,
   entity support >= 2 of 4) - SLU(lock) on the construction rows of iteration t.

**Then CONFIRMED** only if, in the Fold A flight of the accepted state, the widened CI95 lower bound of t-v2 - lock is > 0
(M2). The optimism term (Fold B delta - Fold A delta) is reported always.

**Kill and stop:**
- iteration 1 construction gain < +1 pp: stop the family;
- an accepted iteration fails Fold A: revert to the previous accepted state and report the optimism term;
- an iteration accepts nothing: stop;
- at most 3 iterations (PLAN section 4, rows "select" and "iterate", and its kill paragraph).

The expected outcome is +2 to +4 pp if P5 (consumption, not supply) is the bottleneck, else 0 (PLAN section 4).

## 8. Caps (port-package formulas: L3F:576-585 with fallback = contacts; FIT as in D section 6)

| run | rows | contacts / row | generation calls | speech-seconds | FIT (code rule, raw) | FIT (port-package formula) | card-hours (PROVISIONAL; illustration at r = 1.95) |
|---|---|---|---|---|---|---|---|
| R01C encoding P1 | 625 | tool | 625 encoder contacts | 2,727 | - | - | 2.0 h wall ceiling, 0 NPU (as `docs/readiness/2026-10-28-slurp-tfrl-fold-b-speech-query-encoding-registration-npu-port-v1.json`) |
| construction smoke | 4 | 7 | 56 | 239 | 28 | 112 | 0.5 x instances x T |
| construction i<t> (625 rows; the i2 / i3 halves sum to this, plus at most 1 s rounding per half) | 625 | 7 | 8,750 | 38,176 | 4,375 | 17,500 | ceil_0.1(1.3 x 625 x 3.60 x T / r / 60); 25.0 x T |
| reflector i<t> | 18 contacts | text | 36 (18 + 18 fallback) | 0 | 37 (prepared-text rule 1 + 2 x 18, PF:657-669) | - | 0.5 x instances x T allowance |
| Fold B smoke | 4 | 10 | 80 | 204 | 40 | 160 | 0.5 x instances x T |
| Fold B i<t> | 1,080 | 10 | 21,600 | 63,096 | 10,800 | 43,200 | ceil_0.1(1.3 x 1,080 x 5.04 x T / r / 60); 60.5 x T |

Upper bound for 3 iterations: 26,306 construction calls, 108 reflector calls and 64,880 Fold B calls. Speech-seconds:
114,767 (+2) construction and 189,492 Fold B.

Mixes (lock-row equivalents; `l338_cost_inputs.json` node seconds, rounding as L3F:291): construction (1 x 59.0 + 6 x
64.07) / 123.07 = 3.60; Fold B (4 x 59.0 + 6 x 64.07) / 123.07 = 5.04.

PLAN section 5 estimates "25-30 sampling + 10 confirmation" card-hours at the knee. That assumes the port's measured rate,
far above the historical 1.95. The registered bound comes from the sweep.

## 9. Exposure-ledger DRAFT row texts (iteration 1 in full; i2 / i3 identical with `I2` / `I3` and the `H1` / `H2` halves)

## Cell T NPU port: the agent system plus TFRL-v2 on the BF16 / vLLM-Ascend Thinking core (construction R01C sampling, contrast-pair reflector, Fold B selection) (DRAFT, owner countersignature pending)

DRAFT rows for the T family of the three-way comparison (docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md sections 4-5), drafted under owner decision 2026-10-28 items 4 and 6. Package docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2. No row is countersigned and no row authorises a contact. Gpu-hours cells of model flights are PROVISIONAL card-hour formulas (the exposure gate refuses them until PORT_DEPLOYMENT_PROFILE is filled); FIT calls carry the raw-arm code rule (rows x contacts x 1).

| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-30 | SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-SPEECH-QUERY-ENCODING-NPU-PORT-V1 | bounded-discovery-probe | Port twin of SAEA-SLURP-STAGE3-CONSTRUCTION-R01-SPEECH-QUERY-ENCODING-V1 restricted to the 625 R01C rows: complete-speech retrieval-query encoding for the TFRL-v2 construction roster; no answers or scoring | Frozen local Omni-Embed-Nemotron-3B retrieval tool on the CPU float32 worker pool; no answering core | slurp SLURP-STAGE3-CONSTRUCTION-R01C TRAIN construction | discovery | f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515 | no | 625 | 2727 | 2.0 | The historical R01 / R01C roster, packet and query-encoding exposures carried forward; this row adds only the port's pinned retrieval-encoder contacts | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/r01c-speech-query-encoding-registration-npu-port-v1.json | DRAFT TEXT. CPU float32 retrieval encoder (declared difference 8.7): pinned pool of 4 workers x 16 threads on cores 128-191 of the Ascend host, batch-one forwards; planned_gpu_hours is the attempt's wall-clock ceiling and the actual GPU/NPU hours are 0. Attention geometry FORCED CAUSAL (declared difference 8.4). One encode-v1 attempt; no retry, fallback, scorer, answer read, or cross-fold feedback. Owner countersignature pending; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-ALIAS-NPU-PORT-V1 | model-free-check | Port twin of SAEA-SLURP-STAGE3-CONSTRUCTION-R01-ALIAS-V1: carry the port's sealed R01C content-dimension speech-query segment vectors into the intent-boundary retrieval dimension, values unchanged, segment identities preserved. Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts. | no model or tool contact; local arithmetic over receipt-pinned files | slurp SLURP-STAGE3-CONSTRUCTION-R01C 625 | discovery | f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515 | no | 0 | 0 | 0 | No new speech exposure: every vector read was sealed by SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-SPEECH-QUERY-ENCODING-NPU-PORT-V1. Prior exposure inherited in full. | prepared/SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-ALIAS-NPU-PORT-V1/query-dimension-alias.json (sha256 <filled after the run>) | DRAFT TEXT; to be RUN BEFORE THE ROW IS WRITTEN (model-free, saea-base container); output vectors/slurp-stage3-construction-r01c-npu-port-v1/query-vectors-speech-segments.jsonl. It constructs no ExecutionPlan, reserves no contact, releases no store beyond the one named, re-pins no committed config and authorises nothing. |
| 2026-10-30 | SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-INPUTS-PREPARE-NPU-PORT-V1 | model-free-check | The port's model-free CONTENT DERIVATIVE of the R01C speech-query vectors (scripts/build_slurp_query_content_derivative.py), the exemplar channel's query source. Zero model, server, encoder, embedder, scorer, tokenizer, network, GPU and paid-API contacts. | no model or tool contact; local arithmetic over receipt-pinned files | slurp SLURP-STAGE3-CONSTRUCTION-R01C 625 | discovery | f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515 | no | 0 | 0 | 0 | No new speech exposure (as the alias row). | vectors/slurp-stage3-construction-r01c-npu-port-v1/query-vectors-speech-content-v1.provenance.json (sha256 <filled after the run>) | DRAFT TEXT; model-free; authorises nothing. |
| 2026-10-30 | SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-R01C-NPU-PORT-V1 | model-free-check | Port twin of SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-R01C-V1: the pre-computed, self-excluded exemplar-intent agreement table for R01C, arithmetic over the port minus-r01 pool snapshot, the port pool key vectors and the port R01C speech query vectors through the repository's own recall/rank/fuse. Zero model contacts. | no model or tool contact; local arithmetic over receipt-pinned files | slurp SLURP-STAGE3-CONSTRUCTION-R01C 625 | discovery | f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515 | no | 0 | 0 | 0 | No new speech exposure. | prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-R01C-NPU-PORT-V1/agreement.jsonl (sha256 <filled after the run>) | DRAFT TEXT; model-free; authorises nothing. |
| 2026-10-30 | SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-HALVES-NPU-PORT-V1 | model-free-check | Two frozen, scenario-stratified halves of R01C (R01C-H1, R01C-H2) for the TFRL-v2 cross-fit of iterations 2-3: a pure selection of the parent receipt's items in parent order, no rng beyond the registered seed, each half with its own roster receipt and split identity. Zero model contacts. | no model or tool contact; local arithmetic over receipt-pinned files | slurp SLURP-STAGE3-CONSTRUCTION-R01C 625 | discovery | f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515 | no | 0 | 0 | 0 | No new speech exposure. | docs/receipts/rosters/slurp-stage3-construction-r01c-h1-v1.json, docs/receipts/rosters/slurp-stage3-construction-r01c-h2-v1.json | DRAFT TEXT; model-free; authorises nothing. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-RULE-BANK-I0-RELEASE-NPU-PORT-V1 | model-free-check | Release of the hand-authored seed rule bank rule-bank-i0 (at least 4 consumption-instruction x evidence-ordering variants) as a knowledge store under the construction scope R01C with forbidden rosters Fold A v2, Fold B v2 and DEV-1000; consumed by the s1..s4 arms of the i1 construction flight only. Zero model contacts. | no model or tool contact; local arithmetic over receipt-pinned files | slurp SLURP-STAGE3-CONSTRUCTION-R01C (scope only) | discovery | f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515 | no | 0 | 0 | 0 | No speech exposure. | knowledge/slurp/tfrl-v2/rule-bank-i0-npu-port-v1/release.json (sha256 <filled after the run>) | DRAFT TEXT; authored by Opus 5.5 under Fable 5.1 guidance; authorises nothing. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-SMOKE-4-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- REGISTERED. Four-row form smoke of the TFRL-v2 construction flight: A's lock harness (the plain review-off module graph on the port lock, construction variant r01c with the port minus-r01 exemplar pool) with a Solver arm group lock / aa / s1..s4 (program phase-native-module-v1-c7, 7 contacts per row; s1..s4 at temperature 0.7, each with one rule-bank-i0 variant), TRAIN construction roster R01C, served by the BF16 / vLLM-Ascend Thinking core over the original recordings (native-raw-speech-query-prefix-v1) with the port's re-authored knowledge (intent-boundaries-npu-port-v1, slot-contracts-npu-port-v2) and the port's regenerated retrieval vectors; declared differences (design doc section 8): 8.1 BF16 weights instead of GGUF Q4_K_M; BF16 KV instead of q8_0; 8.2 audio features computed by the server's audio tower from the raw audio instead of llama.cpp mmproj packets; 8.3 vLLM sampling and kernels: temperature 0 with a fixed seed is deterministic per server build, not bit-identical to llama.cpp; 8.4 causal retrieval encoder (forced and measured); every sealed retrieval vector regenerated; 8.5 forced-close whitespace (vLLM reasoning_end_str) until verified; 8.6 re-authored semantic knowledge (intent-boundaries-npu-port-v1, slot-contracts-npu-port-v2; decision 9.5); 8.7 CPU float32 retrieval encoder instead of CUDA bfloat16 | Frozen Qwen3-Omni-30B-A3B-Thinking BF16 safetensors (qwen3-omni-30b-a3b-thinking-bf16) on vLLM-Ascend, BF16 KV, per-sequence window 24,576 tokens, max_tokens 4096 (reasoning budget 3072 + answer reserve 1024 on both nodes), --reasoning-parser qwen3 with the registered forced-close reasoning_end_str (model fragment bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1); deployment (instances, tensor-parallel size, sequences per instance, devices) = PORT_DEPLOYMENT_PROFILE (pending; filled from the measured speed sweep); the server's own audio tower over the original recordings under native-raw-speech-query-prefix-v1; module program phase-native-module-v1-c7; Solver arm group with per-arm decoding (s1..s4 temperature 0.7; lock and aa greedy, seed 20260913); no tool authority | slurp SLURP-STAGE3-CONSTRUCTION-R01C 4 | discovery | e9dcea536028d17737b999c65267e47645ca7b3f6f9c4f57622961cf29363ed5 | no | 56 | 239 | PROVISIONAL: 0.5 x instances x T NPU card-hours; T and instances from PORT_DEPLOYMENT_PROFILE (pending) | TRAIN construction rows already exposed by prior SLURP cells (the R01C construction-source column and its smoke included); this cell opens no split. Prior exposure is inherited in full. | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/construction-smoke-generation-plan-registered-v1.json | DRAFT at study commit <commit>; plan SHA256 <plan sha>; drafted by the agent on owner decision 2026-10-28 item 4, owner countersignature pending; decision receipt coordination/slurp-tfrl-v2-npu-port-v1/root-admission-v1.json not minted; PROVISIONAL gpu-hours; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-SMOKE-4-FIT-NPU-PORT-V1 | bounded-discovery-probe | (scope of the construction smoke row) -- FIT spend identity | (model/tool of the construction smoke row) | slurp SLURP-STAGE3-CONSTRUCTION-R01C 4 | discovery | e9dcea536028d17737b999c65267e47645ca7b3f6f9c4f57622961cf29363ed5 | no | 28 | 239 | PROVISIONAL: 0.5 x instances x T NPU card-hours; T and instances from PORT_DEPLOYMENT_PROFILE (pending) | (as the construction smoke row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/construction-smoke-fit-plan-registered-v1.json | DRAFT; FIT cap 4 x 7 x 1 (raw rule, core/preparation_fit.py:1130-1137); owner countersignature pending; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- REGISTERED. TFRL-v2 iteration 1 trajectory sampling on the 625-row construction roster R01C: as the construction smoke row (program phase-native-module-v1-c7; lock / aa / s1..s4; rule-bank-i0), producing the per-row credit, contrast pairs and the exemplar distillation material of iteration 1; Fold A and Fold B are never knowledge sources; declared differences 8.1-8.7 (as the construction smoke row); no benefit claim from this column | (model/tool of the construction smoke row) | slurp SLURP-STAGE3-CONSTRUCTION-R01C 625 | discovery | f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515 | no | 8750 | 38176 | PROVISIONAL: ceil_0.1(1.3 x 625 x 3.60 x T / r / 60) NPU card-hours; T (cards per instance) and r (measured lock rows per minute per instance) from PORT_DEPLOYMENT_PROFILE (pending) | (as the construction smoke row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/construction-i1-generation-plan-registered-v1.json | DRAFT at study commit <commit>; plan SHA256 <plan sha>; generation cap 625 x (7 + 7 fallback); speech cap ceil(2,726.8471875 x 14); owner countersignature pending; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-FIT-NPU-PORT-V1 | bounded-discovery-probe | (scope of the construction i1 row) -- FIT spend identity | (model/tool of the construction smoke row) | slurp SLURP-STAGE3-CONSTRUCTION-R01C 625 | discovery | f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515 | no | 4375 | 38176 | (as the construction i1 row) | (as the construction smoke row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/construction-i1-fit-plan-registered-v1.json | DRAFT; FIT cap 625 x 7 x 1 (raw rule); nothing authorised. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-CREDIT-I1-NPU-PORT-V1 | model-free-check | Per-row credit of the i1 trajectories against lock (reward = row F1 of the trajectory minus row F1 of lock, from the sealed official per-row scores; counted only where lock and aa agree), the contrast pairs, and the exemplar distillation (each positive's evidence-to-answer path, keyed by scenario, capped at B_max per scenario). Zero model contacts. | no model or tool contact; local arithmetic over receipt-pinned files | slurp SLURP-STAGE3-CONSTRUCTION-R01C 625 | discovery | f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515 | no | 0 | 0 | 0 | No new speech exposure: reads only SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-NPU-PORT-V1 outputs. | prepared/SAEA-SLURP-TFRL-V2-CREDIT-I1-NPU-PORT-V1/credit.jsonl, contrast-pairs.jsonl, receipt.json (sha256 <filled after the run>) | DRAFT TEXT; model-free; authorises nothing. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-REFLECTOR-I1-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- REGISTERED. The contrast-pair reflector of TFRL-v2 iteration 1: at most one text-only contact per SLURP scenario that has at least one same-row positive/negative contrast pair (at most 18), each returning at most a bounded number of consumption rules that must cite their contrast-pair ids; a rule without a pair is refused by script; a deterministic curator ranks rules by script credit. No audio, no Fold A or Fold B material. | (model/tool of the construction smoke row, text-only, no audio part, no module program) | slurp SLURP-STAGE3-CONSTRUCTION-R01C (text material only) | discovery | f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515 | no | 36 | 0 | PROVISIONAL: 0.5 x instances x T NPU card-hours; T and instances from PORT_DEPLOYMENT_PROFILE (pending) | No new speech exposure. | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/reflector-i1-generation-plan-registered-v1.json | DRAFT; generation cap 18 + 18 answer-only fallback; its FIT row (calls 37 = 1 + 2 x 18, prepared-text rule core/preparation_fit.py:657-669) is SAEA-SLURP-TFRL-V2-REFLECTOR-I1-FIT-NPU-PORT-V1 with the same scope; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-BANK-RELEASE-I1-NPU-PORT-V1 | model-free-check | Release of the iteration-1 exemplar bank (knowledge/slurp/tfrl-v2/exemplar-bank-i1-npu-port-v1) and rule bank (knowledge/slurp/tfrl-v2/rule-bank-i1-npu-port-v1) as knowledge stores under the construction scope R01C, forbidden rosters Fold A v2 (1fe7908c...), Fold B v2 (47b89b9e...), DEV-1000 (653c7556...); both are pinned by the i1 Fold B baseline fragment, whose hash therefore moves. Zero model contacts. | no model or tool contact; local arithmetic over receipt-pinned files | slurp SLURP-STAGE3-CONSTRUCTION-R01C (scope only) | discovery | f56bc386776421aec2d8001d8697611b01c9b0562885a462759b15531e2f4515 | no | 0 | 0 | 0 | No speech exposure. | knowledge/slurp/tfrl-v2/exemplar-bank-i1-npu-port-v1/release.json, knowledge/slurp/tfrl-v2/rule-bank-i1-npu-port-v1/release.json (sha256 <filled after the run>) | DRAFT TEXT; model-free; authorises nothing. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-FOLD-B-SMOKE-4-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- REGISTERED. Four-row form smoke of the TFRL-v2 Fold B selection flight: A's lock harness plus the arm t-v2 (the iteration's exemplar bank as scenario-conditioned Solver supply, the iteration's rule bank, Solver vote n = 3 over the top-3 rules at temperature 0.6 / top_p 0.95 / top_k 20, commit saea-commit-vote-v1), the twin aa and the second replicate ab of lock, and the delivering lock arm last (program phase-native-module-v1-c10, 10 contacts per row), TRAIN Fold B; declared differences 8.1-8.7 (as the construction smoke row) | (model/tool of the construction smoke row, with module program phase-native-module-v1-c10 and the t-v2 sampling axis) | slurp TRAIN Fold B 4 | discovery | bdd025a1f1c0a4a1ebd1b1204e7d3e781c3ae2543db7be324420f85171576e32 | no | 80 | 204 | PROVISIONAL: 0.5 x instances x T NPU card-hours; T and instances from PORT_DEPLOYMENT_PROFILE (pending) | TRAIN Fold B discovery rows already exposed by prior SLURP cells (the historical L338 flight and its smoke included); this cell opens no split. Prior exposure is inherited in full. | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/fold-b-smoke-generation-plan-registered-v1.json | DRAFT; its FIT row SAEA-SLURP-TFRL-V2-FOLD-B-SMOKE-4-FIT-NPU-PORT-V1 carries calls 40 (4 x 10 x 1) and audio 204; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-FOLD-B-I1-NPU-PORT-V1 | bounded-discovery-probe | DRAFT -- REGISTERED. TFRL-v2 iteration-1 selection flight on TRAIN Fold B 1,080: as the Fold B smoke row with the i1 banks; read by the acceptance rule of docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2-cell-draft-v1.md section 7 (within-run bar 1.49 pp with the replicate-widened CI95, intent guard, per-row delta >= 0, twin within 1.20 pp, construction gain >= +1 pp) and confirmed only on Fold A; declared differences 8.1-8.7 | (model/tool of the Fold B smoke row) | slurp TRAIN Fold B 1,080 | discovery | 4407794c4a5d2e8883d93fdf242deb33c0987405d62e05a7d8416f5251a3484a | no | 21600 | 63096 | PROVISIONAL: ceil_0.1(1.3 x 1080 x 5.04 x T / r / 60) NPU card-hours; T (cards per instance) and r (measured lock rows per minute per instance) from PORT_DEPLOYMENT_PROFILE (pending) | (as the Fold B smoke row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/fold-b-i1-generation-plan-registered-v1.json | DRAFT at study commit <commit>; plan SHA256 <plan sha>; generation cap 1,080 x (10 + 10 fallback); speech cap ceil(3,154.797375 x 20); owner countersignature pending; nothing authorised. |
| 2026-10-30 | SAEA-SLURP-TFRL-V2-FOLD-B-I1-FIT-NPU-PORT-V1 | bounded-discovery-probe | (scope of the Fold B i1 row) -- FIT spend identity | (model/tool of the Fold B smoke row) | slurp TRAIN Fold B 1,080 | discovery | 4407794c4a5d2e8883d93fdf242deb33c0987405d62e05a7d8416f5251a3484a | no | 10800 | 63096 | (as the Fold B i1 row) | (as the Fold B smoke row) | docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/tfrl-v2/fold-b-i1-fit-plan-registered-v1.json | DRAFT; FIT cap 1,080 x 10 x 1 (raw rule); nothing authorised. |

The writer expands each "(as ...)" cell. Rows for i2 and i3 are written only after the previous iteration is read. They are
registered before their own contact, never in bulk.

## 10. Open items specific to T (hardest first)

- **H1. The thinking budget (M3) and the exemplar bank compete for the same 24,576-token window.** With R1 adopted, the
  completion ceiling becomes 6,144 + 1,024 = 7,168. The Solver limit falls from 17,406 to 24,576 - 3,072 - 7,168 - 2 =
  14,334 (limit basis in `l338_cost_inputs.json`). The measured lock Solver max is already 13,465, and 13,918 over the
  five-arm column. That leaves about 400-870 tokens for the bank and the rules. The options all move the base:
  - a larger per-sequence window (a protocol fact; it lowers the knee again);
  - a smaller supply (fewer train exemplars than k = 20, i.e. a change of A);
  - a tiny bank.

  This needs an owner and Fable decision before T3 sizes B_max.
- **H2. The knowledge-source guard (section 5.3).** The cross-fit halves are a design change to PLAN section 4. They halve
  the material per bank at t >= 2.
- **H3. Reading Fold B up to three times re-creates P2 on Fold B.** Confirming every accepted iteration on Fold A turns Fold
  A into a second selection fold. Fold A is not fresh either: ROLL-2 and VER-1 read it (L338 spec
  `docs/superpowers/specs/2026-10-19-slurp-338-lever-column-design.md`:42-45). Recommendation: use the construction gain as
  the per-iteration gate, read Fold B once per iteration, and confirm on Fold A once, for the final accepted state only.
- **H4. Rails features T3 must add before any T baseline composes:**
  - per-arm sampling temperature in a Solver template arm group;
  - a second twin (T2);
  - store roles `tfrl-exemplar-bank` and `tfrl-rule-bank` with scenario-conditioned supply;
  - commit policy `saea-commit-vote-v1` (its exact entity rule, "union-vote" in the PLAN, is to be fixed before contact; this
    draft assumes support >= 2 of 3);
  - B_max.
- **H5. The FIT cap rule (D draft F1).** The same rule applies here.
