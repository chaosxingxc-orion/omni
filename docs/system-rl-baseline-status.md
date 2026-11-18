# System RL baseline and startup repair

Numerical calibration remains frozen: 98/98 language boundaries passed, with zero measured token/sequence log-probability difference using reference audio features. Natural-audio integration, optimizer updates and adapter export/reload remain separate acceptance items.

Current observation (2026-11-18 13:58 UTC): fresh diagnostic capture has closed 4 of 8 planned episodes and 1 of 2 groups. Closed episodes contain 50 calls (P 4, S1 40, S2 6, D 0), 3,796 output tokens, 17 valid terminals, 22 protocol errors and 11 length truncations. Incomplete episodes are excluded. Accepted optimizer updates remain zero; DEV reward gain is unknown. A subsequent resource observation found all eight training-host AI Cores at 0%, with two ordinary services still resident.

## Critical path to the first update

A controlled handoff is now underway: the original lease drain has closed new capture admission without killing or replaying the in-flight request. The coordinator has exited through the original fence, the diagnostic service exited successfully, and the original service has restarted; restoration certification and ledger reconciliation are still pending. A sealed successor adds terminal-event wakeup for ordinary remote stages, retaining the original closed-result verifier and 900-second backoff on observation failure. Five wakeup checks and nine original controller checks pass; resident first-command polling is unchanged. A fixed journal-inventory script reproduces the same 13 historical actual requests with zero unresolved outcomes and no ledger mutation. No optimizer update is claimed.

The current controller has five remaining stages: close diagnostic generation, run the eight-device actor diagnostic, generate the eight-call P training cohort, finalize current/reference same-action scores, then perform the optimizer update. Adapter export/reload and updated-policy acceptance follow. This is finite engineering acceptance, not completion of the frozen 5,000-draw training dataset. Actual actor-diagnostic/finalize/update timings are not yet available.

The dominant scheduling defect is that numerical diagnostics force a complete system rollout, while D generation depends on successful upstream business output. Independent frozen-node input mode is now explicitly configured in a sealed successor. The original selector, Clock, official annotation checks and endpoint/capture transport preparation pass in the fixed image (70.221 seconds). Isolated original consumers also regenerate all four node inputs exactly for two actual TRAIN records: P1/S1 10/S2 4/D1 and P1/S1 10/S2 2/D1. No historical response is substituted for fresh behavior. This mode still implies 120 fresh calls at G4; the 16-call full-logits quota is an observation quota, not the total sampling workload. The successor is not deployed, and fresh accelerator generation/update remain unaccepted. Original request completion and reconciliation must precede service handoff.

A sealed, undeployed simplification candidate passes 22 targeted fixed-image checks: lease-local incremental accounting replaces per-request historical packet rescans; bounded prompt-length caching avoids repeated processing of identical audio/messages; node training filters its target packets before hydration and avoids unrelated S2/D consumers; diagnostic selection now matches the existing finite-versus-strict coverage gate. Original request reconciliation, durable writes, numerical thresholds and strict coverage remain enforced. These CPU checks do not prove accelerator speedup or training success.

Source inspection confirms duplicate original-control operator work, synchronization, and serialized diagnostic generation. Full-vocabulary capture is already restricted to parity requests, not all training calls. No infinite MoE wrapper recursion was found. Historical dependency traversal and repeated source-tree copying are separate engineering costs. Disabling control computation requires output and state/cache equivalence checks; it has not been done.

Two otherwise idle four-device ordinary services completed a ten-request efficiency measurement with the same real audio input: two simultaneous requests took 1.539 seconds, then eight warm-cache requests took 0.675 seconds (individual latency 0.433–0.642 seconds). Services were not modified. Different arithmetic/image settings and repeated warm input prevent interpreting this as an aligned-runtime speedup or quality result. The active diagnostic service still reports approximately 0.3–0.4 output tokens/s with one running request.

The held-out transport candidate passes with zero closure errors, 596 mappings and 178 read-only inputs. Fresh child processes restore closed inputs and preserve control-host placement. Actual full held-out evaluation remains pending. Frozen numerical evidence remains valid only within its measured scope.

The comparison contract is A: direct Instruct, B: frozen TFRL system, C: the same system with DEV-selected LoRA. DEV uses all 8,690 recordings and final TEST all 13,078. Full TEST comparison is not a prerequisite for the first finite optimizer update.

## Root causes and systematic checks

|Class|Verified failure pattern|Repair and verification|
|---|---|---|
|Dependency closure|Dynamic imports and host-only package references missing from remote bundles|Sealed packages; actual fixed-image imports and isolated consumers|
|Final configuration|Validated release differed from the final template; input representations confused|Original launcher validates resolved deployment and snapshots the HTTP template|
|Content identity|Working-directory dependence and cross-host inode/device mismatches|Explicit roots; content/source identity separated from local read stability|
|Data contracts|Real-only assumptions and missing synthetic/incremental bindings|Unified domain bindings; real/synthetic and held-out identity checks|
|Lifecycle|Aggregate budgets leaked into operation timeouts; terminal states inconsistently consumed|Independent operation timeouts; consistent terminal accounting; unknown responses never replayed|
|Acceptance scope|Component checks were treated as whole-pipeline proof|Explicit static, CPU-consumer, accelerator, update and DEV evidence boundaries|

The main integration-verification design allowed CPU-detectable defects to reach accelerator startup. The revised fixed regression entry aggregates failures by category. The original eight groups comprising 69 checks passed in the actual fixed image: final configuration, journal imports, index relocation, controller, terminal accounting, domain handling, receipt binding and training API imports. Initial checker failures remain recorded; incorrect source paths and incomplete import environments were repaired. Import tests map no devices and load no model; vendor libraries may query device availability. These results do not prove accelerator kernels or training gains.

The original configuration compiler and launcher also completed isolated preparation successfully. The HTTP template is copied into immutable launch inputs and consumed by the original preparation path. No second scheduler was introduced. Remote end-to-end training and long-run capacity admission are still outstanding.

## Data and final comparison

TRAIN contains 119,881 recordings across real and synthetic domains. Training admission requires the gold intent to be present in the actual pre-P retrieval result. The frozen 5,000-draw design uses 80% error trajectories and 20% successful trajectories, with fresh behavior groups. It has not completed training. The expanded CPU preparation audit covered 128 records: 127 initially passed and one failed case passed a targeted repair; original failures are preserved.

DEV uses all 8,690 recordings for selection. Final TEST uses all 13,078 recordings and compares A: direct unchanged Instruct, B: frozen TFRL system, and C: the same system with DEV-selected LoRA. Report SLU-F1, intent accuracy, joint exact, valid-answer rate and cost, including B−A, C−B and C−A. Training eligibility does not filter evaluation denominators. The direct-model full-population backend and final results remain pending.

## Storage boundary

Full sampled-training diagnostic capacity is still not admitted. Excess storage is diagnostic logits, not repeated base-model checkpoints. Candidate retention and verified restoration mechanisms have CPU evidence, but resident-source retirement and permanent archive capacity remain incomplete. No production deletion or reclaimed-capacity credit is claimed. A finite first-update recovery remains separate from full sampled-training capacity admission.

## Classified repair follow-through

The fixed suite now includes a ninth group that consumes 12 retained actual P/S1/S2 responses and rejects paths outside the configured journal roots. D coverage is explicitly a fixture. All nine CPU groups passed on the integrated repair candidate; this does not certify fresh D generation or an optimizer update.

Actual transport preparation exposed another configuration/identity defect before accelerator dispatch: active business code pins still referenced the previous release. The original runtime builder now derives executable pins from the sealed source manifest, preserves external frozen-package/data/knowledge identities, and records a migration receipt. Six targeted checks passed. A related transport fix treats the frozen system document as identity evidence during capture while validating the actual derived business dependency graph. The latest candidate completed fixed preparation; actual transport and isolated consumer validation are ongoing. Original failures are retained, and no incomplete release is promoted.

The next local launch exposed the same frozen-system pin issue in endpoint preparation before any remote stage or model request. Endpoint and capture identity handling are now unified, and the fixed preparation checker exercises both transport graphs. Both pass on the current recovery candidate, as do nine classified CPU groups. The current isolated real/synthetic consumer check is still running; no optimizer update is claimed.

The same-category review also found stale executable references in the DEV/TEST policy derivation. A separate candidate reuses the sealed executable binder for business, system and module references; seven checks on actual frozen configurations and nine original controller checks pass. This candidate is not deployed. Fourteen original resident lifecycle checks also pass after updating stale test fixtures to the current phase contract; this is CPU evidence and gives no production storage-reclamation credit.

Current recovery update: the exact isolated real/synthetic consumer check passed. The fixed launcher staged and started the remote endpoint controller successfully, and remote Docker state confirms it is running. This passes the previous pre-start failure point. Accepted optimizer updates remain zero and DEV gain remains unknown; routine progress observation uses the configured 15-minute cadence.

## Classification-first review update

The common failure mechanism is disagreement between preparation, transport, execution and response-consumption dependency boundaries. Repairs are tracked by dependency closure, final configuration, content identity, data contracts, lifecycle and acceptance scope; a component pass does not close an entire category.

A DEV preparation audit exposed an additional configuration defect in the audit itself: its interpreter used a newer release than its configured dispatch bundle. The CPU-only audit was stopped after traversing 33,800 historical references, with partial evidence retained and no remote dispatch. A candidate guard now compares 145 execution modules, source manifests and runtime configuration against the dispatch bundle before dependency traversal or accounting snapshots. Seven checks pass, including the original packer and actual mixed-release rejection; nine original controller checks also pass. The original preparation entry now rejects the mismatch before traversing history. This candidate is not deployed and does not change frozen numerical calibration.

DEV/TEST transport closure remains incomplete: current traversal pulls in old training artifacts. Its repair must preserve actual business/knowledge consumers, predecessor terminal reconciliation and resource guards. Simply treating all references as leaves would hide missing dependencies. Full held-out evaluation, a real optimizer update, adapter export/reload and permanent diagnostic-capacity admission remain outstanding.

Held-out consumer review update: the actual original controller source/config and policy-hook checks now run before preparation publishes READY. Fixed-image namespace validation found one missing pinned controller settings template, which was added without changing its historical identity. Preparation packages the current release and derives a new execution configuration, uses the existing target-local output-path generator, and checks the requested dataset against the actual held-out split. Eight positive/negative contract checks and nine original controller regressions pass. The fixed audit runner reuses launcher mounts, makes prior outputs read-only, and exposes only a fresh writable audit directory, with no network, devices or Docker socket. These changes remain an undeployed candidate; remote dependency closure and actual evaluation are not yet accepted.

Latest runtime event: the active endpoint controller completed with exit code zero. The original client consumed a non-fixture complete response and automatically advanced to fresh parity capture. The original closed-result reader verified 22 files (323,210 bytes), without additional model calls. The next capture has not yet been confirmed started; accepted optimizer updates remain zero.


## Current classification-driven self-check

A complete CPU-only dependency audit aggregated 1,011 errors rather than stopping at the first missing file. These are error records, not 1,011 independent causes. The fixed configurable classifier retains every original chain and source index, without model calls.

Actual consumer inspection showed that projection readers open shard tasks and bindings, while the business session restores sealed selected inputs without invoking the legacy collection constructor. Shared typed traversal now follows those readers. The original transport audit decreased from 1,011 to 133 errors and from 16,487 to 429 mappings. It still fails and is not eligible for deployment. Remaining groups are cache lineage (70), materializer lineage (36), held-out input views (13), an old endpoint fixture's evidence (5), and other dependencies (9).

A separate lifecycle defect was found before remote execution: newly derived policy programs inherited a one-time failed-start migration restricted to an old run and budget. The candidate records that original lineage without replaying its migration; original predecessor terminal reconciliation, unknown-request checks and resource guards remain in force. Policy-derived programs retaining the obsolete migration are rejected before resource acquisition.

Validation: six projection checks authenticate 128 shards and 256 runtime references, ten original held-out preparation/configuration checks pass, and nine original controller regressions pass. These CPU results do not establish remote evaluation or training success. The candidate remains undeployed. Numerical calibration, business model/prompts, knowledge and official evaluation denominators are unchanged.
