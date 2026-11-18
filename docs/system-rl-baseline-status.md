# System RL baseline and startup repair

Numerical calibration remains frozen: 98/98 language boundaries passed, with zero measured token/sequence log-probability difference using reference audio features. Natural-audio integration, optimizer updates and adapter export/reload remain separate acceptance items.

As of the latest observation, the existing accelerator trajectory capture is running. Accepted optimizer updates remain zero; DEV improvement is unknown. The startup repair candidate has not replaced the active release.

## Root causes and systematic checks

|Class|Verified failure pattern|Repair and verification|
|---|---|---|
|Dependency closure|Dynamic imports and host-only package references missing from remote bundles|Sealed packages; actual fixed-image imports and isolated consumers|
|Final configuration|Validated release differed from the final template; input representations confused|Original launcher validates resolved deployment and snapshots the HTTP template|
|Content identity|Working-directory dependence and cross-host inode/device mismatches|Explicit roots; content/source identity separated from local read stability|
|Data contracts|Real-only assumptions and missing synthetic/incremental bindings|Unified domain bindings; real/synthetic and held-out identity checks|
|Lifecycle|Aggregate budgets leaked into operation timeouts; terminal states inconsistently consumed|Independent operation timeouts; consistent terminal accounting; unknown responses never replayed|
|Acceptance scope|Component checks were treated as whole-pipeline proof|Explicit static, CPU-consumer, accelerator, update and DEV evidence boundaries|

The main integration-verification design allowed CPU-detectable defects to reach accelerator startup. The revised fixed regression entry aggregates failures by category. Eight groups comprising 69 checks pass in the actual fixed image: final configuration, journal imports, index relocation, controller, terminal accounting, domain handling, receipt binding and training API imports. Initial checker failures remain recorded; incorrect source paths and incomplete import environments were repaired. Import tests map no devices and load no model; vendor libraries may query device availability. These results do not prove accelerator kernels or training gains.

The original configuration compiler and launcher also completed isolated preparation successfully. The HTTP template is copied into immutable launch inputs and consumed by the original preparation path. No second scheduler was introduced. Remote end-to-end training and long-run capacity admission are still outstanding.

## Data and final comparison

TRAIN contains 119,881 recordings across real and synthetic domains. Training admission requires the gold intent to be present in the actual pre-P retrieval result. The frozen 5,000-draw design uses 80% error trajectories and 20% successful trajectories, with fresh behavior groups. It has not completed training. The expanded CPU preparation audit covered 128 records: 127 initially passed and one failed case passed a targeted repair; original failures are preserved.

DEV uses all 8,690 recordings for selection. Final TEST uses all 13,078 recordings and compares A: direct unchanged Instruct, B: frozen TFRL system, and C: the same system with DEV-selected LoRA. Report SLU-F1, intent accuracy, joint exact, valid-answer rate and cost, including B−A, C−B and C−A. Training eligibility does not filter evaluation denominators. The direct-model full-population backend and final results remain pending.

## Storage boundary

Full sampled-training diagnostic capacity is still not admitted. Excess storage is diagnostic logits, not repeated base-model checkpoints. Candidate retention and verified restoration mechanisms have CPU evidence, but resident-source retirement and permanent archive capacity remain incomplete. No production deletion or reclaimed-capacity credit is claimed. The active finite first-update run remains independent.
