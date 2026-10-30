# Bad case: `verify_declared_module` refuses the L338 Fold B NPU-port flight (2026-10-30)

Analyst note (Opus 5.5). Read-only: no model call, no card, no source edit, no commit. Study HEAD `389686d6`.
Scope: the THIRD refusal of the port flight `check` on
`configs/experiment/slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1.json`. The first two refusals are
separate: the exposure-ledger rows (c15 vs c17; see `ledger-successor-draft-c17.md`) and the missing runtime receipt
`docs/receipts/runtime-vllm-bf16-thinking-npu.json`.

## 1. What the check is

- `rails/consumption_module.py:2382-2402` `verify_declared_module(raw_block, values)`. It parses the
  config's `native_episode.consumption_module` block. It resolves the carrier hooks (`profile: slurp` ->
  `rails/adapters/slurp_module_hooks.py`) and compiles the block onto the LOCK LAYER the block names. It then diffs
  `composed.values_on_roster_of(values)` against `policy_owned_values(values)` (`:1492`). The diff covers every
  leaf of `POLICY_OWNED_KEYS` (`:68`: `policy`, `knowledge_stores`, `native_episode`, `phase_material`). Any
  difference raises (`:2399`).
- Caller on the flight path: the config loader, `rails/native_episode_config.py:1916-1926`. Any config that declares a
  module block reaches it. The `check` CLI loads the cell config, so the refusal is raised there.
- "Lock layer" here means the carrier lock of the evidence-consumption module. It is NOT `docs/datasets.lock.json`.
  The port block names `"lock": "v2.2"` (port baseline fragment line 266). `consumption_module.py:1620-1624` selects
  `lock_versions["v2.2"]`. That is `slurp_module_hooks.py:216-217` = v2.1 + the v4g normaliser
  (`slurp_module_lock_v2_2.py:66`). v2.1 is v2.0 + the G5 Reviewer template. v2.0 is the policy-owned keys of the
  historical lock fragment `configs/baseline/slurp-native-default-plan-lock-baseline-v1.json`
  (`slurp_module_hooks.py:103, 185-189`). Neither v2.1 nor v2.2 touches `knowledge_stores`. So the compiled
  `knowledge_stores` are the historical 268-B pins byte for byte.
- The only exemption is `ROSTER_PIN_PATHS` (`slurp_module_hooks.py:123-126`). `_rebind_roster`
  (`consumption_module.py:1478`) takes these two leaves from the cell, not from the lock:
  `knowledge_stores.intent-boundaries.phase_supply.perception.query_source` and
  `knowledge_stores.train-exemplars.phase_supply.solver.query_source`. Every other store pin must equal the lock's.

## 2. The failing comparison (reproduced)

Reproduced in `saea-base` on npu2 with `CPUSET=176-183`, CPU only, through `port_py.sh` and a scratch script
(`/mnt/cephfs/xingchao/omni/scratch/badcase_vdm/repro.py`). The script calls `verify_declared_module` on the baseline
fragment `slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1`. Result:
`ConsumptionModuleConfigError ... differ ... at [10 leaves]`. The control,
`slurp-native-stage3-l338-lever-column-fold-b-baseline-v1` (historical 268-B), PASSES. So does its smoke-4 twin.

| leaf (`knowledge_stores.` ...) | compiled from lock v2.2 (lock fragment line) | declared by port fragment (line) |
|---|---|---|
| `intent-boundaries.release.relpath` | `.../semantic/intent-boundaries-v1/release.json` | `.../intent-boundaries-npu-port-v1/release.json` |
| `intent-boundaries.release.sha256` | `2e51c14b...beb4` (:46) | `22242e83...dc06` (:53) |
| `intent-boundaries.snapshot.relpath` | `.../intent-boundaries-v1/snapshot.json` | `.../intent-boundaries-npu-port-v1/snapshot.json` |
| `intent-boundaries.snapshot.sha256` | `5520b844...48fe` (:50) | `cb6ef966...eabc` (:57) |
| `slot-contracts.release.relpath` | `.../semantic/slot-contracts-v2/release.json` | `.../slot-contracts-npu-port-v2/release.json` |
| `slot-contracts.release.sha256` | `f8e42d4f...d11b` (:70) | `2f11a748...c63c` (:77) |
| `slot-contracts.snapshot.relpath` | `.../slot-contracts-v2/snapshot.json` | `.../slot-contracts-npu-port-v2/snapshot.json` |
| `slot-contracts.snapshot.sha256` | `d5992bd1...bdec` (:74) | `e035c1b8...2a5e` (:81) |
| `train-exemplars.phase_supply.solver.facet_source.sha256` | `5dfe7523...fc1e` (:89) | `24f0cdf7...c2c4` (:96) |
| `train-exemplars...facet_source.tool_contact_ledger_row` | `SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-V1` (:90) | `...-NPU-PORT-V1` (:97) |

The facet_source relpath is the same on both sides (`vectors/slurp-kus3/facet-vectors.jsonl`); only its hash and
ledger row moved.

Where the port values come from. The port fragment got them from the re-pin commit `7db9d7cd`. They were extracted
once into the writer's pins `scripts/flights/nx/l338_port_pins.json` (`_note` line 2; values at :81-88, :105-112,
:122-127). `scripts/flights/nx/l338_port_derive_cells.py:236-240` writes `knowledge_stores` from those pins. The
writer KNEW about the clash and stepped around it. At `:204-207` it deliberately does not call `composed.apply_to(base)`,
"that replaces EVERY policy-owned key with the lock layer's, including the knowledge-store pins the port re-authored".
No writer step ran `verify_declared_module` on its output. `registered-facts-v1.json` carries no store hashes. It
declares the move only as prose differences 8.4 and 8.6 (lines 141-149).

The port values are internally consistent on disk (data root):
- release `22242e83...` and `2f11a748...` = file sha256.
- The snapshot pins are CONTENT hashes by design (`rails/knowledge_stores.py:403-404`:
  `snapshot.content_hash()` and `release["snapshot_sha256"]`). Both release files carry `cb6ef966...` / `e035c1b8...`.
  The file sha256 of the snapshots (`08976279...`, `557080ad...`) is correctly NOT what is pinned.
- `vectors/slurp-kus3/facet-vectors.jsonl` hashes to `24f0cdf7...`, matching its receipt
  `receipts-npu-port-v1/vectors.json`.
- The historical `intent-boundaries-v1/` and `slot-contracts-v2/` files are ABSENT from our data root. The historical
  kus3 file (`5dfe7523...`) is not on disk: the port vectors sit at the same relpath.

## 3. Root cause, ranked

1. **(a), but in the reverse direction: CONFIRMED.** The port writer pins the RE-AUTHORED stores (correct, per
   decided differences 8.4/8.6 and decision 9.5). The module declaration still names `lock: v2.2`, and the v2.2 layer
   carries the HISTORICAL 268-B stores. There is no port lock version, so the compiler restores the historical pins
   and the diff fires. This is a missing layer, not a wrong pin.
2. **(b) re-pinned lock layer: REJECTED.** The lock fragment was last touched in `fba45821` (2026-10-25). That is
   before the port re-pin `7db9d7cd` (2026-10-28 20:37) and the package `a50581d4` (2026-10-28 22:21).
   `slurp_module_lock_v2_2.py` and `slurp_module_hooks.py` have no port-era commits.
3. **(c) path or hash-format false alarm: REJECTED.** The relpaths really differ, and so do the store directories.
   Each side's hashes are self-consistent in their own format (file sha256 for release and vectors, content hash for
   snapshot). The 2026-10-30 inventory confusion (file sha vs content hash) does not apply here.

## 4. Smallest correct fix and its owner

- **Owner: the lock layer** (the carrier hooks, `rails/adapters/slurp_module_hooks.py` `lock_versions`, `:249`).
  Change nothing in the writer pins: they are correct and match disk.
- Fix: register a port successor lock, e.g. `v2.2-npu-port`. Its layer = `lock_layer_v2_2()` with exactly these
  leaves replaced, and it should be fail-closed on the bytes, like `lock_layer_v2_1`:
  - `intent-boundaries` release `22242e83...dc06`, snapshot `cb6ef966...eabc`
    (`knowledge/slurp/semantic/intent-boundaries-npu-port-v1/`)
  - `slot-contracts` release `2f11a748...c63c`, snapshot `e035c1b8...2a5e`
    (`knowledge/slurp/semantic/slot-contracts-npu-port-v2/`)
  - `train-exemplars.phase_supply.solver.facet_source` sha256 `24f0cdf7...c2c4`, ledger row
    `SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-NPU-PORT-V1` (also its `embedded_signal`, if the port text differs)
- The module declaration then moves one key: `"lock": "v2.2"` -> `"lock": "v2.2-npu-port"` in every port block.
  That changes `config_block_sha256` and the `composed_config_hashes` in `registered-facts-v1.json`
  (arm `8ccf5578...`, smoke `5d528040...`), so the port packages must be re-minted (rows stay DRAFT).
- Rejected alternatives:
  - Adding the store pins to `ROSTER_PIN_PATHS` would exempt knowledge from the check for EVERY SLURP cell. It
    would also mislabel knowledge as roster identity.
  - Dropping the module block from the port cells would remove the guard.
  - Using `apply_to` would restore the historical stores.
- Registry note: the TFRL writer's finding (`scripts/flights/nx/tfrl_v2_port_write_registration.py:136-138`;
  T `registered-facts-v1.json:459`) attributes the remedy to "owner decision 9.4". That is a misattribution:
  decision 9.4 is the `datasets.lock.json` amendment (`docs/readiness/2026-10-28-npu-port-lock-entries-v1/`). A
  module lock version needs its own registry entry. It records decided differences 8.4/8.6 (decision 9.5), so it
  asks for no new science.

## 5. Do the Fold A, D and T packages hit the same refusal?

Same sweep, all baselines named in each package's `registered-facts-v1.json`:

| package | baseline fragments | `verify_declared_module` |
|---|---|---|
| Fold A confirmation (`2026-10-30-slurp-l338-fold-a-confirmation-npu-port-v1`) | `...-fold-a-baseline-raw-port-v1` | FAIL, the same 10 leaves: **blocked** |
| D, direct-only (`2026-10-30-slurp-direct-only-fold-b-npu-port-v1`) | `slurp-direct-only-fold-b{,-aa,-ab,-smoke-4}-baseline-raw-port-v1` | not reached: no `consumption_module` block. **Not blocked** by this refusal |
| T, TFRL-v2 Fold B (`2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1`) | `...-tfrl-v2-fold-b-{i1,smoke-4}-baseline-raw-port-v1` | FAIL, the same 10 leaves: **blocked** |
| T, construction R01C | `...-construction-r01c-{i1-h1,i1-h2,smoke-4}-baseline-raw-port-v1` | FAIL, 14 leaves: the 10 above + `train-exemplars.{release,snapshot}.{relpath,sha256}` (the minus-r01 twin `83620eb0...` / `6eaa03ae...` vs lock `171aa4a8...` / `4c498dc0...`): **blocked** |
| (not flown) port comparators, Fold A and Fold B | `...-comparator-npu-port-v1` | FAIL, 15 leaves: the 10 + the written-key `facet_source` (port `30e29867...` vs `76e8d16a...`) and `perception.k` 60 vs lock 8 |

- **T.** The T writer's claim "none for T" (`tfrl_v2_port_write_registration.py:138`) is wrong for the flight
  `check`. T avoids `apply_to`, but the LOADER (`native_episode_config.py:1924`) still verifies every declared block.
- **T construction.** The construction cells also need the minus-r01 store pins. They need either the `r01c`
  construction variant generalised to the port names (T finding at `registered-facts-v1.json:449`), or a second port
  lock leaf set.
- **Comparators.** The comparators keep `k: 60` with every component off. The lock compiles 8. This is a separate
  check-only defect of a never-flown input, and does not block any flown cell.

## Hand-off

The fix is a lock-layer source change (new lock version plus test), followed by a re-mint of the L338, Fold A and T
port packages. That belongs to an engineer (Sonnet) once Fable has decided. Nothing here authorises a model call or
a readiness flip.
