# Operator flight steps for the Fold B NPU port (host 181 only)

Nine separate commands. Each one is run by a person; nothing advances to the next step by itself. The authority steps
(countersign, receipt mint, readiness flip) are the owner's and are NOT scripted. Run everything on `npu2` (host 181).
Host `npu` (175) is not ours.

```
ssh npu2
cd /mnt/cephfs/xingchao/omni/work/scripts/flight_steps
python3 preflight.py --stage smoke        # read-only report; exit 1 lists every blocker
bash step_<name>.sh                       # dry-run: prints the exact study command(s) and the gate verdict
bash step_<name>.sh --confirm             # executes (only if every gate passes)
```

## Order

| # | who | step | what it does | detached | marker (in `work/flight_state/fold-b-npu-port/`) |
|---|---|---|---|---|---|
| 0 | operator | `preflight.py --stage smoke` | registration check of the cell, `core.flight.cli check --read-only`, `npu-smi` (8 deployment cards resident and idle), `/health` + `/metrics` of every endpoint of the profile, runtime receipt agrees with the live containers (schema, ports, image id, newer than the newest container start), resident serving session record, CephFS free space, clean study tree, host is 181 | no | none |
| O | OWNER | countersign the DRAFT rows (`countersign-row-texts-v1.json`), mint the receipt (`l338_port_mint_receipt.py --dry`, then without), re-pin and flip with `repin_cell.py`, write the runtime receipt from the running servers, record the resident serving session | | | none; `preflight` turns green when they are done |
| 1 | operator | `step_lease.sh` | pins the study HEAD (`commit.txt`), starts `lease_cli --stage worker` detached in `saea-base` (`docker run -i --name saea-flight-lease`), waits for the token | holder only | `lease.done` |
| 2 | operator | `step_smoke_worker.sh` | `core.flight.cli worker` for the smoke-4 cell (120 generation calls; the native contact FIT runs inline). On exit 0 it sends `release` to the lease holder, on failure it sends `preserve` | yes | `smoke_worker.done` or `.failed`, `lease.released` |
| 3 | operator | `step_fit.sh` | FIT acceptance, read-only: FIT directory and flight receipts exist, then `batch_inspect.py smoke` (4 rows, ceilings, tool surface, study commit) | no | `fit.done` |
| 3b | operator | `step_smoke_read.sh [--read-id ID]` | the port smoke reader (`slurp_l338_port_column_read.py --stage smoke`) writes `analysis/<port READ id>/smoke-decode-cost.json` (write-once; nothing is scored). Needs `fit.done` | no | `smoke_read.done` |
| 4 | operator | `step_signoff.sh [--table P]` | `smoke_signoff.py` against `signoff-conditions-v1.json` over that table. Needs `smoke_read.done` | no | `signoff.done`, `signoff.marker` (PASS) or `signoff.marker.refused` |
| 5 | operator | `step_campaign.sh` | `core.flight.cli campaign` for the flight cell (1,080 rows); takes its own custody, no lease step. Needs `signoff.marker` | yes | `campaign.done` or `.failed` |
| 6 | operator | `step_seal.sh` | seal acceptance, read-only: `batch_inspect.py arm` (attempt FINALIZED, manifest COMPLETED at the pinned commit, 1,080 rows, ceilings, one score job). The campaign seals the run itself; there is no separate seal command | no | `seal.done` |
| 7 | operator | `step_score.sh [--read-id ID]` | `stage3_rollout_score.py --mode perception-branches --stage all`: one complete official scoring per arm column | yes | `score.done` or `.failed` |
| 8 | operator | `step_read.sh --recipe PATH [--reader SCRIPT] [--pass-recipe]` | refuses unless the recipe's READ run id has a root-countersigned ledger row (`read_gate.py`), then runs the reader | no | `read.done` |
| 9 | operator | `step_ledger_close.sh [--today D]` | `ledger_mark_consumed.py` dry-run, then `--apply` for the four run ids. Edits `docs/exposure-ledger.md`; commits nothing, the main thread reviews and commits | no | `ledger_close.done` |

Owner decisions between 0 and 1: the owner countersigns, mints and flips (README of the package, "What the main thread
does"). `preflight.py` shows which of them is still missing: `not-ready`, `row-not-countersigned`, `receipt-missing`
come from `flight_registration_check.py`; `runtime-receipt` and `resident-session` from the serving side.

## Gates every step enforces (with `--confirm`; without it they are only reported)

1. `<step>.done` already exists: refuse (exit 5). Remove the marker by hand to re-run.
2. The previous step's `.done` marker is absent: refuse (exit 3).
3. `preflight.py` fails: refuse (exit 4) and print its report. Steps 1, 2 use `--stage smoke`, step 5 uses `--stage flight`
   (full checks). Steps 3, 4, 6-9 run after a flight has consumed its rows, so they use `--stage post` (offline subset:
   profile, host, clean tree, commit pin, free space); the live checks would be wrong by then.
4. The study HEAD differs from `commit.txt` (written by `step_lease.sh`): refuse (exit 7). All steps fly from one commit.
5. Step-specific inputs (lease token, sign-off marker, decode-cost table, recipe): refuse (exit 6).

Exit codes: 0 ok or dry-run, 2 usage, 3 previous marker absent, 4 preflight failed, 5 done or running, 6 other refusal,
7 commit moved, 8 the step's own command failed.

## Where things land

| what | where |
|---|---|
| markers, logs, pin | `/mnt/cephfs/xingchao/omni/work/flight_state/fold-b-npu-port/` (`<step>.done`, `.failed`, `.running`, `.log`, `commit.txt`, `lease.fifo`, `lease.log`, `signoff.marker`) |
| lease token | `$OMNI/coordination/<smoke cell>.worker-lease.json` (removed on release) |
| run bytes, attempt records | `$OMNI/runs/<run>/<attempt>/`, `$OMNI/attempts/<run>/` |
| FIT evidence | `$OMNI/prepared/<FIT run>/<FIT attempt>/` |
| per-arm official scores | `$OMNI/analysis/<READ run id>/columns/` |
| read outputs | `$OMNI/analysis/<READ run id>/` |
| ledger | `docs/exposure-ledger.md` in the study tree, uncommitted after step 9 |

`$OMNI` = `/mnt/cephfs/xingchao/omni`. Study commands run in `saea-base` through `work/scripts/port_py.sh` with
`SPEECHRL_DATA_DIR=$OMNI` on cpuset `160-191` (override with `CPUSET=...`; never `128-159`, which is the cpuset of
instance `saea-omni-i1`).

## Watching a detached step

`worker`, `campaign` and `score` return at once. Watch the watchdog (`work/watchdog/status-npu2.json`, `events-npu2.log`),
not the raw log. The step's own `<step>.running` holds the pid of the detached shell; `<step>.done` or `<step>.failed`
appears when it ends. A failed worker leaves the lease preserved: reconcile it by hand (`lease_reconcile_*.sh` pattern)
before any new lease.

## Known limits (read before the first `--confirm`)

- The smoke decode-cost table (`analysis/<port READ id>/smoke-decode-cost.json`) is written by the port reader
  (`scripts/analysis/slurp_l338_port_column_read.py --stage smoke --data-root $OMNI`, run in saea-base through `port_py.sh` after the
  smoke is sealed); `step_smoke_read.sh` runs it (dry-run by default) and `step_signoff.sh` refuses until `smoke_read.done` and the table exist (or you pass `--table`). A null `registered_gpu_hours_ceiling` in the table is not read by any sign-off condition. `step_read.sh` runs the same script with
  `--stage read` and writes `read.json`, `README.md` and `port-verdicts.json` (parity, twin admissibility, improvement) beside it.
- FIT is not a separate model run: the cells permit only `check`, `prepare-supply`, `worker`, `campaign` (no `prompt-fit`),
  and the native contact FIT is an inline gate of the worker and campaign. `step_fit.sh` therefore verifies the evidence.
- `--stage smoke` preflight will report `resident-session` until a session record exists at
  `$OMNI/serving/<service_session>/session.json`; creating it is the multi-endpoint transport and session work (gaps A-D),
  not a step.
- `step_lease.sh` needs `PORT_DOCKER_FLAGS` support in `port_py.sh` (added 2026-10-30, default empty; the pre-change copy is
  `work/scratch/port_py.sh.pre-flight-steps`).

## Files

`preflight.py`, `cellinfo.py` (identities from the committed fragments), `fit_evidence.py`, `read_gate.py` (runs in the
container), `_common.sh`, `_detached.sh`, `_lease_holder.sh`, `step_*.sh`, `test_flight_steps.py` (tests, fake ops, CPU
only: `ssh npu2 'cd .../flight_steps && taskset -c 176-191 python3 -m pytest -q -p no:cacheprovider test_flight_steps.py'`).
