# Fold B: comparison baseline and improvement plan (Fable 5.1, 2026-10-30)

Owner redirect (2026-10-29, translated): "what you must finally compare is the performance metrics on the task dataset,
and the improvement in those metrics, against the repo's baseline." This plan sets the comparison baseline, the bars,
the engineering path to the first scored flight, and the lever ladder. The serving work of
`2026-10-29-lock-engineering-base-and-efficiency-plan.md` is the instrument; it is referenced, not repeated.
Inputs: `work/scratch/brief/2026-10-29-comparison-baseline-fold-b.md` (Opus, sealed-record read) and the engineering
readiness report (Sonnet, 2026-10-30; scratch `/mnt/cephfs/xingchao/omni/scratch/eng-readiness/`).
Grants nothing: no readiness flip, no exposure entry, no model call on a dataset row until the owner countersigns.

## 1. The task, the metric, the baseline

| item | value | source |
|---|---|---|
| dataset | SLURP TRAIN Fold B, 1,080 utterances (18 scenarios x 60; 527 headset, 553 far-field; 3,154.8 s audio) | `configs/dataset/slurp-speech-native-evo-fold-b-1080.json` |
| task per row | scenario, action and entities from speech (SLU) | preregistration 2026-10-19 |
| program | `phase-native-module-v1-c15`: 15 contacts per row over 4 arms (lock 2, aa 2, l1-support 6, l2-union 5) | fit plan registered |
| primary metric | official SLURP SLU-F1 (pooled micro, pinned `evaluate.py`) | `analysis-recipe-v1.json` (baseline package) |
| secondary | intent accuracy (exact scenario_action) | same |
| within-run bar | arm vs `lock`, row-paired, CI95 from 10,000 bootstrap draws (seed 20261018); floor 1.49 pp SLU, 0.65 pp intent | same + `stage3_effect_floor_check.py` |
| cross-run bar | >= 2.5 pp SLU at n = 1,080 AND a duplicate twin arm in the same run | effect-floor registration |

**Decision 1 (comparison reference).** The L338 Fold B cell was never flown (ledger rows 9144-9146 unconsumed), so
it has no number. Its `lock` arm is byte-identical to the 268-B Thinking system, whose Fold B readings are the baseline:

| reference | SLU-F1 | intent | source |
|---|---|---|---|
| recipe-pinned reference (268-B Thinking lock arm) | **71.61** | **85.00** | `analysis-recipe-v1.json` reference pin |
| noise band, 8 readings of the same system | 71.61-72.75 | 84.91-85.74 | column reads 2026-10-09..25 |
| nearest same-lock read (R3, 2026-10-25) | 72.04 | 85.28 | `docs/results/2026-10-25-slurp-nxt-r3-solver-template-column-read.md` |

The port is read against the BAND, never against a point: "parity" means the port lock arm lands inside
71.61-72.75 SLU / 84.91-85.74 intent; "improvement" means an arm clears the cross-run bar above the band's top
(>= 75.25 SLU) with its lower CI95 above zero against the port's own lock arm, and the twin arm agrees.

**What "improvement" has meant here.** Instruct -> Thinking: +7.15 SLU [+5.08, +9.26], +2.31 intent (the only lever
that ever cleared a bar). R1-R3 solver/template columns and the training-free RL loop: neutral or refused. The lever
column L338 (aa, l1-support, l2-union retrieval arms) is registered and unflown. Expectations below are calibrated to
this history: most levers are neutral; the plan budgets for reading, not for winning.

## 2. Stages and bars

| stage | question | arms | bar | cards |
|---|---|---|---|---|
| S0 parity | does BF16/vLLM-Ascend land in the band? | port `lock` + `lock-twin` (same protocol, second seed) | both inside the band; twins within 1.1 pp | 1 flight (8.5 h bound on 4 instances) |
| S1 lever column | the registered L338 column on the port base | aa, l1-support, l2-union vs lock (the registered cell) | within-run bar per arm | same flight (all 4 arms are one run) |
| S2 new levers | levers the BF16/vLLM base makes affordable | one lever per registered cell, each with a twin | within-run bar; cross-run bar for the headline | 1 flight per lever |

S0 and S1 are ONE flight: the registered port cell already carries the four arms. The twin arm for S0 is a protocol
addition (a second `lock` seed) and must be registered before the flip (Decision 4). If S0 fails the band (port lock
outside 71.6-72.8), S2 does not start: the deviation under test (BF16, vLLM, raw audio, regenerated vectors) is read
first, by arm and by contact node, and the owner decides whether the port is the comparison base at all.

## 3. S2 lever ladder (hypotheses written before any flight; each is a registered cell with kill criteria)

| id | lever | mechanism and evidence | expected | cost per flight | kill |
|---|---|---|---|---|---|
| V1 | thinking budget 3,072 -> 6,144 | the synthetic sweep showed most contacts hit the 3,112-token thinking cutoff; truncated reasoning on far-field rows is the most likely loss | +0.5..+2 SLU on far-field; intent flat | worst-case KV doubles: admitted knee drops (re-measure R-A3 at the new budget; expect 16-24/instance), ~2x wall | lower CI95 <= 0 after one flight |
| V2 | self-consistency n = 3 on the Solver node (temperature 0.6, majority on scenario_action, union-vote on entities) | the base has 4 instances of headroom; SLU error is dominated by near-tie decisions (D3 showed greedy flips at top-2 gaps of 0.0-0.25) | +1..+3 SLU | 3x Solver contacts (+8 per row) ~ +55 % wall | lower CI95 <= 0 |
| V3 | tool-bearing retrieval arm (owner ruling 2: after the baseline) | native tool call to the frozen Nemotron retriever instead of pre-supplied evidence; the tools-400 path is now understood (design doc 5.5) | +0..+2 SLU on entity slots | requires the Q2 remedy (per-request `chat_template`) and a new registered arm | tool-call parse rate < 95 % in smoke, or neutral read |
| V4 | BF16 precision alone | Q4_K_M -> BF16 recovers quantisation loss | +0.3..+1.5 SLU; below the cross-run bar by itself | none: it is S0's reading | n/a (reported, not decided) |

Order: V1 first (cheapest to register, strongest mechanism evidence), V2 second, V3 only after V1/V2 are read and the
owner confirms the tool arm. V4 is not a flight; it is how S0's reading is reported. Levers are never combined before
each is read alone. Every S2 cell carries a twin.

## 4. Engineering path to the first scored flight (the critical path; card work waits on it)

Status (Sonnet readiness report, 2026-10-30): datasets GO (7/7 annotation hashes, 1,080/1,080 audio pins), models GO
(26/26 Thinking files, Nemotron present), sealed vectors GO (regenerated 2026-10-28, port pins `fe23703a`/`24f0cdf7`
match), prepared artifacts GO after the flip re-pin (phase supply stale on `config_sha256` by design), flight driver
NO-GO. Gaps, owner, effort, acceptance:

| gap | what | role | effort | acceptance |
|---|---|---|---|---|
| A | runner sets `decoding_params["id_slot"]` on every contact; `VllmServerTransport` refuses it (`vllm_transport.py:60,150`) | SONNET | 2-3 h | the router strips `id_slot` and uses it as the instance key; a contract test sends a runner-shaped contact through the vllm transport |
| B | multi-endpoint transport: one `base_url` in the hashed fragment; one session per flight | SONNET | 12-20 h | `VllmRouterTransport`: instance = slot mod N, one session and one runtime receipt per instance, per-send identity check per instance, deployment record in the run manifest; `base_url` stays the protocol fact (instance 0), the other endpoints come from the deployment profile; `protocol_sha256` unchanged (test) |
| C | flight width = baseline `serving_slots` (4); nothing reads the admitted level | SONNET after Decision 2 | 4-6 h | width = instances x admitted level from the deployment layer; `serving_slots` stays the hashed geometry fact; a test shows 128 in flight on the 4xTP2 `-v2` profile and 4 on a profile without `admitted_level` |
| D | `vllm_identity._verify_namespaces` requires harness and server in one mount/network namespace; 4 containers, no `--pid host` | SONNET + OWNER (9.2) | 6-10 h | Decision 3 below; test on the live R0 service |
| E | driver chain (`batch_chain_l338.sh`, `l338_flip.sh`, `l338_read.sh`) bound to WSL paths and llama.cpp tooling | SONNET | 8-12 h | `work/scripts/flight_chain_l338_port.sh`: lease, smoke worker, FIT, sign-off, campaign, score, ledger close; dry-run prints every command; detached, resumable |
| F | no analysis recipe and no READ row for the port package | OPUS (recipe) + SONNET (writer) | 3-4 h | `analysis-recipe-v1.json` in the port package = baseline recipe with the port cell ids, the twin arm and the band comparison; `flight_registration_check.py` no longer lists it |
| G | S5 findings: `docker_run_argv` emits `--restart unless-stopped` (plan refuses `--restart`); the fragment hash string `a35b2740` was reported not to reproduce; RECOMPUTED 2026-10-30: it IS correct (`a35b274058d7…`), the report was wrong; verifier `versions` schema; CANN paths | SONNET | 3-4 h | `--restart no` + test; docstring and docs corrected by successor lines; versions check reads the serving interpreter; CANN paths read once on 181 |

Total ~40-60 h of engineer time; four parallel Sonnet streams on disjoint files finish it in about two working days.
Streams: B+A (transport), C+G (deployment/runner width and fixes), E (driver chain), F (recipe writer). Then re-pin.

**Decision 2 (flight width).** Width is a deployment fact: instances x admitted level (128 today). `serving_slots: 4`
remains the hashed geometry fact and is not reinterpreted. No hash movement.

**Decision 3 (namespace check, for the owner).** Recommend: run the four servers with `--pid host` and relax
`_verify_namespaces` to "same host, server pid visible, image id and argv match", because the receipt gate already
pins image, argv and env per instance; the alternative (one container for four TP2 instances) loses per-instance
cpuset and crash isolation. Owner item 9.2 says "same container"; the owner must say which reading stands.

**Decision 4 (twin arm).** S0 needs a `lock-twin` arm (second seed, same protocol). It is a registration change to
the port cell before the flip (writer re-run, re-mint); the smoke-4 cell gets it too. Cost: +2 contacts per row.

**Decision 3, applied (owner YES, 2026-10-30).** `namespace_policy: shared-pid-host` is set in the `-4xtp2-v2` and `-v3` profiles; `docker_run_argv` emits `--pid host`; the relaxed check requires same host, visible pid, pinned interpreter, matching cmdline and the endpoint listing the recorded model, per instance. Two R0 items remain to be settled on the live service: the harness container must run with `--network host --pid host` (`PORT_DOCKER_FLAGS`), and the uid-1008 harness must be able to read `/proc/<pid>` of root-owned servers (`--user 1008:1008` on the servers if the card devices allow it, else `--cap-add SYS_PTRACE` on the harness). Option (b), one container for four instances, is not expressible in the schema (250-400 lines) and is dropped. O1 is closed.

**Decision 4, revised (Fable, 2026-10-30, after the recipe read).** The registered `aa` arm is already a request-identical repeat of `lock` in the same run (the rails accept exactly one twin, `supply_variants.py:2749-2754`, and no arm may carry its own seed); at temperature 0 a second seed changes nothing. So `aa` IS the twin: no `lock-twin` arm, no re-registration, 15 contacts per row and the registered caps unchanged; owner item O2 is withdrawn. Admissibility of the run as a cross-run comparison: |aa - lock| <= 1.2 pp SLU (the measured band width, 72.75 - 71.61 = 1.14, rounded up; the 268-B twin at +1.15 passes). S0 reports `lock` and `aa` against the band.

**Decision 5 (ceiling-bound rule restated).** Preliminary `waiting` peaks are non-zero well below the ceiling (5 at 32,
13 in the worst case), so "waiting = 0" is wrong. A level is ceiling-bound only if `running == ceiling` and
`waiting > 0` in more than 20 % of the 5 s samples. S8 applies this to R-A2.

## 5. Order of work and calendar (cards on 181 only)

1. Now, parallel: engineering gaps A-G (Sonnet x4), recipe F (Opus); arm B P3/P3b and E3 on the cards when the owner
   launches them (the launch command is in the owner's hands; the chains are built and tested).
2. Base lock: R0 on the resident arm with the final profile (`-v2`, or `-v3` if E3 adopts shm), 11/11 verification,
   smoke re-run; R-A1..A3 recorded; S8 and S5 filled; R3 review. ~1 day of cards, interleaved with 1.
3. Registration: writer re-run with the measured profile and the twin arm; OWNER countersigns the 4 DRAFT rows and the
   6 model-free rows; receipt minted; `repin_cell.py` flip; `flight_registration_check.py` clean.
4. Smoke-4 on the resident service (60 contacts, ~0.5 h): sign-off conditions, tool-call parse, cost measurement.
5. S0+S1 flight: 16,200 contacts + twin, bound 8.5 h at 4 x 32 (67.7 card-hours). Score, READ by the recipe, ledger
   close. Report: port lock vs band; each arm vs lock with CI95; V4 reading.
6. S2: V1 then V2 (each one flight after its own worst-case admissibility re-measure); V3 after the owner's go.

Earliest first scored read: engineering (2 days) + base lock (1 day) + countersign + smoke + flight (1 day) = about
4 working days after the owner's countersign, assuming the cards stay ours.

## 6. What the owner decides

| id | question | recommendation |
|---|---|---|
| O1 | closed: `--pid host` + relaxed check (Decision 3 applied) | none |
| O2 | withdrawn: `aa` is the twin (Decision 4 revised) | none |
| O3 | Countersign the port rows once the profile is filled (step 3) | when the R0 verification is 11/11 |
| O4 | Launch P3/P3b and E3 now (commands already given) | yes; both chains restore arm A |
| O5 | S2 order V1 -> V2 -> V3 | yes |

## 7. Report format from here on

Every status report leads with: port lock SLU-F1 / intent vs the band; each arm's delta vs lock with CI95; what the
next flight tests; then the serving numbers only as cost (wall, card-hours, crash count).
