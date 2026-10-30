# TFRL history, knowledge supply, recognition errors and systematic problems (analyst brief, 2026-10-30)

Opus 5.5, read-only, model-free (no host, card, scorer or commit). Paths are relative to the study tree
`/mnt/cephfs/xingchao/omni/work/dev/umbrella/studies/speech-aware-evidence-acquisition` (branch `npu-vllm-port`). All numbers are
development numbers (TRAIN Fold A / Fold B or DEV; claims_benefit false), quoted from the repo's notes, never re-scored.
Abbreviations. R = `docs/results/`; COV = R/2026-10-11-slurp-tfrl-coverage-preread-fold-b-offline.md; I1 = R/2026-10-11-slurp-tfrl-loop-i1-fold-b-column-read.md;
I3 = R/2026-10-12-slurp-tfrl-loop-three-iterations-read.md; ACAD = R/2026-09-26-slurp-stage3-locked-tfrl-academic-baseline-v1.md;
LIN = docs/readiness/2026-10-11-slurp-tfrl-lineage-and-algorithm-baseline-analysis-v1.md; PROT = docs/readiness/2026-10-11-slurp-tfrl-loop-v1/tfrl-loop-protocol-v1.json;
INS = docs/readiness/2026-09-26-slurp-stage3-measurement-instrument-v1.md; 268B = R/2026-10-13-slurp-268b-thinking-system-fold-b-read.md;
266E = R/2026-10-12-slurp-266e-thinking-vs-instruct-bare-core-read.md; GAP = R/2026-10-19-slurp-v3-gap-to-finetuned-sota-decomposition.md;
L338 = docs/superpowers/specs/2026-10-19-slurp-338-lever-column-design.md; CMP = R/2026-10-29-fold-b-comparison-baseline-read.md;
PLAN = docs/superpowers/plans/2026-10-30-fold-b-comparison-baseline-and-improvement-plan.md; TFD = docs/superpowers/specs/2026-09-21-slurp-stage3-training-free-rl-design.md;
CFG = configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json; SV = src/speech_aware_evidence_acquisition/rails/supply_variants.py.

## A. The TFRL loop as built (two generations)

The writable policy has exactly two parts:
- the CONTEXT each node sees;
- the COMMIT, i.e. which of the episode's own products is delivered.

Every TFRL operator ends in one of those two writes (ACAD:95-103).

**A1. Generation 1, Sept (TF-GRPO and GEPA; Instruct Q4_K_M core; 4-node plan).**
- **What was sampled:** the Solver only, N = 5 per row at T 0.7, on the 625-row R01C construction roster (ACAD:135-139). The other nodes ran once (ACAD:151-155).
- **Credit:** the frozen core wrote a "semantic advantage" on the 225 rows where the group disagreed. The other 400 rows carried no signal (ACAD:142-145, 160-161).
- **Update:** a prose store (379 values), routed into node context per candidate (ACAD:146-147, 156-159). GEPA ran 68 reflective contacts, one per (scenario, node) (ACAD:207-215).
- **Design:** T = 3 iterations, with per-stratum Pareto selection (TFD:35-58).
- **Offline best-of-5:** oracle +6.55 [+5.56, +7.56]; support-3 filter +1.40 [+0.55, +2.29] (ACAD:280-283, 424).

**A2. Generation 2, decision 263-A, Oct 11-12 (sealed-credit loop with an ACE comparator; Instruct core; lock v2.2).**
- **What was sampled:** proposals, not trajectories. Typed proposer: 3 carriers × 1 contact at T 0. Re-sampling and best-of-N were forbidden (PROT:146-151, 221-252). ACE reflector: 18 carriers, one per scenario, at most 3 adds each (I3:91-96).
- **Material:** one lock column on R01C (625 rows), reused as aggregate tables in all 3 iterations. Fold B contributes counters only (I3:79-83; PROT:192, 199).
- **Selection rule:** scripted, and the model never judges (PROT:167). A child is accepted iff all of these hold (PROT:264-270): (a) SLU CI95-LB > 0; (b) SLU point >= 1.49 pp; (c) SLU point above the same-flight twin's; (d) above the parent (vacuous in this loop); (e) intent CI-LB >= -0.65 and intent point >= 0.
- **Rules for other candidates:** delivery and scope rules are read offline, cross-fitted, and need CI-LB > 0 and point >= 0.67 (PROT:275-278); supply and construction candidates fly only if their predicted gain is >= 1.5 pp (PROT:281-284); parent score = LB_slu - 0.25 × flown children; a class is masked after 2 refusals (PROT:213-219).
- **What gets updated:** typed arm: hash-pinned artefacts (exemplar weights, card sets, k, v4g rules, scope routing; no free text) (LIN:191-198); ACE arm: an itemised prose playbook, released as a store and swapped into the Solver supply under T_cap 15,608 tokens (I3:91-96).
- **Iterations:** exactly 3, no early stop (PROT:169-171).
- **Downstream consumption:** none. The parent stayed `s0-lock-v2.2` (I3:56-57), and 268-B carried "lock v2.2 + v4g" unchanged (268B:17-18, 75-78).
- **No autonomous acceptance ever.** Every stable gain (B1, B4b, COMP-1b, v4g, review-off) was chosen from the credit tables by a person (LIN:160-173).

## B. What it measured

| cell (core, roster) | SLU-F1 delta pp [CI95] | intent pp | word / reading | cite |
| --- | --- | --- | --- | --- |
| TF-GRPO CELL-1 (Instruct, Fold A) | -0.48 [-2.18, +1.24] | -0.74 | no verdict (cross-flight, no replicate) | ACAD:167, 419 |
| TF-GRPO CELL-2 (Instruct, Fold A) | +0.64 [-0.76, +1.92] | -0.09 | no verdict | ACAD:169, 420 |
| GEPA CELL-1-GEPA (Instruct, Fold A) | -0.04 [-1.58, +1.39] | -0.09 | no verdict | ACAD:237, 423 |
| TF-GRPO CELL-3 gated (Instruct, DEV) | -1.44 [-2.88, -0.05] | 0.00 | "not distinguishable from flight variance" | INS:129-134 |
| REV-TFRL-1 Reviewer (Instruct, Fold A) | -0.55 [-1.79, +0.69] | -0.46 | descriptive | R/2026-09-28-slurp-revtfrl-1-fold-a-read.md:1 |
| ACE i1 / i2 / i3 vs lock (Instruct, Fold B) | +0.87 [-0.40, +2.10] / **+1.38 [+0.15, +2.57]** / +0.27 [-1.15, +1.71] | 0.00 / -0.56 / +0.46 | REFUSED x3 | I3:36-50 |
| typed arm i1-i3 (offline delivery rules) | +0.00, 0 rows edited | 0 | empty; class masked | I3:213-221 |
| decision-265 coverage pre-read (offline) | composite -1.00; best honest +0.27 | +0.07 | no candidate over 1.5 | COV:32-41 |
| VER-1 support rule, Fold A confirmation | +0.56 [-0.40, +1.51] | -- | NOT CONFIRMED | L338:42-45 |

**Best delta: ACE i2, +1.38 pp SLU.** It is the only reading with CI-LB > 0, but it is 0.11 pp under the floor, carries an intent cost, and did not repeat at i3 (I3:41, 49, 263-264).

**Why the loop was neutral or refused, in the repo's own words:**
- ACE: "The prose makes the Solver emptier; it does not add values the lock misses" (I3:177-181). Its gain is "a mass the locked v4g normaliser already harvests" (I3:253-254).
- Typed arm: the proposer "reached for delivery rules even when not offered that class". It never proposed a construction, supply or scope candidate (I3:209-212, 222-224).
- Generation 1: the semantic advantage is "decoding variance dressed as a lesson" (ACAD:516-517); "You cannot climb a gradient you cannot see" (ACAD:455); the twin reproduces the arm, so the prose's cost cancels its content (ACAD:465-468, 488-490).
- Overall: system structure gives +27.30 pp over the bare core, while "the largest effect any of them produced is under 1 pp" (ACAD:75, 82-85).

## C. Knowledge side

- **Stores** (port lock; same form as v2.2/v3.0): `train-exemplars`: Solver, k 20, pure cosine (CFG:86-118); `intent-boundaries`: Perception, k 60 fused into an 8-entry block, score w_cos 1 + w_agree 1 (CFG:7-48); `slot-contracts` cards: at most 3 candidates × 16 values, 28 values total (CFG:62-72); supply cap: 55 items (CFG:277).
- **Keys and queries:** Exemplar keys: Nemotron-3B 'passage:' vectors over 11,514 TRAIN texts (CFG:92-93). Queries: the row's own speech vector (CFG:35-36, 107-111). That a query must carry speech is enforced in code (`rails/phase_supply.py`:67-80). Centroid keys: the mean TRAIN vector per intent (CFG:20-21). Agreement: the share of the row's top-20 exemplars per intent (CFG:9-10). Perception block = cosine + 1.0 × vote, top 8 (COV:86-87).
- **Pool:** TRAIN minus the folds = 9,354; the legal construction set = 8,554. There is **zero expansion material left in TRAIN** (COV:236-248).
- **Measured supply quality, Instruct v2.2 (COV):** the gold boundary is already in the block on 142 of 190 intent-wrong rows; P(intent correct) is 86.17 % when gold is in the block and 9.43 % when it is not (COV:176, 182); 61 % of MISSING slot values are already in the supply, which makes this a consumption failure (COV:53-54); 36 % of gold-side error values never appear in TRAIN (COV:55-56); only 18 reachable items remain, spread over 14 types (COV:57-58).
- **Measured supply quality, Thinking:** shortlist recall 88.15 % (L338:85); gold was in the block but Perception dropped it on 78 rows; retrieval missed it on 40 rows (GAP:41-43); key-recall levers are closed (GAP:171).
- **Inefficiencies the reads name:** every-card value lists cost +651 tokens per row for a ceiling of +0.27 (COV:214); pool selection pushes other rows' gold out (COV:298-300); a deeper block (8 to 12) harms gold-on-list by -1.39 (R/2026-09-30-slurp-psv3-fold-b-branch-read.md:1); "the channel binds before the information does" (LIN:166); 8-15k tokens of knowledge make Thinking reason about 2x longer (268B:180-182).
- **L338 arms (registered, never flown, CMP:8-9):** `l1-support`: 1 Perception + 5 Solver samples, a support filter at s = 2, and an N1 gate. Offline +1.09 [+0.53, +1.63] (L338:16, 29-33, 56). `l2-union`: 4 Perception samples, union capped at 3. Recall 88.15 to 89.81, intent 0 to +1.0, "Likely word NEUTRAL" (L338:17, 74-79, 93, 100-102).

## D. Agent recognition

- **Nodes:** Perception lists 1-3 intent candidates from the 8-entry block, plus one 128-character distinction field (CMP:21-22; CFG:367-370). The Solver picks the intent and extracts entities from 20 exemplars plus the cards (CMP:20-22). c2 = 2 contacts per row; Reviewer and Planner are retired; `max_revisions` 0 (268B:75-77; CFG:288). c15 = 7 Perception + 8 Solver contacts over 4 arms (CMP:25).
- **Intent errors, Thinking, Fold B:** 75 % have gold off the shortlist, and the Solver gets only 7.8 % of those right (GAP:41, 49, 58); Perception commits to a single candidate on 81 % of rows (GAP:60); 75 % of errors cross scenarios (GAP:62); `general_quirky` accounts for 22 % of errors (GAP:64-67).
- **Slot items (repair value):** MISSING 112 items, +6.22 pp; false fill on 81 of 392 empty-gold rows, +3.20; extra slot, +3.61; BOUNDARY / NEAR, +2.35 / +2.05 (GAP:75-85); missed-slot proposal is closed (recall ceiling 0.086, GAP:172).
- **Forced close:** the 525 forced-close rows hold 65 % of intent errors and 72 % of slot items, but this is "confounded with difficulty" (GAP:102-112); doubling the budget gave -0.81 (R/2026-10-18-slurp-nxt-r2b-solver-budget-column-read.md:1).
- **Scenario:** the weakest are general (intent 38.3), play, music and lists (GAP:126). Music lost -4.20 SLU in 268-B (268B:115).
- **Channel (headset vs far-field):** DEV direct to system intent: far-field 60.4 to 73.9, headset 63.3 to 79.8 (R/2026-09-22-slurp-b6-dev2000-added-direct-vs-plan-read.md:57-58); headset leads by 8.09 intent / 6.30 SLU (R/2026-09-27-slurp-roll-3-dev2000-added-pool-read.md:225-226); the ACE gain sat on headset rows (+0.41 vs +0.02, I1:134-135; +1.40 vs 0.00, I3:183-184); no channel split exists for the Thinking core: not a fact here.
- **Taxonomy:** SPURIOUS / MISSING / WRONG-TYPE / BOUNDARY / NEAR / WRONG-VALUE / FORMAT (LIN:99); row classes (COV:103-104); supply modes (a)-(d) (COV:92-97); intent classes a1 / a2 / b (GAP:41-46).

## E. Systematic problems in the experiments and the analysis

1. **Noise is the size of the effects.** A request-identical A/A pair across flights reads +1.87 [+0.41, +3.36], and SLU and intent can move in opposite directions (INS:44, 50-54). Within one flight, 8.5 % of rows differ (INS:29-33). The floors are 1.49 / 0.65 (same flight) and 2.5 (cross flight) (INS:119-120; CMP:35-39); the effects chased are 1-3 pp (LIN:104). Eight readings of the same system span 1.14 pp (CMP:56).
2. **One selection fold is reused.** Fold B was read 3 times: the winner's curse applies (I3:13-15), and no Fold A or DEV confirmation was run (I3:261-262). Selection optimism is 1.3 pp (COV:339-340). VER-1 shrank from +1.29/+1.43 to +0.56, NOT CONFIRMED (L338:42-45).
3. **The lock is re-measured per run, and comparisons cross flights.** The lock read 64.00 / 64.51 / 64.45 and its twin offset swung -0.49 / +0.32 / -0.37 (I3:63-66, 192-193). The 268-B comparator is cross-flight: "session drift ... is not measured by the twin" (268B:59-62). Port vs baseline is cross-engine, with no registered recipe (CMP:92).
4. **The CI misses part of the variance.** The bootstrap resamples utterance clusters only, not decoding draws: "no arm is read as an expectation" (L338:179-181). Clause (c) compares against one twin draw (PROT:267), even though the ACE interval is about 2x the twin's (I1:236-238). Pooled micro F1 rewards emptying rows: r1-c3 reads +0.96 pooled but -2.22 per row (R/2026-10-09-slurp-nx-r1-solver-fold-b-column-read.md:89-94).
5. **The twin rule is narrow.** The rails accept exactly one request-identical twin, and no arm can carry its own seed (SV:2749-2754; PLAN:98). At T 0 the twin measures only batch nondeterminism, not sampling variance.
6. **The thinking budget truncates.** Forced close hits 17.6 % of Perception and 45.9 % of Solver contacts (268B:154-155). A bigger budget did not help (see D). Far-field truncation is untested (PLAN:58).
7. **Quantisation and engine are confounded.** All history ran on Q4_K_M llama.cpp (ACAD:25; CMP:47). The port is BF16 on vLLM, with regenerated vectors and re-authored stores (CMP:72-80). The BF16 effect is unmeasured (PLAN:61).
8. **The loop's search was weak.** The proposer returned 3 byte-identical responses (I1:82-87), and 8 of 18 reflector answers were empty (I1:96). Class restrictions were not enforced (defect 9, still open), and `credit.vs_twin` is misnamed (I3:238-239). The reflector managed 0 of 100 repairs (LIN:101). Every iteration reused the same R01C tables (I3:81).
9. **Instruct-era closures were never re-tested on the Thinking core** (GAP:159-160, 166).

## F. The three-way comparison as the repo defines it

| arm | registered cell | Fold B SLU / intent | cite |
| --- | --- | --- | --- |
| direct model | `phase-direct-only-v1` (1 contact, no Perception or supply); 266-E `SAEA-SLURP-DIRECT-ONLY-FOLD-B-THINKING-V1` (`configs/baseline/slurp-direct-only-fold-b-thinking-baseline-v1.json`) | 45.39 / 76.76 (Instruct 33.49 / 72.41); 36 abstentions at cap | ACAD:50-51; 266E:33-34, 248; 268B:166-167 |
| agent system | 268-B lock, `phase-native-module-v1-c2`, `SAEA-SLURP-STAGE3-THINKING-SYSTEM-FOLD-B-LOCK-V1` | 71.61 / 85.00 (band 71.61-72.75); +26.21 SLU over direct | 268B:32-33; CMP:50-56 |
| agent + TFRL | none admitted. Instruct only: ACE i2 65.88 vs lock 64.51. Never flown on the Thinking core | -- | I3:41, 56-57 |

- **Instruct analogue on DEV:** direct 38.09, system 65.40, plus TF-GRPO CELL-3 -1.44 (ACAD:75, 171).
- **Nearest TFRL-style cell on the port:** L338 (sampling plus deterministic selection, the D5 route). It is registered but unflown (CMP:8-9; LIN:146).
- **Missing on the port:** Only L338 has port configs: there are none for direct-only, the TFRL loop or the ACE arm. The flight driver is NO-GO (gaps A-G, PLAN:69-82). The loop tooling assumes the PowerShell driver and 9p flips (PROT:157; I3:236). The R01C material column must be re-flown because the lock changed (PROT:199). There is no port read recipe (CMP:92), and the historical `read.json` files are absent (CMP:95-96).
