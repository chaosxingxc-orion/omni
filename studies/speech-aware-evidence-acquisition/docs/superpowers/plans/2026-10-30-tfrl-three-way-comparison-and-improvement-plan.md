# Three-way comparison and training-free RL improvement plan (Fable 5.1, 2026-10-30)

Owner direction (2026-10-30, translated): after the base lock, run a large-scale comparison on Fold B of (D) the direct
model, (A) the agent system, (T) the agent system plus training-free RL (TFRL); the end goal is a better TFRL algorithm.
TFRL here means: no parameter training; sampling finds the best trajectories, which then improve prompt engineering,
knowledge construction and consumption, and the harness. Three questions were asked: the rollout strategy (knowledge
supply efficiency, recognition accuracy), the systematic problems of the current experiments and analysis, and the
improved scheme. Evidence: `docs/results/2026-10-30-tfrl-history-and-systematic-problems-read.md` (Opus, every claim
cited). Companion: `2026-10-30-fold-b-comparison-baseline-and-improvement-plan.md` (band, bars, engineering gaps).
Grants nothing: every flight below is a registered cell behind the owner's countersign.

## 1. What the repo measured (Fold B, Q4_K_M llama.cpp unless stated)

| arm | cell | SLU-F1 | intent | note |
|---|---|---|---|---|
| D direct model | 266-E `SAEA-SLURP-DIRECT-ONLY-FOLD-B-THINKING-V1`, one call per row, no knowledge | 45.39 | 76.76 | |
| A agent system | 268-B lock, `phase-native-module-v1-c2`; band over 8 readings 71.61-72.75 / 84.91-85.74 | 71.61 | 85.00 | the comparison reference |
| T agent + TFRL | none on Thinking; Instruct-era ACE iteration 2: +1.38 pp [+0.15, +2.57] vs its lock, REFUSED (floor 1.49; intent -0.56; fell to +0.27 at iteration 3) | - | - | no TFRL state was ever admitted |

The agent system is worth +26 pp SLU over the direct model; TFRL as built added at most +1.4 pp and did not survive its
own bar. The last loop (decision 263-A) sampled prompt-rule PROPOSALS (3 typed-proposer + 18 ACE reflector contacts per
iteration against one fixed 625-row construction column), not trajectories; the September loop sampled 5 Solver
trajectories per row and let the frozen model write lessons. Both updated only delivery rules.

## 2. Answer to question 1: the systematic problems, from the record

| # | problem | evidence | consequence |
|---|---|---|---|
| P1 | effects are the size of the noise | two identical runs in different flights differ by +1.87 pp; the 8-reading band is 1.14 pp; the bars are 1.49 (within run) and 2.5 (cross run); TFRL deltas were 1-1.4 pp | every TFRL verdict so far is a coin toss against noise |
| P2 | one selection fold, reused three times | Fold B read three times, no Fold A or DEV confirmation; measured selection optimism 1.3 pp; the one confirmed rule shrank from +1.3/+1.4 to +0.56 | accepted gains are inflated by about the size of the gain |
| P3 | the CI ignores decoding noise | bootstrap resamples utterances only; the lock is re-measured per run (64.00 / 64.51 / 64.45); the twin offset swung -0.49 / +0.32 / -0.37; the twin is one request-identical copy, so it measures batching nondeterminism only | the CI95 is too narrow by the decoding term |
| P4 | the pooled metric rewards emptying | +0.96 pooled vs -2.22 per row on one arm; the playbook's only effect was to make the Solver answer "emptier" | a lever can pass pooled F1 while hurting rows |
| P5 | the bottleneck is consumption, not supply | 61 % of missed slot values were already in the supplied knowledge; the proposer only ever produced delivery rules; no TRAIN material is left to add | more knowledge cannot help; using it can |
| P6 | the Solver is truncated | the thinking budget cuts 45.9 % of Solver contacts; the port's synthetic sweep saw most contacts hit the 3,112-token cutoff | a systematic defect of the base, not a lever |
| P7 | the loop's enforcement had open defects; Instruct-era closures were never re-tested on Thinking; all history is Q4_K_M | read, section E | T has never been measured on the model and precision we now run |

**Fixes, in the order they must land (measurement before method):**
- **M1 noise model.** Each cell carries k = 2 decoding replicates of `lock` (the twin plus a second seed-identical copy), so the within-run decoding variance is estimated, not assumed; the CI95 of every arm delta is the row-paired bootstrap PLUS the replicate variance; the within-run floor stays 1.49 pp but the acceptance uses the widened CI. Per-row F1 is reported beside pooled F1 and an arm whose per-row delta is negative cannot be NOMINATED (closes P3, P4).
- **M2 two-fold rule.** Selection on Fold B, confirmation on Fold A (held out, same cell family, one flight); a lever is accepted only if the Fold A lower CI95 > 0. The optimism term (1.3 pp) is reported as selection minus confirmation (closes P2).
- **M3 budget as base.** The thinking budget is a base parameter, not a lever: if V1 (6,144) is admissible under the worst-case rule at the knee, it becomes the locked base before D/A/T run, because comparing arms on a truncated Solver measures truncation (closes P6). Cost: re-measure R-A3 at the new budget (about 4 card-hours) and a lower knee (expected 16-24 per instance).
- **M4 full-chain observability** (owner requirement). Every contact logs: row id, node, arm, supply item ids and their retrieval scores, thinking tokens and a truncation flag, the answer, the per-row score, and for TFRL the sampled trajectory id, its reward and whether it was selected. One JSONL per flight, schema-pinned, summarised per flight by the watchdog (hit rate, truncation rate, reward distribution, acceptance). No raw log is read by a model.

## 3. Answer to question 1: rollout strategy

**Knowledge supply.** Because the missing answer is usually already supplied (P5), supply efficiency is precision, not
recall: supply fewer, better-ranked items. Two measured quantities per contact: supply recall (gold entity present in
the supplied items) and supply usage (supplied item cited or copied into the answer). Target: recall unchanged, usage
up. Mechanism: scenario-conditioned supply. The Perception node's scenario prediction gates the Solver's knowledge
(filter the facet and centroid keys by predicted scenario, then rank), so the Solver sees the k items that matter for
that scenario instead of the global top-k. This is a supply-variant lever (registered as a cell), hash-moving.

**Recognition.** Errors are attributed per node and per field (scenario, action, entities) and per channel (headset,
far-field) from the M4 trace, before any lever is chosen. Levers in order: budget (M3), self-consistency n = 3 on the
Solver with scenario_action majority and entity union-vote (greedy flips at top-2 gaps of 0.0-0.25 were measured on the
port, so near-tie decisions are the target), and exemplar supply (section 4).

## 4. Answer to question 2: the TFRL algorithm, improved (TFRL-v2)

The built loop sampled rules and credited them at the column level. TFRL-v2 samples TRAJECTORIES and credits them at
the row level, then distils; the harness changes are explicit outputs, not side effects.

| step | TFRL as built | TFRL-v2 |
|---|---|---|
| sample | 3 + 18 rule proposals per iteration on a fixed 625-row column | K = 4 Solver trajectories per row on the construction column, temperature 0.7, with varied consumption instructions and evidence orderings drawn from a rule bank |
| credit | column-level SLU vs lock | per-row reward = row F1 of the trajectory minus row F1 of lock; positives are trajectories that fix a row lock gets wrong; negatives are trajectories that break a row lock gets right |
| update: knowledge | none | exemplar bank keyed by scenario, distilled from positives (the solved row's evidence-to-answer path); consumed by the Solver as scenario-conditioned exemplars (construction + consumption in one object) |
| update: prompt | reflector writes delivery rules from column reads | reflector writes a rule only from a positive/negative CONTRAST pair on the same row; a rule without a contrast pair is refused |
| update: harness | none | vote n = 3 over the top-ranked rules at inference (self-consistency) |
| select | within-run bar on Fold B | within-run bar on Fold B with M1's widened CI AND Fold A confirmation (M2); twin within 1.2 pp |
| iterate | 3, fixed | at most 3; stop when an iteration accepts nothing or the construction-column gain is under +1 pp |
| observability | column reads | M4 trace: every sampled trajectory, reward, selection decision |

Kill criteria: iteration 1 under +1 pp on the construction column (stop the family); any accepted iteration that fails
Fold A (revert, report the optimism term). Expected: +2 to +4 pp SLU over A if P5 is the real bottleneck; 0 if it is
not. Both outcomes are informative under M1/M2; the current loop could not distinguish them.

## 5. Answer to question 3: the three-way comparison on the locked base

| arm | registered cell (new, port family) | what it is | contacts per row | card-hours at the knee |
|---|---|---|---|---|
| D | `...-DIRECT-ONLY-FOLD-B-NPU-PORT-V1` | one Solver call, no knowledge, no program; plus twin | 2 | about 1 |
| A | `...-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1` (registered, unflown) | the native module program, lock + aa + l1-support + l2-union | 15 | 67.7 (bound) |
| T | `...-TFRL-V2-FOLD-B-NPU-PORT-V1` | A's lock harness plus the TFRL-v2 outputs (exemplar bank, rule bank, vote); plus twin | 2 + 3 iterations of sampling on the 625-row column (K = 4) | about 25-30 sampling + 10 confirmation |
| confirmation | the three on Fold A, one pass | M2 | as above | about 80 |

Order: M4 trace and M1 replicates land in code (card-free); M3 budget decision from R-A3 at 6,144; D and A in one
campaign on Fold B (A is the registered S0/S1 flight; D is 1,080 cheap contacts); T's sampling on the construction column,
three iterations; T on Fold B; the three on Fold A. Total about 185 card-hours after the base lock, about 23 hours of
wall time on four instances, spread over the iterations. Reporting (owner rule): every report leads with D, A, T on
SLU-F1 and intent with CI95, the A-to-D gap, the T-to-A gap, Fold A confirmation, then cost.

## 6. What is missing before T can run (engineering, card-free, in order)

| gap | work | role | effort |
|---|---|---|---|
| T1 | M4 trace writer in the runner and the Solver node; schema; watchdog summariser | SONNET | 8-12 h |
| T2 | M1 replicates: a second request-identical `lock` copy as a registered arm; CI code adds the replicate variance; per-row F1 in the reader | SONNET (code) + OPUS (recipe) | 6-8 h |
| T3 | TFRL-v2 sampler: trajectory sampling on the construction column through the router transport (K = 4, temperature, rule-bank variation), per-row credit, exemplar distillation, contrast-pair reflector, vote harness | SONNET | 24-32 h |
| T4 | D cell registration (direct-only on the port), T cell registration, Fold A dataset fragment for the port | OPUS (texts) + SONNET (writer) | 6-8 h |
| T5 | loop tooling off the Windows-side driver; construction column re-flown on the port (the lock changed) | SONNET | 8-10 h, plus one 625-row flight |

## 7. Owner items

| id | question | recommendation |
|---|---|---|
| R1 | M3: make the 6,144 thinking budget part of the base if admissible, accepting a lower knee | yes |
| R2 | M2: adopt the two-fold rule (Fold A confirmation flight per accepted lever) | yes |
| R3 | T's sampling budget: K = 4, 3 iterations, 625-row column | yes |
| R4 | start T1-T5 now, in parallel with the base lock | yes |

## 8. Owner answers (2026-10-30, translated) and the planner's resulting rules

- **R1 and R3: delegated to the planner, "lock a clear baseline configuration from experimental feedback, accuracy first".**
  Rule: among admissible configurations (zero worst-case preemptions, KV peak <= 90 %, every 3.4 threshold), choose the one
  with the highest expected task accuracy; efficiency breaks ties only. Applied to the thinking budget: the budget is
  raised to 6,144 and the worst case is re-measured at 32, then 24, then 16 per instance; the base is the highest
  admissible level at 6,144. Only if NO level >= 16 is admissible at 6,144 does the base stay at 3,072 and 32. The budget
  lives in the hashed model fragment (`...-budget3072-...` -> a `-budget6144-` successor fragment), so the port package
  is re-minted by the writer before the countersign, and "thinking budget" is added to the declared differences against
  the 268-B reference; S0 parity is still read against the band, because the band is the system's reading and the
  budget is part of our system. TFRL-v2 sampling starts at K = 4, three iterations; after iteration 1 the planner may
  raise K to 8 on the rows where the reward variance is highest, within the same card budget, and records the change.
- **R2: YES.** Fold A confirmation is required for every accepted lever and for the D/A/T comparison.
- **R4: launch authorised** (arm B P3/P3b and E3).

## 9. Rulings on the T4 registration drafts (Fable, 2026-10-30; drafts under `docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/`)

1. **L338 port FIT cap defect.** The registered port cell carries `native_contact_fit.contact_cap` 64,800, but the raw-audio
   arm's FIT check (`preparation_fit.py:1130-1137`) requires 16,200 (smoke: 60, not 240). Fix in the writer
   (`l338_port_write_registration.py` / `l338_port_family.py`), re-mint with the next writer run; rows stay DRAFT. SONNET.
2. **D has no FIT path.** The live-phase fit refuses the raw-audio arm and the native-contact fit refuses the direct-only
   program. Add a direct-only FIT path (one contact per row, raw audio, the same envelope checks) behind the program id;
   no change to any existing path's bytes (snapshot test). SONNET, 4-6 h.
3. **T's banks and their own rows.** The runner's knowledge-source guard refuses a construction flight whose bank was built
   from the same rows. Ruling: split the 625-row column R01C into two fixed halves (R01C-H1: 313 rows, R01C-H2: 312 rows,
   by the roster order, pinned); iteration t samples on both halves, and the bank each half consumes in iteration t+1 is
   distilled from the OTHER half's positives (cross-fitted banks). Section 4's "update: knowledge" row reads accordingly.
   This also removes within-column leakage from the reward. The Fold B flight of iteration t consumes the union bank.
4. **Window competition (budget 6,144 vs the exemplar bank).** Under R1 the Solver prompt limit is 14,334 tokens and the
   measured maximum lock prompt is 13,465, so the bank has no room on top of 51 supplied items. Ruling: in the T arm,
   exemplars REPLACE supplied items, they do not add to them; T's supply bound is 43 items plus an exemplar block capped at
   2,000 tokens (about 8 items' worth); A keeps 51 items and no bank. The difference is the thing T tests (consumption over
   supply, P5); it is a declared T-vs-A protocol difference, not a confound. If the bank proves worth more than 8 items,
   the trade is re-measured as its own lever. The rule bank is text in the Solver instruction, capped at 400 tokens.
5. **Fragment hash clarification.** `a35b2740…` is `fragment_hash` WITH `SAEA_DATA_ROOT` set (as flights run);
   `bcb98207…` is the same function without a data root (the materialization path collapses to a placeholder). Flights and
   the record carry `a35b2740…`. The earlier note in the base-lock skeleton is amended by this line.
6. **D writer.** D is emitted by the 266-E writer (`direct_fb_write_registration.py`) with the port settings imported from
   `l338_port_family`; T and Fold A by the L338 port writer with a family switch and a per-roster preparation table. SONNET.
