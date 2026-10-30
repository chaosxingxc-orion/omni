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

## 10. Rulings on the delivered T1 and T2 streams (Fable, 2026-10-30)

1. **T1 trace switch.** Tracing is enabled by `SAEA_ROLLOUT_TRACE_READ_ID` set by the launcher, not by the manifest; accepted
   for the base-lock recorded runs and the Fold B three-way. The launch step scripts under `work/scripts/flight_steps/`
   must export it from the READ id they already carry (engineering item, before the first recorded run). Promoting it to a
   deployment-profile field is deferred; it would move the manifest, not the protocol hash.
2. **T2 replicate arm.** `lock-rep2` is a request-identical fifth arm (17 contacts per row: 8 Perception, 9 Solver;
   program `phase-native-module-v1-c17`). Accepted. The replicate spread is reported; the twin-admissibility rule stays on
   `aa` alone.
3. **T2 widened CI.** `SE_t = sqrt(SE_b^2 + V_rep)` with `V_rep` the mean squared pairwise delta over {lock, aa, lock-rep2}
   (two degrees of freedom; recorded as such). Accepted as the M1 fix; the point estimate is unchanged and the copies keep
   their raw intervals.
4. **NOMINATED rule.** The engineer's design choice (pooled AND per-row lower bound above zero, and no resolved per-row loss)
   is accepted under the owner's accuracy-first rule, with two conditions: the per-row F1 (2TP / (2TP + FP + FN) over
   summed word and char counts, 1.0 for empty gold and empty prediction) is a repo diagnostic, never reported as an official
   SLURP number; and the official pooled SLU-F1 and intent accuracy remain the only numbers placed beside public baselines
   (comparison plan Decision 6). The rule is written into `analysis-recipe-v1.json` by the re-mint.
5. **Re-mint order.** The T2 re-mint predates the FIT-cap fix; exactly one combined re-mint is run on the committed package
   after both streams are in the tree, then the package, writer files and tests are staged together by path.
6. **Sweep failures.** The 35 + 68 failures in the T2 regression shards are not accepted as pre-existing until the same files
   are run at clean HEAD `56d3c1c3`; the combined gate carries that table.

### Section 10 addendum (2026-10-30): re-mint result and what the flight `check` now needs

The combined re-mint landed (commit after `2466f300`): flight composed config `8ccf5578…`, protocol `1bb05a55…`; smoke
`5d528040…` / `26e8bd51…`; protocol document `c255b182…`; analysis recipe `1ef6e660…` (pinned in the read-row draft).
The read-only flight `check` refuses both configs, as expected for HELD rows, and now for one more reason: the committed
exposure-ledger rows (lines 9177 and 9179) still describe the c15 draft (32,400 / 120 calls) while the plans say 36,720 /
136. The ledger needs the `--append-ledger` successor block for the c17 rows and the owner's countersign; after that the
next gate asks for the runtime receipt, which the base-lock R0 step mints. Both are on the base-lock path, not blockers now.

## 11. Rulings on the delivered T3 (TFRL-v2 split, sampler, distiller; Fable, 2026-10-30)

1. **Split mode: interleaved, not contiguous.** The R01C roster is sorted by sample id and SLURP ids cluster by scenario,
   so contiguous halves share scenarios for only 16-26 % of rows (H1 holds 25 alarm rows, H2 one; H2 holds 31 cooking
   rows, H1 none) and a cross-fitted scenario-keyed bank would cover almost nothing. Section 9 ruling 3 is amended:
   R01C-H1 = even roster positions (313), R01C-H2 = odd positions (312); 100 % scenario coverage both ways; no draw, no
   seed. Written write-once under `docs/receipts/rosters/` (`slurp-stage3-construction-r01c-h1.json`, `-h2.json`, split
   receipt), R01C file sha256 `e8abc65b…`, H1 identity `1a88c40b…`, H2 identity `c7b16e13…`.
2. **Sampling shape.** One fresh Perception contact plus K=4 Solver contacts per row per iteration; trajectory k differs
   by decoding seed only in iteration 1 (`seed-only`); from iteration 2 the opt-in `rule-variants` mode is used when the
   rule bank is non-empty, so a rule's effect is sampled and not assumed. Rows where `lock` and `aa` disagree are not
   counted as positives.
3. **Budget.** The 4,372-token figure is the 268-B lock's median Solver completion, applied by the sampler to every
   contact; measured cost per iteration is read from the trace after iteration 1 and the budget line is re-derived.
   Planning numbers: 21-28 card-hours per iteration (both halves, K=4), 64-84 over three iterations, above the 25-30
   first estimate in section 5; accepted under the keep-NPUs-loaded rule, with K raised to 8 only if iteration 1 shows a
   positive rate above 20 % of rows.
4. **Exemplar text (open engineering item T3b, SONNET).** The trace carries supplied item ids, not text; the distiller
   must resolve ids to the knowledge-store text when it renders an exemplar, and the exemplar block cap (2,000 tokens)
   must be measured with the repo's tokenizer measure, not `chars / 3.5`. The union bank for the Fold B flight (both
   halves' banks) and the loop-driver wiring (`tfrl_port_loop.py` Sampler/Distiller protocols take contact lines and one
   bank per source half) are the same item.
5. **Runner input.** The sampler's run list is the pre-registration table from which the T writer derives the c7 arm
   group; no runner consumes it directly. Correct; the T cell draft reads accordingly.

### Section 11 addendum: rulings on T3b (exemplar text, measure, union bank, loop adapters; Fable, 2026-10-30)

1. Exemplars now quote the exact `[evidence <id>] <content>` line the Solver read (resolver over the pinned knowledge
   stores, content sha256 per item, unknown ids refused). The two fixture edits in the existing sampler and distiller
   tests are accepted; no assertion was weakened. The RANKVIS rank label is not reproduced (`rank_label: none`); fine.
2. The plan builder refuses estimate-measured or ids-only banks; a real bank is measured with the FIT gate's tokenize
   measure through an injected registered client. Open engineering (T3c, SONNET): a driver that injects the client, and
   the loop command line wired to the V2 adapters (the stubs stay as the default until the first flight exists).
3. `used` flag defect in the trace writer: `core.rollout_trace.item_strings` judges only the JSON exemplar form, so on
   the real pipe-form `train-exemplars` store every `used` is null. Ruling: fix in the trace writer (T3c) with the same
   rule on the pipe form and a `used_basis` field; the resolver's interim judgement stays until then and is recorded.
4. Vote pooling across halves is not done (survivor banks only); accepted, the union keeps the per-half 3-row rule.
5. `rule-variants` mode for iteration 2 and later: trajectory k uses (k=0) no rule bank, (k=1) the full bank, (k=2) the
   rules of the row's scenario only, (k=3) the top half of the bank by support. Same seed scheme. This makes a rule's
   effect measurable per row; T3c implements `variants_fn` accordingly.
6. The construction-column gain and the +1 pp stop rule are measured from the bank-consuming flight, never from the
   adapter; accepted.

### Section 11 addendum 2: rulings on T3c (Fable, 2026-10-30)

1. `used_basis` is a required closed field on trace supply items (`json-form` | `content-pipe-form` | `none`, `none`
   iff `used` is null); no schema bump because no trace file existed yet. Accepted.
2. Rule variants per ruling 5 are implemented; K must be 4 in `rule-variants`; K=8 is only reachable in `seed-only`.
   Accepted. When a bank's rules all sit in one scenario, k1/k2/k3 coincide in content and the per-rule effect is not
   isolated; that is reported in the iteration record, not patched.
3. Defaults `--variation rule-variants` and `--measure vllm-tokenize` accepted; estimate banks are refused in real runs.
4. Trace location: the adapters expect `<analysis-root>/v2/flights/i<t>-<half>/rollout-trace.jsonl` while the writer
   emits `<data root>/analysis/<READ id>/rollout-trace.jsonl`. Ruling: the sampling flights' READ ids are
   `<T cell READ id>-I<t>-<H1|H2>` and the adapters read from the data-root analysis directory of that id; no copy, no
   symlink. Implemented with the T writer work (section 9 ruling 6), not as a separate item.

## 12. Rulings on the T4b registration packages (Fable, 2026-10-30)

1. **Sampling mechanism.** The base decoding is greedy, so K samples that differ only by seed would be near-identical.
   The T construction cells therefore use program c7 with the Solver decoding group top_p {0.80, 0.90, 0.95, 1.00} at the
   fragment's sampling temperature as the K=4 trajectories; this is the operational meaning of section 11 ruling 2's
   "seed-only" for iteration 1. The Fold B flight of T uses program c10 (lock, aa, and the bank-consuming arms).
2. **Caps per cell** (calls / FIT cap / audio seconds): Fold A arm 36,720 / 18,360 / 104,874; D lock, aa, ab 2,160 /
   1,080 / 6,310 each, D smoke 8 / 4 / 20; T construction i1-h1 4,382 / 2,191 / 18,474, i1-h2 4,368 / 2,184 / 19,703,
   T Fold B i1 21,600 / 10,800 / 63,096; T family total 30,486 / 15,243 / 101,716 plus R01C encoding (625 calls) and the
   reflector (36 generation, 37 FIT calls). All rows DRAFT, HELD.
3. **Budget-6,144 interaction.** Under the budget fragment the L338 package would move to config `925ec28d…` / protocol
   `d3da01fc…` (arm) and `db324bab…` / `ba2f1739…` (smoke); T does not fit at 51 items (14,828 > 14,334) and fits at 40.
   T's budget variant is derived only if the base decision picks that fragment; then T's supply bound is 40 minus the
   measured exemplar block.
4. **Fold A** is the Fold B port cell with a declared roster swap; its preparations (encode, alias, content, agreement,
   supply) are zero-sentinel pins until they run on CPU, which is queued engineering (no card).
5. **Open rows-only items in T** (R01C artifacts, `rails_pending` block, rule-bank release, credit/distillation,
   reflector, no stage-3 label, no reader) are the next engineering stream after the base lock; the registration is
   complete enough to HOLD the rows.
6. The existing `verify_declared_module` failure on the L338 port flight (re-authored stores differ from the lock layer)
   is a known control defect, tracked separately; not touched by the writers.

## 13. Rulings on the T rails (T5b; Fable, 2026-10-30)

1. Stage-3 label `tfrl-v2` (family `training-free-rl-v2`, status planned), the `tfrl_v2_bank` consumption declaration
   with its flight-start refusals (pending sentinel, hash mismatch, estimate measure, roster overlap, sources outside
   the construction scope: T0, Fold A, Fold B and DEV-1000 rows are never bank sources), the T analysis recipe
   (`303d238a…`, pinned in the read-row draft and the facts), the reader (construction and Fold B stages, write-once),
   and `bank-release-v1.json` (iteration 1, sentinel hashes until a bank is released) are accepted. The package marker
   is `rails_declared`: the rails exist; the cells adopt them at the flip re-mint together with the `v2.2-npu-port` lock.
2. Stop rule reading confirmed: the loop stops when the construction gain is below +1 pp (+1.0 continues, +0.9 stops),
   and never past three iterations. Mean best-of-K gain is the mean over usable rows of the best trajectory's reward
   minus lock, negatives included.
3. Remaining T engineering before any T flight (T5c, SONNET, after the base lock): (a) the runner renders the exemplar
   block and rule bank into the Solver request through `compose_solver_messages` behind the declaration (snapshot test
   on an existing cell proves bytes unchanged without the frame); (b) a scorer driver for the derived `vote` column so the
   construction gain is a number, not UNDETERMINED; (c) T cells switch to the port lock, and the construction cells get
   the minus-R01 exemplar-store leaves layered over it; (d) the flight-start hook re-checks Fold B row disjointness
   against the bank's source rows, not only the split identity.
