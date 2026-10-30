# SLURP: public test-split baselines, metric alignment and the gaps before a test-set headline (2026-10-30)

Analyst read (Opus 5.5) for Decision 6 of `docs/superpowers/plans/2026-10-30-fold-b-comparison-baseline-and-improvement-plan.md`
(owner rule 2026-10-30, translated: "The final evaluation metric is the test-set metric, and it must be compared against
publicly reported industry baselines."). Model-free: no model, server, encoder, card or scorer contact; no test gold opened.
Test-split counts below come from identity metadata (`recordings[].file`) and FLAC headers only. Paths are relative to the
study tree `/mnt/cephfs/xingchao/omni/work/dev/umbrella/studies/speech-aware-evidence-acquisition` (branch `npu-vllm-port`)
unless absolute. Grants nothing: no readiness flip, no exposure entry, no flight.

## 1. What Fold B and Fold A are

**Fold B** = roster `SLURP-EVO-FOLD-B-V2`, `docs/receipts/rosters/slurp-evo-fold-b-v2.json` (generator
`scripts/build_slurp_evo_fold_rosters.py`). The flight `SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-V1` pins it
(`configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-v1.json`:453, sha256 `47b89b9e…a944a8`; `what_this_is`:487
"4 arms over the same 1,080 Fold B rows").

| fact | Fold B | Fold A (`slurp-evo-fold-a-v2.json`) | source |
| --- | --- | --- | --- |
| official SLURP split | **train** (every id `slurp:train:*`; `non_train_ids_fold_b` 0) | train | roster :14 |
| rows | 1,080 recordings = 1,080 distinct utterances (one recording per utterance) | 1,080 / 1,080 | :31; `max_recordings_per_utterance_*` 1 |
| strata | 18 scenarios x 60 | 18 x 60 | :9765 `per_stratum` |
| channel / audio | 527 headset / 553 far-field; 3,154.8 s | 523 / 557; 3,084.5 s | :2, :36 |
| role | `roster_role` construction, `split_role` discovery | same | :9895, :9908 |
| any official test row | **no**: `shared_with_test_fold_b` 0, `shared_with_devel_fold_b` 0 | no (`shared_with_test_fold_a` 0) | :29 |

**How the folds were drawn** (`selection_rule`, :9899; seed 20260906, :9898): `core.roster.freeze_roster` with one shared
`random.Random(seed)` over each scenario's id-sorted train recording pool; at most one recording per utterance; Fold A drawn
first from the whole train pool, Fold B from the pool minus Fold A's utterances, then 430 exemplars from the rest. The 739
train utterances the study had already touched are excluded before the draws (`pool.touched_train_utterances`). The V2 draw
(60 per scenario) supersedes V1 (30 per scenario, 540 rows) by coordinator decision 1 of 2026-09-06 (`supersedes`).

**Tuning exposure (from `docs/exposure-ledger.md`).**
- *Knowledge construction: no.* The exemplar pool, slot stores and intent-boundary centroids are built from train utterances
  minus every utterance in Fold A, Fold B or DEV-1000 (the 9,354-utterance pool, baseline config `what_the_train_exemplars_store_is`;
  protocol draft `docs/readiness/2026-09-19-slurp-test-split-one-read-protocol-v1-draft.md` section 2.3). The TFRL
  construction roster R01 (800 train rows) and its knowledge-source-free subset **R01C (625 rows)** are disjoint from both folds
  (`docs/receipts/rosters/slurp-stage3-construction-r01.json` `disjointness_proof`: `slurp-evo-fold-b-v2` shared 0); R01/R01C,
  not the folds, are where the loop learns. The Fold B agreement table is a self-excluded vote over the frozen pool, not a label.
- *Method selection: yes, heavily.* 27 consumed full-roster Fold B flight run ids (P-SV-1..4, FILLER C01-C06, NX R1/R1-I2/R6,
  VER-1A, TFRL loop I1-I3, direct-only Instruct x2 and Thinking, 268-B lock and twin, NX-T R1/R2/R2b/R3, ASR339; ledger rows
  5731-9133), each followed by a READ that decided adopt / refuse. Fold A has about 70 consumed run ids (the September v2-base
  program, Stage-3 W1-W5, COMP-1/1b, ROLL-2 and others). Both folds are therefore **development populations with
  selection exposure**; every number on them is a development number (owner decision 28, protocol draft section 1).
  The plan's "Fold A = confirmation tier" (Decision 6, T2) should note that Fold A carried the earlier discovery program.

**The official test split** (`datasets/slurp/metadata/test.jsonl`, sha256 `fe3449af…0783cc`; roster receipt
`docs/receipts/rosters/slurp-test-full-13078-v1.json`:9 `counts`; re-derived here from the jsonl and FLAC headers):

| split | prompts (`slurp_id`) | recordings | headset / far-field | audio |
| --- | ---: | ---: | --- | ---: |
| train | 11,514 | 50,628 | 24,210 / 26,418 | -- |
| devel | 2,033 | 8,690 | 4,173 / 4,517 | -- |
| **test** | **2,974** | **13,078** | 6,281 / 6,797 | **36,911.9 s = 10.25 h** (headset 4.84 h, far-field 5.42 h; mean 2.82 s) |
| train_synthetic | 19,711 | 69,253 | 0 / 69,253 | -- |

**Scoring unit.** SLURP has several recordings per prompt (test: 4.40 on average). The official `evaluate.py` default mode keys
gold by `recordings[].file` (`util.py`:78-79, `release2prediction`), so a published test number is over **13,078 recordings**,
not 2,974 prompts. The repo scores the same unit: one output row per recording `sample_id`, written in the default
file-keyed mode (`src/speech_aware_evidence_acquisition/reproduction/slurp_official_eval.py`:15-46, :310-372). On the folds the
two units coincide (one recording per utterance).

**The test split is already partly exposed in this repo.** Under owner decision 141 (2026-09-27,
`docs/readiness/2026-09-16-slurp-next-stage-owner-decisions-v1.md`:357) the split was partitioned by prompt into
`SLURP-TEST-SAMPLE-3010-V1` (682 prompts, 3,010 recordings; milestone checks) and `SLURP-TEST-CLOSURE-10068-V1` (2,292 prompts,
10,068 recordings; "the roster of the ONE final read"), both in `docs/receipts/rosters/`. Test gold was read offline for the
stratified draw (`target_gold_opened_before_receipt` true). The sample was then flown four times and analysed per row:
successor-1 PLAN and COMP-1b (ledger 6109, 6111), successor-2 composed (8803), successor-3 Thinking system v3.0 (9033:
**73.05 SLU-F1 / 86.33 intent**, post-stratified; raw 72.70 / 86.94), and a row-level gap decomposition (9132;
`docs/results/2026-10-19-slurp-v3-test-sample-gap-decomposition-transfer.md`). The closure roster has no contact. The full-split
owner allowance `docs/receipts/owner/2026-09-23-slurp-test-split-allowance-v1.json`:4 is `pending-owner`.

## 2. Metric alignment

**The repo runs the official script itself.** The registered scorers (`slurp-slu-official`, `-train`, `-test`;
`core/scorers.py`:772 registers `slurp-slu-official-test`) write gold and prediction JSONL and run the **unmodified**
`scripts/evaluation/evaluate.py` of `github.com/pswietojanski/slurp` at pinned commit `8eb16545762be97ace75334109d73824217311f1`
as a subprocess (`--average micro --errors --table-layout csv`) from the local clone
`/mnt/cephfs/xingchao/omni/repos/slurp` (HEAD = that commit, 2022-04-20), in its own pinned environment
(`slurp_official_eval.py`:1-46, :117-124). A per-sample port (`reproduction/slurp_scoring.py`) must agree with it to
`_IDENTITY_TOLERANCE = 1e-6` (`slurp_official_eval.py`:557) or the score fails. The 2026-09-04 audit measured a maximum difference
of 1.1e-16 on six arms (`docs/readiness/2026-09-04-scoring-alignment-audit-slurp-earnings.md`:157-166).

| element | official `evaluate.py` (read at the pinned commit) | repo | verdict |
| --- | --- | --- | --- |
| scenario / action accuracy | micro `FMeasure` over labels (`evaluate.py`:68-69, 84-85); with one label per example micro P = R = F1 = accuracy | the official F-measure column, documented as identical to accuracy (`slurp_official_eval.py` `OfficialSluResult`); the port's per-row exact match agrees | match |
| intent | `"{scenario}_{action}"` string, micro F1 (`evaluate.py`:86-87) | `intent_accuracy` = official "intent (scen_act)" = per-row `scenario_action_exact` | match |
| span F1 ("entities") | exact `{type, filler}` match (`metrics.py`:150-162) | `span_f1` / `slot_f1_exact`; never called SLU-F1 | match |
| word / char span-distance F1 | per predicted entity whose type is in gold: TP += 1, FP and FN += lowest distance to a same-type gold filler (WER via `jiwer.wer`; char = `textdistance` normalised Levenshtein); unmatched type = FP 1; leftover gold = FN 1 (`metrics.py`:206-258, `distance.py`:39-66) | port reproduces it (`slurp_scoring.py`:131-219); headline number is the official subprocess | match |
| SLU-F1 | `SLUF1` sums the word and char TP/FP/FN, then P/R/F1 (`metrics.py`:165-182; `evaluate.py`:123-131) | `slu_f1` = official "SLU F1" row | match |
| entity normalisation | gold fillers lowercased and space-joined from tokens (`util.py`:73-74); predictions used as-is | same asymmetry (`data/slurp.py`:252-260 lowercases gold; predicted fillers are not lowercased) | match |
| empty / unparsed prediction | scored if the file key is present: a `null` scenario/action is a wrong label; no entities gives FN for every gold entity | `parse_slu_response` fails closed to `(None, None, [])` and the row is still scored (counted wrong) | match |
| missing prediction | **skipped silently**: only gold ids present in the prediction file are scored (`evaluate.py`:80-83; warning at :138) | gold file built from the delivered rows only, so the official script cannot see a missing row; completeness is enforced separately (`core/scoring_receipt.py`:178-183 "per_sample rows do not cover the complete roster exactly once") | **stricter**; equal to official when every row is delivered |
| averaging | `--average micro` default | micro | match |
| label space | none imposed | none at scoring; the system projects to the 60 train intents and normalises fillers *inside the agent* before delivery (`saea-deliver-normalise-v4g`, baseline config :282-283; `rails/delivery_normalisers.py`:1262-1276) | system behaviour, not a scorer change; projected scorers (`-projected-v1/v2/v3`) are diagnostics only (protocol draft 3.3) |

**Differences that remain.** (1) *Population, not definition*: published rows use all 13,078 test recordings; Fold B is 1,080
train recordings (one channel per prompt, equal 60 rows per scenario, not the natural mixture). (2) *Shared-fixture check*
(plan Decision 6 rule 3): the identity check is built into every scoring call, so a separate fixture adds little; the open item is
the absence of a `-test`-scored row on the full split, not the scorer. (3) Papers that report a "SF F1" / "slot F1" instead of
the official `SLUF1` are not the same metric; section 3 flags them.

## 3. Public baselines on the SLURP test split

All rows below were read on 2026-10-30 from the source named (PDF through `pdftotext`, or the raw README), by this analyst
or by a research sub-agent whose quotes were spot-checked against the downloaded text (SLURP Tables 7-8, UniverSLU Tables
3/4/6, Seo Table 2, Wang Table 2, SpeechVerse Table 4, AIR-Bench Table 6, SpeechBrain README, LauraGPT Table 1, Choi et al.
Table 2, He & Garner Tables 1/4/5). Split = official test (13,078 recordings) unless stated; metrics = official `evaluate.py`
definitions unless flagged. "Intent" in the SLURP paper is its "Scen_Act" column. Empty cell = not reported.

**3.1 Official test split, official metrics**

| # | system | source | input | training data | scen | act | intent | SLU-F1 | word-F1 | char-F1 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Gold / HerMiT | Bastianelli et al., "SLURP: A Spoken Language Understanding Resource Package", EMNLP 2020, arXiv 2011.13205, T7-8 | gold text (upper bound) | SLURP train text | 90.15 | 86.99 | 84.84 | -- (exact span F1 78.19) | -- | -- |
| 2 | Gold / SF-ID | same | gold text | SLURP train text | 86.48 | 83.69 | 82.25 | -- (exact span F1 69.87) | -- | -- |
| 3 | Multi-SLURP / HerMiT (best pipeline) | same | ASR -> NLU pipeline | external ASR data + SLURP-adapted ASR (16.2 % test WER) | 85.69 | 81.42 | 78.33 | 70.84 | 69.34 | 72.39 |
| 4 | Multi / HerMiT | same | pipeline | external ASR, SLURP NLU | 83.73 | 79.70 | 76.68 | 69.53 | 67.78 | 71.38 |
| 5 | SLURP / HerMiT | same | pipeline | SLURP-only ASR + NLU | 82.31 | 78.07 | 74.62 | 66.99 | 65.48 | 68.56 |
| 6 | Google / SF-ID | same | pipeline (commercial ASR) | SLURP NLU | 78.87 | 74.31 | 72.06 | 64.02 | 62.73 | 65.37 |
| 7 | ESPnet-SLU, Conformer + SLURP-synth | Arora et al., "ESPnet-SLU: Advancing Spoken Language Understanding through ESPnet", ICASSP 2022, arXiv 2111.14706, T2/T5 | audio, E2E | SLURP real + synth | | | 86.3 | 71.9 | | |
| 8 | SpeechBrain "Direct" | SpeechBrain `recipes/SLURP/README.md` (raw.githubusercontent.com/speechbrain/speechbrain/develop) | audio, E2E | train_real + train_synthetic | 81.73 | 77.11 | 75.05 | 63.26 | 61.24 | 65.42 |
| 9 | SpeechBrain "Direct (HuBERT)" | same | audio, E2E | real + synth + HuBERT-base SSL | 91.24 | 88.47 | 87.54 | 75.10 | 72.93 | 77.40 |
| 10 | SpeechBrain "NLU" | same | gold text | real text | 90.81 | 88.29 | 87.28 | | | |
| 11 | CTI (A+S+N) | Seo et al., "Integration of Pre-trained Networks with Continuous Token Interface for End-to-End SLU", ICASSP 2022, arXiv 2104.07253, T2 | audio, E2E (wav2vec2 + RoBERTa) | SLURP real | | | 82.93 | 71.12 | | |
| 12 | CTI + extra data (all) | same | audio, E2E | real + synth | | | 86.92 | 74.66 | | |
| 13 | NLU (RoBERTa) | same | gold text | text | | | 87.73 | 84.34 | | |
| 14 | EF-hbt-large / PF-hbt-large | Wang et al., "A Fine-tuned Wav2vec 2.0/HuBERT Benchmark for Speech Emotion Recognition, Speaker Verification and SLU", arXiv 2111.02735, T2 | audio, E2E | HuBERT-large fine-tuned on SLURP | | | 89.38 / 89.22 | 78.43 / 78.92 ("SF F1"; set against CTI's SLU-F1, identity assumed) | | |
| 15 | Conformer deliberation (two-pass) | Arora et al., Interspeech 2022, arXiv 2207.06670, T2 | audio, E2E | SLURP (+ pretrained LM) | | | 86.6 | | | |
| 16 | 3-pass SLU w/ cross attention | Arora et al., arXiv 2307.11005, T1 | audio, E2E | SLURP + BART | | | | 78.5 | | |
| 17 | UniverSLU-17, task specifier | Arora et al., "UniverSLU: Universal SLU for Diverse Tasks with Natural Language Instructions", NAACL 2024, arXiv 2310.02973, T3/T4 | audio, E2E | Whisper, multi-task on 17 datasets incl. SLURP train | | | 90.3 | 79.5 | | |
| 18 | UniverSLU-3 task-type specifier / Whisper task-specific | same, T6 | audio, E2E | Whisper + SLURP | | | 90.5 / 89.6 | 80.5 / 79.7 | | |
| 19 | LauraGPT | Du et al., arXiv 2310.04673, T1 (training mix T5: "SLURP x10") | audio, speech LLM | SLURP train in the training mix | | | 87.9 | 73.5 | | |
| 20 | Qwen2-Audio-7B-Instruct SFT, 100 % speech (direct) | Choi et al., "Exploring Fine-tuning of Large Audio Language Models for SLU under Limited Speech Data", arXiv 2509.15389, T2 | audio, speech LLM | SLURP real, fine-tuned | | | 88.13 | 76.75 | | |
| 21 | same, text-only fine-tune scored on speech / oracle text | same | audio / gold text | SLURP text | | | 83.60 / 91.29 | 72.07 / 85.93 | | |
| 22 | SpeechVerse Task-FT* | Das et al., "SpeechVerse", arXiv 2405.08295, T4 | audio, speech LLM (Flan-T5) | SLURP train, all intents | | | 84.6 | 76.7 | | |

**3.2 SLURP results under other protocols (not comparable to 3.1; listed so they are not mistaken for it)**

| # | system | source | condition | number |
| --- | --- | --- | --- | --- |
| 23 | ChatGPT, 0-shot / ASR -> ChatGPT 10-shot (20-shot 73.96) | He & Garner, "Can ChatGPT Detect Intent?", arXiv 2305.13512, T1/T5 | prompt-only LLM; gold text / Whisper text; test split; intent only | intent 79.25 / 72.89 |
| 24 | ChatGPT slot filling, 10/20/30-shot | same, T4 | gold text; slot F1 (not `SLUF1`) | 12.03 / 13.00 / 13.35 |
| 25 | SALMONN (activation-tuned) | Tang et al., ICLR 2024, arXiv 2310.13289, T3b | zero-shot speech LLM; slot-value accuracy | 0.41 (Whisper+Vicuna reference 0.46) |
| 26 | AIR-Bench SLURP intent (1k items, 4-way multiple choice) | Yang et al., ACL 2024, arXiv 2402.07729, T6 | zero-shot speech LLMs | Qwen-Audio 77.8 %, SALMONN 36.7 %, Whisper+GPT-4 87.7 % |
| 27 | Dynamic-SUPERB Phase-2 "SUPERB Intent Classification (SLURP)" | arXiv 2411.05361, T5 | zero-shot, GPT-4o-judged | best Whisper-LLaMA 36.50 %; Qwen2-Audio-7B-Instruct 26.00 %; WavLLM 28.00 % |
| 28 | WHISMA + MR | Li et al., SLT 2024, arXiv 2408.16423, T3 | held-out slot-type split (not the official test) | SLU-F1 63.3 |
| 29 | SpeechVerse Task-FT, 10-intent / 4-slot subset | arXiv 2405.08295, T2 | subset | 90.4 / 82.4 |

**3.3 Checked and absent; unverified.** No SLURP number in the PDFs of Qwen-Audio (2311.07919; its pretraining lists
unnamed intent and slot data under 1k h), Qwen2-Audio (2407.10759), Qwen2.5-Omni (2503.20215), **Qwen3-Omni (2509.17765)**,
WavLLM (2404.00656), LTU-AS, Pengi, AudioPaLM, SpeechGPT, MMAU, Kimi-Audio, Dynamic-SUPERB Phase-1. No GPT-4o or Gemini
SLURP number was found. *Unverified, no numbers used*: Wang et al. ICASSP 2020 (predates SLURP); the NeMo
`slu_conformer_transformer_large_slurp` card (not re-opened; split not stated in the repo's 2026-10-12 record).

**Counts.** 22 rows in 3.1 and 7 in 3.2 carry numbers read from source (29 verified); 2 entries are unverified and carry
no number.

**Corrections to `docs/readiness/2026-10-12-sota-comparator-table-v1.md` (not edited here).** (1) "Gold transcripts +
HerMiT ... SLU-F1 78.19" is the exact-span F1 column of SLURP Table 8; the paper prints "--" for SLU-F1 on gold text.
(2) The SLURP 90.3 / 79.5 row is UniverSLU-**17** (task specifier), not UniverSLU-14 (UniverSLU Table 3/4). (3) The
UniverSLU-3 task-type-specifier row is 90.5 / 80.5; 90.4 / 80.4 is the dataset-specifier row (Table 6).

## 4. Positioning of our numbers

**Our condition.** Frozen Qwen3-Omni-30B-A3B-Thinking (no parameter update, no SLURP fine-tuning), audio-only input, but
**not data-free**: every row reads SLURP-train-derived knowledge at run time (top-20 retrieved train exemplars with their
labels, slot contracts, an intent-boundary store, the 60-intent label inventory, and the train-statistics delivery
normaliser v4g). That is in-context use of SLURP train, the analogue of few-shot prompting (row 23), not zero-shot. Whether
SLURP is in Qwen3-Omni's pretraining cannot be ruled out: its report names no SLURP data and reports no SLURP number.

| our reading | population | input / training | scen | act | intent | SLU-F1 | source |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 268-B Thinking system, lock arm (the repo reference) | TRAIN Fold B, 1,080 rec., 60 per scenario (development) | audio; frozen Q4_K_M core + train-derived knowledge | | | **85.00** | **71.61** | `docs/results/2026-10-13-slurp-268b-thinking-system-fold-b-read.md`:30-33 |
| 268-B twin and later same-lock reads (band) | same | same | | | 84.91-85.74 | 71.61-72.75 | `docs/results/2026-10-29-fold-b-comparison-baseline-read.md`:47-56 |
| bare Thinking, direct only (266-E) | same | audio; frozen core, no knowledge | | | 76.76 | 45.39 | `docs/results/2026-10-12-slurp-266e-thinking-vs-instruct-bare-core-read.md`:31-34 |
| v3.0 Thinking system on the test sample (successor 3; the same lock v2.2 + v4g, program c2, Q4_K_M core) | TEST-SAMPLE-3010 (682 prompts), post-stratified to the full-test intent mix; raw 86.94 / 72.70 | same as row 1 | | | **86.33** | **73.05** | ledger 9033; `docs/results/2026-10-19-slurp-v3-test-sample-gap-decomposition-transfer.md`:9 |

None of these is a test-split headline: Fold B is train and selection-exposed; the test sample is a 23 % subset, read four
times and analysed per row, and its number is post-stratified.

**Like-for-like (class a, zero-shot or prompt-only speech LLM, no SLURP fine-tuning, official test, official metric):
there is no published row.** The nearest published lines use other protocols: AIR-Bench 4-way multiple choice (Qwen-Audio
77.8 %), Dynamic-SUPERB Phase-2 LLM-judged intent (at most 36.5 %), ChatGPT on gold text (79.25 % intent, 0-shot) or
Whisper text (72.89 %, 10-shot). Against these, intent 85-86 is above every prompt-only line, but the protocols differ
(closed 60-way list or multiple choice vs our free-form label, text vs audio). No published SLU-F1 exists in class a, so a
T0 SLU-F1 would be the first such row; it can be placed, not ranked.

**Not like-for-like (class b, trained on SLURP train; usually also SLURP-synth or a SLURP-weighted mix).** For reference only,
the distance from our best available number (test sample 86.33 / 73.05) to the class-b rows: SLURP paper best pipeline
78.33 / 70.84 (ours +8.0 / +2.2); ESPnet-SLU 86.3 / 71.9 (+0.0 / +1.2); CTI real-only 82.93 / 71.12; LauraGPT 87.9 / 73.5
(-1.6 / -0.5); SpeechBrain HuBERT 87.54 / 75.10 (-1.2 / -2.1); SFT Qwen2-Audio-7B 88.13 / 76.75 (-1.8 / -3.7);
HuBERT-large EF 89.38 / 78.43 SF-F1; UniverSLU-17 90.3 / 79.5 (-4.0 / -6.5); UniverSLU-3 90.5 / 80.5 (-4.2 / -7.5). Data
caveat: every class-b system was optimised on the SLURP train labels and audio (and SLURP-synth for rows 7-9, 12); ours saw
no gradient on SLURP but does read SLURP train labels in context. The Fold B reference (85.00 / 71.61) is on train
recordings with equal scenario weights, so it is not placed in this list.

**Class c (gold-text upper bounds).** Gold/HerMiT 84.84 intent; SpeechBrain NLU 87.28; RoBERTa NLU 87.73 / 84.34;
text-trained Qwen2-Audio on oracle text 91.29 / 85.93. Our test-sample intent (86.33) already exceeds the original paper's
gold-text HerMiT intent; our SLU-F1 is about 11-13 points under the gold-text SLU-F1 lines, which is where the remaining
distance is (entities).

## 5. Gaps that block a test-set headline

**5.1 What exists and what does not**

| item | state | source |
| --- | --- | --- |
| full-split roster | exists, frozen, label-free: `SLURP-TEST-FULL-13078-V1` (13,078 rec. / 2,974 prompts / 36,911.9 s), disjointness CLEAN over 60 sources | `docs/receipts/rosters/slurp-test-full-13078-v1.json`:9, :91597 |
| final-read roster under decision 141 | exists: `SLURP-TEST-CLOSURE-10068-V1` (10,068 rec. / 2,292 prompts / 28,705.5 s), never contacted | `slurp-test-closure-10068-v1.json`:13, :60522 |
| runtime view | `configs/dataset/slurp-speech-native-test-full.json` (full split only) | -- |
| scorer | `slurp-slu-official-test` registered; refuses non-test ids; one complete job, completeness enforced | `core/scorers.py`:772; `slurp_official_eval.py`:1048-1090 |
| owner allowance | full split: `pending-owner` (decision 108 and the venue, Remedy A or B, open); closure: none exists | `docs/receipts/owner/2026-09-23-slurp-test-split-allowance-v1.json`:4 |
| flight configs | none for the port and none for the full or closure roster. `configs/experiment/slurp-test-full-preprocess.json` is a Q4_K_M llama.cpp packet preparation (not authorised, obsolete for the raw-audio port); the test-sample configs are Q4_K_M milestone cells | `grep test configs/experiment` |
| port preparations | missing: the regenerated forced-causal Nemotron speech-query vectors (boundary segments and content) and the self-excluded agreement table for every test recording; ledger rows 4980-4982 (Q4_K_M era) were never consumed | `/mnt/cephfs/xingchao/omni/vectors/` has no test entry |
| D / A / T arms | not registered for any test roster; the protocol draft names only `SAEA-SLURP-TEST-FULL-LOCKED-SYSTEM-V1` and optional `-DIRECT-V1` | protocol draft 7.1 |
| engineering | the same blockers as the Fold B flight (multi-endpoint transport, flight width, namespace check) | plan section 4 rows A-G |

**5.2 A rule conflict the owner must resolve before T0.** Plan Decision 6 says T0 is "the whole official test split" flown
once per method (D, A, T). Decision 141 instead made the 10,068-row closure "the roster of the ONE final read", because the
3,010-row sample was to be compared against repeatedly; it has since been flown four times and decomposed per row. The
choices are: (i) **full 13,078**, the published population, with the 3,010 exposed rows disclosed and a closure-only number
reported beside it; or (ii) **closure 10,068 only**, clean but not the published population (post-stratify to full-test intent
shares as the sample was; still a different prompt set). Either way the allowance, the venue and a successor protocol
(the draft allows one read of one system; three methods need it amended) are owner items. Text-overlap note: 10 closure
prompts match a train construction sentence exactly and 261 share a word 6-gram (`slurp-test-closure-10068-v1.json`:61270);
this is inherent to SLURP and applies equally to the trained public systems.

**5.3 Cost at the measured rate.** Inputs: host output throughput **1,422.59 tok/s** at 32 admitted per instance over 4 TP2
instances (8 cards), and r = 5.368 lock rows per minute per instance (`docs/results/2026-10-29-npu-port-synthetic-speed-baseline-v1.md`:80-89;
preliminary, synthetic load); 268-B lock completion tokens p50 **1,964.5 (Perception) + 2,407.5 (Solver) = 4,372 per row**,
prompt tokens mean 3,589.6 + 8,563.6, historical 58.34 / 67.15 s per contact on llama.cpp at 4 slots (268-B read:152-155);
bare Thinking direct mean **1,032.0** completion tokens per contact (266-E read:200-201). Method: tokens / 1,422.59 = wall
seconds on 8 cards; card-hours = wall x 8; the registered form is ceil_0.1(1.3 x rows x 2 x T / r / 60), T = 2.

| roster | arm | contacts | completion tokens | wall h (8 cards) | card-h | card-h x 1.3 | registered form (lock) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full 13,078 | A: agent lock (1P + 1S) | 26,156 | 57.2 M | 11.2 | 89.3 | 116.1 | 81.2 -> 105.6 |
| full 13,078 | D: direct (1 contact) | 13,078 | 13.5 M | 2.6 | 21.1 | 27.4 | -- |
| closure 10,068 | A | 20,136 | 44.0 M | 8.6 | 68.8 | 89.4 | 62.5 -> 81.3 |
| closure 10,068 | D | 10,068 | 10.4 M | 2.0 | 16.2 | 21.1 | -- |

A T arm with the lock's contact shape costs as A. D + A + T on the full split is therefore about **200 card-hours (about
25 h wall on the 8 cards)** without margin, **about 260 with the 1.3 margin**. Caveats: the 1,422.59 tok/s was measured on
synthetic prompts of about 2.0k completion tokens; the real Solver prefill (8.6k prompt tokens mean) and the 3,072-token
reasoning budget (forced close on 46 % of Solver contacts) can lower it; the per-row p50 is used because the read gives no
mean. Re-derive from R-A2 and the port's own smoke before registering. The CPU encoder pass for the test vectors
(36,912 s of audio) is extra and card-free.

**5.4 Test-set hygiene (to carry into the T0 registration).**
1. Folds are the development instrument: Fold B chooses, Fold A confirms; nothing is chosen on a test row.
2. The method (protocol hash, knowledge-bank hashes, prompt-fragment hashes, decoding, harness) is frozen and recorded before
   the T0 rows are registered; T0 is flown once per frozen method, scored once as one complete job with `slurp-slu-official-test`,
   with no smoke, re-score or re-run (`slurp-test-full-13078-v1.json`:91597 `one_read_rule`).
3. Test rows are never exemplar, bank or rule sources; the knowledge builder already refuses any devel/test id (protocol draft 2.3).
4. Report the headline next to section 3.1 with the data-condition columns, plus channel (headset / far-field) and the
   exposure disclosure of 5.2; no claim of parity outside class a.
