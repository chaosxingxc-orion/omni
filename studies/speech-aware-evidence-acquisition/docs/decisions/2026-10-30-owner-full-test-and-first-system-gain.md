# Owner decision: full-test comparison and first-system gain

Status: DECIDED by the owner in the active conversation on 2026-10-30. Recorded by Codex; this is a new decision, not a historical signature.

## Owner instruction (English translation)

- D1: The proposed experiment is the assistant's planned experiment and may proceed.
- D2: Report metrics on the complete test set. We have not tuned on the test set; previously reporting aggregate results on samples of it does not make those samples contaminated or unusable.
- D3: First establish and lock the initial system's gains, then iterate the system with training-free RL.
- Follow the assistant's recommendations for the other items.

Source: the owner's response to D1/D2/D3 in `repro/reports/2026-10-30-task-status-decisions-next-plan.md` in the controlling workspace. The translation above is the authorization; it does not claim the owner signed earlier per-run draft rows.

## D1: approved comparison

Proceed with Direct (D), Agent (A), and the best development-selected TFRL system (T). For the final comparison, freeze all three method identities before any final test read, evaluate each once on the same official test roster, and report A minus D and T minus A alongside absolute task metrics and costs. This supersedes decision 141's single-system restriction for this comparison.

## D2: full test is the headline population

Use all 13,078 official SLURP test recordings for the headline metrics and public-baseline comparison. The owner confirms that prior test-sample results were aggregate evaluation reports, not inputs to tuning. Record the 3,010-recording evaluation history accurately; do not infer tuning, contamination, or ineligibility merely from prior evaluation. The 10,068-recording closure subset may be reported as supplementary analysis, not substituted for the full-test headline and not treated as a condition of full-test validity.

Keep test labels out of runtime, prompt/knowledge construction, candidate selection and tuning. Freeze candidates using construction/development evidence. Use the official scorer and disclose evaluation conditions. The authorization applies to this repository's NPU reproduction and system-comparison work; no paper submission or public release is requested. Prepare a successor machine allowance and concrete registered run identities under this decision; preserve the old pending receipt and its hashes as history. Mechanical registration and repinning are implementation work, not a request to approve D1/D2 again.

## D3: establish gains before recursive iteration

Sequence the work as follows:

1. Complete engineering verification and actual D/A measurements. Lock the initial system A0 only after its gains, repeatability, error profile and cost are measured; retain D0 and the method/configuration/code identities.
2. If A0 has no reliable gain, diagnose and repair the system on development data before representing it as an improved starting point.
3. Start TFRL from the locked A0. For the first TFRL candidate, measure and lock T1 minus A0 before treating T1 as the parent of another iteration. Keep A0 as a permanent control so later results cannot hide accumulated regressions.
4. Admit subsequent candidates only under the registered acceptance and stop rules. Report both incremental and cumulative gains. No positive result is assumed in advance.
5. After development and method freeze, run the authorized final D/A/best-T full-test comparison. Do not feed those results into subsequent tuning.

This supersedes the earlier open-ended choice between merely executing a loop and demonstrating sustained improvement: the immediate research milestone is an evidenced first-system gain, followed by bounded TFRL iteration. Initial K=4 and at most three iterations, official scoring, source isolation, negative-result reporting, detached execution and checkpointed recovery remain in force.

## Execution state at decision recording

Engineering strict verification is 11/11 PASS; four-instance synthetic smoke passed. The normal recorded sweep is running. No current-port SLURP D/A/T score exists. Remaining mechanical prerequisites include a clean pinned study revision, measured budgets, consistent supply-plan pins, current runtime receipt/session binding, and concrete generation/FIT/READ ledger entries. None of these prerequisites overturns the decisions above.
