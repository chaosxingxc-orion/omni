# Current-A0 Fold A agreement preparation

The standard exemplar-agreement builder writes its output files directly. A caller retry after interruption could overwrite partial evidence. `scripts/flights/nx/fold_a_current_agreement.py` adds source custody and exclusive fresh-output admission around that existing builder; recall, ranking, fusion and utterance-level self-exclusion are unchanged.

Before reading model-free results, the wrapper requires a clean committed comparator source and the exact derivation hashes for its baseline and experiment. The comparator remains held/check-only. It checks all 1,080 unique TRAIN Fold A identities, records an intent binding the source commit, comparator completion, dataset lock, builder hash and worker count, then exclusively claims a new artifact directory. No model, encoder, scorer or network contact is admitted.

After the standard builder returns, the wrapper verifies source evidence again, requires all rows exactly once, checks finite vote shares and their integer vote counts, checks the receipt's zero-contact declarations and source recall/self-exclusion identities, and publishes completion atomically. Failure preserves the partial artifact, log, intent and failure record. Reusing either output directory is refused; recovery requires inspection rather than automatic replay.

Validation in the accepted network-isolated container: 43 tests passed in 4.72 seconds, covering this wrapper and the existing agreement builder. The wrapper tests use synthetic tables and a substituted builder process; the standard builder tests exercise real recall and vote arithmetic. This does not prove the actual Fold A preparation has run or that the complete runtime/config path is ready.

Next execution requires the queued comparator to finish, its generated configurations to be reviewed and committed in their owning tree, and this wrapper to be integrated into that clean source. Use eight CPU workers during a campaign. The fixed artifact identity is `SAEA-SLURP-CURRENT-A0-FOLD-A-AGREEMENT-NPU-V1`; output is `prepared/SAEA-SLURP-CURRENT-A0-FOLD-A-AGREEMENT-NPU-V1`. Record the resulting receipt in the preparation ledger and register the current Fold A evaluation with matching agreement and phase-supply pins. No evaluation flight or TFRL promotion is authorized by this preparation tool.

Public source verification: the same 43 tests passed in 4.60 seconds in the accepted container.
