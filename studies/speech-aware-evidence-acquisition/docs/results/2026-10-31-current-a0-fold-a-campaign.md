# Current A0 Fold A campaign coordinator

`scripts/flights/nx/fold_a_current_campaign.py` connects the independently registered Fold A evaluation to the standard flight CLI. It validates the exact full roster, generation/FIT identities, ready cell, own supply hash, minted decision, registered delivered-score READ and accepted serving proof. It refuses pre-existing attempts or FIT and a held generation lease.

The detached caller pins a clean source commit. The coordinator records an exclusive start and runs registration check, read-only preflight and the shared campaign in order. Before each stage it rechecks the source and recipe/supply/decision/data-lock hashes; before campaign it repeats admission validation and lease checks. After campaign it audits complete native FIT coverage and reads the finalized complete official score without rescoring. Failures preserve intent, stage log, exit status and failure evidence. An existing coordinator output directory cannot be replayed.

Successful completion means one sealed final-delivery Fold A campaign and its FIT evidence. It does not mean system benefit, full branch confirmation, paired errors/cost lock or TFRL acceptance. Those analyses still require their separately registered READs. This coordinator does not yet recover a crashed worker or survive a node reboot by itself; the established detached queue/guardian handles SSH separation and runner adoption.

Research validation: 15 container tests passed in 46.71 seconds. Tests exercise real registered READ validation and command construction, refusal of altered readiness/roster/decision/supply/recipe or spent identities, and sequential failure handling with substituted CLI/FIT/scoring processes. This is not an actual dataset campaign or an end-to-end real preflight result. No model call was made and the live A0 source was untouched.

Public source plus FIT/scoring/serving/recovery dependencies: 82 tests passed in 49.37 seconds in the accepted container.
