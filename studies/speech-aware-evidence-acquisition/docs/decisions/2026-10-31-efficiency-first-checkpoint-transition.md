# Efficiency-first checkpoint recovery

The owner explicitly requires experiment efficiency and useful work across all
eight NPUs, with asynchronous CPU analysis and checkpoint recovery. When a known
engineering improvement warrants stopping a slow experiment, preserve completed
rows and all spend, repair the engineering path, and resume missing rows under a
fresh attempt. This supersedes earlier guidance to wait for the slow attempt to
finish and does not authorize changing the scientific method or filtering rows.

## Implemented recovery boundary

The optional `checkpoint_compatibility` cell declaration binds a predecessor Git
commit, its four fragment names, a fresh successor attempt and exact reviewed
source changes. The runtime reconstructs the predecessor configuration from Git,
checks it against the original checkpoint config/protocol/source hashes, checks
all current source files against the reviewed before/after pins, and compares the
actual configuration. Model, dataset, roster and method remain identical. Only
fresh attempt/FIT bookkeeping, readiness and output-identical prepared supply may
change. The current protocol is recomputed, never embedded in its own config.
The existing strict checkpoint reader remains unchanged. Every complete row is
kept and only the complementary roster is dispatched.

The full campaign coordinator accepts named, hashed ABORTED checkpoint
predecessors while refusing an existing successor or any undeclared attempt.
Preflight still checks registration, source cleanliness, supply and exposure.
This implementation is not evidence that a real successor has run: its admission,
FIT ancestry, supply preparation and detached queue remain to be completed.

## Verified A0 checkpoint

The controlled-stop attempt retained 427 of 1,080 rows. The shared reader verified
all 427 as keepable with matching request/response record names, no torn tail,
and an unchanged SHA-256:
`8fc5edc100096930c9cd78a0eed582d463249343111da766bf04c34220eea091`.
The materialized evidence trace hash is
`e5546d0f6bef4658a7d55181ec138b4c690aa4c8adc7af68d18729dff93eadc7`.

The attempt ledger records 8,108 calls and 21,550.3005 audio seconds; complete-row
checkpoints account for 7,259 calls and 19,367.59425 audio seconds. The difference
(849 calls, 2,182.70625 audio seconds) remains charged to the interrupted attempt;
it is not reusable complete-row output. Remaining rows: 653. No full A0 score or
post-fix throughput result exists yet. Four rows can test functionality only;
a sufficiently wide continuation window must establish measured throughput.

Publication verification also exposed and repaired a missing campaign-family
argument in the public column reader. Its default family remains unchanged;
successor readers can now pass their registered family explicitly.
