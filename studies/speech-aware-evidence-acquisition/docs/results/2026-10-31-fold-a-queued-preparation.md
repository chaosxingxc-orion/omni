# Queued current-A0 Fold A preparation

The previous queue stopped after the comparator, leaving source integration, agreement and held evaluation derivation dependent on a manual session. `fold_a_current_prepare_queued.py` now runs these steps in a separate pinned worktree after the comparator queue job succeeds.

Before writing it verifies the upstream queue, exact A0 source and model, completion ancestry and every retrieval artifact. It reproduces the comparator transform from the actual source, compares its full output bytes and imports only the two configurations and derivation record. A real Git commit establishes clean custody before the standard agreement builder executes. The held actual-A0 evaluation is then derived and committed separately. Any unrelated tracked, staged or untracked file refuses the commit. Intent, terminal evidence and partial files survive failure; there is no automatic replay.

This job is model-free preparation, not a ready NPU flight. Registration, mint, own supply, preflight and campaign remain outstanding. It does not alter the current A0 or the pinned comparator producer, and it does not relax the encoder's global lease or parent dependency.

Validation: 51 tests passed in 23.48 seconds in the accepted container, covering real Git custody, comparator reconstruction/refusal, coordinator failure preservation, agreement and evaluation. Coordinator tests substitute agreement/evaluation work; actual queued preparation awaits its upstream jobs. No real Fold A inference or sustained eight-card utilization is established by these tests.
