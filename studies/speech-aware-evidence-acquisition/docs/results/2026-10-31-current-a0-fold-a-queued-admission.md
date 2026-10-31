# Queued current-A0 Fold A admission and dispatch

The queue now has an executable bridge from committed, model-free preparation to independent admission and a full 1,080-row campaign. A separate pinned controller runs `fold_a_current_admit_queued.py`; it must not change the preparation checkout before its existing pinned job runs.

## Admission

Require the exact queue identity and a successful preparation job. Verify the clean preparation HEAD and its ancestry: only the six comparator/evaluation configurations and derivation records may differ from the registered base. Require a released generation lease and pin the preparation completion, dataset lock and accepted deployment binding.

Run the target checkout's registration tool, commit only the exact registered output manifest, mint the independent decision, and prepare its own phase supply. Commit only the owned package/configuration/ledger changes. Preserve exclusive intent, stage logs, exit records and any failure; never automatically replay a partially spent admission. The completion binds the clean ready-source commit and all inputs, but does not assert launch readiness.

## Dispatch

Require successful admission in the same queue, the exact clean admitted commit and unchanged inputs. Invoke the target checkout's existing campaign coordinator, which performs registration checks and actual read-only preflight before the shared runtime. The NPU queue job also waits for the no-other-owner card policy. Official full-manifest scoring and FIT evidence follow generation; diagnostic READs and the Direct/A0 benefit lock remain separate work.

## Validation and limits

The research-source subset passed 50 tests in 83.55 seconds, covering real Git commits, exact upstream identity, input drift, stage failure, unrelated changes, replay refusal and dispatch gates. Admission subprocesses are substituted in these chain tests: they do not prove real registration, repin, preflight or inference. An independent model-free environment check validated the accepted four-instance deployment proof and Git/lock access with zero model contacts.

Queueing these stages does not prove eight-card saturation. The live A0 remains unchanged; the corrected thread-pool executor applies to the future Fold A source. CPU encoding still waits for the existing global generation lease and parent analyses, so an idle gap remains possible until resource isolation is implemented.
