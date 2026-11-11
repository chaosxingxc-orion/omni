# Configured S1 launch and result contract

The configured entry delegates to the existing native runtime and detached host.
The generation journal, Admission, service identity, shared guard and lease stay
with those implementations. This bridge does not introduce an HTTP transport or
an experiment scheduler.

## Immutable preparation

A task names its workspace, source/native/helper roots, model root and small model
assets, lane and service identity, manifest, knowledge, runtime policy, legacy host
and runtime. Dependencies have relative paths and SHA-256 pins. Prepare and restore
consume the same index. READY binds task/lane/index/image, artifacts and inherited
audio stat pins. A failed prepare revokes READY; identical inputs return an explicit
NOOP. Audio or metadata identity changes require source verification before new
pins are accepted. No weights or query audio belong in the publication tree.

This candidate intentionally permits only an instruction change from its pinned
baseline. It is a diagnostic configuration, not a general knowledge optimizer or
TFRL implementation. Schema equality with a successful baseline is recorded; it
is not proof of general backend compiler compatibility.

## Detached host and driver integration

Before packaging a new immutable execution tree, apply the companion
`configured-s1-host-driver.patch` to the existing driver revision that builds one
`common` list for bind and run. The patch provides the dependency mount call,
BIND-ARGV receipt and CPU-only formal-driver check. The imported shared module must
be available to the host Python process through the task's packaged source root.
The public patch is the reproducible change, rather than a copy of an unrelated
historical driver with deployment-specific defaults.

`configured_s1_entry.host(root)` injects the task's explicit model root into the
driver task. `readonly_dependency_mounts` validates absolute existing sources,
read-only targets and target collisions. Both bind and run reuse the same common
mount/environment list. The formal CPU check must call this same driver with
`cpu_bind_only=true`, fixed image and network disabled, and capture both argv.
Remove the required model mount from the captured bind argv as a negative check:
it must fail before any model contact. Do not add extra test-only dependencies.
The shared entry persists ERROR.json and rethrows delegated host failures.

Deployment never mutates a running hash-bound tree. An overlay that references a
prior read-only release retains that release as a live dependency until the
successor is retired. Actual launch/spec receipts are the resource truth when a
scientific lane file retains its old preparation root. Restore checks must run in
the final mount layout; an extra source mount can hide broken links.

## Offline terminal processing

After all lane terminal archives are committed, `configured_s1_results.census`
verifies original intent/response identity, request IDs, base64 bytes and response
hashes, then counts every known response, usage, error, unknown and absent request.
A CPU collector scores the frozen full cohort and retains incomplete samples.
Protocol PARTIAL is distinct from unfinished execution.

`scope_costs` records current inference at the top level for scorer compatibility.
Its `all_attempt_*` fields cover only the current scientific run. Pre-model failure
receipts are separately pinned with zero model calls, measured bind wall duration,
and unknown CPU consumption/reserved-device cost. Parent references include both
the complete baseline success and its rejected attempts. Known lineage totals
explicitly exclude unmeasured costs; parent files are never rewritten.

Call `result_interface(successor, terminal_path, archival_complete=True)` after the
collector succeeds. `successor` explicitly supplies scores, summary, predictions
and A/B/C manifests. Unknown or absent requests prevent READY. The interface pins
all seven inputs, checks score/cost inventory consistency, and permits known failed
terminals or incomplete protocol results. Atomically save this interface last using
`configured_s1.save`; it fsyncs the file and parent directory. No consumer infers
paths from a directory naming convention.

Notify the existing paired watcher with `notify_watcher(background_receipt)` only
after boot/start-ticks/argv verification; pidfd prevents signaling a reused PID.
If notification cannot be verified, record that failure and retain its existing
600-second fallback. A finalization exception produces an explicit
FAILED_FINALIZATION receipt and non-READY interface; never fabricate empty scores
or resend inference to complete the interface.

The current archive import configuration remains a documented transition: the
existing archive module is configured locally for the current lanes and roots.
It has not been presented as a fully general archive framework.

## Validation

From the study root, with its source on PYTHONPATH:

```sh
PYTHONPATH=src python -B -m unittest discover -s tests/unit -p 'test_configured_s1*.py' -v
```

Run this command inside the pinned saea image from the task. Tests cover READY
invalidation and NOOP, audio restore, raw journal corruption and unknowns, parent
cost separation, complete interface hashes, failure receipts, mount validation
and verified process notification. Fixed CPU tests validate engineering behavior;
real first-response and formal-driver mount receipts remain required for a launch.
