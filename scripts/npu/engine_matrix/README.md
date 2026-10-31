# Resumable engine matrix

This runner is prepared and model-free tested, **not yet run on models**. It never changes a deployment. Current capacity `run-v1` files and active scientific source are untouched.

## Workload and comparison

`plan.json` is a four-round crossover: L24/L28/L32 rotate over all four TP2 instances, with a second L24 control in each round. Each instance processes 128 trajectories × 3 sequential contacts (384 requests). A dynamic queue feeds workers; a trajectory stays on one endpoint to retain prefix affinity. Prompt target lengths and audio durations are identical across rounds; content seed changes per round to avoid previous-round cache reuse. Within a round, all instances receive identical synthetic content. Different generated lengths remain a possible throughput confound and must be reported.

Tokenizer fitting completes before a four-instance barrier and the timed generation window. Fit elapsed/calls are recorded separately. Every tokenizer request and generation request has an fsynced atomic intent/result pair. Interrupted or resumed trials retain costs and progress but cannot themselves establish an uninterrupted performance baseline. Successful requests are skipped; unresolved intents refuse blind replay. A completed trial is a no-op after checking coverage and result hashes. Graceful SIGTERM stops admitting new work and drains current HTTP calls; SSH/SIGHUP does not stop it.

Per-trial outputs include processed/total, seconds/request, ETA, 2-card/1-instance resources, fit time, whole-generation wall time, raw metrics every 5 seconds, and a continuously fed interval between warmup completions and queue depletion. A short or missing interval is unqualified. Summarize all four trials for host throughput; never average individual rates as host rate. Request metrics are not A0 complete-sample throughput.

## Binding and launch

The runner verifies actual Docker image digest, served model, endpoint port, TP2, context window, explicit sequence/batch flags, and startup evidence for prefix caching/chunked prefill. An omitted batching flag is recorded as a default, not inferred numerically. It holds `generation.guard`, refuses `flight.lock`, and requires idle endpoint gauges. Rounds run separately so engine changes can be recorded between them.

After the current probe finishes, use the formal pinned image on `npu2`, `--pid host --network host`, root container user and free CPUSET. Live inspection requires a Docker socket mount (`-v /var/run/docker.sock:/var/run/docker.sock:ro`). Docker API capability remains privileged despite a read-only socket mount; this runner issues only GET inspect/logs via Python HTTP over the Unix socket. The image has no Docker CLI; none is needed. No model request is made before live validation.

Example Python arguments:

```
matrix.py --plan plan.json --out <fresh-or-resumed-external-directory> --round 0
```

The manifest is a measurement specification, not a scientific admission. No test rows or evaluation labels are used. Do not launch alongside another capacity probe. `engine-candidates.json` records later candidate argv changes for parent review; it does not perform restarts or authorize an unreviewed identity.

## Validation

Formal `saea-base` model-free tests: **14 passed**. Tests cover result reuse, uncertain-request refusal, body changes, orphan evidence, image/default/cache mismatches, matched distributions/fresh content, tokenizer-call recovery, stop-before-contact, steady counters, completed-run no-op and generation failure stopping subsequent contacts.

Public runner defaults to this repository’s study source. Set `SAEA_STUDY_ROOT` and `SPEECHRL_DATA_DIR` for an operational checkout. The current wide search starts at 32/48/64/96 and continues upward when throughput improves; the included 24/28/32 crossover plan is a control example, not an upper limit.

## Read-only result dashboard

`summarize.py --plan <measurement-plan.json> --out <trial-directory> --summary <dashboard.json>` reads durable requests and metrics without contacting the engines. It reports successful requests, complete three-contact trajectories, resources, elapsed time, estimated remaining time, output lengths, p95 latency, prefix-counter deltas and KV/preemption evidence. Host throughput uses the union of trial wall intervals, never the sum or mean of per-instance rates. Resumed/multi-segment runs retain counts but are excluded from clean performance comparison. The fed-window screen requires at least 60 seconds; crossover confirmation is still required. Partial ETAs are provisional and can be distorted by the drain tail. Two targeted synthetic tests cover aggregate wall time, partial trajectories, failed/unknown requests and resumed-rate refusal.
