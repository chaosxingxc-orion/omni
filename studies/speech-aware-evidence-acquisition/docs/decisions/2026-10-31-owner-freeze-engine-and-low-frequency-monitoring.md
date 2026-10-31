# Owner decision: bounded engineering screen and low-frequency observation

The owner supersedes the earlier repeated crossover/upward-search requirement on 2026-10-31:

- One complete inference configuration is sufficient for baseline comparison. Stop repeated engineering observation; prioritize system accuracy.
- Use 100 complete trajectories per each of the four instances. Existing work already exceeds this threshold; retain all extra costs and results, stop new dispatch, and drain in-flight calls.
- Cancel the queued matched-load repeat. Do not schedule 128/192/256 sweeps as a prerequisite for scientific continuation.
- Report per-sample average latency decomposed by stage, with unavailable historical fields explicit. Distinguish individual latency from inverse aggregate throughput.
- Routine task observation runs in a detached script approximately every ten minutes. The model reads completion/failure/decision signals, not repeated raw logs. No model or token expense is required for polling.

Select sequence48, batch2048, four TP2 instances, BF16, context24576, prefix caching and chunked prefill as the practical baseline. This is a frozen comparison configuration, not a claim of a global optimum. The profile records 48 admitted slots per instance, 192 across the four instances. Keep the existing 3072 reasoning/1024 answer reserve and model identity until budget contrasts are separately evaluated.

The screen's synthetic trajectories are not SLURP evaluation samples. Scientific continuation still requires correct checkpoint custody and registered serving/FIT bindings; the decision does not waive these checks. Preserve all previous costs and do not replay unresolved requests.
