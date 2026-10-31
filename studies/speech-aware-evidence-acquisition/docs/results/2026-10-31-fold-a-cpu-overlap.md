# Isolated CPU preparation during A0 generation

## Resource finding and design

Live inspection found that the original encoder pool's CPU allocation overlaps one resident vLLM service and the active client. Its existing registration deliberately waits for generation and analysis to finish, so it must not simply be launched concurrently.

A new prospective tool run uses two CPU float32 workers, each retaining 16 threads and batch-one forwards, on CPUs 32–63. That range is disjoint from the four current serving containers and active client. Model files, package versions, causal attention geometry, pooling, prompt, audio boundaries and the full 1,080-row TRAIN Fold A roster remain inherited from the verified original encoder. Runtime concurrency, placement, preparation identity and cache/output namespace are separately registered.

## Custody and boundaries

The shared preparation runtime accepts `cpu_isolation` only inside the hashed registered protocol, only with its verified CPU speech backend, exact worker affinity, no accelerator device nodes, and outputs/cache under its own run. It uses a separate serialized CPU preparation lease through the existing durable lock implementation. Default preparations continue to use the generation lease. Success releases CPU custody; failure preserves CPU metadata for reconciliation and cannot rewrite the live generation lease.

`fold_a_cpu_overlap.py` registers the independent tool, verifies the standard tool-plan gate, then runs the standard speech preparation runtime. The runtime charges each contact before execution and seals the complete output and attempt. The coordinator validates the source identities, all vector dimensions and numeric values, receipt hashes and final attempt state.

This run does not automatically supersede the queued encoder or authorize inference from its outputs. Downstream consumers must first validate and explicitly adopt the sealed result; until then the original queue remains intact. No TFRL optimization or test-set contact is performed.

## Validation

The research subset passed 69 tests, including synthetic preparation while a real generation lease is held, byte-preservation of that lease on success and worker failure, isolated lease exclusion, backend/device/affinity/path refusals, actual registration-tool admission, and existing CPU worker-pool/preparation regressions. Synthetic backends replace the encoder in those lifecycle tests; actual retrieval-vector completion and NPU throughput remain runtime evidence requirements.
