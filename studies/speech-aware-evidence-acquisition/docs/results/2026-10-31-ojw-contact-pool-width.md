# OpenJiuwen blocking contact concurrency

The live corrected-phase A0 admits 96 logical slots but uses `asyncio.to_thread`
for blocking model calls. Python 3.12's implicit default executor has 32 workers.
The observed process had 33 threads and approximately 32 simultaneous requests
across four endpoints. This is an additional concurrency limit after phase width
forwarding; it is not proof of a model-quality defect.

The independent preparation tree now sizes the fresh event loop's default pool
to the minimum of admitted slots, queue depth and roster size (at least one).
The existing slot gates, cancellation draining, prefix routing, local-tool pool
and model request path remain authoritative. `asyncio.run` shuts down the pool.

Validation in the accepted container, with networking disabled: 110 tests passed
in 18.60 seconds. Real blocking barriers reached 4, 40 and 96 concurrent callbacks
without duplicate slots; routing, node pipeline/batching and timing regressions
also passed. No accelerator or dataset contacts were made by these tests.

This repair is not deployed to the running A0. Preserve that flight's source,
checkpoints and true costs. Future experiments require a new pinned execution
commit and measured endpoint concurrency/utilization; CPU callback tests do not
prove NPU saturation or end-to-end throughput improvement.
