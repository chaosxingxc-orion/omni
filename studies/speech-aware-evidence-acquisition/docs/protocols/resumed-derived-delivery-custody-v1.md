# Resumed native delivery custody

Each kept sample is held against the single predecessor trace assigned to it;
request IDs are attempt-local and must never be resolved in a merged trace.
Any request in the recovery trace for a kept sample is a conflict.

MODEL_REQUEST identifies its sample in `payload.sample_id`. A normal answer,
including a finite delivery of an earlier Solver and a delivery-normalised
answer, retains that raw request identity. Normalisation is independently
replayed against its response by I7n.

The native runner writes TOOL request/response pairs for three abstentions:
`native-terminal-abstention-v1`, `native-terminal-seal-refusal-abstention-v1`,
and `native-context-abstention-v1`. Their sample identity is the top-level
`sample_id`, accepted only after reconstructing the exact pair with the
registered delivery constructor and the row's terminal disposition. Finite
and seal-refusal derivatives must name the same source request and contact as
the disposition, and that raw request's sample and request/response hashes
must match in the same predecessor trace. Context abstention has no completed
source contact; its request ID and hashes bind the sample and disposition.
The derived request and response hashes and raw source identities are distinct;
neither is a new model call. Unknown tool schemas and conflicting raw/derived
envelopes are refused. Existing checkpoint, trace and terminal-custody audits
remain responsible for immutable bytes, full episode authority and journal
reconciliation; this identity check does not replace those checks.
