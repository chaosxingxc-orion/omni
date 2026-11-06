# Runtime efficiency overlays

`efficiency-v1.example.json` is an external operational overlay. Copy it to a
run-owned configuration, pin its SHA256 as `SAEA_RUNTIME_CONFIG_SHA256`, and
mount it read-only. It enables the bounded request pipeline and preloads the
frozen roster's raw media before the first model contact. The cache is
identity-bound and fail-closed on missing or mutated files; it does not alter
request bytes or scientific fragments. Tune worker counts only after a real
600-second throughput window.

Native P→Slot→Decision flights use `native-slotfill-efficiency-v1.example.json`.
It carries the same external-config identity and enables recall prefetch,
role-input offload, ready-request admission, validation workers, and bounded
trajectory/model windows. This overlay is accepted only for a native role
episode and vLLM transport; it cannot be mixed with `request_pipeline`, node
batch/pipeline schedulers, or a scientific `dispatch` block. The effective
dispatch is copied into the runtime config hash and is reported in the native
dispatch receipt, while prompts, knowledge, roster, and algorithm parameters
remain in the four scientific fragments.
