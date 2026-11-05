# Runtime efficiency overlays

`efficiency-v1.example.json` is an external operational overlay. Copy it to a
run-owned configuration, pin its SHA256 as `SAEA_RUNTIME_CONFIG_SHA256`, and
mount it read-only. It enables the bounded request pipeline and preloads the
frozen roster's raw media before the first model contact. The cache is
identity-bound and fail-closed on missing or mutated files; it does not alter
request bytes or scientific fragments. Tune worker counts only after a real
600-second throughput window.
