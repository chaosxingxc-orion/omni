# NPU port deployment v1 (2026-10-28)

## Deployment layer and resident serving (draft)

Status: DRAFT. Nothing in this note has run on an NPU card. Host 181's cards were all held by another job when this was
written (about 62.8 of 65.5 GB HBM used and one process on each card). `plan` was exercised read-only on 175 and 181.
`up`, `wait`, `status` against live instances, `down` and the sweep have not run against a real server.

### Why a separate layer

Owner decision, 2026-10-28. The serving facts split in two:

| | hashed (protocol) | recorded, not hashed (deployment) |
|---|---|---|
| what | weights, the 24,576-token per-sequence window (`--max-model-len`), decoding params, reasoning budget 3,072, `max_tokens` 4,096, parsers, `--limit-mm-per-prompt`, `--seed` | instance count, tensor parallel (TP) size per instance, `--max-num-seqs` per instance, `--gpu-memory-utilization`, `--max-num-batched-tokens`, card assignment, ports, container names, cpusets |
| owner | the model fragment (`configs/model/*.json`, `server_args`) | the deployment profile (`configs/deployment/*.json`) |

A deployment change must never look like a protocol change. A protocol change must never hide inside a deployment
profile. We deploy once and leave the deployment up. `--max-num-seqs` is a ceiling that is fixed when the instances
start. Experiments vary only the client-side concurrency below it.

### Profile (`core/deployment.py`, `configs/deployment/`)

- Schema `saea-deployment-profile-v1`. The loader refuses:
  - unknown or missing keys;
  - cards, ports, names or cpusets that overlap between instances;
  - an instance whose card count differs from `tensor_parallel_size`;
  - a `model_relpath` that is not a plain relative path;
  - a `profile_id` that differs from the file stem.

  A profile carries no free-form server flags, so it cannot carry a protocol flag.
- One absolute path. `data_mount` is the data root. It is mounted at the same path in every container, and the weights
  are at `data_mount/model_relpath`. If the data root moves (the planned move to `/mnt/cephfs/xingchao/omni`), the root
  is overridden in one place: `data_root=` / `--data-root` / `OMNI_ROOT`. The profile file does not change.
- `qwen3-omni-thinking-bf16-181-4xtp2-v1` is the resident deployment:
  - four instances, TP2 each;
  - cards [0,1] [2,3] [4,5] [6,7], ports 8200-8203, containers `saea-omni-i0..i3`;
  - `max_num_seqs` 64 (the resident scheduling ceiling, fixed at deploy time and at least the sweep's top level 48; it reserves no KV, and each experiment's client concurrency comes from the sweep's knee), `gpu_memory_utilization` 0.92.

  The cpusets follow each card's PCI NUMA node, as read on 181: NPU 0-1 are on node 6, 2-3 on node 4, 4-5 on node 0
  and 6-7 on node 2. Each instance is pinned to its own node's 32 cores (192-223, 128-159, 0-31, 64-95); the other 128 cores (nodes 1/3/5/7) stay free for the flight harness and CPU encoder runs. `HCCL_IF_BASE_PORT` is set to a different value
  per instance because all four instances share the host network namespace. This is defensive and has not been
  verified.
- `qwen3-omni-thinking-bf16-181-2xtp4-v1` is the TP4 comparison arm for the sweep only. It runs two instances on cards
  [0-3] and [4-7].
- `instance_server_argv` builds the arguments in this order:
  1. `serve <data_mount/model_relpath>`;
  2. the deployment flags that this module owns (`--served-model-name` from the fragment, `--host`, `--port`,
     `--tensor-parallel-size`, `--gpu-memory-utilization`, and `--max-num-batched-tokens` when it is set);
  3. the fragment's `server_args`, passed through the same `core.session.serving.merge_server_args` that sessions and
     flights use. Only `--max-num-seqs` is replaced, with the profile value. `--max-model-len` remains the fragment's
     protocol window.

  A fragment that carries a deployment flag is refused, so each fact has one owner. llama geometry flags are refused
  by the merge.
- `deployment_record(profile)` (`saea-deployment-record-v1`, `hashed: false`) contains:
  - the profile id and sha256;
  - the data mount, model path and image;
  - instance count, TP, `max-num-seqs`, memory utilisation and card count;
  - for each instance: name, cards, port, base URL, cpuset, environment, and the sha256 of its exact `vllm serve`
    argv.

  It also copies `protocol_window_tokens`, so a reader can see that the deployment did not change the window. This
  record is what the run manifest and the runtime receipt should carry.

### Card allocator

`allocate_cards` / `allocate_from_npu_smi` take the rows from the study's own `telemetry.parse_npu_smi_table`, plus
the process rows that `npu_smi_process_cards` reads. They refuse every requested card that:

- is absent from the table;
- does not report health `OK`;
- uses more HBM than the profile's `busy_threshold_mib` (4,096);
- holds a process, even if its HBM use is low.

The allocation is all or nothing, and a refusal names every refused card with every reason. An unparseable table
refuses every card. `instances_for_cards(profile, n)` maps an experiment's declared card count to whole instances:
smoke 2 gives `i0`, flight 8 gives all four. A 1-card encoder probe is not a TP2 instance and needs a card outside this
deployment.

### Launcher (`work/scripts/npu_deploy.sh`, host-side, outside the repo)

- `plan <profile>` prints the exact docker command for each instance, with the card verdict when the host has
  `npu-smi`. It starts nothing.
- `up <profile>` runs these steps:
  1. Refuses unless `hostname` equals the profile's `host`.
  2. Refuses if the weights are missing.
  3. Refuses if a container with a profile name already exists, or if a profile port is already listening.
  4. Captures `npu-smi info` and runs the study module's `emit`. `emit` refuses busy cards and prints NUL-delimited
     argv. The launcher executes that argv directly, without `eval`.
  5. Starts each instance as a detached, named container that stays resident: `--restart unless-stopped`,
     `--network host`, only its own `/dev/davinci<N>`, its own cpuset, the data root bind-mounted, deployment labels,
     capped json-file logs, and `HF_HUB_OFFLINE=1`.
  6. Waits for `/health` and for `/v1/models` listing the served name on every port. It fails fast if a container
     exits or restarts.
- `status` prints the container state, `/health`, `/v1/models` and HBM per card.
- `down` stops and removes only containers whose `saea.deployment.profile` label equals the profile id.

The profile logic runs inside a throwaway `saea-base` container with no devices and `--network none`. Every path
derives from `OMNI_ROOT`, or from `SAEA_SRC` for the source tree. Nothing depends on llama.cpp slots.

### Concurrency sweep (`scripts/serving/synthetic_sweep.py`, stdlib only)

The load is synthetic and shaped like the Fold B mix. It contains no dataset row, so it needs no exposure ledger entry.

- Prompt length: per-contact prompt-token targets come from a lognormal truncated to 3,000-20,000 tokens, with a
  truncated mean of 8,500. The targets are hit exactly through the server's chat-form `/tokenize`, which uses the same
  request shape as `measure_vllm_multimodal_request`, so the audio placeholder tokens are counted. The remainder is
  made up with single-letter pad words in the state suffix.
- Message layout: `native-raw-speech-query-prefix-v1`, i.e. system instruction, then a user turn of [`input_audio`,
  stable evidence, mutable state]. No `tools` are sent.
- Audio: a synthetic 16 kHz WAV of 2-8 s. It is a voiced harmonic buzz under a syllable envelope, plus noise.
- Trajectories: three contacts each, sharing one audio clip and one word-stream prefix, so prefix caching sees the
  reuse that a real trajectory gives it.
- Decoding: temperature 0, `max_tokens` 4,096, `thinking_token_budget` 3,072, `enable_thinking: true`.

Driving and recording:

- Concurrency levels are per instance. Level `c` over `N` instances runs `c*N` closed-loop workers. Worker `w` is
  pinned to instance `w mod N`.
- Each level gets its own seeded content, so it cannot reuse the prefix cache that an earlier level filled. All
  prompts of a level are fitted before its clock starts.
- Per request, the report records:
  - target, tokenized and server-reported prompt tokens;
  - completion tokens, latency and streamed TTFT;
  - status (`ok` / `timeout` / `error`), with the 420 s total deadline.
- Per level, the report records:
  - output tok/s;
  - end-to-end and decode per-sequence tok/s at p50 and p10;
  - latency p95.
- Before and after each level, the sweep reads the `/metrics` counters (sampled during the level as well):
  - `vllm:num_preemptions_total`. vLLM 0.30's counter is `vllm:num_preemptions` in `v1/metrics/loggers.py`;
    prometheus_client adds `_total`.
  - prefix-cache queries and hits.
  - `vllm:kv_cache_usage_perc`, `num_requests_running` and `num_requests_waiting`, sampled during the level.
- The recommended knee is the highest level with:
  - zero errors and zero timeouts;
  - a preemption delta that is readable and zero;
  - p10 end-to-end tok/s >= 10. At that rate, 4,096 tokens fit inside 420 s.

  An unreadable preemption counter fails the level.

Sweep procedure:

1. Deploy each profile.
2. Run `--levels 8,16,24,32,48` against the instances, with `--deployment-profile` so the report embeds the profile.
3. Compare the 4xTP2 and 2xTP4 reports.
4. Set `max_num_seqs` to at least the chosen knee and redeploy once.
5. Validate with the registered 4-row smoke.

The deployment's `max_num_seqs` must be at least the highest level swept. Otherwise the server queues requests
(`num_requests_waiting > 0` in the gauges), and the sweep measures the ceiling instead of the knee.

### What the flight-side integration still needs (not built here)

- A multi-endpoint transport that routes each contact to an instance by the flight's per-trajectory slot
  (`slot mod instance_count`, the same affinity the sweep uses) and holds a session and runtime receipt per instance.
  Today a fragment names one `base_url`.
- `id_slot`. `core/flight/runtime.py` (around line 6686) sets `decoding_params["id_slot"] = slot` on every contact,
  and the vLLM transport refuses `id_slot` (`VLLM_REFUSED_REQUEST_KEYS`). The integration must route by that slot and
  strip `id_slot` before the wire. This is not fixed here.
- The protocol hash must stop covering the fragment's `--max-num-seqs 4` and `profile_slot_count`. Either the hash
  excludes them, or the deployment record supplies them. This belongs to the main thread and the runtime receipt
  work.
- The run manifest and receipt must embed `deployment_record(profile)`.

### Open questions

- The ceiling `max_num_seqs` 64 only bounds scheduling; whether the knee fits the KV budget is the sweep's question. The KV estimate is about 96 KB per token and roughly 25 GB of KV per card after
  the weights at 0.92 utilisation. Both must be checked against vLLM's reported KV block count when the instances
  first start.
- `HCCL_IF_BASE_PORT` per instance, and simultaneous starts of four TP2 instances on one host network: confirm on the
  first `up`, or start the instances one at a time if they collide.
- Resolved (main thread): each instance owns its NUMA node's 32 cores and 128 cores stay free for harness and encoders.
- Does `usage.prompt_tokens` equal the `/tokenize` count for an audio request? The sweep records
  `prompt_tokens_server_matches_tokenize` for each level. The NPU smoke checks the same thing.
