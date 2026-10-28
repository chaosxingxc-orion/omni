# NPU / vLLM-Ascend port of the Thinking Fold B baseline: engineering design

Date: 2026-10-28. Branch: `npu-vllm-port` (off `fba45821`). Status: ENGINEERING ONLY. Nothing in this document or on the
branch authorises a flight, a readiness flip, an exposure entry or an admission. The governance changes this port
needs are listed in section 9 as OPEN owner decisions and are not implemented.

## 1. Target

Reproduce `SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-V1` (4 arms over TRAIN Fold B 1,080) on:

| | historical (the study) | port |
|---|---|---|
| host | WSL2, one GPU | Linux, Ascend 910-class NPU, container `saea-base:20261027` |
| server | patched llama.cpp `llama-server` | vLLM 0.30 + vllm-ascend 0.30.0rc2 |
| core model | Qwen3-Omni-30B-A3B-Thinking GGUF Q4_K_M, mmproj-free decode over pre-materialised speech packets | Qwen3-Omni-30B-A3B-Thinking BF16 (HF safetensors), audio tower in the server |
| retrieval encoder | Omni-Embed-Nemotron-3B via `transformers`, bidirectional remote code, CUDA | the same weights served by vLLM pooling, CAUSAL attention |

The dataset, split, arms, prompts, knowledge stores, policies, budgets and agent engineering (openJiuwen layer, tools,
scoring, state, Thinking governance) are reused unchanged. The port replaces the serving backend and only what is
bound to it.

## 2. Reused unchanged

* `LlamaServerTransport`'s whole request path: payload binding, media confinement, prompt orders, reasoning-content
  guard, decision-286 reasoning projection, envelope, and the per-send session re-verification.
* The fragment knob validation of decisions 275/276 (`reasoning_budget_tokens` + `answer_reserve_tokens` =
  `max_tokens`, forced-close message hash).
* Every scorer, ledger, receipt and config-hash rule.

## 3. Transport (implemented, `core/model/vllm_transport.py`)

`VllmServerTransport` subclasses `LlamaServerTransport` and changes only the wire dialect, at the injected `post` seam:

| concern | llama-server | vLLM 0.30 | port rule |
|---|---|---|---|
| reasoning budget | `thinking_budget_tokens` | `thinking_token_budget` | renamed; a direct `thinking_token_budget` in a payload is refused |
| model | implicit | `model` | always the fragment's `served_model_name`; a different model in the body is refused |
| slot affinity | `id_slot` | none (automatic prefix caching) | refused: a fragment that declares it was written for the other server |
| reasoning channel | `reasoning_content` | `message.reasoning` | moved to `reasoning_content`, the only name the decision-286 projection strips; two different channels are refused |
| null usage details | absent | `*_tokens_details: null` | dropped |
| context overflow | `exceed_context_size_error` | `BadRequestError` 400, message only | the two exact vLLM 0.30 templates (`v1/engine/input_processor.py`), full match, are relabelled to the llama type so E20 `compact-or-abstain` still applies; every other 400 stays hard; a reworded upgrade fails closed |
| forced-close message | `--reasoning-budget-message` | `--reasoning-config {"reasoning_end_str": ...}` | the message is the text before the trailing natural `</think>`; hashed exactly as before |

llama-server-only flags (`--reasoning-budget-message`, `-np`/`--parallel`, `-c`/`--ctx-size`, `--reasoning-format`)
and a `feature_cache_guard` are refused for a `vllm` fragment. Two further refusals come from the runtime-identity
seam (section 9.3): the fragment's `served_model_name` and its PROTOCOL flags must equal the vLLM runtime receipt's,
and `_assert_session_still_valid` -- the check the base class runs at construction and before every single send --
asks the seam, so a `vllm` fragment's session is re-proven by the vLLM identity (section 9.2) at exactly the same
sites and the same frequency as a llama fragment's is by the llama one.

## 4. Reasoning budget

vLLM's per-request `thinking_token_budget` counts tokens after the reasoning start token and, when exhausted, forces
the token sequence of `reasoning_end_str`. Qwen3-Omni Thinking's template does not pre-insert `<think>` when
`enable_thinking` is true, so the count starts at the model's own `<think>`, as llama.cpp's budget sampler does.
vllm-ascend implements the state on NPU (`vllm_ascend/worker/npu_input_batch.py`, upstream e2e test
`tests/e2e/pull_request/one_card/test_thinking_budget.py`). The fragment keeps 3072 + 1024 = 4096.

Forced string: `<the 266-E message>\n</think>\n\n`. The exact whitespace llama.cpp inserts around the message is
verified on the first NPU smoke by reading the reasoning bytes of one forced close; until then it is an ASSUMPTION.

That the template does not pre-insert `<think>` is verified offline (section 5.5); only the forced-close bytes need the
server.

## 5. Speech input

### 5.1 What Fold B sends

The Fold B baseline arm (`speech-feature-query-native-tools`) sends exactly one speech part per request: the query
packet (`native_contact_fit.max_speech_packets = 1`; knowledge stores are text-only, `expose_facets: false`). The
packet is a llama.cpp mmproj output materialised offline from the item's `media_relpath` (bound by `audio_sha256`) and
placed by `speech-query-prefix-v2` as:

`[query packet, stable evidence, key packets (none in Fold B), mutable evidence/state]`

with the stable/mutable split: knowledge roles present: stable = key texts, mutable = the rest plus state; else state
present: stable = all evidence, mutable = state; else stable = evidence[:-1], mutable = the last item.

### 5.2 Port rule

vLLM has no packet input: its audio tower consumes raw audio. The port therefore runs the study's existing raw arm
mechanics (`raw-speech-agent-native-tools`: `raw_speech_query {media_relpath, sha256}` in the payload, original audio
loaded under the data root and re-hashed on every send, no packet promotion) under a NEW prompt order
`native-raw-speech-query-prefix-v1` (`core/model/transport.py`, `_raw_query_prefix_content`). It renders the
`speech-query-prefix-v2` layout bit for bit with the original audio in the query-packet slot; the layout function is
the shared `_ordered_speech_feature_content`, not a copy. The historical `native-raw-speech-v1` (audio last) is not
used: it would change the prompt prefix the agent sees, which is an experiment change, not a backend change.

Tested (`tests/unit/test_vllm_speech_layout_and_fit.py`): layout equality with prefix-v2 across the empty, evidence,
state and key/value-role cases; the audio first on the wire; raw-identity and mixed-input refusals still apply.

### 5.3 Token fit

The native contact fit measured a request by llama-server's multimodal tokenize. For `transport: vllm` it measures
through vLLM's chat-form `POST /tokenize` (`measure_vllm_multimodal_request`): the same `messages`, `tools` and
`chat_template_kwargs` the transport sends, `add_generation_prompt: true`, `add_special_tokens: false`; the server
runs the full multimodal processor, so the count includes the audio placeholder tokens. A response is refused unless
`count` is an integer equal to `len(tokens)`. The row schema is unchanged (`measurement: vllm-chat-tokenize`). That
the count equals the engine's prompt length for an audio request is checked on the NPU smoke (section 11).

### 5.4 Consequences carried by the port

* The raw fit makes one endpoint per contact; the raw contact cap (16,200) applies instead of the packet cap.
* `_phase_seed_inputs` hashes the query identity, so prepared supply is re-minted for the port; the historical
  prepared supply is not reused.
* Knowledge-store supply and prepared recall resolve speech bindings from packet receipts. Fold B's stores are
  text-only, so the port needs no packet bindings there; an arm with speech-bearing stores would need raw bindings
  first (not in scope).

### 5.5 Chat template under vLLM (verified offline)

The shipped Thinking `chat_template.json` was rendered offline with a transformers-style sandboxed jinja2
environment, and vLLM 0.30's own content-format detector (`vllm.renderers.hf._detect_content_format`) was run on it:

| finding | consequence |
|---|---|
| `enable_thinking: true` ends the prompt at `<\|im_start\|>assistant\n`, no `<think>` | the budget counts from the model's own `<think>` (section 4) |
| `enable_thinking: false` ends it with the empty block `<think>\n\n</think>\n\n` | the decision-278 answer-only re-ask is supported as sent |
| an earlier assistant turn's reasoning is not rendered | consistent with `drop-reasoning-v1` |
| one audio part renders exactly one `<\|audio_start\|>...<\|audio_end\|>` span | the raw query slot is one span |
| vLLM detects the content format as `openai` and wraps every string content into text parts | the system message reaches the template as parts |
| with `tools`, the template computes `messages[0].content + '\n\n'`, which raises on parts | a system message beside tools fails on vLLM |

Fold B declares no `phase_tools` (a tool-free native arm), so the baseline never sends tools and is unaffected. A
tool-bearing arm on vLLM needs a decision first: a port chat template that joins the system text parts (a declared
template difference), or `--chat-template-content-format string` (which changes how vLLM places the audio and would
need its own layout proof). Neither is implemented. The NPU smoke records the server's behaviour (section 11).

## 6. Retrieval encoders (implemented)

### 6.1 Serving views (`core/vllm_serving_view.py`)

vLLM 0.30 serves Qwen2.5-Omni Thinker models only under the full Omni architectures, reading
`hf_config.thinker_config` and mapping `thinker.*` tensor names. The three shipped encoders are bare Thinker
checkpoints (`model.*`, `audio_tower.*`, `visual.*`, no `lm_head`). A serving view is a new directory with exactly
these declared differences:

* every tensor name prefixed `thinker.`; the safetensors data regions are byte-identical (SHA-256 recorded both sides);
* `config.json` = the source config as `thinker_config` of a `qwen2_5_omni` config, `auto_map` dropped (no remote
  code), `tie_word_embeddings: true` when no head is shipped (pooling never computes logits; an untied absent head fails
  vLLM's all-weights-loaded check);
* `chat_template.jinja` = the source `additional_chat_templates/sentence_transformers.jinja`;
* attention is causal: Nemotron's bidirectional override (`modeling_nv_omni_embed.py`) is not carried (owner decision
  2026-10-28).

Views (built on CephFS under `models/serving-views/`):

| view | source | pooling | dims |
|---|---|---|---|
| `omni-embed-nemotron-3b-causal-v1` | Omni-Embed-Nemotron-3B | MEAN, L2 | 2048 |
| `lco-embedding-omni-3b-2605-v1` | LCO-Embedding-Omni-3B-2605 | LAST, L2 | 2048 |
| `lco-embedding-omni-7b-v1` | LCO-Embedding-Omni-7B | LAST, L2 | 3584 |

Serve: `vllm serve <view> --runner pooling --convert embed --pooler-config {"seq_pooling_type": ..., "use_activation": true}
--chat-template <view>/chat_template.jinja --chat-template-content-format openai --dtype bfloat16
--limit-mm-per-prompt {"audio":1,"image":0,"video":0}` (`vllm_embedding_serve_args`).

### 6.2 Pooling client (`core/vllm_embedding.py`)

Reproduces `core.speech_embedding.encode` over chat-form `/v1/embeddings`: the same content parts in the same order
(`["query: ", audio]`, `["passage: " + text]`), no generation prompt, the exact `read_audio_span` waveform as a 32-bit
float WAV (lossless). A response is refused unless it is one finite unit vector of the recipe dimension within the
registered prompt-token bound. Before the first request it verifies the view against its pinned receipt hash, the
view's sources against the pinned encoder inventory, and the server's `/v1/models` (exactly one model, the served name,
rooted at the view). `VllmSpeechEncoder` / `VllmPassageEncoder` are drop-in `backend_factory` products for
`speech_preparation_runtime.run_preparation` and the passage run.

Comparison with the historical sealed vectors is by ranking agreement only (owner decision 3, 2026-10-28): the vectors
are regenerated, causal, and from different kernels.

### 6.3 Served-encoder inventory (preparation wiring)

The speech-query and text-facet preparations pin their encoder by inventory (`protocol.encoder`). The historical
schema `saea-pinned-speech-query-encoder-inventory-v1` pins Nemotron's checkpoint and the in-process `torch` /
`transformers` versions and is unchanged. A new schema `saea-vllm-served-encoder-inventory-v1` pins:

| key | meaning |
|---|---|
| `source` | the checkpoint inventory (`model_dir_relpath`, `files`) the view was built from; every file re-hashed |
| `view_relpath`, `view_receipt_sha256` | the serving view, verified against its receipt; it must derive from `source` |
| `served_model_name`, `base_url`, `server_version` | the server: loopback `http://host:port` only, `/version` must equal the pin, `/v1/models` must list exactly the view |
| `packages` | harness packages that shape the request (audio decode, WAV encode) |

`verified_backend` (`core/speech_preparation_runtime.py`) runs a served inventory on its vLLM client instead of the
default in-process encoder, so an in-process forward can never encode under a served pin. The view and source are
verified before the preparation lease; the server is contacted inside the lease, so a wrong or absent server aborts
that attempt with its receipt. The encoder inventory hash already enters every prepared artifact's identity, so the
regenerated vectors can never be mistaken for the historical ones.

### 6.4 CPU encoder runtime (implemented; no NPU needed)

The Fold B cell reads only pre-sealed vectors and never contacts an encoder in flight. All three encoder-built
inputs can therefore be built on the Ascend host's CPUs:

- the kus3 TRAIN texts (11,514);
- the intent-boundary keys (60);
- the Fold B speech queries (1,080 recordings, 3,155 s).

**Measurements** (Kunpeng-920, synthetic inputs only):

- **dtype:** fp32 GEMM reaches 644 GFLOPS on 64 threads. bf16 is software-emulated at 74 GFLOPS, so the CPU runtime is
  `cpu-float32`. On four synthetic inputs, fp32 and bf16 vectors agree to cosine 0.9997-0.9999.
- **threads:** the saea-base image sets `OMP_NUM_THREADS=1`. Every CPU worker must set its own thread count.
- **attention geometry** (measured 2026-10-28 on the loaded weights, synthetic text, strict token-prefix test; log
  `work/logs/geometry-matrix.log`): as shipped, `modeling_nv_omni_embed.py` sets `is_causal=False` on every text
  layer and overrides `_update_causal_mask`. On transformers 5.14.1 the override is never called; the flag reaches
  sdpa only when the mask is skipped. Result: **batch one -> bidirectional** (prefix states differ by 11.7),
  **padded batch -> causal** (0.0). The historical run (transformers 5.12.1, sdpa) therefore mixed geometries:
  batch-one speech queries bidirectional, 16-item text batches causal whenever padded. The port forces
  `is_causal=True` on every text layer (`load_stack(..., attention_geometry="causal")`), which measures causal under
  both mask paths (0.00005 / 0.0); every CPU worker and MF-7b shard re-measures the geometry on its loaded weights
  and refuses a mismatch. The CPU inventory pins `attention_geometry: "causal"`.
- **worker split** on cores 128-191 (NUMA nodes 4 and 5, each worker doing batch-one forwards):

  | workers x threads | text/s | audio (3 s)/s |
  |---|---|---|
  | 1 x 64 | 1.37 | 0.69 |
  | 2 x 32 | 2.19 | 1.05 |
  | 4 x 16 | 2.54 | 1.25 |
  | 8 x 8 | 2.03 | 1.25 |
  | 16 x 4 | 1.47 | 0.95 |

  Throughput peaks at 4 x 16, which puts two workers on each 32-core NUMA node. More nodes add throughput
  roughly linearly.

**Inventory and backend.** The schema `saea-cpu-encoder-inventory-v1` (`core/cpu_encoder_pool.py`) pins:

- the checkpoint files, as the historical inventory does;
- the runtime;
- the worker interpreter (the image's system Python, which has torch; the harness venv does not);
- the worker count, threads per worker and first core;
- `torch`/`transformers` versions, which each worker verifies in its own interpreter.

`verified_backend` runs this inventory only on the default backend's CPU pool.

**Seals.**

- The speech-query and text-facet preparations seal unique items in windows of `workers` contacts.
- `prepare_artifact_batch` persists every start marker of a window before one charged `contact(batch=True)`.
- An interrupted window is therefore never replayed, and the cache bytes, index bytes and accounting equal the
  single-contact path's.
- Each contact is its own batch-one forward on a fixed thread count, so the pool size never changes a vector. A live
  smoke gave bit-identical vectors on two workers.

**MF-7b key pass.** The kus3 TRAIN-text pass keeps its historical 16-item batches and spreads whole batches over pinned
workers (`--runtime cpu-float32 --workers N --cores A-B`). `build_slurp_kus3_vectors.py --facets-only` writes the one
file Fold B pins.

**Cost at 4 x 16 on two NUMA nodes:**

| input | time |
|---|---|
| kus3 | about 75 min at batch one (batch 16 is faster) |
| Fold B speech | about 15 min |
| intent keys | about 1 min |

**Declared difference:** CPU float32, not CUDA bfloat16.

**Inputs and authority (2026-10-28):**

- `scripts/build_slurp_mf7_train_facets.py` rebuilds the kus3 input `vectors/slurp-mf7/train-facets.jsonl` model-free:
  11,514 rows, sha256 `3252d195…`. It refuses unless the `size-full` snapshot rebuilds to its pin `e27b2745…`. No
  historical pin exists for the file itself.
- Decision 9.1 is decided. Registered encoder runs execute in the `saea-base` container (section 9.1).

## 7. Serving geometry (implemented; values verified on NPU)

The study keeps one geometry `(slots, slot_tokens)`. `core/session/serving.py` spells it per server dialect
(`GEOMETRY_FLAGS`, `geometry_flags(transport)`): llama-server `-np slots -c slots*slot_tokens`, vLLM
`--max-num-seqs slots --max-model-len slot_tokens`. `merge_server_args`, `declared_server_geometry` (used by
`resolve_serving_profile`) and the SLURP runtime view's `configured_geometry` read the fragment's `transport`; every
transport other than `vllm` keeps the historical flags and error texts. A vLLM argv carrying llama geometry flags or a
`--flag=value` spelling is refused, so the session reuse predicate compares one spelling with one spelling.

| llama-server (historical) | vLLM |
|---|---|
| `-c 98304 -np 4` (four 24,576-token slots) | `--max-model-len 24576 --max-num-seqs 4` |
| `--cache-prompt`, `--slot-prompt-similarity 0` | automatic prefix caching (default on) |
| `-ctk q8_0 -ctv q8_0` | BF16 KV cache (declared difference) |
| `-b 2048 -ub 512` | `--max-num-batched-tokens` (throughput only) |
| `--reasoning-format deepseek` | `--reasoning-parser qwen3` |
| `--reasoning-budget-message M` | `--reasoning-config {"reasoning_end_str": "M\n</think>\n\n"}` |
| native tools | `--enable-auto-tool-choice --tool-call-parser hermes` |
| mmproj-free packets | `--limit-mm-per-prompt {"audio": 1, "image": 0, "video": 0}` |

Tensor parallel size and NPU placement are chosen when the cards are free.

Telemetry (`core/session/telemetry.py`). The serving resource guard and the preparation monitor read accelerator
memory through `read_gpu`. On a host without `nvidia-smi` it reads `npu-smi info`'s device table (HBM used/total,
power, AI Core utilisation, health) and reports the visible card with the least free HBM, the safe direction for an
admission floor. A table it cannot parse is no reading (the guard then stops), never a guessed one. The serving
container should therefore mount only the serving cards. The guard's host-memory floors are still named for Windows;
their values for a Linux host are part of the runtime receipt decision (section 9.3).

## 8. Declared differences from the historical runs

1. BF16 weights instead of GGUF Q4_K_M; BF16 KV instead of q8_0.
2. Audio features computed by the server's audio tower from raw audio instead of llama.cpp mmproj packets.
3. vLLM sampling and kernels; temperature 0 with a fixed seed is deterministic per server build, not bit-identical to
   llama.cpp.
4. Causal retrieval encoder (forced and measured; section 6.4); all sealed retrieval vectors regenerated. The
   historical vectors were not uniformly bidirectional: as shipped the tower is bidirectional only at batch one.
5. Forced-close whitespace (section 4) until verified.
6. Re-authored semantic knowledge (section 9.5, decided). The intent-boundaries and slot-contracts stores have the
   historical structure, rules and support, but new card text written by root (this port's agent session) on
   2026-10-28. They were not written by the Thinking model. Every knowledge-bound number is measured against these
   cards, not the sealed ones (`docs/readiness/2026-10-28-slurp-semantic-authoring-root-review-npu-port-v1.md`).
7. CPU float32 retrieval encoder (section 6.4) instead of CUDA bfloat16; the port's vectors are all built on it.

## 9. Owner decisions (9.1-9.5 decided)

1. DECIDED 2026-10-28 (owner, two messages: "继续吧，授权执行权限，注意生成的过程后台运行" and then
   "WSL2这种约束都得删了，我们目前是用之前做的docker来拉起实验的"): the WSL2 authority and every WSL2 constraint
   are REMOVED, not supplemented. The formal execution environment is the `saea-base` Docker container.
   `contracts.plan.assert_execution_environment` admits a process inside a container (`/.dockerenv`) that carries
   the image's layout (`/opt/saea-venv/bin/python`, `/opt/umbrella`); `FrozenCoreGate`, the tool-plan admission
   and the server spawn call it. `assert_wsl_execution_environment` no longer exists. The pytest marker
   `requires_wsl2_execution_authority` became `requires_saea_container` (same 115 tests; CI still deselects and
   counts them).
2. DECIDED 2026-10-28 (owner, on the proposal below: "按你建议走" -- go with the proposal). Session identity.
   `vllm serve` runs in the SAME container as the harness, and the pinned "binary" is the Python interpreter plus the
   vLLM and vllm-ascend source trees plus the image digest. The transport re-verifies on every send, exactly as
   strictly as the llama path does, that a LIVE process with that identity owns the listening socket.
   Implemented in `core/session/vllm_identity.py`, reached through the seam of decision 9.3. Per ENDPOINT, never per
   server: liveness and state from `/proc/<pid>/stat`; the mount and network namespaces equal to this process's (the
   proof that the harness and the server share one container, so the files the gate hashed are the files the server
   loaded, and `/proc/net/tcp` is the server's own network); `/proc/<pid>/exe` re-hashed against the pinned
   interpreter; `/proc/<pid>/cmdline` equal to the receipt's protocol plus THIS instance's deployment flags;
   `/proc/<pid>/cwd` the pinned working directory; `/proc/<pid>/environ` carrying exactly the receipt's pinned
   behaviour-relevant variables and this instance's own `ASCEND_RT_VISIBLE_DEVICES`; `/proc/<pid>/maps` with every
   pinned runtime library mapped, no second libpython and nothing deleted under a pinned root; and the llama path's
   own address-aware socket-ownership proof (`/proc/net/tcp`(6) plus `/proc/<pid>/fd`) for that endpoint's exact
   (address, port). Nothing is cached between calls. A vLLM session is recorded in the UNCHANGED `SessionReceipt` v1
   (role `legacy`), with the interpreter as its pinned binary.
3. DECIDED 2026-10-28 (owner: "可以，但是最好用adapter模式兼容" -- approved, but preferably compatible via an adapter
   pattern). Runtime receipt. A new schema `saea-runtime-receipt-vllm-v1` (`contracts/gate_runtime_vllm.py`) sits
   BESIDE the llama one: served weights by file hash, the interpreter, its runtime libraries, the site directory's
   import-path files, both editable source trees as exact-set manifests, each package's METADATA/RECORD identity, the
   server argv, the pinned environment and the attested image id. The llama schemas v2/v3 and their validator are not
   loosened by one byte.
   The adapter is the whole shape of it: ONE runtime-identity interface, split across the layers by the rule that
   `contracts` never imports `core` -- the receipt half in `contracts/runtime_identity.py`
   (`validate_runtime_receipt`) and the live half in `core/session/runtime_identity.py` (`verify_session`,
   `assert_session_alive`, `assert_sessions_alive`, `start_session`, `attach_session`, `stop_session`). The model
   fragment's `transport` field selects the adapter (`core.config.runtime_transport` -> `FrozenCoreGate.transport`);
   an unregistered transport closes the gate rather than defaulting to anything. The gate, the transport and the
   session lifecycle each call the seam at ONE site instead of branching on the backend, and the llama
   implementation is a delegation to the existing functions, called exactly as before.
   MULTI-INSTANCE (coordinator design input, 2026-10-28, folded in): serving is N instances (e.g. 4 x
   tensor-parallel 2 on 8 cards), each with its own port and device group, and `--max-num-seqs` is per instance.
   The receipt therefore records WHAT is computed once (`protocol`: served model name, interpreter invocation,
   decoding and reasoning flags, cwd, pinned environment -- hashed into `protocol_sha256`) and WHERE/HOW WIDE each
   instance runs in `deployment.instances` (host, port, device ids, tensor-parallel size, `--max-num-seqs`,
   `--gpu-memory-utilization`, `--max-num-batched-tokens` -- deliberately outside that hash), so a deployment change
   never looks like a protocol change. Device groups must be disjoint and endpoints unique. The receipt is PRODUCED
   from the running servers by `scripts/write_vllm_runtime_receipt.py` (it observes each pid, requires all instances
   to share one protocol, re-hashes the weights, and refuses to write anything that does not pass both the gate-time
   validator and the per-endpoint live check). The multi-endpoint transport and the card allocator are not built yet;
   the interfaces are list-of-instances ready.
4. DECIDED 2026-10-28 (owner: "go ahead", translated from the owner's Chinese message; the owner asked the agent to
   draft the entries): lock entries for the BF16 Thinking model and the three serving views. Four entries were added
   to the umbrella lock's `models`, together with a new profile `speech-aware-npu-port` and one `amendments` record.
   No existing entry changed.

   | entry | files | bytes | identity |
   |---|---|---|---|
   | `qwen3-omni-30b-a3b-thinking-bf16` (`models/qwen3-omni-30b-a3b-thinking`) | 26 | 63,447,489,451 | hf `Qwen/Qwen3-Omni-30B-A3B-Thinking` @ `2f443cfc` |
   | `omni-embed-nemotron-3b-causal-v1` | 14 | 9,423,118,096 | receipt `a3ddac02…` |
   | `lco-embedding-omni-3b-2605-v1` | 14 | 9,423,121,465 | receipt `4c95f853…` |
   | `lco-embedding-omni-7b-v1` | 16 | 17,879,801,172 | receipt `c77fd25b…` |

   - Each entry records the sha256 of every file.
   - The BF16 files match the official LFS oids and Git blob SHA-1s at the revision.
   - The view files match their receipts, and the receipts' source hashes match the sources on disk.
   - The BF16 entry is the port model fragment's `model_lock_key`.
   - The lock is not versioned on the Ascend hosts. The exact entries, the apply script and the lock hashes are in
     `docs/readiness/2026-10-28-npu-port-lock-entries-v1/`: before `eb39a1ca…`, after `61f3d111…`. The pristine
     bytes are kept as `datasets.lock.json.pre-npu-port-v1`.
5. DECIDED 2026-10-28 (owner: "re-author following the documented procedure"): the authored semantic knowledge is
   re-authored by root from the governed TRAIN support. This is a new knowledge release and declared difference 8.6.

## 10. Port cell (draft) and its offline check

Draft fragments (committed, nothing registered):

* model `bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1`: the historical model
  fragment's geometry, decoding, knobs, forced-close hash, context policy and guard in the vLLM dialect;
  `prompt_order: native-raw-speech-query-prefix-v1`; no `id_slot`, packet root or llama server env.
* baseline `slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1`: the historical baseline with
  `arm: raw-speech-agent-native-tools` and the packet promotion receipt removed; everything else unchanged.
* experiment `slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1`: the historical cell under its own
  `-NPU-PORT-V1` run id and `-npu-port-v1` attempt id.

`core.flight.cli check --read-only` builds the flight runtime (compose, lock, SLURP adapter, geometry) and stops at
the gate's execution-environment check (section 9.1, since decided). A diagnostic probe outside the repo, which records each gate
refusal instead of stopping and changes no repo code, shows what remains:

| stage | result |
|---|---|
| gate: execution environment | refused outside the container at the time; decided 2026-10-28 (section 9.1) |
| gate: exposure preregistration | no row for the port run id (governance: written by the owner before any contact) |
| gate: runtime receipt | schema, validator, seam and producer exist (decision 9.3, decided 2026-10-28); the receipt itself is written from the running servers by `scripts/write_vllm_runtime_receipt.py` and therefore waits for free cards |
| gate: lock, carrier dataset entry, E0 | pass |
| preflight past the gate | passes on 1,080 rows: config, split identity, prompt-budget estimate (ALLOW, 15,090-token operational margin in a 24,576 window), prepared-artifact identity |
| `prepare-supply` (offline, no gate) | stops at the first missing data-root asset, `knowledge/slurp/semantic/intent-boundaries-v1/release.json` |

The audio files themselves are present and match the dataset fragment's `audio_sha256` pins.

### 10.1 Data-root assets the cell needs (none shipped)

The study ships pins, not data. Build graph of the Fold B cell's data-root inputs (leaves first; "B" is the
historical baseline fragment):

| asset | builder | contact | rebuildable here |
|---|---|---|---|
| `prepared/SAEA-SLURP-NATIVE-CANDIDATE-CONTRACTS-PREPARE-V1/catalog.json` | `rails/slurp_candidate_contracts.py` provider `source_document` (no file writer in the repo) | none | content yes (canonical `deb4e097`); file bytes by reconstruction |
| `knowledge/slurp/saea-kb-distmodel-v1/size-pool-annotated-exemplars/{snapshot,release}.json` | `scripts/build_slurp_exemplar_pool_assets.py` (release: reconstruction) | none | yes |
| `knowledge/slurp/semantic/intent-boundaries-v1/*`, `slot-contracts-v1/*`, `slot-contracts-v2/*` | `core.driver.cli prepare-semantic-knowledge` over `prepared/SAEA-SLURP-SEMANTIC-AUTHORING-PREPARE-V{1,2}/authored-*.json` | the compile is model-free; the sources were AUTHORED by the study's root agent from the support rows | historical text NO (the sources `1950c4a3`, `34f6c3dc`, `fc34b243` exist only on the author's data disk). RE-AUTHORED and prepared 2026-10-28 as `intent-boundaries-npu-port-v1`, `slot-contracts-npu-port-v1` and `-v2` (snapshot content `cb6ef966`, `2ac1666b`, `e035c1b8`) |
| `vectors/slurp-kus3/facet-vectors.jsonl` | `scripts/slurp_mf7b_nemotron_embedding_pass.py` then `scripts/build_slurp_kus3_vectors.py` | encoder (11,514 passages) | regenerated; the converter also requires MF-7b DEV and fused sets and needs narrowing to the keys set |
| `preparation-runs/SAEA-SLURP-INTENT-BOUNDARY-KEY-ENCODING-V1/*` (+ prepared facets) | `run_text_facet_preparation` | encoder (60 passages) | after the intent-boundaries store |
| `preparation-runs/SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-V1/*` (+ queries) | `run_preparation` | encoder (1,080 utterances) | regenerated |
| `prepared/SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-B-ALIAS-V1/*`, `vectors/slurp-intent-boundaries-train-fold-b-1080-v1/*` | `core.preparation_vectors` | none | from the regenerated queries |
| `vectors/slurp-stage3-filler-fold-b-v1/query-vectors-speech-content-v1.*` | `scripts/build_slurp_query_content_derivative.py` | none | from the above |
| `prepared/SAEA-SLURP-INTENT-BOUNDARY-CENTROID-KEYS-V1/*` | `scripts/build_slurp_intent_boundary_centroid_keys.py` | none | after the stores and facet vectors |
| `prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-V1/*` | `scripts/build_slurp_intent_exemplar_agreement.py` | none | after the above |

Consequences:

* Every vector and everything derived from one is regenerated (owner decision 3), so its pins are re-minted in port
  fragments; receipts carry timestamps and builder hashes and are never byte-comparable anyway.
* The authored semantic knowledge is the one input that cannot be regenerated by a deterministic or encoder step.
  It was re-authored (section 9.5, decided; difference 8.6). The three port stores are prepared under the
  registrations `docs/readiness/2026-10-28-slurp-{intent-boundaries,slot-contracts}-preparation-registration-npu-port-v*.json`.
  Downstream (key encoding, centroid keys, agreement table) now waits only on the encoder.
* The authoring support preparation and the encoder runtimes carry the same execution-environment gate and
  historical host paths (`/mnt/e`, `/mnt/d`, `/home/chao/...`) as the flight; section 9.1 covers them.

## 11. NPU verification plan (when the cards are free)

1. Serve the Nemotron view; embed 20 SLURP queries; compare rank agreement with the in-process `transformers` causal
   forward on the same NPU (same weights, same causal mask) to separate kernel noise from the attention change.
2. Serve Thinking BF16 and run `npu_smoke_thinking.py`: forced close (budget 16) to fix the section-4 string; context
   overflow to confirm the section-3 relabel; one hermes tool call; `/tokenize` against the engine's prompt length for
   an audio request (section 5.3); the rendered prompt in both thinking modes; the answer-only re-ask; and a RECORD of
   the system-plus-tools request (section 5.5).
3. A four-row dry run of Fold B under the port fragment.
