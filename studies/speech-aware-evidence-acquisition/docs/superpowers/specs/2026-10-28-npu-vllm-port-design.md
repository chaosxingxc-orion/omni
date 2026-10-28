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
and a `feature_cache_guard` are refused for a `vllm` fragment.

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
4. Causal retrieval encoder; all sealed retrieval vectors regenerated.
5. Forced-close whitespace (section 4) until verified.

## 9. OPEN owner decisions (not implemented)

1. Execution authority. `CLAUDE.md` names WSL2 as the sole formal execution authority and its gate must not be
   weakened. The port needs a dated owner decision that ADDS an Ascend NPU container authority (the WSL2 check and its
   guard tests unchanged), or a ruling that port runs are unofficial and outside the gate.
2. Session identity. The transport re-verifies, on every send, that a live process running the pinned binary owns the
   listening socket. Proposal: run `vllm serve` in the same container as the harness and pin the Python interpreter,
   the vLLM and vllm-ascend source trees and the image digest as the "binary".
3. Runtime receipt. A new vLLM runtime-receipt schema (served weights by file hash, server argv, package versions,
   image digest) beside the llama one, without loosening the existing schema.
4. Lock entries for the BF16 Thinking model and the three serving views.

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
the gate's WSL2 execution-authority check (section 9.1). A diagnostic probe outside the repo, which records each gate
refusal instead of stopping and changes no repo code, shows what remains:

| stage | result |
|---|---|
| gate: execution authority | open owner decision 9.1 |
| gate: exposure preregistration | no row for the port run id (governance: written by the owner before any contact) |
| gate: runtime receipt | the vLLM runtime receipt does not exist (owner decision 9.3) |
| gate: lock, carrier dataset entry, E0 | pass |
| preflight past the gate | passes on 1,080 rows: config, split identity, prompt-budget estimate (ALLOW, 15,090-token operational margin in a 24,576 window), prepared-artifact identity |
| `prepare-supply` (offline, no gate) | stops at the first missing data-root asset, `knowledge/slurp/semantic/intent-boundaries-v1/release.json` |

The audio files themselves are present and match the dataset fragment's `audio_sha256` pins.

## 11. NPU verification plan (when the cards are free)

1. Serve the Nemotron view; embed 20 SLURP queries; compare rank agreement with the in-process `transformers` causal
   forward on the same NPU (same weights, same causal mask) to separate kernel noise from the attention change.
2. Serve Thinking BF16 and run `npu_smoke_thinking.py`: forced close (budget 16) to fix the section-4 string; context
   overflow to confirm the section-3 relabel; one hermes tool call; `/tokenize` against the engine's prompt length for
   an audio request (section 5.3); the rendered prompt in both thinking modes; the answer-only re-ask; and a RECORD of
   the system-plus-tools request (section 5.5).
3. A four-row dry run of Fold B under the port fragment.
