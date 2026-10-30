# Lock the engineering base and fix the experiment-efficiency configuration (2026-10-29)

Planner: Fable 5.1. Branch `npu-vllm-port`. Plan only: it grants no contact authority, no readiness flip and no
exposure entry. Owner goal (translated from Chinese): "Lock the engineering base, and determine the optimal
experiment-efficiency configuration and run plan."

> **Revision 2 (2026-10-29, after the independent S0 brief `work/scratch/brief/2026-10-29-task-understanding.md`,
> every blocker re-verified by the planner against the cited lines).** Changed: (1) facts L1, L3, L4, L5, L9, L12,
> L13 corrected (L5 read methods were partly invalid; `--served-model-name` is deployment-only; the HCCL interface
> variables and extra mounts live only in `work/scripts/npu_serve.sh`, not in `core/deployment.py`). (2) KV
> arithmetic in 3.1 redone: the measured run already ran at the default 0.92, 25.83 GiB is per card, KV shrinks at
> `max-num-seqs 64`. (3) Card-hour total restated as 60-66. (4) Three blockers get streams: deployment argv parity
> (S10: env, mounts, entrypoint form), per-instance env out of `protocol_sha256` (S12), S6's eight dependents (S13).
> (5) Hour bound re-derived in the REGISTERED form (margin 1.3, 7.52 lock-row equivalents); the plan's contact-count
> formula is a cross-check only. (6) Role rule applied: S5 and S8 are OPUS; FABLE review gates R1-R4 after each
> wave. (7) 2.2c is a per-request `chat_template` probe, no scratch server; `--chat-template` is a refused flag.
> (8) Q1-Q6 re-derived: Q3 answered by facts, Q2 narrowed, Q5/Q6 restated as protocol-hash questions. (9) Waves
> re-ordered: no card is touched before S10, S12 land and the verification script passes offline.
>
> **Revision 3 (2026-10-29, after the preliminary arm A sweep).** Section 3.7 records the result, the next-level
> predictions and the planner decisions (knee bracketing at the ceiling, arm B at equal host concurrency, arm C dropped,
> arm D conditional, preliminary-versus-recorded run list, card time). Wave 2 is split into 2a (preliminary, now) and
> 2b (recorded, after R2). Where 3.7 differs from 3.2 or 3.6, 3.7 wins.

Scope rails for every stream. Synthetic load only; no dataset row is read (the registered 4-row smoke waits for an
exposure row). No GitHub push. No design-doc section 9 decision beyond 9.1-9.5 is implemented. Never use a busy card
(`npu_deploy.sh` and `npu_serve.sh` refuse them). All NPU work on host 181 (`ssh npu2`). CephFS is 93 % full: sweep
reports are JSON under 20 MB, no weight copies, no trace dumps. Only the main thread commits. Streams are briefed on
the WORKING TREE, not HEAD: S1, S2+S3, S4, S6 have uncommitted output in it (brief G1).

Measured state this plan builds on (host 181, 2026-10-29, log
`/mnt/cephfs/xingchao/omni/work/logs/saea-serve-thinking-20261029T063554.log`, launched by `npu_serve.sh`, not by
`npu_deploy.sh`): vLLM 0.30.0 + `vllm_ascend 0.30.0rc2`, torch 2.10.0 / torch_npu 2.10.0.post7, TP2 on cards 0,1,
`max_num_seqs 4`, `gpu_memory_utilization 0.92` (the vLLM default; log line 229 "Desired GPU memory utilization is
(0.92, 56.08 GiB)"); available KV **25.83 GiB per card** (Worker_TP0, line 209) = **564,224 tokens per instance =
22.96x** the 24,576-token window (line 210); weights 29.19 GiB per card, peak activation 0.35, non-torch 0.72, graph
0.31 GiB (line 229); engine init 76.33 s of which 32.28 s compilation (warm compile cache); `Application startup
complete` 3 min 45 s after launch; prefix caching and chunked prefill on; `cudagraph_mode FULL_AND_PIECEWISE` with
capture sizes `[1,2,4]` (from `max_num_seqs`); HCCL operator timeout 1836 s with the engine's advice
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000`; the `causal_conv1d_update` fallback warning about stalled ACL graph capture
at decode-FULL (lines 92-93), yet capture completed and replay is active (lines 225-228, 299). The run used
`--cpuset-cpus 128-191` and the engine skipped CPU binding (NUMA conflict, lines 231-234); the profiles pin each
instance to its cards' own NUMA node, so startup and prefill timings will move (brief G11).

---

## 1. What "locked engineering base" means

A fact is locked when (a) it is written in a file under version control or in a hash-pinned receipt, and (b) a
script re-derives it from the live system and compares, failing closed on any difference. Facts split in two by
owner decision 2026-10-28 part 1: the **protocol** half is hashed; the **deployment** half is recorded and not hashed.

### 1.1 Facts to pin, and where (corrected)

| # | fact | value source (what is actually readable) | pinned where | hashed? |
|---|---|---|---|---|
| L1 | image identity, PER HOST | `docker image inspect saea-base:20261027` -> `Id`, `Created`; `RepoDigests` is `[]` (locally built, never pushed). Ids differ between `npu` and `npu2`; the record carries both, the check compares the host it runs on | base-lock record; receipt `image_id` | recorded |
| L2 | base image provenance | `quay.io/ascend/vllm-ascend:v0.30.0rc2`, local ID `4d683043e4bd` on both hosts | base-lock record | recorded |
| L3 | interpreter | ONE executable, `/usr/local/python3.12.13/bin/python3.12` (3.12.13), reached by two paths: `/opt/saea-venv/bin/python` (symlink; venv has `include-system-site-packages = false`) and `/usr/local/python3.12.13/bin/python3`; one `exe` sha256 | runtime receipt (`saea-runtime-receipt-vllm-v1`) | hashed |
| L4 | serving stack, interpreter NAMED | system python: `vllm 0.30.0+empty`, `vllm_ascend 0.30.0rc2`, `torch 2.10.0+cpu`, `torch_npu 2.10.0.post7`, `transformers 5.14.1`, `triton 3.6.0`, `triton-ascend 3.6.0+dev20260923230212`; harness venv `/opt/saea-venv`: `transformers 5.21.0`, no vllm/torch. Exact strings with local suffixes | receipt package identity; base-lock record table | hashed |
| L5 | CANN / driver / firmware | host: `/usr/local/Ascend/driver/version.info` (`Version=26.0.rc1`), `/etc/ascend_install.info`, `npu-smi info -t board -i 0` (Software Version 26.0.rc1; Firmware reports `NA`), host toolkit dir `/usr/local/Ascend/ascend-toolkit/8.2` (no `latest`). Image: `/usr/local/Ascend/ascend-toolkit/latest -> /usr/local/Ascend/cann` = `cann-9.2.0`, `cann-9.2.0/compiler/version.info` (`Version=9.2.0`). Firmware version is recorded as NOT READABLE (dir unreadable as our uid, npu-smi says NA); no check pretends otherwise | base-lock record | recorded |
| L6 | host identity | `hostname` `A800I-03`, 8 x 910B4-1 x 65,536 MiB, 256 cores, 2011 GiB RAM, NUMA: NPU 0-1 node 6 (cpus 192-223), 2-3 node 4 (128-159), 4-5 node 0 (0-31), 6-7 node 2 (64-95) = the profiles' cpusets | deployment profile + base-lock record | recorded |
| L7 | weights | 26 governed top-level files, 63,447,489,451 B, per-file sha256, hf revision `2f443cfc`; `.hfd/` is declared not governed and is ignored | umbrella lock entry `qwen3-omni-30b-a3b-thinking-bf16` (lock sha `61f3d111`); re-hashed by the receipt producer | hashed |
| L8 | encoder weights | lock entries `omni-embed-nemotron-3b-causal-v1`, `lco-embedding-omni-3b-2605-v1`, `lco-embedding-omni-7b-v1` | same lock | hashed |
| L9 | protocol serving flags | fragment `bf16-thinking-...-port-v1` `server_args`: `--max-model-len 24576`, `--dtype bfloat16`, `--reasoning-parser qwen3`, `--reasoning-config {...}`, `--enable-auto-tool-choice`, `--tool-call-parser hermes`, `--limit-mm-per-prompt {...}`, `--seed 0`. `--served-model-name` is NOT in `server_args` (it is a `DEPLOYMENT_ONLY_FLAGS` member, refused inside a fragment); the served name is the fragment key hashed via receipt `protocol.served_model_name` | fragment; receipt `protocol_sha256` | hashed |
| L10 | decoding and reasoning | `temperature 0`, `seed 20260913`, `max_tokens 4096`, `reasoning_budget_tokens 3072`, `answer_reserve_tokens 1024`, `forced_close_message_sha256 aab2abbe...`, `reasoning_context_policy drop-reasoning-v1`, `timeout_seconds 420` | same fragment | hashed |
| L11 | deployment layer | instances, cards, cpuset, port, `--tensor-parallel-size`, `--max-num-seqs 64`, `--gpu-memory-utilization 0.92`, `--max-num-batched-tokens null`, per-instance `HCCL_IF_BASE_PORT` | `configs/deployment/*.json`; receipt `deployment.instances`; `core/deployment.py deployment_record` | NOT hashed |
| L12 | HCCL / network env and mounts | TODAY only `work/scripts/npu_serve.sh:40-49` sets `HCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`/`TP_SOCKET_IFNAME` = NIC with the host's global IPv4 (`ip -o -4 addr show scope global`; 181: `enp61s0f2`) and mounts `/etc/hccn.conf`, `/etc/ascend_install.info`, `/usr/local/Ascend/firmware`, `/usr/local/dcmi` read-only. `core/deployment.py` emits none of them (`grep IFNAME src/` is empty), so `npu_deploy.sh up` cannot bring a multi-card instance up. S10 fixes this; after S12 the four interface/port keys are DEPLOYMENT env (recorded per instance, excluded from `protocol.env`) | `core/deployment.py` argv + deployment record; base-lock record | recorded (after S12) |
| L13 | engine-derived facts | per instance: KV tokens, max concurrency, prefix caching, chunked prefill, cudagraph mode and capture sizes, from the REAL profile's startup log; block size read from the cache config / `/metrics`, NOT asserted (the measured log never prints `block_size`) | base-lock record | recorded |
| L14 | tokenizer / template | `chat_template.json` sha256 `3afb9519...` (one key, 5,715 chars); `tokenizer_config.json` has NO `chat_template` key; decoded `/tokenize` tails in both thinking modes; one audio span per audio part; `/tokenize` count == engine `prompt_tokens` | base-lock record + smoke tails | recorded; the file hash is part of L7 |
| L15 | execution authority | `/.dockerenv`, `/opt/saea-venv/bin/python`, `/opt/umbrella` (`contracts/plan.py:308-321`) | `assert_execution_environment` | hashed via code manifest |
| L16 | launch form (new) | live argv must be `<interpreter> -m vllm.entrypoints.cli.main serve <model_dir> ...` (`gate_runtime_vllm.py:75,352-357`). Both launchers use `--entrypoint vllm`, giving `python3 /usr/local/python3.12.13/bin/vllm serve ...`, which the gate refuses. S10 changes the launchers, not the gate | receipt `protocol.argv0/entrypoint` | hashed |

### 1.2 The acceptance test

`scripts/serving/verify_engineering_base.py` (study tree, UNTRACKED today, 1,299 lines, S4 in flight) driven by
`work/scripts/npu_verify_base.sh` (exists). Takes a deployment profile plus a base-lock record, emits
`saea-engineering-base-verification-v1` JSON, attaches to the resident deployment, never starts or stops a server.
Checks, all fail-closed on unreadable input: 1 `image` (per host, L1); 2 `versions` (per interpreter, exact strings,
L4); 3 `cann` (the readable files of L5 only); 4 `weights` (26 governed files, `.hfd/` ignored, behind
`--rehash-weights`); 5 `endpoints`; 6 `protocol` (`/proc/<pid>/cmdline` protocol subset identical across instances,
`protocol_sha256` recomputed; passes only after S10 and S12); 7 `deployment` (cards, cpuset, flags, disjoint device
groups, unique ports, per-instance env equals the deployment record); 8 `env` (interface variables equal the NIC
with the global IPv4); 9 `kv` (within 1 % of the recorded value from the SAME profile's log); 10 `template` (decoded
tails, audio-span count, `/tokenize` == `prompt_tokens` on one synthetic audio request); 11 `receipt`
(`validate_runtime_receipt` + per-endpoint live identity). Until S10/S12 land, checks 6 and 11 report the two known
defects by name instead of being relaxed. `/metrics` gauges sit in a delimited `volatile` section so two runs on an
untouched deployment are byte-identical outside `generated_at` and `volatile`.

### 1.3 The base-lock record (OPUS)

`docs/readiness/2026-10-29-engineering-base-lock-npu-port-v1.md` plus `.../base-lock-v1.json`: every fact L1-L16
with value, read method, and hashed/recorded; the winning deployment pair from 3.5; a statement that it is a port
engineering record and flips no readiness.

---

## 2. Triage of the smoke failures (all three fixes are in the working tree; this section records the facts)

### 2.1 `overflow` FAIL: serving-layer wording, parser matched the engine wording

The observed 400 is raised by `vllm/renderers/params.py:494-517` (`TokenizeParams._token_len_check`,
`parameter="input_tokens"`), with the ` (parameter=..., value=...)` suffix appended by `VLLMValidationError.__str__`
(`vllm/exceptions.py:48-55`); the two committed `fullmatch` patterns in `core/model/vllm_transport.py:64-77` are the
engine's `input_processor.py` wording and never fire for a chat request with explicit `max_tokens`. Not a server
defect; a fail-closed guard that mislabels a `FrozenCoreContextSizeError` as a generic error. The third pattern (S1,
uncommitted) fullmatches the recorded body. OPEN: `params.py:354-375` has a character-level sibling
(`parameter="input_text"`); S1b adds it with a test or records it out of scope. Declared difference? No owner
decision: wire-dialect translation; OPUS adds one informational line to design-doc section 8 (S15).

### 2.2 `tool-call` FAIL and `system-tools` RECORD: one cause, read from vLLM source

`vllm/renderers/hf.py:263-299 resolve_chat_template`: (1) per-request `chat_template`; (2) the AutoProcessor template,
**skipped when `tools is not None`** (comment at hf.py:276); (3) `tokenizer.get_chat_template` in a bare `try/except`,
which raises because `tokenizer_config.json` has no `chat_template` (`transformers/tokenization_utils_base.py:3276-3288`);
(4) model-type fallback finds nothing; `hf.py:804-809` raises the observed "As of transformers v4.44 ..." error. So on
this build ANY request carrying `tools` fails, system message or not, and tool-free requests pass through the
processor template. Design-doc 5.5's "mechanism is an inference" caveat (lines 140-142) closes with this citation
(S15). Fold B sends no `tools`: not blocked. Remedy facts: `--chat-template` is in `REFUSED_SERVER_FLAGS`
(`gate_runtime_vllm.py:89-106`), so Q2's first option is refused by the receipt gate, not merely a hash change.
Diagnostic (S7, part of the smoke re-run): resend the recorded `tool-call` request to the RESIDENT service with a
per-request `"chat_template"` field holding the file's template; record 200/400. No scratch server, no flag added.

### 2.3 `template` FAIL: undecoded byte-level tokens

`/tokenize` `return_token_strs` gives GPT-2 byte-level forms (`Ġ` space, `Ċ` newline). The fix is in
`work/scripts/npu_smoke_thinking.py` (`_byte_level_decoder`, `decode_byte_level`, `template_tails_ok`, uncommitted,
host script outside git) and the brief ran it over the recorded tails: `template_tails_ok -> True`. Design-doc 5.5's
prediction holds (no pre-inserted `<think>` in thinking mode, empty think block otherwise, one audio span). The same
decoder goes into check 10 of 1.2.

### 2.4 Owner questions, re-derived

Answered by facts (no decision needed): **Q3** the overflow relabel is internal wire translation; one informational
line in design-doc section 8. Narrowed: **Q2** option 1 (`--chat-template` in `server_args`) is refused by
`REFUSED_SERVER_FLAGS`; only options 2 and 3 remain, and Q2 is moot until Q1 is answered yes.

Genuine owner decisions, each with a recommendation:
- **Q1.** Is a tool-bearing arm in scope for this port? Recommend NO for this plan; keep `--enable-auto-tool-choice
  --tool-call-parser hermes` in the hashed `server_args` for geometry fidelity with the historical mapping (provably
  inert for Fold B on this build).
- **Q4.** If TP4 (arm B) beats TP2 (arm A) on aggregate throughput by more than 5 %, which is resident? Recommend A
  unless B wins by more than 5 %: four instances isolate a failure and match the declared resident profile.
- **Q5 (restated).** `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` starts with `VLLM_`, a pinned prefix, so setting it puts
  it in `protocol.env` and changes `protocol_sha256`; it would be uniform across instances (set in `COMMON_ENV`), so
  the identity check still passes. Question: may a protocol-hash-affecting watchdog value be set? Recommend NOT for
  the sweep (the measured run came up without it); if a level shows a watchdog abort, re-ask with the log line.
- **Q6 (restated).** `--enforce-eager` is not in `DEPLOYMENT_FLAGS`, so under the receipt gate it is a protocol flag:
  arm E changes `protocol_sha256`, not only the deployment. Recommend NOT running arm E: graph capture completed and
  replay is active in the measured run; revisit only if the sweep shows decode stalls.
**Owner rulings, 2026-10-29 (translated from Chinese; recorded by the coordinator):**
- **Commit and publish:** commit the finished wave-0 work on `npu-vllm-port`, then push to `main` of the public
  engineering repo `chaosxingxc-orion/omni` (the study tree's own remote is a local path, so this is the only
  `main`). Scrub before the push, as always.
- **Q1 RULED yes, sequenced:** a tool-bearing arm IS in scope for this port, but only after the baseline metrics of
  this plan are out. The tool-free base and sweep run first, unchanged. Q2 is therefore live for a later stage: its
  two remaining options (a port chat template that joins system text parts, or `--chat-template-content-format
  string` with its own layout proof) need a Fable proposal and an owner choice before a tool arm runs.
- **Q4 RULED as recommended:** the four-instance TP2 layout (arm A) is resident unless arm B wins on aggregate
  throughput by more than 5 %.
- **Q5 RULED:** do NOT set `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` for the sweep; set it after the configuration is
  locked. Setting it re-mints `protocol_sha256`, so it lands as its own recorded change after the base lock, with
  the base-lock record and verification re-run.
- **Q6 RULED as recommended:** no arm E (eager versus graph).

Owner FYI (engineering under the 2026-10-28 split, not a question): S12 moves the four per-instance interface/port
env keys out of `protocol.env` into the recorded deployment layer; the design doc section 7 records it (S15).

---

## 3. Efficiency programme

### 3.1 Predicted capacity (written down before any card is used)

Per token, whole model: 48 layers x 4 KV heads x 128 x 2 (K,V) x 2 B = 98,304 B; per card 49,152 B at TP2 and
24,576 B at TP4. Measured at TP2 / `max-num-seqs 4` / 0.92: 25.83 GiB per card -> 564,224 tokens per instance.
Corrections to revision 1: the default utilisation IS 0.92, so there is no 0.92 uplift; the resident profile's
`max-num-seqs 64` grows peak activation and graph memory (capture sizes derive from it), so KV per card at the real
profile will be LOWER than 25.83 GiB. Predictions for S7 to confirm or refute from the real profile's startup log:
- TP2 instance: KV 24-25.5 GiB per card, 525-557 k tokens; ~21-22 full-length (24,576) sequences; ~41-44 at the
  synthetic mix (8,500 prompt + 4,096 completion = 12,596 tokens). Level 32 fits (403 k tokens); level 48 (605 k)
  exceeds KV -> preemption predicted at 48. Host aggregate KV (bytes) 4 x 2 x ~25 GiB = ~200 GiB.
- TP4 instance: weights ~14.6 GiB per card, KV ~39-40 GiB per card -> ~1.7 M tokens per instance (3x TP2); ~70
  full-length, ~135 at the mix; all five levels fit in KV, so B's knee is compute-bound, not memory-bound. Host
  aggregate ~2 x 4 x ~39.5 = ~316 GiB.
- Arm C (0.95): request ~57.9 GiB of 60.53 GiB free; with the measured ~0.3 GiB overshoot, ~2.3 GiB free per card at
  peak, right at the 2 GiB threshold -> predicted to fail or barely pass the HBM gate.
- 4,096 / 420 s = 9.75 tok/s, so the p10 >= 10 tok/s threshold has ~2.5 % margin; `max latency < 420 s` is the
  binding gate.
Caveat: 8,500 is the Solver prompt mean (`l338_cost_inputs.json`); 7 of 15 contacts per row are Perception, so the
mix represents the Solver population (conservative for cost).

### 3.2 Arms

| arm | profile | levels (per instance) | purpose |
|---|---|---|---|
| A | `qwen3-omni-thinking-bf16-181-4xtp2-v1` (4 x TP2) | 8, 16, 24, 32, 48 | candidate resident deployment |
| B | `qwen3-omni-thinking-bf16-181-2xtp4-v1` (2 x TP4) | 8, 16, 24, 32, 48 | the owner-named comparison |
| C | winner, `--gpu-memory-utilization` 0.92 (already measured) vs 0.95 | two levels around the knee | HBM headroom vs KV |
| D | winner, `--max-num-batched-tokens` null vs 8192 | knee level | does capping prefill cut p95 at equal throughput |
| E | winner, graph vs `--enforce-eager` | knee level | ONLY if the owner answers Q6 yes (recommended no) |

`--max-num-seqs` stays 64 in every arm; the measured quantity is client concurrency below it. C, D, E need redeploys
and run after A/B decide the shape.
Revision 3: A adds levels 56 and 64 (the ceiling); B runs 16, 32, 48, 64, 96 per instance with a 128 ceiling (equal
host concurrency to A); C is not run; D is conditional; see 3.7.

### 3.3 Load shape

Exactly what `scripts/serving/synthetic_sweep.py` generates, unchanged: lognormal prompt targets in [3000, 20000],
mean 8,500, fitted via the server's `/tokenize`; `native-raw-speech-query-prefix-v1` layout; no `tools`; synthetic
16 kHz mono WAV 2-8 s; 3 sequential contacts per trajectory sharing one clip; `temperature 0`, `max_tokens 4096`,
`thinking_token_budget 3072`, `enable_thinking true`; per-trajectory instance affinity inside the tool. Seed
`20261028`. No dataset row, no recording. Parameters: default `--requests-per-level` for the 5-level scan, then the
two candidate levels at 3x; `--min-tokens` unset, plus one `--min-tokens 4096` confirmation at the knee.
Note: the tool's affinity is NOT the flight transport. The flight-side multi-endpoint transport (route a trajectory to
an instance, strip `id_slot`, which `VLLM_REFUSED_REQUEST_KEYS` refuses) is still to build; it does not block the
sweep and the flight is out of scope here, so it is a FOLLOW-UP (section 7), and the baseline document must say the
knee is not usable by a flight until it exists.

### 3.4 Metrics, thresholds, stopping and selection

Per level: aggregate output tok/s; per-sequence e2e tok/s p50/p10 and decode tok/s; latency p50/p95/max; TTFT;
`vllm:num_preemptions` delta; prefix-cache queries/hits; sampled `kv_cache_usage_perc`, running, waiting; plus
`npu-smi` samples (HBM used/total, AI Core %) per card into the report directory. Thresholds: p10 e2e >= 10 tok/s;
max latency < 420 s; zero timeouts, zero errors; preemption delta exactly 0 with a readable counter; >= 2 GiB HBM
free per card at peak. Stopping: stop an arm at the first failing level. Selection (applied by S8 to the per-level
table; the tool's `recommended_concurrency_per_instance` is the highest PASSING level and is NOT the answer):
highest aggregate output tok/s among passing levels, ties within 5 % -> the lower level; between arms, higher host
throughput at the knee, ties within 5 % -> A (Q4). Record the losing arm too.

### 3.5 How the result becomes the recorded baseline

1. The winning `(profile_id, client concurrency)` goes into the base-lock record and into
   `docs/results/2026-10-29-npu-port-synthetic-speed-baseline-v1.md` (OPUS, S8) with the per-level table, the knee,
   the thresholds met and each report's sha256. Raw reports: `work/logs/sweep-<arm>-<timestamp>.json`.
2. **Hour bound in the REGISTERED form** (`docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/
   registered-facts-v1.json`, `accelerator_bound.formula.arm`): `card-hours = ceil_0.1(1.3 x 1,080 rows x 7.52
   lock-row equivalents x T / r / 60)`, T = cards per instance, r = lock-only rows per minute of ONE instance at its
   chosen client concurrency. S8 derives r from the sweep as `completed synthetic contacts per minute per instance at
   the knee / 2` (one lock row = one Perception + one Solver contact, both costed at the Solver-shaped synthetic
   contact; conservative since historical Perception is 59.0 s vs Solver 64.07 s). Cross-check only: `16,200
   contacts x mean completion tokens / aggregate output tok/s` as wall hours x 8 cards. If the two differ by more than
   20 %, S8 reports both and the discrepancy; it does not choose. The margin is the registered 1.3, not 1.5. The
   registration's gate stays closed until the owner acts (plan field note in the same file).
3. Client concurrency is a deployment fact: run manifest + receipt `deployment.instances`, outside `protocol_sha256`.
   S6 removes `--max-num-seqs 4` / `profile_slot_count 4` from the hashed fragment; S13 moves the eight dependents
   listed in `scripts/flights/nx/l338_port_family.py:73-85` and re-runs the registration writer, before S8 quotes a hash.

### 3.6 Card time on 181

| step | wall time |
|---|---|
| arm A deploy + verify base + 5 levels | 2.0 h |
| arm B redeploy + 5 levels | 2.0 h |
| winner's 2 candidate levels at 3x requests + `--min-tokens 4096` run | 1.5 h |
| arms C and D (two redeploys, knee level) | 1.5 h |
| smoke re-run (overflow, template, tool-call probe) on the resident service | 0.5 h |
| **subtotal** | **7.5 h wall x 8 cards = 60 card-hours** |
| arm E, only if Q6 is answered yes | +0.75 h = +6 card-hours (66 total) |

Levels get slower as concurrency rises and four instances compiling in parallel start slower than the measured single
instance (`DEPLOY_WAIT_SECONDS` default 2400 s covers it). The resident deployment stays up afterwards.

---

### 3.7 Preliminary arm A result and next sweep steps (revision 3, 2026-10-29, planner decision from the summary)

Input: the preliminary arm A sweep (`docs/results/2026-10-29-npu-port-preliminary-arm-a-sweep-v1.md`; launched by
`work/scripts/npu_serve.sh`, 4 x TP2 on cards 0-7, `--max-num-seqs 64`, NUMA-pinned, levels 8-48 per instance, one
pass = 3 contacts per worker, seed 20261028). Every level passed every 3.4 threshold: zero errors, timeouts and
preemptions; KV peak 80 % at 48; throughput still +26 % from 32 to 48; p10 11.4 tok/s at 48 (14 % above the floor);
max latency 249 s (gate 420); min HBM free 4.1 GiB per card on 30 s samples. The knee is NOT bracketed. 3.1
predictions marked: "preemption at 48" REFUTED (80 % of 564 k = ~451 k tokens for 192 host sequences = ~9.4 k per
sequence, below the 10.5 k mix; prefix reuse across the three contacts of a trajectory was not in the estimate);
"knee at or below 32" REFUTED; "KV shrinks at `max-num-seqs 64`" REFUTED (25.83 GiB per card, unchanged).

**Correction for S8 (OPUS) before any number is quoted.** The summary's "output tok/s per instance" column is the
HOST aggregate: `scripts/serving/synthetic_sweep.py:527-543` sums `completion_tokens` over every ok result of the
level (all instances) and divides by the level's wall time, and the driver copies that field as
`aggregate_output_tok_s`. Arithmetic agrees: at level 8, 904.9 tok/s per instance would need 113 tok/s from 8
sequences whose p50 is 39.6; at 48, 13.1 contacts/min/instance x ~2.0 k tokens = ~437 tok/s per instance = 1,795 / 4.
The preliminary result document's "host output tok/s" column (3,620 ... 7,181) is therefore a 4x over-count; the
per-instance figures are ~226 ... ~449 tok/s and the host figures are the first column (905 ... 1,795). S8 corrects
the result document with a successor edit and uses the host figure in the hour-bound cross-check (3.5 item 2).
Contacts/min and r (6.5 lock rows/min/instance at 48) are unaffected; no decision below depends on this.

**Predictions for the next levels (written before the runs; S8 marks each confirmed/refuted).** Per-sequence rate
fell 20 % from 32 to 48 (x1.5 concurrency), so p10 is expected at 10.3-10.6 tok/s at 56 and 9.5-9.8 at 64; KV peak
~93 % at 56 and over 100 % at 64 (linear from 80 % at 48). So 64 FAILS on p10 and preempts; 56 PASSES narrowly with
a p10 margin under 6 %; throughput +8-12 % at 56 over 48 (outside the 5 % tie band), +0-10 % at 64 if it runs. The
`--min-tokens 4096` worst case (every completion 4,096 tokens; ~12.6 k KV tokens per sequence) PASSES at 48 with max
latency 370-420 s (the binding gate) and FAILS at 56 (max latency over 420 s or preemption). Expected outcome:
knee 48 per instance after the worst-case confirmation; 56 only if the worst case passes there. Arm B (2 x TP4):
AI Core is already 83 % mean / 100 % max at A's 48, so the host is compute-bound; TP4 buys 2x KV per card at the
price of all-reduce over four cards and half the number of independent schedulers; B lands within +-5 % of A or
below at equal host concurrency, so A stays resident (Q4).

**Decisions.**
1. **Bracket the knee at the ceiling; do not raise it.** Run 56 and 64 per instance on the deployment that is up
   (64 equals `--max-num-seqs`, so the client never exceeds the scheduler ceiling; at 64 the sampled `waiting`
   gauge must stay 0, otherwise the level is reported as ceiling-bound, not as a knee). The ceiling is a deployment
   fact (`DEPLOYMENT_FLAGS`, profile `max_num_seqs`), so raising it would not re-mint `protocol_sha256`, but it is a
   profile revision and changes the graph capture sizes; it happens only if 64 PASSES every threshold (then a `-v2`
   profile with 96, after the base lock). The p10 risk: 11.4 at 48 leaves 14 %; one pass of 576 requests puts p10 on
   the 58th-worst sequence, so a narrow pass at 56 is a tail statistic and is decided by the 3x run, not by one pass.
   Stopping rule unchanged: 56 fails -> 64 is not run.
2. **Requests per level.** The one-pass scan is completed by 56 and 64. The two candidate levels for the 3x re-run
   are the highest passing level and the next lower measured level (the fallback keeps a 3x measurement if the top
   level fails at 3x): expected 56 and 48; if 56 fails one pass, 48 and 32. The `--min-tokens 4096` worst case runs
   at the highest passing level, one pass; if it fails max latency, p10 or preemption, the knee drops one level and
   the worst case repeats there. The 3x re-runs are NOT run preliminarily: they are the recorded measurement and the
   engine bytes are the same, so a preliminary 3x is paid twice. The worst case IS run preliminarily: it is the run
   most likely to fail and its outcome fixes the recorded level list.
3. **Arm B is still measured** because Q4 is a measured rule, and it runs AFTER (1) and the preliminary worst case:
   those use the deployment already up, B needs all eight cards redeployed. B is compared at equal HOST concurrency:
   B levels per instance 16, 32, 48, 64, 96 = host 32, 64, 96, 128, 192 (A's 48 = host 192). B's profile ceiling 64
   would cap it at host 128, so the preliminary B launch uses `MAX_NUM_SEQS=128` (launcher env override; the B
   profile JSON is revised to 128 only if B is ever recorded). B is preliminary-only unless it beats A's host
   throughput at the knee by more than 5 %, in which case B AND A are re-measured on the locked base before the
   residency changes. A preliminary B loss confirms the owner's default and is cited in the base-lock record as such,
   labelled preliminary: the only thing a preliminary number may decide is "do not displace the default".
4. **Arm C (0.95): NOT run.** 4.1 GiB free at 0.92 on 30 s samples; 0.95 asks ~1.8 GiB more and leaves ~2.3 GiB at
   a sampled peak, at the 2 GiB gate, for a 7 % KV gain, and the knee is not KV-bound (80 % at 48). Reopened only
   if 64 fails SOLELY by preemption with p10 >= 10. **Arm D (`--max-num-batched-tokens 8192`): conditional.** TTFT
   p95 13.8 s at 48 is 6 % of the p95 latency; capping prefill cannot raise throughput and only trades TTFT for the
   decode tail. D runs at the knee (one redeploy, one level, one pass) only if the recorded 3x run at the knee shows
   p10 within 5 % of the floor; otherwise S8 records it as not run with this reason. E stays ruled out (Q6).
5. **Preliminary now versus recorded.** NOW, on the preliminary path (`npu_serve.sh`; `--label` and file names carry
   `prelim`; cards must be free; no dataset row): **P1** levels 56, 64 on arm A; **P2** the worst case at the highest
   passing level (repeated one level down if it fails); **P3** arm B redeploy with `MAX_NUM_SEQS=128`, levels 16, 32,
   48, 64, 96, one pass, stopping rule; **P4** arm A redeployed so the cards are in the resident shape (or left down
   if S7 is imminent). Sampler cadence for P1-P3: `npu-smi` every 10 s (30 s can miss a peak at the ceiling). S10
   does not edit `npu_serve.sh` while a P3/P4 launch is in progress (the coordinator sequences it; the sweep client
   is unaffected). ONLY after S10 and S12 land and R2 passes (the RECORDED run, S7: `npu_deploy.sh up`, receipt,
   `npu_verify_base.sh` all 11 PASS, smoke re-run, then the sweep): **R-A1** one pass at the non-candidate levels
   (8, 16, 24, 32 and whichever of 48/56/64 were measured but are not candidates, so the S8 table is complete from
   the locked base); **R-A2** the two candidates at 3x requests; **R-A3** the worst case at the knee; **R-D** only
   under the condition in (4); **R-B** only if B won preliminarily. The recorded baseline and the hour bound quote
   R-* reports only; preliminary reports appear in the baseline document as context, labelled, never in the bound.
6. **Card time remaining** (8 cards; spent so far ~0.8 h wall = ~6.5 card-hours for the preliminary scan).

| run | wall h | card-hours |
|---|---|---|
| P1 56 + 64 one pass (672 + 768 requests at ~55/min host) | 0.5 | 4 |
| P2 worst case, one level, possibly a second | 0.5-0.9 | 4-7 |
| P3 arm B redeploy + 5 levels, P4 arm A redeploy | 1.2 | 10 |
| R0 deploy + receipt + verify + smoke re-run | 0.8 | 6 |
| R-A1 one pass at 5 non-candidate levels | 0.6 | 5 |
| R-A2 two candidates at 3x (~1,700-2,000 requests each) | 1.2 | 10 |
| R-A3 worst case at the knee | 0.5 | 4 |
| R-D conditional | 0.4 | 3 |
| **remaining** | **5.7-6.1** | **46-49** (52-56 with the spent scan; inside the 60-66 of 3.6) |
| R-B only if B wins preliminarily | +1.5 | +12 |

Run list in order: P1 (56, then 64) -> P2 -> P3 (B: 16, 32, 48, 64, 96) -> P4 -> [R2 gate] -> R0 -> R-A1 -> R-A2 ->
R-A3 -> R-D (conditional) -> R-B (conditional). This section supersedes the level lists of 3.2 and the timings of
3.6 where they differ.

### 3.8 Preemption crash and revised run list (revision 4, 2026-10-29)

Input: the Opus bad-case note `work/scratch/badcase/2026-10-29-arm-a-exit-under-worst-case.md` (planner read the note,
no raw log) and P1. Facts: P1-L56 FAILED (p10 9.13, 1 preemption; host 1,952.8 tok/s = +8.8 % over 48; max latency
265 s); 64 skipped by rule. P2 worst case at 48: preemptions on all four instances (11 and 14 on the survivors, one
with 9 timeouts); instances 0 and 3 exited on `AssertionError ... len(mm_embeddings)=0` in `gpu_model_runner.py:3187
_execute_mm_encoder` while recomputing a preempted audio request from `num_computed_tokens=0`; 476/576 ok; the chain
stopped at p2-L32 against dead ports, so 32 is UNMEASURED. The engine default utilisation is 0.92
(`vllm/config/cache.py:103`): the preliminary numbers are at 0.92, R2's Arm C re-derivation is moot, the sidecar now
passes the flag. 3.7 predictions: "56 passes narrowly" REFUTED; "p10 10.3-10.6 at 56" REFUTED (9.13); "KV ~93 % at
56" REFUTED (preemption = 100 %); "+8-12 % at 56" CONFIRMED (+8.8 %); "worst case PASSES at 48, max latency 370-420 s"
REFUTED (all instances preempted, two crashed); "knee 48" REFUTED; "B within +-5 % of A" untested, now doubtful.

**Decision 1, operating rule: a preemption is a CRASH risk for the locked base, not a latency penalty.** In the shipped
engine any preemption of an audio request can kill the instance (2 of 4 survived 11-14 preemptions, so not every
resume is fatal) and with `--restart no` it stays down. Rules: (a) a level is admissible for recording only if the
one-pass `--min-tokens 4096` worst case shows a preemption delta of exactly 0 on EVERY instance with a readable counter
AND a sampled KV peak <= 90 %, so another seed or real prompts cannot cross 100 %; (b) client concurrency of every
recorded run and of any flight is at most the admitted level; (c) a dead endpoint mid-level is a level FAIL and a chain
STOP, never absorbed. KV-safe counts per TP2 instance (564,224 tokens): full window 24,576 -> 22 sequences (guaranteed,
but level 24 is -30 % host throughput: REJECTED); mean mix 12,596 -> 44 (an expectation, not a bound); measured
worst-case footprint 99.8 % at 48 = ~11.7 k tokens per sequence with prefix reuse -> 90 % at 43, 83 % at 40, 66 % at
32. Expected admitted level 40 per instance (host 160), fallback 32. `--max-num-seqs` is a deployment fact (3.5 item
3): lowering it from 64 to the admitted level in a `-v2` profile is defence in depth with no `protocol_sha256` movement
(S12's equality test; capture sizes shrink, so S7 re-reads KV per card from the startup line) but NOT a guarantee (that
needs 22) and does not replace (a). Consequence: knee 48/56 -> 40 or 32; host ~1,610 tok/s interpolated at 40 (-10 % vs
48) or 1,423 at 32 (-21 %); r ~5.8 or 5.2 lock rows/min/instance instead of 6.5, hour bound +12 % or +26 %. C not run.

**Decision 2, the engine bug: understand, reproduce, mitigate in the deployment layer; no engine patch before the lock**
(patching vLLM in the image is a new base: follow-up, section 7).
- **E0 (OPUS, card-free, today).** Read the vLLM / vllm-ascend source in the image (`/vllm-workspace/vllm`, device-less
  `docker run --rm saea-base` on 181): `_execute_mm_encoder`, the resumed-request path that schedules
  `scheduled_encoder_inputs` at `num_computed_tokens=0`, `encoder_cache_manager`, the MM processor cache (82 % hit rate
  seen) and any vllm-ascend override. Deliverable: English note (<= 60 lines) in `work/scratch/badcase/` naming the path
  by which a resumed request's audio input is empty (or exactly what stays unknown) and each candidate knob
  (`--mm-processor-cache-gb 0`, `--disable-mm-preprocessor-cache`, encoder cache budget, preemption mode) with its
  protocol-or-deployment placement. Acceptance: line-cited; no output-changing knob proposed for the deployment layer.
- **E1 / E2 (SONNET, 181 only, free cards only, detached, `prelim` label, in parallel on cards 0,1 and 6,7).** E1: one
  `npu_serve.sh` TP2 instance, worst case at 48 then 56, one pass; accepted if the same assertion recurs, else 3 passes
  with >= 10 preemptions each and no exit = "not reproduced in isolation". E2: same load with E0's top knob
  (`--mm-processor-cache-gb 0` if E0 is not ready); accepted at 3 passes, >= 10 preemptions each, zero exits, and a test
  that the knob leaves `protocol_sha256` unchanged. Bundles under `work/logs/`; OPUS writes the result note.
- **Launcher policy.** `--restart` is REFUSED for recorded runs: a restart drops the in-flight set and resets counters,
  so a run looks complete and is not. SONNET adds a per-level liveness poll (30 s, `/v1/models` per instance) to
  `work/scripts/prelim_chain.py` (host script) and to S7's driver (study change, tested): dead endpoint = FAIL + STOP.

**Decision 3, revised preliminary run list (all `prelim`; cards must read free; 181 only).**
1. **P2b** (0.5-0.9 h, 8 cards): stop arm-a-1/-2, fresh arm A (captured argv, equal cold cache); worst case at 40, one
   pass; if it preempts (a crash is possible) relaunch and repeat at 32. SONNET first changes the chain's P2 descent to
   48 -> 40 -> 32 and adds the poll. P2b runs first: the admitted level gates S7's sweep plan file and S8's bound.
2. **E1 + E2** (1 h, 4 cards) after P2b: they fix R0's deployment flags, which must be final before `up`.
3. **P3** arm B (`MAX_NUM_SEQS=128`; 16, 32, 48, 64, 96 per instance = host 32-192; one pass; stopping rule) plus
   **P3b**, B's worst case at its highest passing level under rule (a) (1.6 h, 8 cards). B matters more now: TP4 holds
   ~1.7 M KV tokens per instance, so host 192 (96 x 11.7 k = 66 %) does not preempt where A's 48 does; if B's admitted
   host level is one step above A's, B can beat A by more than 5 % at the knee and Q4 makes B resident (then R-B and A
   are re-measured on the locked base, 3.7 (3)). Arms compare at equal HOST concurrency, each at its admitted level.
4. **P4** per the R2 gate: skipped if R2 has passed by the end of P3; otherwise arm A relaunched. Cards are left free
   for R0 or holding arm A, never a half shape.

**Decision 4, recorded runs, S7 gate, hour bound.** Candidates 40 and 32 if 40 passes P2b, else 32 and 24. R-A1 one
pass at 8, 16, 24, 48; 56 is NOT re-run on the locked base (it preempted; a recorded crash costs an R0 redo), S8 cites
the preliminary 56 and 48-worst-case rows as labelled context. R-A2 two candidates at 3x; R-A3 worst case at the knee
with the poll; R-D as in 3.7 (4); R-B only if B won. S7 gate additions: a zero-preemption worst-case pass at the
admitted level (P2b or P3b); E0 written and the E1/E2 verdict known before `up`; the `-v2` ceiling profile (if Q-B yes)
and any E2 knob in the tree with the equality test; the driver poll tested. Card time: spent ~10.6 ch (scan 6.5, P1 1.6,
P2 2.5); remaining P2b 4-7, E1/E2 4, P3+P3b 13, R0 6, R-A1 4, R-A2 10, R-A3 4, R-D 3 = 48-51 -> 59-62 ch total, top of
the 3.6 band; R-B +12 (FYI).

**Owner questions (recommendation in brackets).**
- **Q-A.** Accept a crash-safe knee (rule (a)), expected 40 per instance (host 160), ~10 % less host throughput than 48
  and ~12 % more card-hours in the bound, rather than wait for an engine fix? [Yes: the fix is a new image, a base
  change that delays the lock; the knee is re-measured once the engine is fixed.]
- **Q-B.** Lower `--max-num-seqs` 64 -> admitted level in a `-v2` deployment profile for R0 (deployment layer, no
  protocol hash movement)? [Yes if S12's test shows no registered-fact movement; otherwise keep 64 and record (b).]
- **Q-C.** If E2 proves a cache-only knob prevents the crash, carry it in R0's deployment flags? [Yes if output-neutral
  and crash-free over 3 passes; a knob that changes outputs or the hash (chunked prefill off, eager) is NOT adopted.]

### 3.9 Worst-case knee, crash trials and determinism (revision 5, 2026-10-29)

Input: P2b, E1, E2 and the equality probe (chain status files relayed by the coordinator; no raw log read) and the E0
note `work/scratch/badcase/2026-10-29-e0-mm-encoder-preemption-source.md`. Facts: P2b worst case on a fresh arm A, one
pass: 40 -> 9 preemptions (2/4/0/3), KV peak 96.7-100 %, p10 13.85, max latency 306 s, no exit = NOT admissible under
rule (a); 32 -> 0 preemptions on all four, KV peak 80.1-88.1 %, p10 15.76, max latency 266 s = ADMISSIBLE. E1 (one TP2
instance, baseline flags): 13 preemptions at 48, 19 and 22 at 56, many 420 s timeouts, zero exits, no assertion = "not
reproduced in isolation". E2 (`--mm-processor-cache-type shm`): 25/26/22 preemptions, zero exits = passes the 3.8
letter, but the baseline did not crash either, so E2 does NOT discriminate. Equality probe (16 greedy requests,
temperature 0, fixed seed): E1 vs E2 16/16 byte-identical; E1 vs ITSELF 9/16, max |delta logprob| 1.97 (indices 0, 2, 3,
4, 10, 12, 14). 3.8 predictions: "admitted 40" REFUTED, "fallback 32" CONFIRMED, "E1 reproduces the assertion" REFUTED.
Full suite: zero true regressions against HEAD (2 shard-race flakes, 3 identical hangs, 566 shared environment
failures); nothing to decide.

**Decision 1, the admitted knee is 32 per instance (host 128).** Rule (a) holds: 0 preemptions everywhere, 88.1 % <= 90
%. R-A2 candidates are 32 and 24. Rule (b) binds R-A1 too: no recorded run exceeds 32, so R-A1 is one pass at 8 and 16
only; 40, 48 and 56 are NOT re-run on the locked base (a crash costs an R0 redo) and the S8 table quotes the preliminary
48, 56 and worst-case 48/40/32 rows as labelled context. This supersedes 3.8 Decision 4's "R-A1 at 8, 16, 24, 48". Cost:
normal-mix host throughput 1,422.6 tok/s at 32 against 1,795.3 at 48 = -20.8 %; r ~5.2 lock rows/min/instance instead of
6.5; hour bound +26 % (3.8's better case is gone). (The worst-case 2,050 tok/s at 32 is not mix throughput: every
completion is 4,096 tokens.) Card time: spent ~24 ch (3.8's 10.6 + P2b ~7 + E1/E2 ~6; S8 takes exact walls from the
status files); remaining P3+P3b 13, R0 6, R-A1 2, R-A2 10, R-A3 4, R-D 3 = 38 -> ~62 ch total, inside the 60-66 band of
3.6. P3 is now the decisive run: B holds ~1.7 M KV tokens per TP4 instance, so worst-case host 128 is ~44 % and host 192
~66 % of KV. Prediction (S8 marks it): B's admitted host level is 192 and B's host throughput there lands within +-5 %
of A's 1,795 at 48, so B beats A's admitted 1,423 by ~20-25 % and Q4 makes B resident; refuted if B's four-card
all-reduce costs more than ~20 %. If B wins, the recorded run list is re-planned in a revision 6.

**Decision 2, the crash: rule (a) stays, shm is deferred, no discriminating test before the lock.** The assertion
appeared once (P2, 2 of 4 engines, caches warmed by ~5,000 requests) and not in E1, E2 or P2b despite 9-26 preemptions
per pass: E0's path needs a contact-2 request to be preempted AND to miss both the prefix cache and the 2,048-token
encoder cache on resume, a race whose RATE E1 did not reach (cold single instance; 420 s client aborts free KV early),
so E0's path stays the working hypothesis, unconfirmed. Rule (a) does not depend on it: zero worst-case preemptions
remove the precondition, the only guard that needs no understanding of the race. Q-C: shm is NOT carried into R0. It is
non-discriminating; by E0 it sits in the hashed `server_args` fragment today, so adopting it moves `protocol_sha256` or
needs a flag reclassified into `DEPLOYMENT_FLAGS` (a study change) before the lock; and its output-neutrality evidence
is void (Decision 3). The discriminating test, if the owner ever wants shm, is **E3**: fresh arm A 4 x TP2, worst case
at 48, shm on all four, TWO passes (P2 crashed 2 of 4 in one pass; one clean pass has ~6 % chance under no effect, two
~0.4 %) = ~1.2 h wall, ~10 ch, outside the 3.6 band, AFTER P3 and AFTER the lock, gating a `-v3` profile only. It is a
section 7 follow-up with the engine fix; the lock does not wait for it.

**Decision 3, determinism: byte-identity is not an acceptance criterion on this stack.** Two runs of one instance agree
on 9/16 greedy outputs while two configurations agreed on 16/16: the E1-vs-E2 identity is an artefact of the pair
compared, and self-agreement is the ceiling of any equality test. Candidate causes (MoE routing flips from tiny numeric
differences; batch composition with 16 requests in flight; prefix-cache state of a second run; TP2 all-reduce order) are
not separable yet. Consequences: (i) L-facts pin bytes, hashes and flags, never outputs; the base-lock record gains a
measured "greedy self-agreement" fact (rate and max |delta logprob| under a stated send protocol), recorded, not gated.
(ii) Every output-equality acceptance in this plan (3.8 E2, Q-C, any knob test) becomes an A/A/B test: a knob is
output-neutral if its B-vs-A disagreement lies within the A-vs-A band (same probe, same send protocol, >= 3 repeats
each); hash equality tests (S12, registered facts) compare hashes, not outputs, and are unaffected. (iii) Protocol: the
study seeds and `--seed 0` fix sampling, not kernel or batch numerics; S7 records in the manifest the client seed, the
`--seed` server flag and a sequential-send probe before and after each recorded level with its agreement rate; the
baseline document states that outputs are reproducible statistically, not bit-wise. Card-free now:
- **D1 (SONNET).** `greedy_probe.py` (host script, `work/scripts/`): fixed 16-request set, `--send sequential` (one in
  flight, fixed order) or `--send burst`; records text, token ids, logprobs; `compare` reports exact-match rate, first
  divergent token position per request, max |delta logprob| before divergence. Acceptance: unit test on synthetic files;
  `compare` reproduces 9/16 and 1.97 on the two existing E1 probe outputs.
- **D2 (OPUS).** Note (<= 40 lines, `work/scratch/badcase/`) on the three existing probe outputs: which E1 run E2 was
  compared against and whether the other pairing also matches; for each of the seven mismatches the first divergent
  position (0 = prefill numerics, later = decode) and the logprob gap there (near-tie flip or large). Acceptance: the
  compared pair named, positions and gaps tabulated, no cause claimed beyond the evidence.
- **D3 (SONNET, 2 cards, minutes, on the idle arm A before P3 stops it).** 3 sequential-send and 3 burst-send probe
  runs; sequential 16/16 across repeats with burst disagreeing = batch composition; sequential disagreeing = kernel or
  cache nondeterminism. The classification goes into S7's manifest and the base-lock record.

**Decision 4, Q-B: the `-v2` ceiling is 32, not 40.** With no recorded run above 32, headroom buys nothing; a ceiling
equal to the admitted level turns rule (b) from a client convention into an engine-enforced bound (excess requests
queue, they do not preempt); the only cost is 3.7's ceiling-bound check, the sampled `waiting` gauge must read 0 at the
recorded level 32. Conditional, as before, on S12's test showing no registered-fact movement.

**Owner questions (one-word answers; recommendation in brackets).**
- **Q-A.** Accept the crash-safe knee 32 per instance (host 128): -20.8 % host throughput vs 48, hour bound +26 %? [Yes]
- **Q-B.** `--max-num-seqs` 64 -> 32 in a `-v2` deployment profile for R0? [Yes]
- **Q-C.** Carry `--mm-processor-cache-type shm` in R0? [No: deferred to the E3 follow-up after the lock]
- **Q-D.** Accept a locked base that is not bit-reproducible (greedy self-agreement measured and recorded; equality
  acceptance replaced by the A/A band) rather than seek a deterministic engine mode before the lock? [Yes]
- **Q-E.** Permit the P3 launch now (stop arm A, redeploy arm B on 8 cards, ~1.6 h, `prelim`, after D3)? [Yes]

### 3.10 Final configuration and run plan (revision 6, 2026-10-29)

Input: 3.9, the D2 note (`work/scratch/badcase/2026-10-29-d2-greedy-self-agreement.md`) and D3 (idle `saea-arm-a-0`, 16
fixed requests, temperature 0, seed 777, max_tokens 64; sequential S1-S3 then burst B1-B3): S1-S2 9/16, S1-S3 9/16, S2-S3
16/16; B1-B2 8/16, B1-B3 15/16, B2-B3 8/16; sequential-vs-burst 6-7/16; every divergence at decode position 13-58, never
0, top-2 gap 0.0-0.25. Reading: a WARM sequential run repeats exactly, the cold first run differs (cache state), burst
adds batch composition; D2: the E1-vs-E2 16/16 was two cold first runs and 1.97 is a post-divergence logprob. 3.9
Decision 3 stands; the manifest and base-lock classification is "cache-state and batch-composition nondeterminism; warm
sequential greedy repeatable (16/16, n = 1 pair)". This subsection is the whole remaining plan: the P3/P3b branch and the
five owner answers fill the cells and nothing else is re-planned. Reference: A admitted 32 per instance = host 128 =
1,422.6 tok/s normal mix (P2b); B's admitted host level is the highest P3 level whose P3b worst case meets rule (a).

**1. Resident configuration by P3/P3b branch.** Common to every branch: image `saea-base:20261027`, `--gpu-memory-utilization
0.92` (C not run), `--max-num-batched-tokens null`, `--shm-size 16g` (docker, unrelated to the shm knob), `--network host`,
NIC = the host's global-IPv4 interface derived by `npu_deploy.sh` (`enp61s0f2` on 181) into `HCCL/GLOO/TP_SOCKET_IFNAME`,
the four read-only Ascend mounts, bind `127.0.0.1`, hashed fragment unchanged (`--seed 0` stays). Shm knob: ABSENT (Q-C no).

| cell | (i) B beats 1,422.6 by > 5 % (> 1,493.7) | (ii) B within +-5 % | (iii) B worse | (iv) B crashes / no admissible level |
|---|---|---|---|---|
| resident arm, TP, instances | B, TP4, 2 | A, TP2, 4 (Q4 tie -> A) | A, TP2, 4 | A, TP2, 4 |
| cards / ports / `HCCL_IF_BASE_PORT` | i0 0-3 :8200 61000; i1 4-7 :8201 61200 | i0 0,1 :8200 61000; i1 2,3 :8201 61200; i2 4,5 :8202 61400; i3 6,7 :8203 61600 | as (ii) | as (ii) |
| NUMA cpuset | i0 `128-159,192-223`; i1 `0-31,64-95` | i0 `192-223`; i1 `128-159`; i2 `0-31`; i3 `64-95` | as (ii) | as (ii) |
| `--max-num-seqs` ceiling (Q-B yes / no) | B's admitted per-instance level (expected 96) / 128 (the P3 launch value) | 32 / 64 | 32 / 64 | 32 / 64 |
| profile file | `qwen3-omni-thinking-bf16-181-2xtp4-v2.json` (needed in both Q-B cases: -v1 says 64) | `...-4xtp2-v2.json` (Q-B yes) or `-v1` unchanged (Q-B no) | as (ii) | as (ii) |
| admitted level / host concurrency | expected 96 / 192 (measured by P3b) | 32 / 128 | 32 / 128 | 32 / 128 |
| expected host tok/s (normal mix) | measured P3 value, expected 1,705-1,885 (3.9 prediction) | 1,422.6 | 1,422.6 | 1,422.6 |
| r (lock rows/min/instance), T | contacts/min per TP4 instance / 2, expected ~13.1; T = 4 | ~5.2; T = 2 | ~5.2; T = 2 | ~5.2; T = 2 |

Base-port note (coordinator, 2026-10-29): the table's `HCCL_IF_BASE_PORT` values are the profile's (61000-61600 for A), which `npu_deploy.sh up` launches with. The PRELIMINARY arm A launched by `npu_serve.sh` ran with 60000/60100/60200/60300 (captured in `work/logs/prelim-chain-20261029T094933Z/arm-a-launch.json`); arm B preliminary used 60400/60500. Base ports are a recorded deployment fact (S12), outside `protocol_sha256`, so the preliminary and recorded runs differ in this fact and S8 states it when it quotes preliminary rows as context.

Q-C yes changes one cell in every column: the profile becomes `-v3` (= `-v2` plus `--mm-processor-cache-type shm`), which
first needs the flag reclassified into `DEPLOYMENT_FLAGS` with the hash-equality test (study change, SONNET) and E3 (two
clean worst-case passes at 48 on 4 x TP2, ~10 ch) BEFORE R0; the lock then waits for both. Branch (i) is provisional
until item 2's recorded comparison confirms it (3.7 (3): a preliminary number cannot displace the default).

**2. Recorded run list.** All runs: `npu_deploy.sh up` on the branch's profile, receipt, `npu_verify_base.sh` 11/11 PASS,
smoke re-run; `--restart` refused; liveness poll every 30 s (dead endpoint = level FAIL + chain STOP, R0 redo); `npu-smi`
every 10 s; per-level checkpoint and `--resume`; client seed 20261028 and server `--seed 0` in the manifest; the greedy
probe (D3 set, `--send sequential`) once as a discarded warm-up after `up`, then before and after every recorded level,
agreement rate and first-divergence positions recorded, NOT gated (Q-D). Stopping rules: a level failing any 3.4 threshold
stops the arm there; preemption delta > 0 in an R-x3 worst case drops the admitted level one measured step and R-x3
repeats there (+4 ch), the R-x2 row at that level becomes the knee; with the ceiling equal to the admitted level the
sampled `waiting` gauge must read 0 at the knee, else the row is reported ceiling-bound and the ceiling is raised one
step in a profile revision (no hash movement). `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` is NOT set in any run below (owner
ruling Q5); it is the first post-lock recorded change: `COMMON_ENV` -> `protocol.env`, re-minted `protocol_sha256`,
base-lock successor note, verification re-run, no re-sweep; value rule: never below the engine default 300 s, the smallest
multiple of 60 s that is >= 10 x the recorded max TTFT of R-x3 at the admitted level.

| run | (ii)/(iii)/(iv): A resident | (i): B resident, A compared on the locked base |
|---|---|---|
| R0 | deploy A, receipt, verify, smoke: 0.8 h, 6 ch | R0-A as left (6 ch), then after R-A2 `down` and R0-B (0.8 h, 6 ch) |
| R-x1 one pass, non-candidates | R-A1: 8, 16 per instance (0.25 h, 2 ch); 40/48/56 never on the locked base | R-B1: 16, 32, 48 per instance (0.5 h, 4 ch) |
| R-x2 candidates at 3x requests | R-A2: 32 and 24 (1.2 h, 10 ch) | R-A2: 32 only (0.6 h, 5 ch); R-B2: admitted (96) and the next lower measured (64) (1.2 h, 10 ch) |
| R-x3 worst case `--min-tokens 4096`, one pass | R-A3 at 32 (0.5 h, 4 ch) | R-B3 at B's admitted level (0.5 h, 4 ch) |
| R-D (`--max-num-batched-tokens 8192`) | only if R-A2 at 32 shows p10 within 5 % of 10 tok/s (0.4 h, 3 ch) | same rule on R-B2 at the admitted level (3 ch) |
| residency check | none (Q4 default) | recorded R-B2 (admitted) vs recorded R-A2 (32): B stays only if > 5 %; else redeploy A (+6 ch) and run R-A1, R-A3 (+6 ch) |
| remaining / total card-hours | 25 / ~62 ch (24 spent + 13 P3, less in (iv)); wall ~3.1 h | 39 / ~76 ch (88 if the check overturns); wall ~4.9 h; above the 3.6 band, the price of the Q4 rule |

**3. Owner answers (defaults = the 3.9 recommendations; the cells that move if the owner says otherwise).**
- **Q-A yes** (crash-safe knee). No: no lock on this image; items 2, 4 and 5 wait for the engine-fix image (section 7, a new
  base); P3/P3b still run as context; nothing else in this subsection changes.
- **Q-B yes** (ceiling = admitted level). No: the ceiling row reads 64 (A) / 128 (B); rule (b) stays client-enforced; the
  `waiting` = 0 check is dropped; A keeps `-v1` and needs no profile commit.
- **Q-C no** (no shm). Yes: the `-v3` paragraph above; +10 ch and one study change before R0.
- **Q-D yes** (statistical reproducibility). No: same consequence as Q-A no (no deterministic mode exists on this stack, 3.9).
- **Q-E yes** (P3 now). No: branch (iv) applies with "B not measured (owner)" in S8; the 13 P3 ch are not spent (~49 ch total).

**4. Experiment-efficiency plan for the flights.** The locked service is used at exactly the admitted level per instance
(32 on A, B's measured level on B), never above (rule (b)); flight workers = instances x admitted level, each trajectory
pinned to one instance. Until the flight-side multi-endpoint transport exists (section 7: route by profile, strip
`id_slot`) a flight may use ONE instance at its admitted level, 1/4 (A) or 1/2 (B) of the host figure; the baseline
document states this caveat. The flight mix must stay within the synthetic one (prompt mean 8,500 in [3,000, 20,000],
`max_tokens 4096`, budget 3,072, one clip per trajectory, 3 contacts); longer prompts or more audio per prompt raise the
worst-case KV footprint and re-trigger R-x3 before the flight. Hour bound, registered form: `card-hours = ceil_0.1(1.3 x
1,080 x 7.52 x T / r / 60)`. A: T 2, r ~5.2 -> 67.7 ch (~8.5 h wall on 8 cards); cross-check 16,200 x ~2,050 tokens /
1,422.6 tok/s = 6.5 h x 8 = ~52 ch, ~67 with the 1.3 margin, inside the 20 % band. B expected: T 4, r ~13.1 -> ~53.7 ch
(~6.7 h); S8 quotes r from R-B2 only. Re-measure triggers (R0 + R-x2 at the admitted level + R-x3; the full ladder only
for an image change): image id, any hashed `server_args` or `DEPLOYMENT_FLAGS` value, profile revision (`-vN`),
driver/CANN/firmware, any `VLLM_` env except the timeout above (verification re-run only), shm adoption after E3.

**5. Completion criteria ("base locked, configuration and run plan explicit").** [ ] Q-A..Q-E answered and the resolved
cells recorded here; [ ] P3/P3b outcome recorded, branch named; [ ] R2 gate list ticked; [ ] `-v2` (or `-v3`) profile
committed with the S12 equality test; [ ] S7 driver tested offline (poll, probes, checkpoint/resume); [ ] R0 receipt,
11/11 verification, smoke re-run on the resident profile; [ ] R-x1, R-x2, R-x3 (R-D if triggered; the (i) residency
check) done or stopped by rule; [ ] S8 baseline: per-level table, r, registered bound and cross-check, every 3.1-3.9
prediction marked; [ ] S5 base-lock record: L1-L16, the greedy self-agreement fact (D3 + S7 probes), design-doc section 7
post-fix successor cited; [ ] R3; [ ] S9 read-back; [ ] R4; [ ] memory successor note for
`saea-serving-deployment-decision.md`; [ ] commit on `npu-vllm-port`, scrub, publish to `chaosxingxc-orion/omni` `main`
when the owner asks. Post-lock, not gating: the timeout change (item 2), E3/shm, the engine-fix image, the transport.

**6. Card-free work that starts now (disjoint files).** SONNET S7 driver (`scripts/serving/record_runs.py`, not yet
existing; the liveness poll in `synthetic_sweep.py` is in the tree, uncommitted): R0 sequence, poll, seed and sequential
probes, per-level checkpoint/resume, fixture-tested; plus the S7 sweep plan file carrying every prediction above. SONNET
S16: `-v2` profiles for both arms (A ceiling 32; B ceiling 96 with a 128 variant) and the `-v3` shm reclassification as a
separate patch, each with the S12 equality output; nothing committed until Q-B/Q-C. OPUS: S8 skeleton (branch
placeholders, labelled preliminary rows, the bound with A's numbers); S5 skeleton (L1-L16 read methods, receipt
placeholders, the D3 fact); the design-doc section 7 post-fix successor (gates S5, R2 decision 4); the D3 note in
`work/scratch/badcase/` (<= 30 lines). FABLE: R3/R4 and the owner relay only.


**Owner answers (2026-10-29, translated; recorded by the planner).** Q-A knee 32: YES. Q-B `-v2` ceiling 32: YES (the `-v2` profiles are committed with this revision). Q-C shm in the recorded runs: YES, against the recommendation; by item 3 this means a `-v3` profile with `--mm-processor-cache-type shm` reclassified as a deployment flag (SONNET S16b: `DEPLOYMENT_FLAGS` entry, S12-style equality test proving `protocol_sha256` unchanged, refusal of any other `--mm-*` value), and the discriminating test E3 (fresh 4xTP2, worst case at 48 per instance, shm on all four, two passes, PRELIMINARY, ~10 ch) BEFORE R0; E3 is accepted when no engine exits in either pass; if an engine exits, Q-C is re-asked with that evidence. Q-D non-bit-reproducible base: YES. Q-E (P3 launch): the launch command was denied by the session permission system twice; retried after these answers.


### 3.11 Arbitration and final judgement (Fable 5.1, main session, 2026-10-29)

The owner asked for a final ruling on the points discussed today. Evidence: 3.7-3.10, the Opus notes under
`work/scratch/badcase/` (crash, E0 source read, D2), the P1/P2/P2b/E1/E2/D3 preliminary results, and the owner's answers.

1. **Base lock proceeds on this image.** The crash is real but bounded: it needs preemption, and preemption needs the
   worst case above 32 per instance on 4xTP2. The admissibility rule (zero worst-case preemptions, KV peak <= 90 %)
   keeps the recorded runs out of that regime. Waiting for an upstream fix is not justified. FINAL: lock now.
2. **Knee and ceiling.** Admitted knee 32 per instance (host 128) for arm A; `-v2` ceiling 32 so the engine enforces it.
   The 21 % throughput cost against 48 is the price of a base that cannot lose an instance mid-flight. FINAL.
3. **The shm knob (owner YES, planner NO).** Both positions are correct on their own terms: the planner's "no" is about
   evidence (E2 did not discriminate) and hash placement; the owner's "yes" is about robustness over the many hours the
   base will run, where a real flight's mix can preempt even if the synthetic worst case does not. Ruling: the owner's
   choice stands, conditioned on evidence rather than on E2. (a) `--mm-processor-cache-type` is a cache-transport knob
   that by source changes no logits; it is reclassified as a DEPLOYMENT flag (S16b), with the S12-style test that
   `protocol_sha256` is unchanged and a refusal of any other `--mm-*` flag. (b) E3 runs on 4xTP2 in the known crash
   regime (worst case at 48 per instance): one baseline pass, then two shm passes, all four instances. shm is ADOPTED
   into `-v3` if its two passes show zero engine exits AND a warm sequential A/A probe (D3 protocol) agrees 16/16 with
   the baseline instance. If a shm pass exits, shm is DROPPED and the owner is told with the evidence. The baseline pass
   is informative only: a second crash strengthens E0's hypothesis; no crash does not change the ruling, because the
   owner has chosen robustness and the rule in (1) already protects the recorded runs. Cost ~12 ch. E3 precedes R0.
4. **Determinism.** Accepted as answered. The base-lock record states the measured fact: warm sequential greedy output
   repeats (16/16, n = 1 pair); a cold first run and burst batches differ at near-tie tokens. No gate uses byte identity.
5. **Arm B.** P3/P3b is the one remaining measurement; 3.10's table resolves the resident arm mechanically from it, and the
   shm flag, being a deployment flag, carries to whichever arm is resident. Order of card work: P3/P3b -> E3 -> R0 on the
   resident arm with the final profile -> the 3.10 recorded list. The P3 launch command was denied twice by the session
   permission system; the main thread retries it now and otherwise hands the owner the one-line launch command.
6. **What ends the task.** The 3.10 completion checklist, unchanged. The planner's remaining actions are R3/R4 only.

---

## 4. Streams (disjoint files), with ready-to-paste brief headers

Tags per CLAUDE.md principle 5: **OPUS** = anything written or read (reports, records, doc edits, briefs); **SONNET**
= code, scripts, tests, builds, card runs; **FABLE** = plan, evaluation, feedback, owner questions. English only. Only
the main thread commits. Every brief header ends with the rails: "No dataset row, no exposure entry, no readiness
flip, no GitHub push, no busy card, English only, work on the working tree not HEAD, hand back a diff description."

### Done or in flight (do not touch their files until the main thread commits them)

| id | tag | state | files |
|---|---|---|---|
| S0 | OPUS | done | `work/scratch/brief/2026-10-29-task-understanding.md` |
| S1 | SONNET | done, uncommitted | `core/model/vllm_transport.py`, `tests/unit/test_vllm_transport.py` |
| S2+S3 | SONNET | done, uncommitted; host script outside git | `work/scripts/npu_smoke_thinking.py`, `work/scripts/test_npu_smoke_thinking.py`, design doc 5.5 |
| S4 | SONNET | running | `scripts/serving/verify_engineering_base.py`, `tests/unit/test_verify_engineering_base.py`, `work/scripts/npu_verify_base.sh` |
| S6 | SONNET | running | fragment, `core/session/serving.py`, `core/deployment.py`, `rails/adapters/runtime_view.py`, `tests/unit/test_deployment_unhashed.py` |

### New streams

**S1b (SONNET) — second serving-layer overflow wording.** Goal: cover `vllm/renderers/params.py:354-375`
(`parameter="input_text"`, character-level) with a fourth `fullmatch` pattern, or record in the test file why it is
out of scope. Files: `src/.../core/model/vllm_transport.py` (patterns only), `tests/unit/test_vllm_transport.py`.
Acceptance: `fullmatch` and the `BadRequestError` type gate kept; near-miss test asserts `None`; existing tests pass.
Starts after S1 is committed.

**S10 (SONNET) — deployment argv parity with the working launcher.** Goal: make `core/deployment.py docker_run_argv`
produce a container that comes up on multiple cards AND whose live cmdline the receipt gate accepts. (a) Add the four
read-only mounts (`/etc/hccn.conf`, `/etc/ascend_install.info`, `/usr/local/Ascend/firmware`, `/usr/local/dcmi`) to
`ASCEND_READONLY_MOUNTS`; add `HCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME` = a `--nic` value that
`work/scripts/npu_deploy.sh` derives on the host exactly as `npu_serve.sh:41` does; `plan`/`up` REFUSE to emit argv
without a NIC (fail closed); the NIC is written into `deployment_record`. (b) Replace `--entrypoint vllm` with
`--entrypoint /usr/local/python3.12.13/bin/python3` followed by `-m vllm.entrypoints.cli.main serve <model_dir> ...`
so the live argv is exactly the `VLLM_ENTRYPOINT` form; the gate is NOT widened. Align `npu_serve.sh` the same way so
smoke servers are receipt-able. Files: `src/.../core/deployment.py`, `tests/unit/test_deployment.py`,
`work/scripts/npu_deploy.sh`, `work/scripts/npu_serve.sh`. Acceptance: a unit test feeds the emitted argv (after the
image token) to `contracts.gate_runtime_vllm.split_vllm_server_argv` and it parses; a test proves refusal without
`--nic`; emitted env/mount set equals `npu_serve.sh`'s; `deployment_record` carries the NIC; no card touched. Starts
after S6 is committed (same file). No owner question: the gate form is the committed contract, the launcher is ours.

**S12 (SONNET) — per-instance deployment env out of `protocol_sha256`.** Goal: stop `HCCL_IF_BASE_PORT` (different
per instance: 61000/61200/61400/61600) from entering `protocol.env`, which today makes one protocol hash unable to
cover four instances and fails the live identity check on three of them. Fix: a named constant
`DEPLOYMENT_ENV_KEYS = {HCCL_IF_BASE_PORT, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME, TP_SOCKET_IFNAME}` in
`gate_runtime_vllm.py`; `pinned_environment` excludes exactly these (as it excludes the device key); the receipt
producer records them under `deployment.instances[].env`; the live check compares them per instance against the
instance record, fail closed; a profile `instances[].env` key outside the set is REFUSED (no smuggling of `VLLM_` or
other protocol env out of the hash). Making base ports uniform is rejected: all instances share `--network host`.
Files: `src/.../contracts/gate_runtime_vllm.py`, `src/.../core/session/vllm_identity.py`,
`scripts/write_vllm_runtime_receipt.py`, `tests/unit/test_vllm_runtime_receipt_gate.py`,
`tests/unit/test_vllm_session_identity.py`. Acceptance: a four-instance fixture with distinct base ports yields one
`protocol_sha256` and passes identity for all four; a fixture with `VLLM_FOO` in instance env is refused; a fixture
whose live base port differs from its record fails; `protocol_sha256` of a single-instance receipt without these
keys is unchanged; full suite passes. Disjoint from S10 and S4.

**S13 (SONNET) — S6's eight dependents.** Goal: carry the `--max-num-seqs`/`profile_slot_count` removal through
`MAX_NUM_SEQS_DEPENDENTS` (`scripts/flights/nx/l338_port_family.py:73-85`): the two baseline configs' `serving_slots`
/ `serving_slot_tokens` (`configs/baseline/...fold-b-{,smoke-4-}baseline-raw-port-v1.json`), the fragment's
`geometry_basis` text, the registration writer re-run (`composed_config_hash` / `protocol_hash` of every plan in the
2026-10-28 package, DRAFT rows stay DRAFT), the phase-supply re-pin IF any phase supply exists (model-free), and
the `MAX_NUM_SEQS_DEPENDENTS` list itself updated to "moved". Also check `core/driver/speech_native.py
_profile_geometry` (reads `-np`/`-c`) for the path a vllm fragment reaches. Files: the two baseline configs,
`scripts/flights/nx/l338_port_family.py`, the writer's outputs under `docs/readiness/2026-10-28-...-npu-port-v1/`,
`tests/` for the geometry check. Acceptance: `rails.adapters.runtime_view.configured_geometry` passes for the vllm
baselines; the OLD fragment form (`--max-num-seqs` inside `server_args`) is REFUSED by a test, not silently dropped;
llama fragments' derived argv byte-identical before/after (diff shown); registration offline check passes; no
readiness field changes. Starts after S6 is committed.

**S15 (OPUS) — design-doc successor edits.** Goal: (a) close 5.5's "mechanism is an inference" caveat with the
`renderers/hf.py:263-299, 804-809` citation and the `REFUSED_SERVER_FLAGS` fact; (b) section 7: the deployment env
keys and mounts, the launch form (L16), the NIC as a deployment fact; (c) section 8: one line on the overflow
relabel wording (Q3). Files: `docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md` only, after S3's edit is
committed. Acceptance: every sentence cites a file and line; no section 9 decision added or implied; Q1, Q4, Q5, Q6
listed as open with the recommendations of 2.4; English only.

**S7 (SONNET) — NPU sweep execution.** Goal: on `npu2`, record `npu-smi` (cards must read below the profile's
`busy_threshold_mib 4096`; idle today is ~3,440 MiB, so a refusal means STOP, not raise the threshold), run
`npu_deploy.sh up` for arm A, write the runtime receipt, run `npu_verify_base.sh` and require all 11 checks PASS, run
the smoke re-run (overflow, template, tool-call per-request `chat_template` probe) on the resident service, then the
5-level scan; redeploy for B; re-runs; C, D. Files: `work/logs/sweep-*.json`, `work/logs/base-verification-*.json`,
`work/logs/receipt-*.json`, an `npu-smi` sampler `work/scripts/npu_sample_cards.sh`. Acceptance: 3.1 predictions
copied into the sweep plan file before `up`; every level's thresholds evaluated by the tool; CephFS free space
checked before each arm; no dataset row; no busy card; resident service left up; `npu-smi` recorded before and after
each arm. GATE: starts only after R2 passes.

**S5 (OPUS) — base-lock record** (1.3). Inputs: S7's verification JSON and receipts, L1-L16. Files:
`docs/readiness/2026-10-29-engineering-base-lock-npu-port-v1.md` + `.../base-lock-v1.json`. Acceptance: every fact
has value, read method, hashed/recorded; the unreadable firmware fact is stated as unreadable; grants no authority.

**S8 (OPUS) — speed baseline and hour bound** (3.5). Inputs: S7's reports. Files:
`docs/results/2026-10-29-npu-port-synthetic-speed-baseline-v1.md`. Acceptance: every number traces to a report
sha256; the plan's selection rule applied explicitly to the table; r derived and the registered formula shown with
the cross-check and their difference; each 3.1 prediction marked confirmed/refuted; states the knee is not usable by
a flight until the multi-endpoint transport exists; no readiness flip.

**S9 (OPUS) — verification read-back.** Files: `work/scratch/brief/2026-10-29-verification-readback.md`. Acceptance:
each of L1-L16 traced to a pinned location and a check, or reported as a gap; S8's arithmetic re-done independently.

**R1-R4 (FABLE) — review gates**, one after each wave, using section 6. R2 additionally requires: full test suite
green (named environment-only exceptions aside); `verify_engineering_base.py` passes offline on fixtures including
the four-instance case; the S10 dry-argv test passes; llama-path diff empty; no `[\x{4e00}-\x{9fff}]` in touched files.

### Waves

| wave | streams | gate to start | gate to leave |
|---|---|---|---|
| 0 | commit in-flight work in the R1 order: translation (with the section-9 hunks of the design doc), S3 (5.5 hunk), S1, S6, G (stale-package guard), S4, this plan; host scripts stay outside git | now | R1 PASSED 2026-10-29 (conditional, see "R1 review" below): suite green, files disjoint, nothing committed by a stream |
| 1 (off-NPU, parallel) | S10, S12, S13, S1b, S15 | R1 passed; S6 and S3 committed | R2 (see above) |
| 2a (cards, 181, PRELIMINARY) | P1-P4 of 3.7 on `npu_serve.sh` | arm A up, no busy card, S10 holds `npu_serve.sh` edits during launches | reports labelled `prelim`; 3.7 predictions marked; B compared under Q4 |
| 2b (cards, 181, RECORDED) | S7: R0, R-A1..R-A3, R-D/R-B conditional | R2 passed | all recorded runs done or stopped by rule; verification PASS on arm A |
| 3 (parallel) | S5, S8 | S7 reports + receipts exist | R3: numbers trace, formula is the registered one |
| 4 | S9 | S5, S8 committed | R4: gaps listed; main thread writes the memory successor note (G8) |

### R1 review (Fable 5.1, 2026-10-29, working tree over HEAD 9db58c1c)

Verdict: **GO, conditional** on (1) the HEAD-vs-tree full-suite comparison showing zero true regressions and (2) the
one-line fix below landing in the translation commit. Diffs read in full: S1, S3, S4, S6, the translation, the plan.

- Blocking (trivial): `docs/readiness/2026-09-04-e22-esb-shortform-alignment-v1-verdict.md:139` still carries the raw
  Chinese ASR hypothesis beside its new English gloss. The file is touched and is not byte-pinned anywhere (checked:
  no sha256 or blob id of its HEAD bytes appears in the tree), so principle 1 applies: keep the gloss, drop the raw
  characters, point at the sealed check artefact for the bytes. After that, `grep -rlP '[\x{4e00}-\x{9fff}]'` over the
  touched files is clean; the 27 remaining tracked carriers are sealed records, receipts with `english_translation`
  fields, or test fixtures, all covered by `docs/readiness/2026-10-29-english-successors-for-pinned-owner-quotes.md`.
- S6 is committed WITHOUT the re-pin (option i). Why this is safe: every contact path recomputes
  `protocol_hash(composed.config_hash, arm, split_hash)` from the live composed config and refuses a plan whose
  `protocol_hash` differs (`core/driver/speech_native.py:11666-11675`, `:13331-13337`, `core/driver/split_check.py:71-74`),
  so the stale DRAFT package (`protocol_hash 2de63028...` / `521bf157...`) cannot flight; and the admission draft is
  `DRAFT-NOT-MINTED`, minted only by the main thread after countersign. Guard G (new, wave 0, model-free, lands right
  after S6): `tests/unit/test_l338_port_package_staleness.py` recomputes both cells' `composed_config_hash` via
  `core.config.compose` and asserts equality with `protocol-registered-v1.json`, marked `xfail(strict=True, reason=
  "stale since S6; S13 re-mints")`, so the suite stays green now and S13 is forced to remove the marker when the re-run
  makes it pass; and `scripts/flights/nx/l338_port_mint_receipt.py` refuses to mint when the recomputed hashes differ
  from the package (fail closed), with a test. S13 absorbs: the two baselines' `serving_slots` (client concurrency,
  unhashed after S6, still asserted nowhere), the writer re-run, an English successor line for the four superseded
  DRAFT ledger rows (their plan-file sha256 pins go stale when the writer rewrites the plan files), the phase-supply
  re-pin (`prepared/.../phase-supply-v1.json` pins `config_sha256 0e09988f...`), and one sentence in `geometry_basis`
  that `seq4` in the fragment NAME is historical (renaming the fragment is NOT recommended: it moves every reference).
- Non-blocking findings carried into wave 1: (a) `verify_engineering_base.py:549-556 live_vllm_args` accepts only the
  `--entrypoint vllm` form; S10 changes the launcher to the `python3 -m vllm.entrypoints.cli.main serve` form, so S10
  must also update this parser and its fixture (`tests/unit/test_verify_engineering_base.py:267 "Path": "vllm"`); the
  verifier does not yet name the S10/S12 defects (1.2 promised it), R2 checks this. (b) S15 must record Q1, Q4, Q5, Q6
  as RULED (2.4), not open, and correct design-doc 5.5's Q2 text (`--chat-template` is refused, not merely hashed).
  (c) `_refuse_vllm_slot_flag(merged)` in `merge_server_args` passes no fragment name, so the refusal reads `'?'`;
  cosmetic. (d) `l338_port_derive_cells.FORBIDDEN` matches `np4` inside the historical fragment id quoted in
  `geometry_basis`; pre-existing at HEAD, scoped to derived baseline/experiment docs, not a regression.
- Publish scrub (owner ruling: push to public `main` after commit): no internal host address, no proxy host or variable,
  no credential, no employee id, no e-mail in the diff or the new files. Present and already public at HEAD: the data
  root path `/mnt/cephfs/xingchao/omni/...` (3 files; also in a pushed commit subject), hostname `A800I-03` (3 files),
  `--user 1008:1008` (5 files), the fixture literal `10.0.0.5` (RFC 1918, pre-existing). New but harmless: NIC name
  `enp61s0f2`, log name `saea-serve-thinking-20261029T063554`. Cosmetic: this plan says "host 181"; the ssh alias
  `npu2` is the preferred spelling.

### R2 review (Fable 5.1, 2026-10-29)

Scope: working tree over HEAD `32589378`, read with `git diff` / `git status` (28 modified, 1 new file) plus the host
scripts `work/scripts/npu_deploy.sh` and `npu_serve.sh.s10-next`; no card, no log, no model call. Owner rule today:
host 175 is not ours; every run of this plan is on `npu2` (181). Verdict: **PASS, conditional** on (1) the full-suite
run on the tree being green and (2) the HEAD-vs-tree comparison classifying `tests/contract/test_real_receipts.py` and
the `test_flight_repin_requirement_identity.py` collection error as pre-existing at HEAD (both are outside wave 1's
files). Common checks: `grep -rlP '[\x{4e00}-\x{9fff}]'` clean over every touched file and both host scripts; files
disjoint per stream (S10 took the verifier and its test from the committed S4 as R1 (a) assigned; S12 owns the shared
fixture `_vllm_world.py`, S10 only reads it); nothing committed by a stream; readiness `status` values identical HEAD
vs tree (`DRAFT-HELD`, `DRAFT-NOT-MINTED`, 5 x `DRAFT-HELD-REGISTERED...`, 6 x `PENDING`, 7 x `PROVISIONAL`); no
exposure entry, no dataset row, no push. Hash movement: only S13's re-mint (below); S12 showed `protocol_sha256`
`3b79f78a` unchanged on the fixed fixture; S10's NIC lives in `deployment` (test proves `protocol_sha256` equality).

- **S1b: ACCEPT.** Fourth pattern is `fullmatch`, the 400 + `BadRequestError` gate is untouched (near-miss tests refuse
  `NotFoundError`, status 500, a prefix, a suffix, one changed word, the wrong parameter clause). Making both "at
  least " fragments optional is correct: `params.py:494-517` prints the qualifier only at `max_input_tokens + 1`.
- **S12: ACCEPT.** `DEPLOYMENT_ENV_KEYS` is closed and pinned by a test; `pinned_environment` excludes exactly it plus
  the device key, `HCCL_CONNECT_TIMEOUT` stays protocol; a deployment key in `protocol.env` is refused; an instance
  `env` key outside the set is refused (5 cases); the live check is an exact per-instance comparison (missing, extra
  and differing all refused); `start_vllm_session` spawns with the recorded instance env. The OPTIONAL `env` does not
  weaken fail-closed: absent is read as `{}` and the exact comparison then refuses any live process carrying one of
  the four keys (tested); the producer always writes the field. No fix.
- **S10: ACCEPT-with-fix.** Verified: six read-only mounts equal the launcher's; `nic` keyword-only with no default
  (`TypeError` test) and `validated_nic` fullmatch refuses `None`, empty, whitespace, `;`, 16 bytes, leading `-`;
  `plan`/`emit` validate the NIC before any argv exists; profile env outside `DEPLOYMENT_ENV_KEYS` and NIC keys in a
  profile both refused; live argv parses through `split_vllm_server_argv`; verifier `live_vllm_args` accepts only the
  interpreter + `VLLM_ENTRYPOINT` form and `server_argv_sha256` hashes the same `serve <model> <flags>` list as
  `deployment_record`. Fix (SONNET, sidecar only, before the swap): add `--gpu-memory-utilization
  "${GPU_MEM_UTIL:-0.92}"` to `npu_serve.sh.s10-next` so a preliminary server satisfies `REQUIRED_DEPLOYMENT_FLAGS`;
  `--seed 0` stays, it is in the hashed fragment and is not a defect. Finding for S8: neither launcher passes
  `--gpu-memory-utilization`, so the preliminary arm A ran at the engine default, not at 0.92 as 3.7 assumes; S8
  quotes the value from the watchdog status (engine start line), and the Arm C margin in 3.7 (4) is re-derived.
- **S13: ACCEPT-with-fix.** Re-mint is the intended movement only: flight `21bf1af0`->`77185307` / `2de63028`->
  `de9937af`, smoke `14d06e30`->`22f715bf` / `521bf157`->`205ae3fc`, `study_commit_at_registration` and the two
  `*_base_commit` fields `7db9d7cd`->`32589378`; the old hashes survive only as the `STALE_HASHES` fixture of guard G
  and in this plan. Guard G's strict xfail became the live assertion and the stale fixture now uses the pre-move
  hashes (good: it tests the refusal, not the committed bytes). The inline `--max-num-seqs=4` spelling is refused and
  tested, as is `profile_slot_count`. Fix: the ledger successor line, decision (1) below.
- **S15: ACCEPT-with-fix.** `hf.py:263-314`, `804-809`, `599-616` cited; Q2 corrected and its gate lines verified at
  `32589378` (`:76`, `:89-106`, `:333-334`); the section 7 note is explicitly the pre-fix state with line numbers at
  `32589378`; no section 9 text moved (diff hunks end at line 397, section 9 starts at 403). Fix: decision (c).

**Decisions on the open items.**
1. S13 (1), ledger rows 9175-9178: four plan-file pins (`488dac0e`, `90df133e`, `9e482a77`, `a5598d5c`) and `7db9d7cd`
   are stale. The rows are DRAFT, unconsumed, countersignature pending; the writer's `draft-ledger-rows.md` is the
   authority and the rows are re-copied at countersign. OPUS appends one English successor line directly under the
   DRAFT block heading (line ~9170) naming the four superseded pins, the re-minted package and the commit; the rows
   themselves are not edited. MAIN commits it with S15 (commit 5).
2. S13 (2): NO second writer run. `32589378` is the commit the registered facts were computed from (configs unchanged
   since `db798a83`); the precedent is `a50581d4`, registered at `7db9d7cd`. A re-run would move only the commit
   string and need its own commit. The mint step re-checks the live hashes anyway (guard G).
3. S13 (3): the phase supply stays stale and un-pinned. `core/flight/runtime.py:359, 402` binds `config_sha256`
   explicitly, so the stale supply cannot be consumed; `repin_cell.py` flips readiness and rewrites the baseline pin,
   which is the flip step (package README) and needs the OWNER's countersign. MAIN runs it then, on a clean tree.
4. S10 (b): see the S10 fix. Smoke servers from `npu_serve.sh` remain preliminary; the RECORDED smoke re-run (R0)
   runs on the `npu_deploy.sh up` service, as 3.7 (5) already says. S10 (c): OPUS writes the post-fix successor
   paragraph in design-doc section 7 (new line numbers, "done" state) AFTER commits 2-3 land; not an S7 gate, but a
   gate for S5 (the base-lock record cites section 7). Owner FYI only: no question.
5. S15 open-vs-ruled: Q1/Q4/Q5/Q6 are RULED in 2.4 and recorded in this plan; they are serving and sweep rulings,
   not section 9 entries (section 9 is the port's authority registry), so S15's wording is right. Section 6 item 3 is
   read from R2 on as "ruled in 2.4, recorded in the plan, no section 9 entry; a tool arm (Q1) still needs its own
   readiness package". No owner question.

**Commit order (MAIN, study branch).** S12 must precede S10: `core/deployment.py` imports
`undeclared_deployment_env_keys` and the verifier imports `deployment_environment` / `instance_deployment_env`.
1. S1b — "Relabel vLLM's character-level context refusal and the plain serving-layer token wording"
2. S12 — "Move the per-instance HCCL and NIC environment out of protocol_sha256 into the recorded deployment layer"
3. S10 — "Emit the gate's launch form, the HCCL mounts and a required host NIC from the deployment profile"
4. S13 — "Re-mint the L338 NPU-port registration package after the --max-num-seqs move; refuse the inline spelling"
5. S15 + this subsection + the ledger successor line — "Design doc: tool-template mechanism from vLLM source,
   deployment-env successor note, Q1/Q4-Q6 recorded; plan R2 review"
Host scripts stay outside git. After each commit: `git status` clean except the next stream's files.

**R2 gate: what S7 still needs before R0.**
- [ ] Full suite green on the tree (Sonnet run) and the two regression items classified pre-existing or fixed.
- [ ] Commits 1-5 landed; `git diff HEAD` empty; publish scrub (no proxy, no host address, no id) before any push.
- [ ] S10 sidecar fix applied and its test re-pointed (`SONNET`): the swap `mv npu_serve.sh.s10-next npu_serve.sh`
      happens only after the last `npu_serve.sh` launch of P3 has started (P1-P3 never re-launch); the test
      hard-coded to `.s10-next` is re-pointed to `npu_serve.sh`, not dropped, and the parametrised pair becomes one.
- [ ] P4 is SKIPPED if R2 has passed by the end of P3: R0 (`npu_deploy.sh up`) produces the resident shape itself and a
      `npu_serve.sh` service is not receipt-able. Prelim containers are stopped before `up` (it refuses a listening port).
- [ ] `npu-smi` on 181 reads every card below 4096 MiB; a refusal is STOP. Nothing on 175.
- [ ] S7's sweep plan file carries the 3.1/3.7 predictions and the P1-P3 outcomes (knee candidates) before `up`.
- [ ] S8 corrects the preliminary memory-utilisation value (S10 finding) before quoting Arm C's margin.

---

## 5. Risks

| risk | mitigation |
|---|---|
| Level 48 at TP2 preempts | REFUTED by the preliminary sweep (3.7); the risk moves to 64 at the ceiling: `waiting` must read 0, 10 s `npu-smi` samples, stopping rule |
| the worst case (`--min-tokens 4096`) fails at the one-pass knee | predicted in 3.7; the knee drops one level and the worst case repeats; run preliminarily so the recorded level list is known before R2 |
| a preliminary number is treated as recorded | every preliminary report and file name carries `prelim`; the hour bound quotes R-* reports only; the only decision a preliminary number may carry is "do not displace the default" (Q4) |
| the summary's "per instance" throughput column is a host aggregate | S8 reads the report field definition before quoting; the result document gets a successor edit (3.7) |
| `npu_deploy.sh up` fails HCCL init (error 19) | S10 lands before any card; R2 checks the emitted env/mount set equals `npu_serve.sh` |
| receipt refuses the live cmdline | S10 entrypoint form + dry-argv test; S7 writes the receipt before the sweep |
| identity check fails on instances 1-3 | S12; four-instance fixture in R2 |
| CephFS fills | JSON reports, free space check per arm, no traces |
| redeploy leaves containers/ports | `npu_deploy.sh down` is label-scoped; `status` before `up` |
| a card is claimed mid-sweep | launchers refuse busy cards; stop, do not share |
| short generations mask a slow level | `--min-tokens 4096` confirmation |
| S6/S13 re-mint hashes after S8 quotes them | S13 in wave 1; S8 quotes post-S13 hashes only |
| an agent "fixes" tools by editing `server_args` | S3/S15 record-only; `--chat-template` is a refused flag anyway |
| two agents hold one file | S10 waits for S6; S15 waits for S3; S1b waits for S1 |
| measured startup/prefill timings carried as profile facts | S8 quotes only numbers from the profile's own logs |

## 6. Evaluation checklist (Fable's review gates)

1. English only; `grep -rlP '[\x{4e00}-\x{9fff}]'` clean over touched files. 2. No dataset row, exposure entry,
readiness flip, gate flip or push. 3. No section-9 decision beyond 9.1-9.5 implemented; Q1, Q4, Q5, Q6 open. 4. Files
disjoint per stream; nothing committed by a stream. 5. Each fix has a test failing before and passing after; suite
green. 6. Llama path byte-identical (diff shown). 7. Every number traces to a file, line or report sha256. 8. The
acceptance test fails closed and never starts/stops a server. 9. Hashed/unhashed split respected both ways: nothing
deployment-level inside `protocol_sha256` (base ports, NIC, client concurrency), nothing protocol-level moved out
(the `DEPLOYMENT_ENV_KEYS` set is closed and tested). 10. Predictions written before the run and each marked
confirmed/refuted. 11. Card hygiene recorded. 12. CephFS checked; no artifact over 20 MB without a reason.

## 7. Follow-ups (not in this plan)

- Flight-side multi-endpoint transport: route a trajectory to an instance from the deployment profile, strip
  `id_slot`, flight workers from the profile. Needed before any flight uses the knee; not needed for the sweep.
- Memory successor note for `saea-serving-deployment-decision.md` (superseded KV estimate, 93 % CephFS), main thread.
- Q1/Q2 tool-bearing arm remedy, only if the owner answers Q1 yes.

## 3.12 Launch record (2026-10-30, planner session)

Owner rule 2026-10-30 (translated): "We must keep the NPUs fully loaded with experiments as much as possible." The eight
cards had been idle under the resident arm A deployment since the E chain finished on 2026-10-29. Launched detached on
`npu2` from this session after a read-only `--preflight` (all four arm A endpoints healthy, cards at 0 % AI core):

| chain | command (from `/mnt/cephfs/xingchao/omni/work/scripts`) | log | watchdog entries |
| --- | --- | --- | --- |
| P3 + P3b + P4 (arm B 2xTP4 levels, worst case, arm A relaunch) | `nohup setsid python3 prelim_chain.py --p3 --ts 20261030T041529Z` | `work/logs/prelim-chain-p3-20261030T041529Z/chain.log` | `prelim-chain-p3`, `sweep-armB-prelim-P3`, `sweep-armB-prelim-P3b-mintok4096` |
| E3 (shm baseline pass, shm passes, verdict), gated on the P3 chain being FINAL | `nohup setsid python3 e3_chain.py --ts 20261030T041647Z` | `work/logs/e3-chain-20261030T041647Z/chain.log` | `prelim-E3-chain` |

Next in the card queue once E3 is FINAL (the queue runner is being built, plan 2026-10-30 TFRL section 8 R1): the
budget-6,144 worst-case sweep at 32/24/16, then the recorded runs `record_run_list.example-armA-knee32.json` and
`record_run_list.budget6144-worstcase.json`. The base is chosen by the accuracy-first rule (section 8) from those results.

### 3.12 addendum: the card queue (2026-10-30)

A detached queue runner (`work/scripts/card_queue.py`, queue `card_queue.base-lock-v1.json`, state under
`work/card-queue/base-lock-v1/`, watchdog kind `card_queue`) now chains the card work after the E3 chain is FINAL:

1. `R-A3-b6144-prelim`: budget-6,144 worst case at 32, 24, 16 against the resident arm A that E3 restores (min_tokens =
   max_tokens 7,168, thinking budget 6,144, prompt max 17,408, latency rule 735 s); failing levels are data.
2. `stop-idle-arm-a`: stops the four arm A instances only when every one reports zero running and zero waiting.
3. `recorded-base-lock-v1`: `record_runs.py` on the real list `record_run_list.base-lock-v1.json`: R0 once (deploy from
   the profile, receipt, 11/11 base verification, smoke, manifest), then `R-A2-L32` (normal mix, 3 passes), `R-A3-L32`
   (worst case 4,096, admissibility), `R-A3-b6144` (32/24/16 at 7,168, `continue` so the whole descent is measured).
   The first two runs stop the chain on failure; a stop is a bad case for Opus, not a retry.

The base is then chosen by the section 8 rule of the TFRL plan (budget 6,144 if admissible at 16 or more per instance),
the lock record's open fields are filled from the recorded runs, and the record goes to the owner for countersign.
