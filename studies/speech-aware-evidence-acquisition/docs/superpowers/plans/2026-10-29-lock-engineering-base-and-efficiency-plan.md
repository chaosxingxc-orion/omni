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
| 2 (cards, 181) | S7 | R2 passed | all arms done or stopped by rule; verification PASS on arm A |
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

---

## 5. Risks

| risk | mitigation |
|---|---|
| Level 48 at TP2 preempts | predicted in 3.1; stopping rule; utilisation stays 0.92 until arm C |
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
