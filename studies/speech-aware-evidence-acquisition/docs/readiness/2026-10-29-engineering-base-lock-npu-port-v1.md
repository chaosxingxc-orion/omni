# Engineering base lock, NPU port (S5, SKELETON v1, 2026-10-29)

Status: SKELETON. Written before R0 from committed code, the deployment profiles, the umbrella lock and the
preliminary evidence, so that R0 only fills cells. This is a port engineering record. It grants no contact
authority, no readiness flip and no exposure entry, and no gate reads it as permission. Plan:
`docs/superpowers/plans/2026-10-29-lock-engineering-base-and-efficiency-plan.md` sections 1.1-1.3, 3.9-3.11
(study HEAD `a3320041`, working tree with uncommitted S16b output).

Machine-readable sibling: `docs/readiness/2026-10-29-engineering-base-lock-npu-port-v1/base-lock-v1.json`
(schema `saea-engineering-base-lock-v1`; sha256 at writing `dcd5c74f8dbc01dc863b662632aaa84fd6d2f1d88a09f48573195df837e63e1b`;
the hash changes when R0 fills it).

## 0. How the JSON relates to the verifier

`scripts/serving/verify_engineering_base.py` (working tree, sha256 `d70af256...a5410f`, docstring lines 26-55) loads
`--base-lock` only if it is a JSON object with `schema == "saea-engineering-base-lock-v1"` (lines 1256-1265). After that,
every leaf a check observes is looked up in the lock and compared for equality (`kv_tokens`, `max_concurrency` within 1 %).
A missing leaf fails as "base-lock lacks", and a different value fails as "base-lock says X, observed Y" (lines 1176-1186).
Keys that no check claims (`_skeleton` here) are ignored. The keys of `python.interpreters`, `versions`,
`hosts.<host>.cann_files` and `cann.image_files` are read as the list of things to observe (lines 476-483). Notes
therefore never go inside those maps.

Placeholders: every cell that only R0 can supply holds a string beginning `TODO-R0:`. Because comparison is by
equality, each placeholder makes its check FAIL closed until R0 replaces it. No placeholder can pass by accident.

Loadability, tested offline on the dev host with the verifier's own `main()` and a reader that refuses every host
read: the lock is accepted (`inputs.base_lock_schema` set, no setup error) for both `qwen3-omni-thinking-bf16-181-4xtp2-v2`
and `-v3`. All 98 claim paths the ten non-weights checks compare for that profile are present (no "base-lock lacks").
`REQUIRED_DISTS` is covered. The checks fail only because the host readers were refused offline, and the receipt
check fails because `receipt.path` is a placeholder. The `weights` check (`--rehash-weights`) was not exercised.

Two arm A profiles are carried under `profiles`: `-v2` (ceiling 32, committed in `9cb4eacd`) and `-v3` (`-v2` plus
`mm_processor_cache_type: "shm"`, an UNTRACKED S16b file, adopted only if E3 passes, plan 3.11 ruling 3). The verifier
reads only the entry of the profile it is run with. R0 deletes the losing one. Branch (i) of plan 3.10 (arm B
resident) replaces both with a 2xtp4 entry. The committed `qwen3-omni-thinking-bf16-181-2xtp4-v2.json` still says
`max_num_seqs 64`, and its basis calls that a planner placeholder.

## 1. L-facts L1-L16

Classification: H = hashed (in `protocol_sha256`, the umbrella lock or the code manifest); R = recorded, not hashed.
Value state: FILLED (known now, source named), CANDIDATE (known but to be confirmed by R0), TODO-R0.

| # | fact | value (state) | read method | JSON leaf | H/R |
|---|---|---|---|---|---|
| L1 | image identity on 181 | ref `saea-base:20261027` (FILLED). Id `sha256:ebd6a3c104fc76f63b5843429bfa9a398428cfa4e1ca4de0ea9d36c15b19b5ef` (CANDIDATE: `/mnt/cephfs/xingchao/docker/saea-base/build-20261027T120229.log` line 354, file sha256 `24807d25...7cb626`; that this build ran on 181 is inferred from the log clock, since 181 logs in UTC and the other two build logs report the legacy builder id `477228c6fb8b`). `RepoDigests` `[]` (FILLED, plan L1). `Created` TODO-R0. Only 181 is carried: host `npu` is not ours (owner rule 2026-10-29) | `docker image inspect` on npu2, via `npu_verify_base.sh` facts | `image.ref/id/repo_digests/created` | R |
| L2 | base image | `quay.io/ascend/vllm-ascend:v0.30.0rc2`, local id `4d683043e4bd` (FILLED, plan L2) | `docker image inspect` of the base ref | `image.base.ref/local_id` | R |
| L3 | interpreter | version `3.12.13`; both `/opt/saea-venv/bin/python` and `/usr/local/python3.12.13/bin/python3` resolve to `/usr/local/python3.12.13/bin/python3.12` (FILLED, plan L3). Executable sha256 TODO-R0 | `os.path.realpath` + sha256 in the verifier container | `python.version`, `python.interpreters.*` | H (receipt) |
| L4 | serving stack | `vllm 0.30.0+empty`, `vllm-ascend 0.30.0rc2`, `torch 2.10.0+cpu`, `torch_npu 2.10.0.post7`, `transformers 5.14.1`, `triton 3.6.0` (FILLED, system interpreter, plan L4). Harness venv: `transformers 5.21.0`, no vllm/torch. `triton-ascend 3.6.0+dev20260923230212` is not in the verifier's dist list. See decision D1: as launched today the verifier reads the VENV and will observe different values | `importlib.metadata` of the interpreter running the verifier | `versions.*` | H (receipt package identity) |
| L5 | CANN / driver / firmware | host driver `Version=26.0.rc1`, image CANN `9.2.0` (plan L5, recorded as text, not compared). Compared files: the verifier defaults, three host files and one image file, sha256/bytes TODO-R0. **Firmware version: NOT READABLE.** The firmware directory is unreadable for our uid and `npu-smi` reports `NA`. No check reads it or pretends to | sha256 of the listed files (host files via `--host-root /host`) | `hosts.A800I-03.cann_files.*`, `cann.image_files.*` | R |
| L6 | host identity | hostname `A800I-03`, 8 cards, 256 cores (FILLED, profile `host`, `host_card_count`, `host_cpu_count`). NUMA map 0-1 node 6, 2-3 node 4, 4-5 node 0, 6-7 node 2 = the profile cpusets | `hostname`, `ls /dev/davinci[0-9]*`, `nproc --all` | `hosts.A800I-03.card_count/cpu_count` | R |
| L7 | core weights | lock entry `qwen3-omni-30b-a3b-thinking-bf16` in `docs/datasets.lock.json` (umbrella; file sha256 `61f3d111...b2ee20df`), revision `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b`, 26 files, 63,447,489,451 B, per-file sha256 copied from `source.files` (FILLED). `other_entries` `[".hfd"]` (FILLED from a directory listing; `.hfd/` is not governed). Live re-hash and the receipt's `model.files` TODO-R0 | `--rehash-weights` (sha256 of every file); receipt `model.files` | `weights.*` | H (lock) |
| L8 | encoder weights | `omni-embed-nemotron-3b-causal-v1` (14 files, 9,423,118,096 B), `lco-embedding-omni-3b-2605-v1` (14, 9,423,121,465 B), `lco-embedding-omni-7b-v1` (16, 17,879,801,172 B), per-file sha256 from the same lock (FILLED). Directory listing shows no dot entries. See decision D6 on the hashing cost | `--rehash-weights` | `encoders.*` | H (lock) |
| L9 | protocol serving flags | fragment `configs/model/bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-port-v1.json` (file sha256 `bceb079d...fc5`; `core.config.fragment_hash` = `bcb98207bd929e0c21b3f6f3dd2835ff07e6e24d11bad9882520766c9b752447`). Protocol tokens `--max-model-len 24576 --dtype bfloat16 --reasoning-parser qwen3 --reasoning-config {...} --enable-auto-tool-choice --tool-call-parser hermes --limit-mm-per-prompt {"audio": 1, "image": 0, "video": 0} --seed 0`, served name `qwen3-omni-30b-a3b-thinking` (FILLED, derived offline with `gate_runtime_vllm.classify_server_flags`). `protocol_sha256` TODO-R0. See decision D2 on the brief's "a35b2740" | `/proc/<pid>/cmdline` protocol subset; `protocol_sha256(receipt)` | `protocol.server_args`, `protocol.served_model_name`, `protocol.protocol_sha256` | H |
| L10 | decoding and reasoning | temperature 0, seed 20260913, max_tokens 4096, reasoning budget 3072, answer reserve 1024, forced-close sha256 `aab2abbe2767036ab7019b3bc28d76e9ce288359f56c92774fe029069e166adf`, `drop-reasoning-v1`, timeout 420 s, thinking on (FILLED, from the fragment) | fragment read | `protocol.decoding` | H |
| L11 | deployment layer | per instance cards / cpuset / port / TP / ceiling / 0.92 / batched-tokens null / mm cache type (null in -v2, `shm` in -v3) and `server_argv_sha256` (FILLED, derived offline from profile + fragment with `core.deployment.instance_server_argv`). Profile sha256 (canonical JSON): -v2 `bd0c356005830ad52cbcc3bcd822018616650698b7c64d92c38ff4f2a00c1ca6`, -v3 `5d18e04d8f99bd5a1ebc36c8aac072f0f7693990c92173fc8522a63b0674dfc0` (the -v3 value is valid only while the untracked file is unchanged). `HCCL_IF_BASE_PORT` 61000/61200/61400/61600 | `docker inspect` (devices, cpuset, argv), profile load | `profiles.<id>.profile_sha256`, `profiles.<id>.instances.*` | R |
| L12 | HCCL env and mounts | `DEPLOYMENT_ENV_KEYS` = {`HCCL_IF_BASE_PORT`, `HCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`} (`contracts/gate_runtime_vllm.py`, in HEAD), excluded from `protocol.env`. NIC `enp61s0f2` (FILLED, plan L12 and the preliminary launch capture `arm-a-launch.json` sha256 `a481373e...08f3a`). Network mode `host`. Read-only mounts `/etc/ascend_install.info`, `/etc/hccn.conf`, `/usr/local/Ascend/driver`, `/usr/local/Ascend/firmware`, `/usr/local/dcmi`, `/usr/local/sbin/npu-smi` (FILLED, `core.deployment.ASCEND_READONLY_MOUNTS`) | `ip -o -4 addr show scope global`; container env and mounts | `hosts.A800I-03.global_nic`, `network_env.*` | R |
| L13 | engine-derived facts | all TODO-R0 per instance: KV tokens, max concurrency, block size (from `/metrics`, never asserted), prefix caching, chunked prefill, cudagraph mode, capture sizes. Preliminary reference only (ceiling 64, `npu_serve.sh`): 25.83 GiB/card, 564,224 tokens, 22.96x, FULL_AND_PIECEWISE (`docs/results/2026-10-29-npu-port-preliminary-arm-a-sweep-v1.md`). The capture sizes change at ceiling 32, so KV is re-read. `execute_model_timeout_seconds` null (FILLED: not set, owner ruling Q5, plan 3.10 item 2) | startup log lines + `/metrics` `cache_config_info` | `profiles.<id>.instances.<name>.engine.*` | R |
| L14 | tokenizer / template | `chat_template.json` sha256 `3afb9519448b62e252cefeed1c0c1939b01b4e8ed70adbaf3c9f952969aa0ec7` (FILLED, lock `source.files`). `tokenizer_config.json` has no `chat_template` key (FILLED, read on CephFS). One audio span per prompt `[1, 1]` in both modes (FILLED: the check requires it). Decoded 48-character tails TODO-R0. `/tokenize` count == engine `prompt_tokens` is a pass/fail check, not a leaf | `/tokenize` + one 1-token chat per instance | `template.*` | R (file hash part of L7) |
| L15 | execution authority | `saea-container` (FILLED, `contracts/plan.py` `EXECUTION_ENVIRONMENT`; markers `/.dockerenv`, `/opt/saea-venv/bin/python`, `/opt/umbrella`) | `assert_execution_environment()` | `execution.environment` | H (code manifest) |
| L16 | launch form | entrypoint `/usr/local/python3.12.13/bin/python3`, then `-m vllm.entrypoints.cli.main serve <model_dir> ...` (FILLED, `core.deployment.CONTAINER_INTERPRETER` + `VLLM_ENTRYPOINT`). The live argv is TODO-R0 (receipt `protocol.argv0/entrypoint`; the verifier's `live_vllm_args` refuses any other form). The preliminary runs used `--entrypoint vllm` (launch capture), a form the gate refuses | `docker inspect` Path/Args; `/proc/<pid>/cmdline` | covered through `server_argv_sha256` and the receipt check | H |

Additional recorded facts (not verifier leaves):

- **Greedy self-agreement (recorded, not gated; plan 3.9 decision 3, 3.10, 3.11 ruling 4).** D3 on idle `saea-arm-a-0`
  (`/mnt/cephfs/xingchao/omni/work/logs/prelim-D3-greedy-20261029T130527Z.json`, sha256 `ca0af18c...a4836`): 16 fixed
  requests, temperature 0, max_tokens 64; sequential S1-S2 9/16, S1-S3 9/16, S2-S3 16/16; burst B1-B2 8/16, B1-B3 15/16,
  B2-B3 8/16; sequential vs burst 6-7/16; first divergence at decode positions 13-58, never 0. The tool's own
  `classification` string reads "kernel-or-cache nondeterminism: sequential repeats disagree". The planner's reading
  (plan 3.10) is "cache-state and batch-composition nondeterminism; warm sequential greedy repeatable (16/16, n = 1
  pair)". This record carries both, verbatim. The S7 sequential probes before and after each recorded level fill TODO-R0
  agreement rates.
- **Winning deployment pair (plan 1.3, 3.5).** TODO, depends on the P3/P3b branch. In branches (ii)-(iv): arm A,
  `-v2` or `-v3` (decided by E3), client concurrency 32 per instance (host 128). In branch (i): arm B at its admitted
  level, provisional until R-B2 confirms. The speed numbers live in `docs/results/2026-10-29-npu-port-synthetic-speed-baseline-v1.md`.
- **Restart policy.** `core.deployment.docker_run_argv` emits `--restart unless-stopped` (HEAD). Plan 3.8 and 3.10 say
  `--restart` is refused for recorded runs. The verifier's `endpoints` check fails any `RestartCount != 0`, so a
  restart is detected but not prevented. See decision D4.

## 2. TODO-R0 cells (the full list; the JSON holds 72 placeholder strings)

1. `image.created`; confirm `image.id` (CANDIDATE).
2. `python.interpreters.*.sha256` (2).
3. `hosts.A800I-03.cann_files.*.{sha256,bytes}` (3 files x 2) and `cann.image_files.*.{sha256,bytes}` (1 x 2).
4. `protocol.protocol_sha256`.
5. `profiles.<id>.instances.<4 names>.engine.{kv_tokens,max_concurrency,block_size,prefix_caching,chunked_prefill,cudagraph_mode,cudagraph_capture_sizes}` (7 x 4 per profile, two profiles carried).
6. `template.tails.thinking_true`, `template.tails.thinking_false`.
7. `receipt.path`.
8. Confirmations, not placeholders: the live weights re-hash equals L7/L8; the receipt `model.files` equals L7; the live
   argv equals the derived `server_argv_sha256`.
9. Recorded-only: winning pair, greedy probe rates, deletion of the losing profile entry.

The fastest fill is `npu_verify_base.sh <profile> - --print-observed --receipt <R0 receipt> --rehash-weights`, then a
leaf-by-leaf diff against this skeleton. Every FILLED leaf that differs is a finding to report, not a value to overwrite.

## 3. Decisions the planner must take (the verifier schema as it stands)

- **D1 (blocking for check 2).** `npu_verify_base.sh` runs the verifier with `--entrypoint /opt/saea-venv/bin/python`.
  `versions` is read with `importlib.metadata` of that interpreter, which has `transformers 5.21.0` and no vllm, vllm-ascend,
  torch or torch_npu (plan L3/L4). Check 2 therefore fails "required package vllm is not installed" against ANY lock. The
  schema has one flat `versions` map, while plan L4 asks for versions "per interpreter". Options: (a) the verifier reads
  versions of each locked interpreter in a subprocess (schema `versions.<interpreter>.<dist>`); (b) run the verifier under the
  system interpreter (check that `assert_execution_environment` still holds); (c) pin the serving stack only through the
  receipt's package identity and lock the venv versions here. The skeleton holds the SYSTEM (serving) values.
- **D2.** The brief names "the protocol fragment hash a35b2740", from the docstring of `tests/unit/test_l338_port_package_staleness.py`.
  It does not reproduce. `core.config.fragment_hash` of the fragment at `db798a83`, `d7c8f17a` and HEAD gives `bcb98207...`,
  the pre-`db798a83` fragment gives `14308a22...`, and no plain canonical-JSON variant gives `a35b2740` or `df95b65b`. The
  record carries the reproducible `bcb98207` and the file sha256. Which hash the docstring meant is open, and that docstring
  may need correcting.
- **D3.** Host CANN file paths. The verifier default is `/usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux/ascend_toolkit_install.info`;
  plan L5 names the host toolkit dir `.../ascend-toolkit/8.2`. The image default is `.../latest/aarch64-linux/ascend_toolkit_install.info`;
  plan L5 names `cann-9.2.0/compiler/version.info`. An unreadable listed file fails check 3, so the paths must be read once on
  181 before R0 is gated on them.
- **D4.** `--restart unless-stopped` in `docker_run_argv` versus "`--restart` refused" in plan 3.8 and 3.10.
- **D5.** Image id per host: the schema has one `image.id`. Plan L1 wanted both hosts, but host `npu` is no longer ours,
  so one id is enough. Recommend recording that L1 is single-host.
- **D6.** Carrying `encoders` makes `--rehash-weights` also hash ~36.7 GB of encoder views, which no serving instance uses.
  Keep (L8 locked in one pass) or drop the block and lock L8 elsewhere.
- **D7.** Where the JSON lives: placed at `docs/readiness/2026-10-29-engineering-base-lock-npu-port-v1/base-lock-v1.json`
  (a stem directory beside the `.md`); plan 1.3 writes `.../base-lock-v1.json` without naming the directory.

## 4. Sources

Offline derivations ran on the dev host with the study working tree's own modules (`core.deployment`,
`contracts.gate_runtime_vllm`, `core.config.fragment_hash`). Python 3.11 loaded them by bypassing the `core` package
`__init__`, without editing any file. R0 re-derives every value in `saea-base`. Files: plan (study), deployment profiles
`configs/deployment/qwen3-omni-thinking-bf16-181-4xtp2-v{2,3}.json` (sha256 `8d65fba5...a384c`, `99b112eb...ba`),
umbrella `docs/datasets.lock.json` (`61f3d111...`), the image build log above, the launch capture above, the D3 JSON above.
