# CLAUDE.md

Guidance for Claude Code when working in this workspace (`/home/xingchao/workspace/omni`) and in the
engineering trees it drives on the shared CephFS (`/mnt/cephfs/xingchao/omni/`).

## First principles

These override every other convention here and in the trees below.

1. **English only, everywhere, always.** Every request sent to a model API, every prompt, every
   internally processed document, every memory file, every commit message, every code comment,
   every design document, plan, check report and note is written in English. If the user writes
   in another language (for example Chinese), translate the message into English FIRST, then work
   from the English version; the original wording is never passed to a model or copied into a
   document. Any non-English text found in an engineering document is a defect: translate it to
   English when the file is touched. Hash-pinned bytes and sealed records are never rewritten in
   place; add an English successor note instead and point at the sealed source. Verify with
   `grep -rlP '[\x{4e00}-\x{9fff}]'` before committing.
2. **Memory is English.** Files under the Claude memory directory are written in English only,
   including quoted user instructions (quote the English translation, never the original).
3. **Do every non-accelerator step first and harden the engineering base.** Nothing waits on a busy
   card if it can be done on CPU, offline, or model-free.
4. **Concurrency first.** Split independent work into parallel streams with disjoint files; shard
   compute across processes and cores (the Ascend hosts have 256 cores and 2 TB RAM). Only the
   main thread commits. Downloads are the one explicit exception: they run sequentially.
5. **Multi-agent collaboration by default (owner rule 2026-10-29).** Day-to-day tasks are carried
   out by a team of agents, each role bound to a model. Spawn subagents with the `Agent` tool and
   set `model` explicitly; never let a role run on the wrong model.

   | Role | Model (`model` argument) | Work it owns |
   | --- | --- | --- |
   | Planner / reviewer | Fable 5.1 (the main session; `claude-fable-5-1`) | Task planning and decomposition, evaluation of delivered work, feedback, guidance for optimisation, final judgement, commits. |
   | Engineer | Sonnet 5.5 (`sonnet`) | Engineering environment preparation, development, testing, integration, verification; scripts, builds, container runs, test suites. |
   | Analyst / writer | Opus 5.5 (`opus`) | Reading documents and logs, writing reports and notes, optimisation carried out under the planner's guidance. |

   Clarification (owner, 2026-10-29): schemes, plans, evaluations and feedback are ALWAYS produced by
   Fable 5.1; when the current session is not running on Fable, spawn a Fable subagent
   (`model: "fable"`) for them instead of doing them in the session. Opus 5.5 executes the content
   side of a plan: task-understanding briefs, reading, and every written document (base-lock
   records, result and baseline reports, notes). Sonnet 5.5 executes the engineering side only:
   code, scripts, builds, tests, container and card runs. A session that is not Fable only
   coordinates: it dispatches by these roles and relays results.

   Log reading and token economy (owner rule 2026-10-29): routine progress of sub-tasks is
   watched by watchdog scripts, not by models. Watchdogs run detached on the hosts, analyse logs
   on a timer, and write a compact status file plus a change-only event log under
   `/mnt/cephfs/xingchao/omni/work/watchdog/`; the session reads only those. Fable never reads raw
   logs. A bad case (crash, threshold failure, regression, stall) goes to Opus, which reads the
   raw evidence and writes a short bad-case note; Fable then analyses the note and decides.

   Working loop: Fable plans and writes the brief; Sonnet and Opus agents run in parallel on disjoint
   files; Fable evaluates the results, gives feedback, and re-dispatches. Briefs and reports between
   agents are English (principle 1). A one-line lookup or a single tool call stays in the main
   thread; everything larger is delegated.

## What this workspace is

Reproduction of the SAEA (speech-aware evidence acquisition) Thinking Fold B baseline
`SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-V1` with Qwen3-Omni-30B-A3B-Thinking (BF16) served by
vLLM on Ascend NPUs, with `Omni-Embed-Nemotron-3B` as the retrieval core and the LCO-Embedding-Omni
3B/7B encoders as the ladder arms. Sealed vectors are regenerated here; nothing historical is
re-used as evidence.

| path | what |
| --- | --- |
| `exploring-l4-intelligence/` | Read-only clone of the upstream umbrella repo (`master`). Its `wiki/audit/` and `wiki/archive/` are immutable records; never rewrite them. |
| `dl_scripts/`, `dl_logs/` | Download drivers and their logs (sequential queue, resumable, detached by default). |
| `repro/` | Local reproduction notes and tools. |
| `/mnt/cephfs/xingchao/omni/` | Data root (`SPEECHRL_DATA_DIR` / `OMNI_ROOT`): `models/`, `datasets/`, `prepared/`, `vectors/`, `knowledge/`, `coordination/`, `attempts/`, `work/`, `publish/`, `logs/`, `scratch/`. Everything we generate on CephFS goes under this directory, never directly under `/mnt/cephfs/xingchao/`. |
| `/mnt/cephfs/xingchao/omni/work/dev/umbrella/studies/speech-aware-evidence-acquisition` | The study dev tree, branch `npu-vllm-port`. This is where engineering changes are made and committed. |
| `/mnt/cephfs/xingchao/omni/work/umbrella` | Pristine base clone; `work/dev/wt-umb` is the base worktree for comparisons. |
| `/mnt/cephfs/xingchao/omni/work/scripts/` | Host launch scripts (container runners, encoder chains, serving, card watcher). |
| `/mnt/cephfs/xingchao/omni/publish/omni` | Local clone of the PUBLIC GitHub repo `chaosxingxc-orion/omni` (code and scripts only). |
| `/mnt/cephfs/xingchao/docker/saea-base/` | Build context of the `saea-base` image. Stays where it is. |

Not ours, never touch: `/mnt/cephfs/xingchao/docker/rl_ascend_20261026`, `docker/README_RL.txt`,
`/mnt/cephfs/xingchao/experiments/`, and any container we did not start.

## Hosts and execution authority

- Dev host (this machine): no sudo, outbound network only through the proxy exported in
  `~/.bashrc` (never print it). The GitHub git pack protocol hangs here; finish clones over HTTPS
  with object-id verification.
- `npu` (host 175, 910B3) and `npu2` (host 181, 910B4-1), reached only by their ssh aliases: 8 x 64 GB cards each,
  256 cores, 2 TB RAM, same CephFS mount, NO internet. Build images from a staged offline context.
  All NPU experiments run on 181; never use a busy card; wait for the owner's job to finish.
- Formal execution environment is the `saea-base` Docker container (decision 9.1): run study
  commands through `work/scripts/port_py.sh` / `port_cli.sh` over `ssh npu`. Do not pass `-I` to
  Python there (it drops `PYTHONPATH`). Encoders run on torch_npu or CPU fp32, not vLLM.
- One resident vLLM deployment, declared by a profile under `configs/deployment/`; the deployment
  layer (instances, cards, `--max-num-seqs`) is recorded in the run manifest but is NOT part of
  the protocol hash.

## Downloads and experiments run detached

Every experiment, server, sweep, encoder run and long test runs in the background on the host so a
network jitter or an ssh drop never interrupts it (owner rule 2026-10-29): servers as `docker run -d`
(daemon-managed, logs to CephFS), drivers under `nohup setsid` with no controlling terminal, results
checkpointed per step with a resume option, and monitoring that only reads logs and ports. Never hold
an experiment in a foreground ssh session.

Downloads: model first, then datasets, one at a time; every transfer resumable and verified; paths follow the
repo's own lock and fetch scripts; official `huggingface.co` (not a mirror); every driver detaches
itself (`dl_scripts/_daemonize.sh`) so an SSH drop never kills it.

## Git and publishing

- Commit each change in the repo that owns it. The study port branch `npu-vllm-port` is the
  authority for engineering; the umbrella clone is not modified.
- The GitHub repo `chaosxingxc-orion/omni` is PUBLIC. Before any push: no proxy credentials, no
  internal host addresses (use the `npu`/`npu2` ssh aliases), no employee ids, no proxy
  hostnames. Push only when the user asks, over SSH via `~/.ssh/github_proxy.conf`.
- Sync recipe: `git format-patch` from the study branch, `git am --directory=studies/...` in the
  publish clone, compare trees before pushing.

## Research boundary

Never infer execution authority from this guide. A documentation or check repair does not
authorise model calls, flights, readiness flips or exposure entries. Owner decisions are recorded
in the study's decision registry and design document (section 9 of
`docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md`), in English.
