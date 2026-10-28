#!/usr/bin/env bash
# Sequential, resumable download queue: one asset at a time, never in parallel.
# Runs detached (nohup + setsid) by default -- just `bash dl_scripts/run_queue.sh`; it prints the
# pid and log path and returns. Re-run at any time (after a crash, a reboot or a dropped proxy):
# finished steps are skipped by their completion checks and the current step resumes from its
# .part files / git objects. A second copy refuses to start while one is running (flock).
#
# SCOPE (owner directives 2026-10-27), two rules, both narrowing:
#   1. BF16 / vLLM ONLY. Every serving core is the BF16 safetensors release vllm-ascend can
#      load, never a GGUF. The llama.cpp route the study's own receipts pin was dropped with it.
#   2. ONLY what the FINAL experiment baseline actually touches. The baseline moved to
#      Qwen3-Omni-30B-A3B-**Thinking** (exposure-ledger 2026-10-24,
#      SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-V1: "Frozen Qwen3-Omni-30B-A3B-Thinking
#      Q4_K_M ... module program phase-native-module-v1-c15 ... no tool authority").
#
# What rule 2 removed, with the evidence, so nothing is dropped on a hunch:
#   lco-embedding-omni-3b / GLAP / CLSP -- the LOSING arms of the same bake-off nemotron won.
#     P-EMB-4R (2026-08-21) scored the LCO audio arms at 0.1500 / 0.1167 K=5 accuracy, and the
#     2026-09-06 MF-7b lift table ranks them last: +1.084 nemotron-fused, +0.718 nemotron,
#     +0.440 glap, +0.183 lco. No sealed vector file in any surviving cell names them. 10 GB
#     saved. (CORRECTION 2026-10-27: an earlier revision of this file dropped
#     omni-embed-nemotron-3b alongside them on the same sentence. That was wrong -- see step 1.)
#   qwen3-asr-0.6b -- one ledger row, superseded by the 1.7B in the 2026-10-19/24 ASR columns.
#   whisper-large-v3 -- last touched 2026-09-29, a historical transcript comparator, not in the
#     baseline. Cheap to add back (one `hf` line) if a comparison needs it.
#   diar-sortformer / spkrec-ecapa / MMS_FA -- tool models the lock itself marks as needing a
#     tool-smoke ledger row before a first runtime drive; spkrec has ZERO ledger rows, i.e. it
#     has never been driven at all.
#   audio2tool -- last touched 2026-09-21; 71,441 small files through a throttling proxy, and
#     no part of the Thinking baseline.
#
# Layout and revisions otherwise follow exploring-l4-intelligence/docs/datasets.lock.json with
# SPEECHRL_DATA_DIR=/mnt/cephfs/xingchao; every hf step is file-selective via HF_COMPLETE_INCLUDE.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(dirname "$HERE")/dl_logs
export DL_LOG=${DL_LOG:-$LOGS/queue.log}
source "$HERE/_daemonize.sh"

export SPEECHRL_DATA_DIR=${SPEECHRL_DATA_DIR:-/mnt/cephfs/xingchao}
# Official huggingface.co, not hf-mirror (owner 2026-10-27). Measured through the corporate proxy the same
# evening: 14.5-15.1 MB/s per connection, 108 MB/s over 8 ranged connections.
export SPEECHRL_HF_ENDPOINT=${SPEECHRL_HF_ENDPOINT:-https://huggingface.co}
DR=$SPEECHRL_DATA_DIR
mkdir -p "$LOGS"

exec 9> "$LOGS/.queue.lock"
flock -n 9 || { echo "$(date '+%F %T') queue already running"; exit 0; }
echo "$(date '+%F %T') queue started (pid $$)"

step() { # name done-test command...
  local name=$1 done_test=$2; shift 2
  if eval "$done_test"; then echo "$(date '+%F %T') done    : $name"; return 0; fi
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') start   : $name (attempt $attempt, log $LOGS/$name.log)"
    "$@" >> "$LOGS/$name.log" 2>&1
    if eval "$done_test"; then echo "$(date '+%F %T') finished: $name"; return 0; fi
    sleep 60
  done
  echo "$(date '+%F %T') FAILED  : $name (re-run the queue to resume)"; return 1
}

hf() { # name repo dest revision model|dataset conn include...  (no include = whole repo)
  local name=$1 repo=$2 dest=$3 rev=$4 type=$5 conn=$6; shift 6
  local include; include=$(IFS=,; echo "$*")
  step "$name" "[ -f $dest/.hfd/speechrl-complete-revision ]" \
    env HF_COMPLETE_INCLUDE="$include" bash "$HERE/fetch_hf.sh" "$repo" "$dest" "$rev" "$type" "$conn"
}

# --- already complete (checks short-circuit) -------------------------------------------------
# The Instruct BF16 core, the carrier SAEA-E-001 measured against, and SLURP, the carrier the
# CURRENT Thinking baseline measures against.
step qwen3omni "[ -f $DR/models/qwen3-omni-30b-a3b-instruct/.hfd/speechrl-complete-revision ]" \
  bash "$HERE/fetch_hf.sh" Qwen/Qwen3-Omni-30B-A3B-Instruct "$DR/models/qwen3-omni-30b-a3b-instruct" \
  26291f793822fb6be9555850f06dfe95f2d7e695 model 16 || exit 1
step slurp "grep -q SLURP_DONE $LOGS/slurp.log 2>/dev/null" bash "$HERE/fetch_slurp.sh" || exit 1

# --- 0. the superseded text embedder, kept only because it is already in flight ---------------
# Qwen3-Embedding-0.6B embedded direct-decode response TEXT for the DEV-1000 BLEND lock through a
# CPU llama.cpp `--embedding --pooling last` featcache sidecar. The owner's 2026-09-06 rule --
# "a QUERY vector must contain the speech signal and may never be text-only" -- retired that
# route; it has ZERO occurrences in the Fold B baseline chain (verified against both the baseline
# and the experiment fragment). 1.2 GB and already part-fetched, so it finishes rather than being
# abandoned: it is the comparator if an earlier DEV-1000 cell ever has to be re-read.
hf qwen3-embedding-0.6b Qwen/Qwen3-Embedding-0.6B \
  "$DR/models/qwen3-embedding-0.6b" 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 model 8 || exit 1

# --- 1. THE OMNI EMBEDDING: both sides of the baseline's retrieval ----------------------------
# This is the repo's multimodal-knowledge core, and it is REQUIRED, not optional. The final
# Thinking Fold B baseline's own experiment fragment lists as preparation requirement 3 "this
# cell's OWN prepared first-phase Perception supply ... from the sealed TRAIN Fold B speech-query
# segment vectors", and its baseline fragment names the encoder on BOTH sides of the retrieval:
#   QUERY: "speech: complete original TRAIN Fold B recordings represented by sealed full-utterance
#     segments in the intent-boundary retrieval dimension" -- ledger row
#     SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-V1 (2026-09-12), "Frozen local
#     Omni-Embed-Nemotron-3B retrieval tool; no answering core", 1,080 encoder contacts.
#   KEY:   "text: Omni-Embed-Nemotron-3B over the 11,514 governed SLURP TRAIN utterance texts,
#     prompt 'passage: ', mean-pooled and L2-normalised" -- row SAEA-SLURP-MF7B-NEMOTRON-
#     EMBEDDING-V1, file vectors/slurp-kus3/facet-vectors.jsonl.
# Those sealed .jsonl vector files are NOT shipped with the repo -- only their sha256 pins are --
# and nothing under $SPEECHRL_DATA_DIR/vectors or /prepared exists yet, so they must be
# REGENERATED here, which means the encoder itself is on the critical path. It is loaded
# in-process by core/speech_preparation_runtime.py (NemotronSpeechEncoder / NemotronPassageEncoder
# via the repo's own modeling_nv_omni_embed.py), i.e. no serving engine and no NPU is involved --
# so this 8.8 GB unlocks preparation work we can do on CPU TODAY, which is why it is queued ahead
# of the 63 GB Thinking core that is useless until the NPUs are released.
# ModelScope, not HF: docs/datasets.lock.json registers it as source.kind "modelscope",
# id nv-community/omni-embed-nemotron-3b, revision "master (unpinned -- content-fingerprinted)",
# 22 files / 8.8 GB. fetch_modelscope.sh verifies every file against the listing API's own sha256.
step omni-embed-nemotron-3b \
  "[ -f $DR/models/omni-embed-nemotron-3b/.hfd/speechrl-complete-revision ]" \
  bash "$HERE/fetch_modelscope.sh" nv-community/omni-embed-nemotron-3b \
  "$DR/models/omni-embed-nemotron-3b" master 8 || exit 1

# --- 2. THE LONG POLE: the current baseline's frozen core, BF16 -------------------------------
# 63.45 GB at this proxy's rate, so it starts as early as the small assets allow and everything
# cheap is queued behind it rather than in front of it.
step qwen3omni-thinking "[ -f $DR/models/qwen3-omni-30b-a3b-thinking/.hfd/speechrl-complete-revision ]" \
  bash "$HERE/fetch_hf.sh" Qwen/Qwen3-Omni-30B-A3B-Thinking "$DR/models/qwen3-omni-30b-a3b-thinking" \
  2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b model 16 || exit 1

# --- 3. THE EXPERIMENT ARMS: the size ladder the laptop could not hold ------------------------
# Queued AFTER the Thinking core on purpose: nemotron + Thinking are the CONTROL path, and the
# Thinking core is needed the moment the NPUs are released (2026-10-28), while these arms
# cannot fly until the control run is validated. 27.3 GB, disk has 3.1 TB free.
# Owner decision 2026-10-27: fly a 3-arm encoder ladder instead of nemotron alone. These two are
# the EXPERIMENT group; nemotron (step 1) stays the CONTROL, because it is the encoder the sealed
# vectors' identity is anchored to.
#
# Why these are a fair re-test and not a fishing trip. The study's own selection ticket
# (docs/readiness/2026-08-20-omni-embedding-selection.md) ranked LCO FIRST and nemotron FOURTH:
# MAEB Table 2 zero-shot / cross-modal-retrieval -- LCO-7B 64.5 / 50.3, LCO-3B 62.2 / 50.3, while
# nemotron's published audio numbers are video-soundtrack retrieval, "not speech-semantic". LCO
# then LOST every flown comparison -- but every one of those losses is recorded as
# POOLING-OVERRIDE SCOPED, because LCO was served through `llama-server --embedding --pooling
# last` on a build whose --pooling flag provably never reaches the audio side (P-EMB-3M gate (a);
# the defect reproduced on two unrelated models). In the one head-to-head (MF-7b, 2026-09-06) LCO
# ran as a Q8_0 GGUF on that build while nemotron ran bf16 through plain transformers with correct
# mean pooling and its own "query: "/"passage: " prompts. The P-EMB-3M verdict names the remedy
# itself: "A model-level reading of M1 requires ... LCO driven through a non-llama.cpp stack."
# Dropping llama.cpp (owner directive 1) IS that stack change, so this is the first fair test.
# Corroboration that the model, not the space, was fine: in P-EMB-4R the text-mediated T-LCO7 arm
# reached Recall@5 0.8667, second only to T-QW.
#
# There is no larger OMNI nemotron to ladder with instead (checked 2026-10-27):
# nvidia/omni-embed-nemotron-3b is the line's only member. nvidia/Nemotron-3.5-Embed-8B-BF16 is
# Mistral3Model vision+text (pipeline visual-document-retrieval) and Nemotron-3-Embed-8B-BF16 is
# Ministral3Model text-only -- both carry no audio tower, so both fail the owner's 2026-09-06 rule
# that a QUERY vector must contain the speech signal. LCO is the only audio-capable scale-up.
#
# Both are apache-2.0 and ungated, which also drops a licence layer: nemotron carries NVIDIA
# OneWay Noncommercial + Qwen Research on top of SLURP's CC BY-NC 4.0.
#
# Revisions are the exact ones the selection ticket pinned, re-verified live against the mirror.
# 3B-2605 first: its 1_Pooling/config.json reports embedding_dimension 2048 -- IDENTICAL to
# nemotron -- so it needs no change to DIMS (core/speech_embedding.py:14) or EMBEDDING_DIM
# (rails/embedding_client.py:109) and is the clean same-dimension contrast. 7B is 3584 and needs
# those constants opened up.
# Both report pooling_mode "lasttoken", which core/speech_embedding.py:81 currently rejects
# outright (it hard-requires "mean") -- that is encoder-adapter work, tracked separately; the
# bytes are fetched now because download is the long-lead item.
hf lco-embedding-omni-3b-2605 LCO-Embedding/LCO-Embedding-Omni-3B-2605 \
  "$DR/models/lco-embedding-omni-3b-2605" 5f6b5329da5141367da30e06a9826d1322d6c9b2 model 8 || exit 1
hf lco-embedding-omni-7b LCO-Embedding/LCO-Embedding-Omni-7B \
  "$DR/models/lco-embedding-omni-7b" 108f6f1a5de3b2eedd1d1c7b7005aaca6ed3802c model 8 || exit 1

# --- 4. the ASR tool contact still in the current working set ---------------------------------
# SAEA-SLURP-ASRBIAS-FOLD-B-V1 (2026-10-24, the same day as the baseline row) drove 17,280 ASR
# calls through it as a TOOL CONTACT with 0 core calls, on CPU. Transcripts never enter a
# decision, prompt or retrieval key (decision 201), so this is a side channel, not the core.
hf qwen3-asr-1.7b Qwen/Qwen3-ASR-1.7B \
  "$DR/models/qwen3-asr-1.7b" 7278e1e70fe206f11671096ffdd38061171dd6e5 model 8 || exit 1

# --- 5. the SAEA-E-001 evaluation carrier -----------------------------------------------------
# Earnings is the one asset that needs the git protocol, whose pack negotiation hangs
# indefinitely from this host. fetch_earnings.sh works around that by completing the partial
# clone's missing blobs over plain HTTPS (fix_earnings_blobs.py) instead.
step earnings "grep -q EARNINGS_DONE $LOGS/earnings.log 2>/dev/null" bash "$HERE/fetch_earnings.sh" || exit 1

echo "$(date '+%F %T') ALL DONE"
