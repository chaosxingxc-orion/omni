# Base experiment image for the speech-aware-evidence-acquisition (SAEA) reproduction.
#
# What this image is for: run the SAME dataset and the SAME agent base (openJiuwen
# obs-agent-loop) as the registered SAEA study, against OUR OWN serving stack -- vLLM on Ascend
# with BF16 safetensors -- and see whether the result lands near the published one. It is
# deliberately NOT a byte-identical reproduction: the study's own receipts pin a llama.cpp
# binary and Q4_K_M GGUF by sha256, and that whole route was dropped on owner directive
# (2026-10-27, "only BF16 / vLLM models"). Nothing GGUF is built or installed here.
#
# Base: the Ascend vLLM image already present on both NPU hosts (ssh aliases npu and npu2). It
# is NOT pulled at build time -- neither host has a route to the internet (verified: hf.co,
# hf-mirror, raw.githubusercontent and codeload all time out there), so everything this file
# adds arrives through the shared /mnt/cephfs build context. v0.30.0rc2 is the first
# vllm-ascend that registers Qwen3OmniMoeForConditionalGeneration, verified inside this image.
FROM quay.io/ascend/vllm-ascend:v0.30.0rc2

# --- the experiment harness lives in its own venv ---------------------------------------------
# Deliberately NOT the image's site-packages: openjiuwen requires transformers>=4.52.4 and its
# own numpy (the image carries 1.26.4), and resolving those in place would re-resolve the
# torch_npu / vLLM stack this image exists to provide. The harness never shares a process with
# vLLM -- it reaches the serving core over HTTP (/v1/chat/completions) -- so the isolation costs
# nothing and keeps `vllm serve` in this same container working untouched.
ENV SAEA_VENV=/opt/saea-venv
RUN python3 -m venv "$SAEA_VENV"

# Offline wheelhouse, staged on the shared disk by the dev host, which is the only machine with
# a proxy. Why each package, since the study declares `dependencies = []`:
#   openjiuwen 0.1.16.post2  the executor SAEA-E-001 actually ran on (baseline bare-core-ojw,
#                            policy obs-agent-loop-ojw) -- pinned exactly as the ledger records,
#                            and all 13 deep-internal module paths the study imports were
#                            verified present in this wheel before the build.
#   numpy / soundfile / jiwer  the study's declared test-time trio; jiwer>=4.0 is tight on
#                            purpose because jiwer.wer IS the measurement, not a convenience.
#   more-itertools, regex    reached at IMPORT time by reproduction/whisper_normalizer, which
#                            core/__init__ pulls in -- without them the package will not import.
#   mlflow-skinny            core/tracking.py uses only set_tracking_uri / set_experiment /
#                            start_run / log_params / log_metrics / log_artifact against a local
#                            file store; skinny provides all of those without the server stack.
#   pytest                   the runbook's first precondition is a contract-test run.
# The wheelhouse stays at /tmp/wheelhouse for the WHOLE build and is removed in the last RUN:
# every editable install below needs PEP 517 build isolation to find setuptools>=77 (the study's
# own [build-system]), and with --no-index the only place it can come from is --find-links.
COPY wheelhouse /tmp/wheelhouse
# ARG, not ENV: visible to every RUN below as an environment variable, never baked into the
# runtime image, so pip inside a running container behaves normally.
ARG PIP_NO_INDEX=1
ARG PIP_FIND_LINKS=/tmp/wheelhouse
RUN "$SAEA_VENV/bin/pip" install --upgrade pip \
 && "$SAEA_VENV/bin/pip" install \
      "openjiuwen==0.1.16.post2" \
      "numpy>=2.0,!=2.4.0" "soundfile>=0.12" "jiwer>=4.0" \
      "more-itertools>=10" "regex>=2024.0.0" "mlflow-skinny>=2.14" "pytest>=8"

# --- the study, in its registered layout ------------------------------------------------------
# data/lock.py resolves the umbrella lock as <study>/../../docs/datasets.lock.json and
# e0/artifacts.py takes REPO_ROOT from the package's own depth, so the study MUST sit at
# <umbrella>/studies/speech-aware-evidence-acquisition/. A symlink would not do: Path.resolve()
# follows it and the parent chain would break. --no-deps because the package deliberately
# declares none (its "D2 zero-dependency gate"), so this adds no resolution at all.
COPY umbrella /opt/umbrella
RUN "$SAEA_VENV/bin/pip" install --no-deps -e \
      /opt/umbrella/studies/speech-aware-evidence-acquisition

# --- paths the study resolves against ---------------------------------------------------------
# SPEECHRL_DATA_DIR is the asset-root convention from the umbrella's docs/datasets.lock.json;
# dataset and model configs resolve beneath it. Mounted from the shared disk at run time.
# PATH deliberately does NOT put the venv first for the system python: `vllm` and `python3` must
# keep resolving to the image's own interpreter so serving is unaffected. Use
# /opt/saea-venv/bin/python (or `saea-driver`, installed there) for the harness.
ENV SPEECHRL_DATA_DIR=/mnt/cephfs/xingchao \
    SAEA_PYTHON=/opt/saea-venv/bin/python \
    SAEA_DRIVER=/opt/saea-venv/bin/saea-driver
WORKDIR /opt/umbrella/studies/speech-aware-evidence-acquisition

# --- the encoder-preparation side: the SAME study, on the image's torch -----------------------
# The retrieval core of the baseline is Omni-Embed-Nemotron-3B, and unlike the answering core it
# is NOT reached over HTTP: core/speech_preparation_runtime.py loads it IN-PROCESS
# (NemotronSpeechEncoder / NemotronPassageEncoder -> core/speech_embedding.py: load_stack, which
# imports torch + transformers.AutoModel/AutoProcessor against the model's own shipped
# modeling_nv_omni_embed.py). That process therefore needs torch -- which means the image's
# torch_npu build, not a pip wheel -- so the study is installed into the SYSTEM interpreter too.
# Two installs of one source tree, each where its dependencies already are:
#   /opt/saea-venv/bin/python  the agent loop (openjiuwen + measurement deps, no torch)
#   python3                    encoder preparation (torch_npu + transformers from the image)
# --no-deps keeps this from touching the serving stack. Three small additions: soundfile (audio
# decode), and more-itertools + regex, which core/__init__ reaches at IMPORT time through the
# whisper normaliser -- without them `import speech_aware_evidence_acquisition.core` fails in this
# interpreter exactly as it did in the venv (verified in this image 2026-10-27). librosa is
# deliberately NOT installed: speech_embedding only imports it to resample, and SLURP
# is already 16 kHz == SAMPLE_RATE_HZ (verified: 300/300 slurp_real FLACs at 16000), so that
# branch is dead and the numba/llvmlite aarch64 build is avoided.
RUN python3 -m pip install --no-deps "soundfile>=0.12" "more-itertools>=10" "regex>=2024.0.0" \
 && python3 -m pip install --no-deps -e \
      /opt/umbrella/studies/speech-aware-evidence-acquisition

# --- our adapter layer ------------------------------------------------------------------------
# Reuse the study, refactor the base: the study tree is never edited (its receipts hash their own
# source and its "D2 zero-dependency gate" is a property of that package), so every difference
# between the author's stack and ours lives here instead. Installed into BOTH interpreters for
# the same reason the study is: the encoder adapters need the image's torch, and the transport
# adapters belong beside the agent loop.
#   saea_adapters.device         cuda -> npu/cuda/cpu, with torch_npu's import ordering isolated
#   saea_adapters.encoder        pooling mode (mean|lasttoken) and width read from the
#                                checkpoint's own 1_Pooling/config.json instead of DIMS = 2048
#   saea_adapters.attention_gate proves the loaded text tower's information flow matches what the
#                                arm declares -- nemotron's bidirectional override has not bound
#                                since transformers 4.55.0 and nothing raises when it doesn't
#   saea_adapters.backends       drop-in backend_factory classes for run_preparation
# --no-deps: torch/transformers/numpy all come from the base image and must not be re-resolved.
COPY adapters /opt/adapters
RUN "$SAEA_VENV/bin/pip" install --no-deps -e /opt/adapters \
 && python3 -m pip install --no-deps -e /opt/adapters \
 && rm -rf /tmp/wheelhouse
