# omni

Engineering base for reproducing the SAEA (speech-aware evidence acquisition) Thinking Fold B baseline
`SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-V1` with Qwen3-Omni-30B-A3B-Thinking (BF16), served by vLLM on Ascend NPUs
(vllm-ascend), with the openJiuwen agent base.

Code only. Datasets, features, vectors, prepared knowledge and model weights are not in this repository; the
download and preparation scripts that rebuild them are.

## Layout

| path | what |
| --- | --- |
| `studies/speech-aware-evidence-acquisition/` | The study, imported from [chaosxingxc-orion/speech-aware-evidence-acquisition](https://github.com/chaosxingxc-orion/speech-aware-evidence-acquisition) and followed by the NPU/vLLM port commits (vLLM transport, CPU encoder pool, forced-causal nemotron geometry, port cells, deployment layer). Design: `docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md`. |
| `docs/datasets.lock.json` | The umbrella lock the study reads two directories above itself (same layout as [exploring-l4-intelligence](https://github.com/chaosxingxc-orion/exploring-l4-intelligence)). |
| `adapters/` | `saea_adapters`: the encoder/device adapter layer installed into the image. |
| `docker/saea-base/` | Build context of the `saea-base` experiment image (on top of `quay.io/ascend/vllm-ascend:v0.30.0rc2`). `build.sh` expects `umbrella/` (this repo's `docs/` + `studies/`), `adapters/` and an offline `wheelhouse/` staged beside it. |
| `scripts/download/` | Sequential, resumable downloads of the model weights and datasets from huggingface.co / ModelScope / GitHub (`run_queue.sh`, `fetch_*.sh`, `mt_download.py`, `verify_hf.py`); the proxy comes from the environment. |
| `scripts/prep/` | Feature and knowledge preparation not in the study's own `scripts/`: semantic-knowledge authoring, the agreement-free comparator and the downstream build. |
| `scripts/npu/` | Host scripts for the NPU servers: container runners (`port_py.sh`, `port_cli.sh`), encoder runs (`port_encoder_*.sh`), serving (`npu_serve.sh`, `npu_deploy.sh`), card watcher, smoke. |
| `scripts/probe/` | Offline probes and benchmarks (preflight probe, attention geometry, CPU encoder benchmarks, a fake vLLM server). |

## Data root

Scripts take the data root from `SPEECHRL_DATA_DIR` / `OMNI_ROOT`. Under it: `models/`, `datasets/`, `prepared/`,
`vectors/`, `knowledge/`, `coordination/`, `attempts/` -- the study's relative layout. Every artifact is pinned by
sha256 in the study's configs and receipts, so a rebuild is checked against the recorded hashes.

## Serving

One resident deployment, declared by a profile in
`studies/speech-aware-evidence-acquisition/configs/deployment/` (four instances x tensor-parallel 2 on 8 Ascend
910B cards); `scripts/npu/npu_deploy.sh plan|up|status|down <profile>` refuses busy cards. Client concurrency comes
from `scripts/serving/synthetic_sweep.py` (synthetic load only).
