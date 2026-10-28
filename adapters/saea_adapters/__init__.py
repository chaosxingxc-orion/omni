"""Adapters that let the registered SAEA study run on our own stack.

The study is reused, never edited: every module here either wraps a study seam
or re-implements one of its hard-coded assumptions generically. Two axes so far.

platform / engine
    The answering core moves from a llama.cpp ``llama-server`` pinned by binary
    sha256 to vLLM-Ascend serving BF16 safetensors, and the execution host moves
    from WSL2 to a container on Linux. (See the transport adapter work; not in
    this package yet.)

retrieval encoder  (this package)
    ``device``          cuda -> npu/cuda/cpu resolution, with torch_npu's
                        import-ordering side effect isolated.
    ``encoder``         generic ``load_stack``/``encode``: pooling mode
                        (mean | lasttoken), width from the checkpoint's own
                        ``1_Pooling/config.json`` instead of a module constant,
                        and the declined-default-prompt guard.
    ``attention_gate``  proves the loaded text tower's information flow matches
                        what the arm declares, because the nemotron arm's
                        bidirectional patch silently stops binding on
                        transformers 5.x and nothing else would catch it.
    ``backends``        drop-in ``backend_factory`` classes for
                        ``core.speech_preparation_runtime.run_preparation``.

Nothing here imports ``sentence_transformers`` (it pulls torchcodec, which needs
an ffmpeg the study forbids installing), and nothing here reads a dataset,
a label or a transcript.
"""

from __future__ import annotations

__all__ = ["attention_gate", "backends", "device", "encoder"]
