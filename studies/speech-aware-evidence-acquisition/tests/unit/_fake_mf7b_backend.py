"""Fake Omni-Embed entry points for the MF-7b sharding tests (imported by spawned workers by module path).

``encode`` returns float32-derived floats that depend on each text AND on its batch (size and position),
so any regrouping of batches changes the vectors and an equality test catches it.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np


def load_stack(model_dir, *, local_files_only=False, runtime="cuda-bfloat16", attention_geometry=None):
    return "processor", ("model", str(model_dir), runtime, attention_geometry), "template"


def measure_attention_geometry(processor, model, *, device="cpu"):
    """The forced geometry, unless the model dir name asks for a mismatch (``*-mismeasured``)."""

    if model[1].endswith("-mismeasured"):
        return "bidirectional"
    return model[3] or "bidirectional"


def render(processor, template, content):
    return content[0]["text"]


def encode(processor, model, texts, audio, *, device="cuda", **_kwargs):
    if os.environ.get("_FAKE_EXPECT_PINNED") == "1":
        threads = int(os.environ["OMP_NUM_THREADS"])
        if threads != len(os.sched_getaffinity(0)) or os.environ.get("OPENBLAS_NUM_THREADS") != str(threads):
            raise RuntimeError("worker was not pinned before encoding")
    if device != model[2].split("-")[0]:
        raise RuntimeError(f"device {device} does not match runtime {model[2]}")
    out = []
    for position, text in enumerate(texts):
        if "boom" in text:
            raise RuntimeError("fake encoder failure")
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        values = np.asarray([seed % 997 / 997.0, len(texts), position, len(text) / 7.0], dtype=np.float32)
        out.append(tuple(float(v) for v in values))
    return out
