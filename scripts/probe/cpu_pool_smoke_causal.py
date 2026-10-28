"""Live smoke of the forced-causal CpuEncoderPool on synthetic content: workers start, geometry verified, bit identity.

usage: cpu_pool_smoke_causal.py <model-dir> <first-core> <workers> <threads>
"""
import importlib.metadata
import subprocess
import sys

import numpy as np

from speech_aware_evidence_acquisition.core.cpu_encoder_pool import CPU_INVENTORY_SCHEMA, CpuEncoderPool
from speech_aware_evidence_acquisition.core.cpu_encoder_worker import pack_wave

model_dir, first, workers, threads = sys.argv[1:5]
py = "/usr/local/python3.12.13/bin/python3"
versions = dict(zip(("torch", "transformers"), subprocess.check_output(
    [py, "-c", "import importlib.metadata as m; print(m.version('torch')); print(m.version('transformers'))"],
    text=True, env={"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}).split()))
inventory = {
    "schema": CPU_INVENTORY_SCHEMA, "model_dir_relpath": "unused-by-the-pool", "files": {},
    "packages": versions, "runtime": "cpu-float32", "attention_geometry": "causal",
    "python_executable": py, "workers": int(workers), "threads_per_worker": int(threads), "first_core": int(first),
}
wave = (0.05 * np.random.default_rng(0).standard_normal(48000)).astype(np.float32)
text = {"content": [{"type": "text", "text": "passage: switch the kitchen lights to blue"}],
        "bounds": {"max_sequence_tokens": 8192}}
audio = {"content": [{"type": "text", "text": "query: "}, {"type": "audio"}], "audio": [pack_wave(wave)],
         "bounds": {"max_audio_samples": 48000, "require_complete_audio": True, "max_sequence_tokens": 8192}}
pool = CpuEncoderPool(inventory, model_dir)
try:
    vectors = pool.map([text, text, audio, audio])
finally:
    pool.close()
v = np.array(vectors)
print("dims", v.shape, "norms", np.round(np.linalg.norm(v, axis=1), 6).tolist())
print("text bit-identical across workers:", vectors[0] == vectors[1])
print("audio bit-identical across workers:", vectors[2] == vectors[3])
print("text/audio cosine", round(float(v[0] @ v[2]), 4))
