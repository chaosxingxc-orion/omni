"""One pinned CPU encoder worker process (spawned by :mod:`core.cpu_encoder_pool`, never run by hand).

The worker pins itself to its core slice and fixes its thread count BEFORE torch is imported, hides
``torch_npu`` (a CPU worker must never initialise an accelerator backend), loads the pinned stack once
under the inventory's runtime and attention geometry, reports the package versions of ITS interpreter and the
MEASURED geometry for the parent to verify,
then serves batch-one encode requests until its input closes. The protocol is length-prefixed pickle
frames of builtins only (the parent and worker interpreters may carry different numpy versions, so a
waveform travels as :func:`pack_wave` bytes plus its dtype and is rebuilt exactly) on a private copy of
stdout; anything a library prints goes to stderr.
"""

from __future__ import annotations

import os
import pickle
import struct
import sys
from pathlib import Path

_HEADER = struct.Struct("!Q")


def read_frame(stream):
    header = stream.read(_HEADER.size)
    if not header:
        return None
    if len(header) != _HEADER.size:
        raise EOFError("truncated encoder frame header")
    size = _HEADER.unpack(header)[0]
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("truncated encoder frame")
    return pickle.loads(payload)


def write_frame(stream, value):
    payload = pickle.dumps(value, protocol=5)
    stream.write(_HEADER.pack(len(payload)) + payload)
    stream.flush()


def pack_wave(wave):
    """A 1-D waveform as builtins: its dtype name and its contiguous little-endian bytes."""

    import numpy as np

    array = np.ascontiguousarray(wave)
    if array.ndim != 1:
        raise ValueError("waveform must be one-dimensional")
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return {"dtype": array.dtype.str, "data": array.tobytes()}


def unpack_wave(packed):
    import numpy as np

    return np.frombuffer(packed["data"], dtype=np.dtype(packed["dtype"])).copy()


def main(argv=None) -> int:
    model_dir, runtime, cores, threads, packages, geometry = (argv or sys.argv[1:])[:6]
    first, last = (int(value) for value in cores.split("-"))
    os.sched_setaffinity(0, range(first, last + 1))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = threads
    sys.modules["torch_npu"] = None
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    requests = sys.stdin.buffer

    import importlib.metadata

    import torch

    torch.set_num_threads(int(threads))
    from .speech_embedding import encode, load_stack, measure_attention_geometry, render

    device = "cpu" if runtime.startswith("cpu-") else "cuda"
    processor, model, template = load_stack(
        Path(model_dir), local_files_only=True, runtime=runtime, attention_geometry=geometry)
    write_frame(channel, {
        "ready": True,
        "packages": {name: importlib.metadata.version(name) for name in packages.split(",") if name},
        "threads": torch.get_num_threads(),
        "cores": sorted(os.sched_getaffinity(0)),
        "attention_geometry": measure_attention_geometry(processor, model, device=device),
    })
    while (request := read_frame(requests)) is not None:
        try:
            rendered = render(processor, template, request["content"])
            audio = [unpack_wave(packed) for packed in request["audio"]] if "audio" in request else None
            vector = encode(processor, model, [rendered], audio, device=device, **request.get("bounds", {}))[0]
            write_frame(channel, {"vector": vector})
        except Exception as error:  # reported to the parent, which aborts the attempt
            write_frame(channel, {"error": f"{type(error).__name__}: {error}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
