"""Aggregate CPU encoder throughput for P pinned worker processes x T threads (synthetic inputs only).

usage: cpu_pool_bench.py <model-dir> <first-core> <cores> <workers> [items-per-worker]
Each worker pins itself to its own core slice before torch is imported, loads its own float32 stack and
encodes the same synthetic text (~30 tokens) and 3 s audio items at batch one.
"""

import json
import os
import subprocess
import sys
import time

if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    _, _, model_dir, cores, kind, count = sys.argv
    first, last = map(int, cores.split("-"))
    os.sched_setaffinity(0, range(first, last + 1))
    threads = last - first + 1
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    sys.modules["torch_npu"] = None  # CPU only: no Ascend driver in this container
    import numpy as np
    import torch

    torch.set_num_threads(threads)
    from speech_aware_evidence_acquisition.core.speech_embedding import encode, load_stack, render

    processor, model, template = load_stack(__import__("pathlib").Path(model_dir), local_files_only=True, runtime="cpu-float32")
    wave = (0.05 * np.random.default_rng(0).standard_normal(48000)).astype(np.float32)
    if kind == "text":
        rendered = render(processor, template, [{"type": "text", "text": "passage: wake me up at seven tomorrow"
                                                                          " morning and remind me about the dentist"}])
        call = lambda: encode(processor, model, [rendered], None, device="cpu")
    else:
        rendered = render(processor, template, [{"type": "text", "text": "query: "}, {"type": "audio"}])
        call = lambda: encode(processor, model, [rendered], [wave], device="cpu", max_audio_samples=48000,
                              require_complete_audio=True)
    call()
    print("READY", flush=True)
    sys.stdin.readline()  # start barrier
    started = time.time()
    for _ in range(int(count)):
        call()
    print(json.dumps({"seconds": time.time() - started}), flush=True)
    sys.exit(0)

model_dir, first, cores, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
count = int(sys.argv[5]) if len(sys.argv) > 5 else 12
width = cores // workers
for kind in ("text", "audio"):
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", model_dir,
                               f"{first + i * width}-{first + (i + 1) * width - 1}", kind, str(count)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) for i in range(workers)]
    for proc in procs:
        while (line := proc.stdout.readline()) and line.strip() != "READY":
            pass
        if not line:
            sys.exit(f"worker exited before READY (code {proc.wait()})")
    started = time.time()
    for proc in procs:
        proc.stdin.write("go\n")
        proc.stdin.flush()
    results = [json.loads(proc.stdout.readline()) for proc in procs]
    wall = time.time() - started
    for proc in procs:
        proc.wait()
    print(json.dumps({"kind": kind, "workers": workers, "threads_each": width, "items": workers * count,
                      "wall_seconds": round(wall, 2), "items_per_second": round(workers * count / wall, 3),
                      "slowest_worker_seconds": round(max(r["seconds"] for r in results), 2)}), flush=True)
