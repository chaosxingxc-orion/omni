#!/usr/bin/env python3
"""CPU-vs-NPU benchmark of the SAEA port's three retrieval encoders, on SYNTHETIC inputs only.

One protocol, two halves: ``--device cpu`` now, ``--device npu[:N]`` later when cards are free; both emit
the same JSON schema (``saea-encbench-v1``) and a vector file (npz) for the fixed synthetic set, so
``compare`` can report cosine agreement between the halves.

Loader/encoder: ``saea_adapters.encoder`` (installed in saea-base:20261027), used unmodified. Two things are
done on top of its ``load_stack``, both outside the adapter:
  * dtype: the adapter always loads bfloat16 (the checkpoints' stored dtype). For ``--dtype float32`` the
    loaded model is cast with ``model.float()`` (bf16 -> fp32 is exact, so this equals a direct fp32 load).
  * geometry: nemotron is forced causal exactly as the port does (``is_causal=True`` on every text layer,
    see core/speech_embedding.load_stack(..., attention_geometry="causal")). Every arm is then gated
    ``causal`` with ``saea_adapters.attention_gate.require`` on the final dtype/device.

No dataset is read: audio is seeded noise + tones, text is seeded sentences over a fixed neutral vocabulary.

usage:
  encbench.py run --encoder {nemotron,lco3b,lco7b} --device cpu|npu[:N] --dtype float32|bfloat16 --out X.json
  encbench.py compare --ref A.npz --cand B.npz [--out C.json]
  encbench.py summarize results/*.json [--out SUMMARY.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import resource
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "saea-encbench-v1"
DATA_ROOT = Path(os.environ.get("SAEA_DATA_ROOT", "/mnt/cephfs/xingchao/omni"))
SAMPLE_RATE = 16000
# The port's registered bounds (configs/preparation/*-npu-port-v1.json).
MAX_AUDIO_SAMPLES = 1920000
MAX_SEQUENCE_TOKENS = 8192

# Prompts. nemotron: its shipped sentence-transformers prompts, as the study uses them. LCO: frozen explicit
# EMPTY prompts (no instruct prefix, P-EMB-1's binding decision); the checkpoints' default_prompt_name
# ("default" = the Qwen persona string) is declined and recorded by the adapter, never inherited.
ENCODERS = {
    "nemotron": dict(model_relpath="models/omni-embed-nemotron-3b", pooling_mode="mean",
                     embedding_dimension=2048, trust_remote_code=True, force_causal=True,
                     query_prompt="query: ", passage_prompt="passage: "),
    "lco3b": dict(model_relpath="models/lco-embedding-omni-3b-2605", pooling_mode="lasttoken",
                  embedding_dimension=2048, trust_remote_code=False, force_causal=False,
                  query_prompt="", passage_prompt=""),
    "lco7b": dict(model_relpath="models/lco-embedding-omni-7b", pooling_mode="lasttoken",
                  embedding_dimension=3584, trust_remote_code=False, force_causal=False,
                  query_prompt="", passage_prompt=""),
}
PROMPT_POLICY = "encbench-prompts-v1: nemotron query:/passage: (shipped); LCO '' / '' (no instruct prefix, default persona declined)"

# ---------------------------------------------------------------- synthetic inputs (deterministic)

VOCAB = (
    "the a small large red blue green orange quiet loud bright dark early late seven twelve paper metal "
    "window table river mountain garden kitchen station bridge lamp clock door street cloud stone basket "
    "walks moves opens closes turns carries holds finds brings counts paints builds near under above "
    "beside behind across through slowly gently often never always again today tomorrow morning evening"
).split()


def synth_audio(seconds: float, seed: int):
    """Mono 16 kHz float32: low noise floor plus four amplitude-modulated tones, peak 0.5."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = int(round(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    wave = 0.02 * rng.standard_normal(n)
    for _ in range(4):
        freq, amp = rng.uniform(150.0, 3000.0), rng.uniform(0.05, 0.2)
        rate, phase = rng.uniform(0.5, 4.0), rng.uniform(0, 2 * np.pi)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * rate * t + phase))
        wave += amp * envelope * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    wave = 0.5 * wave / np.abs(wave).max()
    return wave.astype(np.float32)


def synth_text(words: int, seed: int) -> str:
    import numpy as np

    rng = np.random.default_rng(seed)
    tokens = [VOCAB[i] for i in rng.integers(0, len(VOCAB), size=words)]
    return " ".join(tokens).capitalize() + "."


def input_set():
    """id -> spec. Latency items + 16-item batch sets; together they are the fixed fidelity set."""
    items = {}
    for s in (2, 3, 5, 10):
        items[f"a{s}s"] = {"kind": "audio", "seconds": float(s), "seed": 1000 + s}
    for w in (10, 30):
        items[f"t{w}w"] = {"kind": "text", "words": w, "seed": 1100 + w}
    for i in range(16):
        items[f"ba{i:02d}"] = {"kind": "audio", "seconds": float((2, 3, 4, 5)[i % 4]), "seed": 2000 + i}
        items[f"bt{i:02d}"] = {"kind": "text", "words": (10, 30)[i % 2], "seed": 3000 + i}
    return items


LATENCY_AUDIO = ["a2s", "a3s", "a5s", "a10s"]
LATENCY_TEXT = ["t10w", "t30w"]
BATCH_AUDIO = [f"ba{i:02d}" for i in range(16)]
BATCH_TEXT = [f"bt{i:02d}" for i in range(16)]


def materialise(spec):
    if spec["kind"] == "audio":
        return synth_audio(spec["seconds"], spec["seed"])
    return synth_text(spec["words"], spec["seed"])


def payload_sha(value) -> str:
    if isinstance(value, str):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f4").tobytes()).hexdigest()


# ---------------------------------------------------------------- process helpers


def parse_cores(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def pin_all_threads(cores):
    """Set the affinity of EVERY thread of this process (sched_setaffinity(0) only moves the caller)."""
    cores = set(cores)
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cores)
        except OSError:
            pass


def mem_gb():
    status = {}
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(("VmRSS", "VmHWM")):
                key, value = line.split(":")
                status[key] = round(int(value.split()[0]) / 1024 / 1024, 3)
    return {"rss_gb": status.get("VmRSS"), "peak_rss_gb": status.get("VmHWM")}


def device_mem(device):
    if not device.startswith("npu"):
        return None
    import torch

    return {"max_allocated_gb": round(torch.npu.max_memory_allocated() / 2**30, 3),  # type: ignore[attr-defined]
            "max_reserved_gb": round(torch.npu.max_memory_reserved() / 2**30, 3)}  # type: ignore[attr-defined]


def stats(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return {"n": int(arr.size), "p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
            "mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}


def early_environment(device: str, threads: int):
    """Must run before torch is imported."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if device == "cpu":
        # A CPU run must never initialise an accelerator backend (no NPU contact).
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        sys.modules["torch_npu"] = None  # type: ignore[assignment]


# ---------------------------------------------------------------- the encoder under test


class _architecture_auto_model:
    """Context manager: transformers.AutoModel -> load the class named in the checkpoint's config.json."""

    def __enter__(self):
        import transformers

        class ArchitectureAutoModel:
            @staticmethod
            def from_pretrained(path, **kw):
                arch = json.loads((Path(path) / "config.json").read_text())["architectures"][0]
                kw.pop("trust_remote_code", None)  # native class; the adapter passes False for LCO anyway
                return getattr(transformers, arch).from_pretrained(path, **kw)

        self.transformers, self.original = transformers, transformers.AutoModel
        transformers.AutoModel = ArchitectureAutoModel
        return self

    def __exit__(self, *exc):
        self.transformers.AutoModel = self.original
        return False


class Arm:
    def __init__(self, name, device, dtype):
        self.name, self.cfg, self.device, self.dtype = name, ENCODERS[name], device, dtype
        self.stack = None
        self.facts = {}
        self.events = {}

    def load(self):
        import torch
        from saea_adapters import attention_gate, encoder

        cfg = self.cfg
        model_dir = DATA_ROOT / cfg["model_relpath"]
        # Time the adapter's own load-time gate separately (a runtime wrapper; the adapter is not edited):
        # on CPU it runs two bf16 forwards, which are software-emulated on CPUs without bf16 instructions.
        gate_clock = {"s": 0.0}
        real_require = encoder.attention_gate.require

        def timed_require(*a, **k):
            g0 = time.perf_counter()
            try:
                return real_require(*a, **k)
            finally:
                gate_clock["s"] += time.perf_counter() - g0

        encoder.attention_gate.require = timed_require
        t0 = time.perf_counter()
        # measure-only at load: the load-time measurement is on the stored bf16 weights and, for nemotron,
        # BEFORE the causal forcing; the binding gate is the one below, on the final dtype and geometry.
        kwargs = dict(pooling_mode=cfg["pooling_mode"], embedding_dimension=cfg["embedding_dimension"],
                      attention_geometry="measure-only", local_files_only=True,
                      trust_remote_code=cfg["trust_remote_code"], attn_implementation="sdpa",
                      requested_device=self.device)
        self.loader = "saea_adapters.encoder.load_stack"
        self.adapter_error = None
        try:
            self.stack = encoder.load_stack(model_dir, **kwargs)
        except ValueError as error:
            if "Unrecognized configuration class" not in str(error):
                raise
            # ADAPTER BUG (reported, not patched): load_stack calls AutoModel.from_pretrained, and transformers
            # 5.14.1's AutoModel has no mapping for Qwen2_5OmniThinkerConfig (the LCO checkpoints' native
            # architecture). Re-run the SAME adapter call with transformers.AutoModel swapped, for this call
            # only, for a shim that resolves the class named in the checkpoint's own config.json
            # "architectures". Every adapter check, the gate and encode/pooling stay the adapter's.
            import traceback

            self.adapter_error = "".join(traceback.format_exception_only(type(error), error)).strip()[:600]
            self.loader = ("saea_adapters.encoder.load_stack with transformers.AutoModel scoped to "
                           "config.json architectures[0] (encbench workaround for the adapter bug)")
            gate_clock["s"] = 0.0
            t0 = time.perf_counter()
            with _architecture_auto_model():
                self.stack = encoder.load_stack(model_dir, **kwargs)
        t1 = time.perf_counter()
        encoder.attention_gate.require = real_require
        model = self.stack.model
        if self.dtype == "float32":
            model = model.float()
        elif self.dtype != "bfloat16":
            raise SystemExit(f"unsupported dtype {self.dtype}")
        self.stack.model = model.eval()
        t2 = time.perf_counter()
        if cfg["force_causal"]:
            layers = getattr(getattr(model, "model", None), "layers", None)
            if layers is None:
                raise SystemExit("encoder has no text tower layers to force causal on")
            for layer in layers:
                layer.self_attn.is_causal = True
        gate_error = None
        try:
            gate = attention_gate.require("causal", self.stack.processor, model, device=self.stack.device)
        except attention_gate.AttentionGeometryError as error:
            gate_error = str(error)
            gate = attention_gate.measure_text_tower(self.stack.processor, model, device=self.stack.device)
        t3 = time.perf_counter()
        param = next(model.parameters())
        self.facts = {
            "loader": self.loader, "adapter_load_stack_error": self.adapter_error,
            "adapter_facts": {k: v for k, v in self.stack.facts.items()},
            "final_param_dtype": str(param.dtype), "final_param_device": str(param.device),
            "forced_causal": bool(cfg["force_causal"]),
            "final_attention_gate": gate, "final_attention_gate_error": gate_error,
            "padding_side": getattr(self.stack.processor.tokenizer, "padding_side", None),
        }
        if cfg["force_causal"]:
            try:  # cross-check with the port's own geometry probe, if this study tree carries it
                from speech_aware_evidence_acquisition.core.speech_embedding import measure_attention_geometry

                self.facts["study_measure_attention_geometry"] = measure_attention_geometry(
                    self.stack.processor, model, device=self.stack.device)
            except ImportError as error:
                self.facts["study_measure_attention_geometry"] = f"unavailable: {error}"
        timing = {"load_s": t3 - t0, "adapter_load_s": t1 - t0, "adapter_load_gate_s": gate_clock["s"],
                  "cast_s": t2 - t1, "gate_s": t3 - t2,
                  # what a loader that reads the final dtype directly and gates once would pay
                  "load_s_operational": (t1 - t0 - gate_clock["s"]) + (t2 - t1) + (t3 - t2)}
        return timing, gate_error

    def _event(self, name):
        self.events[name] = time.perf_counter()

    def encode(self, kind, payloads):
        """End-to-end encode of a list of same-kind payloads (one forward). Returns (vectors, total_s, fwd_s)."""
        from saea_adapters import encoder

        stack, cfg = self.stack, self.cfg
        self.events = {}
        t0 = time.perf_counter()
        if kind == "audio":
            content = [{"type": "text", "text": cfg["query_prompt"]}, {"type": "audio"}]
            rendered = encoder.render(stack.processor, stack.template, content)
            if len(payloads) == 1:
                vectors = encoder.encode(stack, [rendered], list(payloads), max_audio_samples=MAX_AUDIO_SAMPLES,
                                         require_complete_audio=True, max_sequence_tokens=MAX_SEQUENCE_TOKENS,
                                         forward_event=self._event)
            else:  # the adapter's complete-audio check is batch-one only; batched agreement is measured instead
                vectors = encoder.encode(stack, [rendered] * len(payloads), list(payloads),
                                         max_sequence_tokens=MAX_SEQUENCE_TOKENS, forward_event=self._event)
        else:
            rendered = [encoder.render(stack.processor, stack.template,
                                       [{"type": "text", "text": cfg["passage_prompt"] + text}]) for text in payloads]
            vectors = encoder.encode(stack, rendered, None, max_sequence_tokens=MAX_SEQUENCE_TOKENS,
                                     forward_event=self._event)
        total = time.perf_counter() - t0
        fwd = self.events.get("encoder_forward_finished", t0) - self.events.get("encoder_forward_started", t0)
        return vectors, total, fwd

    def token_count(self, kind, payload):
        from saea_adapters import encoder

        stack, cfg = self.stack, self.cfg
        if kind == "audio":
            rendered = encoder.render(stack.processor, stack.template,
                                      [{"type": "text", "text": cfg["query_prompt"]}, {"type": "audio"}])
            inputs = stack.processor(text=[rendered], audio=[payload], sampling_rate=SAMPLE_RATE,
                                     return_tensors="pt", padding=True,
                                     audio_kwargs={"max_length": MAX_AUDIO_SAMPLES, "truncation": False})
        else:
            rendered = encoder.render(stack.processor, stack.template,
                                      [{"type": "text", "text": cfg["passage_prompt"] + payload}])
            inputs = stack.processor(text=[rendered], return_tensors="pt", padding=True)
        return int(inputs["attention_mask"].shape[-1])


def cosine(a, b):
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def agreement(ref: dict, cand: dict):
    import numpy as np

    cos = {k: cosine(ref[k], cand[k]) for k in cand if k in ref}
    diff = {k: float(np.abs(np.asarray(ref[k]) - np.asarray(cand[k])).max()) for k in cos}
    if not cos:
        return None
    return {"n": len(cos), "cos_min": min(cos.values()), "cos_mean": float(np.mean(list(cos.values()))),
            "max_abs_diff": max(diff.values()), "worst": min(cos, key=cos.get)}


# ---------------------------------------------------------------- pool workers


_HEADER = struct.Struct("!Q")


def write_frame(stream, value):
    payload = pickle.dumps(value, protocol=5)
    stream.write(_HEADER.pack(len(payload)) + payload)
    stream.flush()


def read_frame(stream):
    header = stream.read(_HEADER.size)
    if len(header) != _HEADER.size:
        return None
    size = _HEADER.unpack(header)[0]
    data = stream.read(size)
    if len(data) != size:
        return None
    return pickle.loads(data)


def worker_main(args):
    """One pool worker: own core slice or own device; regenerates the synthetic inputs from their seeds."""
    cores = parse_cores(args.cores) if args.cores else None
    if cores:
        os.sched_setaffinity(0, cores)
    early_environment("cpu" if args.device == "cpu" else "npu", args.threads)
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    import torch

    torch.set_num_threads(args.threads)
    arm = Arm(args.encoder, args.device, args.dtype)
    timing, gate_error = arm.load()
    if cores:
        pin_all_threads(cores)
    items = input_set()
    payloads = {k: materialise(v) for k, v in items.items()}
    write_frame(channel, {"ready": True, "load": timing, "gate_error": gate_error, "mem": mem_gb(),
                          "device_mem": device_mem(arm.device), "threads": torch.get_num_threads(),
                          "cores": sorted(os.sched_getaffinity(0)), "device": arm.device})
    stdin = sys.stdin.buffer
    while (req := read_frame(stdin)) is not None:
        ids, batch = req["ids"], req["batch"]
        kind = items[ids[0]]["kind"]
        out, t0 = {}, time.perf_counter()
        for start in range(0, len(ids), batch):
            chunk = ids[start:start + batch]
            vectors, _, _ = arm.encode(kind, [payloads[i] for i in chunk])
            out.update(zip(chunk, vectors))
        write_frame(channel, {"vectors": out, "busy_s": time.perf_counter() - t0, "mem": mem_gb()})
    return 0


class Pool:
    def __init__(self, args, slots):
        """slots: list of (cores or None, device)."""
        env = dict(os.environ)
        self.procs = []
        for cores, device in slots:
            cmd = [sys.executable, os.path.abspath(__file__), "worker", "--encoder", args.encoder,
                   "--device", device, "--dtype", args.dtype, "--threads", str(args.pool_threads)]
            if cores:
                cmd += ["--cores", f"{cores[0]}-{cores[-1]}"]
            self.procs.append(subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env))
        t0 = time.perf_counter()
        self.reports = []
        for proc in self.procs:
            report = read_frame(proc.stdout)
            if not report or not report.get("ready"):
                self.close()
                raise RuntimeError("pool worker failed to start (see log for its traceback)")
            self.reports.append(report)
        self.concurrent_load_s = time.perf_counter() - t0

    def run(self, ids, batch):
        """Split ids round-robin across workers; wall-clock from first send to last reply."""
        shares = [ids[i::len(self.procs)] for i in range(len(self.procs))]
        t0 = time.perf_counter()
        for proc, share in zip(self.procs, shares):
            write_frame(proc.stdin, {"ids": share, "batch": batch})
        replies = [read_frame(proc.stdout) for proc in self.procs]
        wall = time.perf_counter() - t0
        if any(r is None for r in replies):
            raise RuntimeError("pool worker died mid-run")
        vectors = {}
        for reply in replies:
            vectors.update(reply["vectors"])
        return vectors, wall, replies

    def close(self):
        for proc in self.procs:
            try:
                proc.stdin.close()
            except Exception:
                pass
        for proc in self.procs:
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------------------------------------------------------------- the run


def run_main(args):
    device_kind = "cpu" if args.device == "cpu" else "npu"
    threads = [int(t) for t in args.threads.split(",")]
    allowed = sorted(os.sched_getaffinity(0))
    if args.cores:
        declared = parse_cores(args.cores)
        if set(allowed) != set(declared):
            raise SystemExit(f"process affinity {allowed[0]}..{allowed[-1]} ({len(allowed)}) is not the declared "
                             f"--cores {args.cores}; refusing to run on other cores")
    if max(threads + [args.batch_threads]) > len(allowed):
        raise SystemExit("more threads requested than allowed cores")
    early_environment(device_kind, max(threads))
    import numpy as np
    import torch

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": SCHEMA, "status": "running", "failures": []}

    def save():
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=1, sort_keys=False))
        tmp.replace(out_path)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

    import importlib.metadata as md

    packages = {}
    for name in ("torch", "transformers", "torch_npu", "numpy", "saea-adapters"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    cpu_model = None
    try:
        cpu_model = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        cpu_model = next((l.split(":", 1)[1].strip() for l in cpu_model.splitlines() if "Model name" in l), None)
    except Exception:
        pass
    result["run"] = {
        "encoder": args.encoder, "device": args.device, "dtype": args.dtype, "label": f"{args.device}-{args.dtype}",
        "host": socket.gethostname(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv,
        "python": sys.executable, "python_version": platform.python_version(), "packages": packages,
        "cpu_model": cpu_model, "affinity": f"{allowed[0]}-{allowed[-1]} ({len(allowed)} cores)",
        "loadavg_start": open("/proc/loadavg").read().strip(),
        "protocol": {"threads": threads, "warmup": args.warmup, "calls": args.calls, "batch_threads": args.batch_threads,
                     "batch_rounds": args.rounds, "pool": args.pool, "pool_threads": args.pool_threads,
                     "pool_devices": args.pool_devices, "sections": args.sections,
                     "ref_threads": args.ref_threads, "max_audio_samples": MAX_AUDIO_SAMPLES,
                     "max_sequence_tokens": MAX_SEQUENCE_TOKENS},
    }
    cfg = ENCODERS[args.encoder]
    result["encoder"] = {
        "name": args.encoder, "model_dir": str(DATA_ROOT / cfg["model_relpath"]), "pooling_mode": cfg["pooling_mode"],
        "embedding_dimension": cfg["embedding_dimension"], "prompt_policy": PROMPT_POLICY,
        "query_prompt": cfg["query_prompt"], "passage_prompt": cfg["passage_prompt"],
        "prompts_sha256": hashlib.sha256(json.dumps([cfg["query_prompt"], cfg["passage_prompt"]]).encode()).hexdigest(),
    }
    items = input_set()
    payloads = {k: materialise(v) for k, v in items.items()}
    result["inputs"] = {"generator": "encbench synth_audio/synth_text v1 (seeded numpy default_rng)",
                        "items": {k: dict(v, sha256=payload_sha(payloads[k])) for k, v in items.items()}}
    save()

    # ---- 1. load
    torch.set_num_threads(max(threads))
    log(f"loading {args.encoder} on {args.device} {args.dtype}")
    arm = Arm(args.encoder, args.device, args.dtype)
    timing, gate_error = arm.load()
    result["encoder"].update(arm.facts)
    if arm.adapter_error:
        result["failures"].append({"stage": "adapter_load_stack", "error": arm.adapter_error,
                                   "handled_by": arm.loader})
    result["load"] = dict(timing, **mem_gb(), device_mem=device_mem(arm.device))
    if gate_error:
        result["failures"].append({"stage": "attention_gate", "error": gate_error})
    for k in LATENCY_AUDIO + LATENCY_TEXT:
        result["inputs"]["items"][k]["tokens"] = arm.token_count(items[k]["kind"], payloads[k])
    log(f"loaded in {timing['load_s']:.1f}s; {mem_gb()}")
    save()

    reference, b16_vectors, thread_vectors = {}, {}, {}
    sections = set(args.sections.split(","))

    # ---- 2. batch-1 latency with thread scaling
    if "latency" in sections:
        result["latency"] = {}
        for n in threads:
            if device_kind == "cpu":
                pin_all_threads(allowed[:n])
            torch.set_num_threads(n)
            cell = {"cores": f"{allowed[0]}-{allowed[n - 1]}" if device_kind == "cpu" else None}
            for k in LATENCY_AUDIO + LATENCY_TEXT:
                if n in args.skip_cells.get(k, ()):
                    continue
                kind = items[k]["kind"]
                for _ in range(args.warmup):
                    arm.encode(kind, [payloads[k]])
                totals, fwds, first, last = [], [], None, None
                for c in range(args.calls):
                    vec, total, fwd = arm.encode(kind, [payloads[k]])
                    totals.append(total)
                    fwds.append(fwd)
                    first = vec[0] if first is None else first
                    last = vec[0]
                cell[k] = dict(stats(totals), forward_p50=stats(fwds)["p50"],
                               repeat_max_abs_diff=float(np.abs(np.asarray(first) - np.asarray(last)).max()))
                thread_vectors.setdefault(n, {})[k] = first
                log(f"threads={n} {k}: p50 {cell[k]['p50']:.3f}s p95 {cell[k]['p95']:.3f}s")
                result["latency"][str(n)] = cell
                save()
        # best thread count: lowest summed p50 over the items measured at every thread count
        common = [k for k in LATENCY_AUDIO + LATENCY_TEXT if all(k in result["latency"][str(n)] for n in threads)]
        score = {n: sum(result["latency"][str(n)][k]["p50"] for k in common) for n in threads}
        best = min(score, key=score.get)
        result["latency_best"] = {"threads": best, "score_items": common,
                                  "per_item": {k: min(threads, key=lambda n: result["latency"][str(n)][k]["p50"])
                                               for k in common}}
        ref_n = args.ref_threads if args.ref_threads in thread_vectors else threads[0]
        reference.update(thread_vectors[ref_n])
        result["thread_determinism"] = {
            "ref_threads": ref_n,
            "vs_ref": {str(n): agreement(thread_vectors[ref_n], vecs) for n, vecs in thread_vectors.items() if n != ref_n},
        }
        save()

    # ---- 3a. batched throughput, one process x batch_threads
    if "batch" in sections:
        n = args.batch_threads
        if device_kind == "cpu":
            pin_all_threads(allowed[:n])
        torch.set_num_threads(n)
        result["batch"] = {"threads": n, "cores": f"{allowed[0]}-{allowed[n - 1]}" if device_kind == "cpu" else None}
        for kind, ids in (("text", BATCH_TEXT), ("audio", BATCH_AUDIO)):
            entry = {"items": len(ids)}
            if kind == "audio":
                entry["audio_seconds"] = sum(items[i]["seconds"] for i in ids)
            # (i) the study pool's per-item semantics: 16 sequential batch-one forwards
            arm.encode(kind, [payloads[ids[0]]])
            walls = []
            for r in range(args.rounds):
                t0 = time.perf_counter()
                for i in ids:
                    vec, _, _ = arm.encode(kind, [payloads[i]])
                    if r == 0:
                        reference[i] = vec[0]
                walls.append(time.perf_counter() - t0)
            entry["seq_b1"] = {"wall_s": stats(walls), "items_per_s": len(ids) / stats(walls)["p50"]}
            log(f"batch {kind} seq_b1: {entry['seq_b1']['items_per_s']:.2f} items/s")
            save()
            # (ii) one padded batch-16 forward
            arm.encode(kind, [payloads[i] for i in ids])
            walls = []
            for r in range(args.rounds):
                vecs, total, _ = arm.encode(kind, [payloads[i] for i in ids])
                walls.append(total)
                if r == 0:
                    b16_vectors.update(zip(ids, vecs))
            entry["padded_b16"] = {"wall_s": stats(walls), "items_per_s": len(ids) / stats(walls)["p50"],
                                   "agreement_vs_b1": agreement({i: reference[i] for i in ids}, b16_vectors)}
            log(f"batch {kind} padded_b16: {entry['padded_b16']['items_per_s']:.2f} items/s "
                f"cos_min vs b1 {entry['padded_b16']['agreement_vs_b1']['cos_min']:.6f}")
            result["batch"][kind] = entry
            save()
    result["mem_after_single_process"] = mem_gb()
    result["device_mem_after_single_process"] = device_mem(arm.device)

    # ---- 3b. pool layout
    if "pool" in sections and args.pool:
        slots = []
        if args.pool_devices:
            per = args.pool_threads
            for index, dev in enumerate(args.pool_devices.split(",")):
                cores = allowed[index * per:(index + 1) * per] if per * (index + 1) <= len(allowed) else None
                slots.append((cores, dev))
        else:
            workers, per = (int(v) for v in args.pool.lower().split("x"))
            if workers * per > len(allowed):
                raise SystemExit("pool layout exceeds allowed cores")
            slots = [(allowed[i * per:(i + 1) * per], "cpu") for i in range(workers)]
        # free the single-process model first so the pool's memory peak is its own
        # (on npu this also frees npu:0 for pool worker 0, which shares that card)
        if not args.keep_main_model:
            arm.stack = None
            import gc

            gc.collect()
            if device_kind == "npu":
                torch.npu.empty_cache()  # type: ignore[attr-defined]
        log(f"starting pool {args.pool} {args.pool_devices or ''}")
        pool = Pool(args, slots)
        result["pool"] = {"layout": args.pool, "devices": args.pool_devices, "concurrent_load_s": pool.concurrent_load_s,
                          "workers": [{k: r[k] for k in ("load", "mem", "device_mem", "threads", "device", "gate_error")}
                                      | {"cores": f"{r['cores'][0]}-{r['cores'][-1]}"} for r in pool.reports]}
        save()
        try:
            for kind, ids in (("text", BATCH_TEXT), ("audio", BATCH_AUDIO)):
                entry = {}
                for mode, batch in (("b1", 1), ("b4", 4)):
                    pool.run(ids, batch)  # warm-up round
                    walls, last = [], None
                    for _ in range(args.rounds):
                        vecs, wall, replies = pool.run(ids, batch)
                        walls.append(wall)
                        last = vecs
                    ref = {i: reference[i] for i in ids if i in reference}
                    entry[mode] = {"wall_s": stats(walls), "items_per_s": len(ids) / stats(walls)["p50"],
                                   "agreement_vs_single_b1": agreement(ref, last) if ref else None}
                    log(f"pool {kind} {mode}: {entry[mode]['items_per_s']:.2f} items/s")
                result["pool"][kind] = entry
                save()
            result["pool"]["worker_mem_end"] = [r["mem"] for r in replies]
        finally:
            pool.close()

    # ---- 4. fidelity record
    vec_path = out_path.with_suffix(".npz")
    meta = {"schema": SCHEMA, "encoder": args.encoder, "device": args.device, "dtype": args.dtype,
            "ref_threads": result.get("thread_determinism", {}).get("ref_threads"),
            "inputs": {k: result["inputs"]["items"][k]["sha256"] for k in reference},
            "prompts_sha256": result["encoder"]["prompts_sha256"]}
    arrays = {f"b1__{k}": np.asarray(v, dtype=np.float32) for k, v in reference.items()}
    arrays.update({f"b16__{k}": np.asarray(v, dtype=np.float32) for k, v in b16_vectors.items()})
    np.savez(vec_path, __meta__=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
    result["vectors"] = {"path": str(vec_path), "sha256": hashlib.sha256(vec_path.read_bytes()).hexdigest(),
                         "b1_items": len(reference), "b16_items": len(b16_vectors)}
    result["peak_rss_gb_end"] = mem_gb()["peak_rss_gb"]
    result["run"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result["run"]["loadavg_end"] = open("/proc/loadavg").read().strip()
    result["status"] = "ok" if not result["failures"] else "ok-with-failures"
    save()
    log(f"done -> {out_path}")
    return 0


# ---------------------------------------------------------------- compare / summarize


def load_npz(path):
    import numpy as np

    data = np.load(path)
    meta = json.loads(bytes(data["__meta__"]).decode())
    return meta, {k: data[k] for k in data.files if k != "__meta__"}


def compare_main(args):
    import numpy as np

    ref_meta, ref = load_npz(args.ref)
    cand_meta, cand = load_npz(args.cand)
    out = {"schema": SCHEMA + "-compare", "ref": args.ref, "cand": args.cand,
           "ref_run": {k: ref_meta[k] for k in ("encoder", "device", "dtype")},
           "cand_run": {k: cand_meta[k] for k in ("encoder", "device", "dtype")}}
    if ref_meta["encoder"] != cand_meta["encoder"]:
        raise SystemExit("different encoders")
    bad = [k for k in ref_meta["inputs"] if k in cand_meta["inputs"] and ref_meta["inputs"][k] != cand_meta["inputs"][k]]
    out["input_hash_mismatch"] = bad
    out["prompts_match"] = ref_meta["prompts_sha256"] == cand_meta["prompts_sha256"]
    for prefix in ("b1__", "b16__"):
        keys = sorted(k for k in ref if k.startswith(prefix) and k in cand)
        cos = {k[len(prefix):]: cosine(ref[k], cand[k]) for k in keys}
        if cos:
            out[prefix.rstrip("_")] = {"n": len(cos), "cos_min": min(cos.values()),
                                      "cos_mean": float(np.mean(list(cos.values()))), "worst": min(cos, key=cos.get),
                                      "per_item": cos}
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def summarize_main(args):
    runs = []
    for path in args.results:
        data = json.loads(Path(path).read_text())
        if data.get("schema") == SCHEMA:
            runs.append(data)
    runs.sort(key=lambda d: (list(ENCODERS).index(d["run"]["encoder"]), d["run"]["label"]))

    def get(d, *keys, fmt="{:.2f}"):
        try:
            for key in keys:
                d = d[key]
            return fmt.format(d)
        except (KeyError, TypeError, IndexError):
            return "-"

    out = ["## Load and memory", "",
           "| encoder | run | status | load s (first, as measured) | load s (operational) | pool worker load s (warm) "
           "| peak RSS GB (1 process) | pool worker RSS GB | device mem GB |",
           "|---|---|---|---|---|---|---|---|---|"]
    for d in runs:
        r, load, pool = d["run"], d.get("load", {}), d.get("pool", {})
        workers = pool.get("workers") or []
        wload = "/".join(get(w, "load", "load_s", fmt="{:.0f}") for w in workers) or "-"
        wrss = get(workers, 0, "mem", "peak_rss_gb", fmt="{:.1f}") if workers else "-"
        out.append(f"| {r['encoder']} | {r['label']} | {d.get('status')} | {get(load, 'load_s', fmt='{:.1f}')} | "
                   f"{get(load, 'load_s_operational', fmt='{:.1f}')} | {wload} | {get(d, 'peak_rss_gb_end', fmt='{:.1f}')} | "
                   f"{wrss} | {get(load, 'device_mem', 'max_allocated_gb', fmt='{:.1f}')} |")
    items = LATENCY_AUDIO + LATENCY_TEXT
    out += ["", "## Batch-1 latency, seconds p50 / p95 (end-to-end encode: processor + forward + pooling)", "",
            "| encoder | run | threads | " + " | ".join(items) + " |", "|---|---|---|" + "---|" * len(items)]
    for d in runs:
        r, best = d["run"], d.get("latency_best", {}).get("threads")
        for n, cell in sorted(d.get("latency", {}).items(), key=lambda kv: int(kv[0])):
            vals = [f"{cell[k]['p50']:.2f} / {cell[k]['p95']:.2f}" if k in cell else "-" for k in items]
            mark = " (best)" if best is not None and int(n) == best else ""
            out.append(f"| {r['encoder']} | {r['label']} | {n}{mark} | " + " | ".join(vals) + " |")
    out += ["", "## Batched throughput (16 items; items/s, median of rounds)", "",
            "| encoder | run | 1x16t seq b1 text | 1x16t b16 text | 4x16 pool b1 text | 4x16 pool b4 text "
            "| 1x16t seq b1 audio | 1x16t b16 audio | 4x16 pool b1 audio | 4x16 pool b4 audio "
            "| b16 vs b1 cos min text / audio | thread-count vector max abs diff |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in runs:
        r, b, p = d["run"], d.get("batch", {}), d.get("pool", {})
        if not b and not p:
            continue
        det = d.get("thread_determinism", {}).get("vs_ref", {})
        detv = max((v["max_abs_diff"] for v in det.values() if v), default=None)
        out.append(
            f"| {r['encoder']} | {r['label']} | {get(b, 'text', 'seq_b1', 'items_per_s')} | "
            f"{get(b, 'text', 'padded_b16', 'items_per_s')} | {get(p, 'text', 'b1', 'items_per_s')} | "
            f"{get(p, 'text', 'b4', 'items_per_s')} | {get(b, 'audio', 'seq_b1', 'items_per_s')} | "
            f"{get(b, 'audio', 'padded_b16', 'items_per_s')} | {get(p, 'audio', 'b1', 'items_per_s')} | "
            f"{get(p, 'audio', 'b4', 'items_per_s')} | "
            f"{get(b, 'text', 'padded_b16', 'agreement_vs_b1', 'cos_min', fmt='{:.6f}')} / "
            f"{get(b, 'audio', 'padded_b16', 'agreement_vs_b1', 'cos_min', fmt='{:.6f}')} | "
            f"{'-' if detv is None else f'{detv:.1e}'} |")
    fails = [(d["run"]["encoder"], d["run"]["label"], f) for d in runs for f in d.get("failures", [])]
    if fails:
        out += ["", "## Failures", ""] + [f"- {e} {l}: stage {f.get('stage')}: {str(f.get('error'))[:170]}... (handled by: {f.get('handled_by', 'not handled')})"
                                    for e, l, f in fails]
    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    r.add_argument("--device", default="cpu", help="cpu | npu | npu:N")
    r.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    r.add_argument("--out", required=True)
    r.add_argument("--cores", default="", help="declared core set; the run refuses if its affinity differs")
    r.add_argument("--threads", default="8,16,32,64", help="batch-1 thread counts (host threads on npu)")
    r.add_argument("--ref-threads", type=int, default=16)
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--calls", type=int, default=20)
    r.add_argument("--batch-threads", type=int, default=16)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--pool", default="4x16", help="CPU pool WORKERSxTHREADS; '' to skip")
    r.add_argument("--pool-threads", type=int, default=16)
    r.add_argument("--pool-devices", default="", help="npu pool: one worker per listed device, e.g. npu:0,npu:1")
    r.add_argument("--keep-main-model", action="store_true")
    r.add_argument("--sections", default="latency,batch,pool")
    r.add_argument("--skip", default="", help="item@threads,... cells to skip, e.g. a2s@8,a5s@8")
    w = sub.add_parser("worker")
    w.add_argument("--encoder", required=True)
    w.add_argument("--device", default="cpu")
    w.add_argument("--dtype", default="float32")
    w.add_argument("--threads", type=int, default=16)
    w.add_argument("--cores", default="")
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--cand", required=True)
    c.add_argument("--out", default="")
    s = sub.add_parser("summarize")
    s.add_argument("results", nargs="+")
    s.add_argument("--out", default="")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        if args.pool_devices and not args.pool:
            args.pool = f"{len(args.pool_devices.split(','))}xdevice"
        skip = {}
        for token in filter(None, args.skip.split(",")):
            item, n = token.split("@")
            skip.setdefault(item, set()).add(int(n))
        args.skip_cells = skip
        return run_main(args)
    if args.cmd == "worker":
        return worker_main(args)
    if args.cmd == "compare":
        return compare_main(args)
    return summarize_main(args)


if __name__ == "__main__":
    sys.exit(main())
