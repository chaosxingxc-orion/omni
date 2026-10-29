#!/usr/bin/env python3
"""Synthetic concurrency sweep for the resident vLLM deployment (deployment layer, npu-vllm-port 2026-10-28).

    python scripts/serving/synthetic_sweep.py \
        --base-url http://127.0.0.1:8200 --base-url http://127.0.0.1:8201 ... \
        --model qwen3-omni-30b-a3b-thinking --levels 8,16,24,32,48 --seed 20261028 \
        [--requests-per-level N] [--deployment-profile configs/deployment/<id>.json] --out report.json

Why it exists. ``--max-num-seqs`` is a deployment ceiling fixed when the resident instances start; the owner decided
(2026-10-28) that the client concurrency below it is MEASURED, not inherited from the historical 4. This tool measures
it on SYNTHETIC load shaped like the real Fold B mix, so the measurement touches no dataset row and needs no exposure
ledger entry:

* prompts: per-contact prompt-token targets from a lognormal truncated to ``[--prompt-min, --prompt-max]`` whose
  truncated mean is ``--prompt-mean`` (defaults 3,000 / 20,000 / 8,500), hit EXACTLY through the server's own
  chat-form ``POST /tokenize`` (the same request shape ``core.preparation.measure_vllm_multimodal_request`` measures:
  ``messages`` + ``chat_template_kwargs``, ``add_generation_prompt: true``, ``add_special_tokens: false``), so the count
  includes the audio placeholder tokens the server's multimodal processor expands;
* message layout: the ``native-raw-speech-query-prefix-v1`` shape the port sends -- ``[system: instruction]``,
  ``[user: input_audio (query slot), stable evidence text, mutable state text]`` -- with no ``tools`` (Fold B is a
  tool-free arm, and a system string beside tools fails under vLLM's detected content format; design doc 5.5);
* audio: a synthetic 16 kHz mono 16-bit WAV of ``--audio-min-seconds``..``--audio-max-seconds`` (default 2-8 s), a
  voiced harmonic buzz under a syllable-rate envelope plus low noise, generated from the seed; no recording is read;
* text: shuffled common English words from :data:`WORDS`; no dataset text is read;
* trajectories: ``--contacts-per-trajectory`` sequential contacts (default 3) that share one audio clip and one word
  stream, each contact a strict extension of the previous one's evidence (so prefix caching sees the reuse a real
  trajectory gives it), with a short per-contact state suffix after it;
* decoding: ``temperature 0``, ``max_tokens`` 4,096, ``thinking_token_budget`` 3,072 (vLLM's spelling of the
  fragment's reasoning budget, ``core/model/vllm_transport.py``), ``chat_template_kwargs {"enable_thinking": true}``.

How it drives. Concurrency levels are PER INSTANCE: level ``c`` over ``N`` base URLs runs ``c * N`` closed-loop
workers. Worker ``w`` is pinned to instance ``w mod N`` (per-trajectory affinity, the routing the flight-side
multi-endpoint transport will use) and owns trajectories ``t`` with ``t mod workers == w``; it sends each trajectory's
contacts one after another. Every level gets its own seeded content, so a level never hits the prefix cache a previous
level filled. All prompts of a level are fitted BEFORE its clock starts.

What it records. Per request: worker, instance, target / tokenized / server-reported prompt tokens, completion tokens,
latency, time to first streamed token (streaming is on unless ``--no-stream``), status (``ok`` / ``timeout`` /
``error``) and error text. Per level: aggregate output tok/s over the level's wall time, per-sequence end-to-end tok/s
(completion tokens / latency -- the quantity the 420 s timeout bounds: 4,096 tokens need >= ~9.8 tok/s) and decode
tok/s at p50/p10, latency p50/p95/max, TTFT, and the instances' ``/metrics`` before and after
(``vllm:num_preemptions_total`` -- vLLM 0.30's ``vllm:num_preemptions`` counter, ``v1/metrics/loggers.py`` --, prefix
cache queries/hits, prompt/generation token counters) plus ``vllm:kv_cache_usage_perc`` / ``num_requests_running`` /
``num_requests_waiting`` sampled every ``--metrics-interval`` seconds.

Liveness (owner rule, plan 3.8 item 2 / 3.9): a :class:`LivenessPoller` thread runs beside the metrics sampler and polls
``GET /v1/models`` of every instance every ``--liveness-interval`` seconds (default 30; 0 disables), then once more when
a level's workers finish. A dead endpoint (unreachable, non-200, or no longer serving the model, two attempts in a row)
ABORTS the level: the report carries an explicit ``endpoint_died`` field on the level and at the top, no later level is
run, a level with ``endpoint_died`` is never a knee candidate, and the CLI exits with status 3 AFTER writing the report.
A dead endpoint is never absorbed as "some errors"; the tool never restarts anything.

The knee (:func:`recommend_knee`): the highest level with zero errors, zero timeouts, a READABLE and zero preemption
delta, and per-sequence end-to-end tok/s p10 >= ``--min-seq-tok-s`` (default 10). An unreadable counter fails the
level closed. The report is JSON (``indent=2, sort_keys=True``).

Standard library only, runnable with the harness venv (``/opt/saea-venv/bin/python``) or any Python >= 3.10.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import io
import json
import math
import random
import socket
import statistics
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

SCHEMA = "saea-synthetic-serving-sweep-v1"
SAMPLE_RATE = 16000

#: The synthetic vocabulary. Common English function and content words only; nothing here comes from a dataset.
WORDS = tuple(
    """the of and to in is that it for on with as was at by be this have from or one had not but what all were when we
    there can an your which their said if do will each about how up out them then she many some so these would other
    into has more her two like him see time could no make than first been its who now people my made over did down only
    way find use may water long little very after words called just where most know get through back much go good new
    write our used me man too any day same right look think also around another came come work three word must because
    does part even place well such here take why things help put years different away again off went old number great
    tell men say small every found still between name should home big give air line set own under read last never us
    left end along while might next sound below saw something thought both few those always looked show large often
    together asked house world going want school important until form food keep children feet land side without boy
    once animals life enough took sometimes four head above kind began almost live page got earth need far hand high
    year mother light parts country father let night following picture being study second eyes soon times story boys
    since white days ever paper hard near sentence better best across during today others however sure means knew try
    told young miles sun ways thing whole hear example heard several change answer room sea against top turned learn
    point city play toward five using himself usually money seen car morning""".split()
)
SYSTEM_TEXT = (
    "You are a careful assistant. A short audio clip is followed by background notes and the current step. "
    "Think about the notes, then answer with one short JSON object naming the most likely topic and your confidence."
)
METRIC_COUNTERS = {
    # report key: candidate exposition names, first readable wins (prometheus_client appends _total to counters).
    "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    "prefix_cache_queries": ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"),
    "prefix_cache_hits": ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
    "prompt_tokens": ("vllm:prompt_tokens_total", "vllm:prompt_tokens"),
    "generation_tokens": ("vllm:generation_tokens_total", "vllm:generation_tokens"),
}
METRIC_GAUGES = {
    "kv_cache_usage_perc": "vllm:kv_cache_usage_perc",
    "num_requests_running": "vllm:num_requests_running",
    "num_requests_waiting": "vllm:num_requests_waiting",
}


#: Exit status of the CLI when an endpoint died during the sweep (the report is still written).
EXIT_ENDPOINT_DIED = 3
#: Seconds between the two probes that must both fail before an endpoint counts as dead.
LIVENESS_RETRY_GAP_S = 2.0


class SweepError(RuntimeError):
    """A sweep the tool refuses to run or to report (bad arguments, an unreachable or wrong server, an unfit prompt)."""


# ---------------------------------------------------------------------------
# Synthetic content.
# ---------------------------------------------------------------------------


def synthetic_wav(seed: object, seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    """A deterministic speech-shaped WAV: a voiced harmonic buzz (f0 100-220 Hz, gliding) under a 3-6 Hz syllable
    envelope with short pauses, plus low white noise. 16-bit mono PCM; no recording is read."""

    rng = random.Random(f"wav:{seed}")
    frames = int(round(seconds * sample_rate))
    f0, glide = rng.uniform(100.0, 220.0), rng.uniform(-0.15, 0.15)
    syllable_hz = rng.uniform(3.0, 6.0)
    harmonics = [(k, rng.uniform(0.2, 1.0) / k) for k in range(1, 6)]
    norm = sum(amplitude for _, amplitude in harmonics)
    phase = 0.0
    samples = bytearray()
    for n in range(frames):
        t = n / sample_rate
        pitch = f0 * (1.0 + glide * math.sin(2 * math.pi * 0.4 * t))
        phase += 2 * math.pi * pitch / sample_rate
        envelope = max(0.0, math.sin(math.pi * syllable_hz * t)) ** 2
        voiced = sum(amplitude * math.sin(k * phase) for k, amplitude in harmonics) / norm
        value = 0.55 * envelope * voiced + 0.02 * rng.uniform(-1.0, 1.0)
        samples += struct.pack("<h", max(-32767, min(32767, int(value * 32767))))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(samples))
    return buffer.getvalue()


def _truncated_lognormal_mean(mu: float, sigma: float, low: float, high: float, bins: int = 4000) -> float:
    normal = statistics.NormalDist()
    values = [math.exp(mu + sigma * normal.inv_cdf((k + 0.5) / bins)) for k in range(bins)]
    kept = [value for value in values if low <= value <= high]
    return sum(kept) / len(kept) if kept else float("nan")


def lognormal_mu_for_mean(mean: float, sigma: float, low: float, high: float) -> float:
    """The ``mu`` whose lognormal, truncated to ``[low, high]``, has mean ``mean`` (bisection; deterministic)."""

    if not low < mean < high or sigma <= 0:
        raise SweepError(f"prompt mean {mean} must lie strictly inside [{low}, {high}] with sigma > 0")
    lo, hi = math.log(low) - 3 * sigma, math.log(high) + 3 * sigma
    for _ in range(80):
        mid = (lo + hi) / 2
        value = _truncated_lognormal_mean(mid, sigma, low, high)
        if math.isnan(value) or value < mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sample_prompt_targets(rng: random.Random, count: int, *, mu: float, sigma: float, low: int, high: int) -> list[int]:
    """``count`` truncated-lognormal prompt-token targets (rejection inside ``[low, high]``)."""

    out: list[int] = []
    while len(out) < count:
        value = math.exp(rng.gauss(mu, sigma))
        if low <= value <= high:
            out.append(int(round(value)))
    return out


@dataclass(frozen=True)
class Trajectory:
    """One synthetic trajectory: its own audio and word stream; contacts are ascending prompt-token targets."""

    index: int
    seed: str
    audio_seconds: float
    targets: tuple[int, ...]

    def words(self, count: int) -> list[str]:
        rng = random.Random(f"words:{self.seed}")
        return [rng.choice(WORDS) for _ in range(count)]

    def audio_b64(self) -> str:
        return base64.b64encode(synthetic_wav(self.seed, self.audio_seconds)).decode("ascii")


def contact_messages(audio_b64: str, words: Sequence[str], n_words: int, contact: int, contacts: int,
                     pad: int) -> list[dict[str, object]]:
    """The ``native-raw-speech-query-prefix-v1`` shape: system instruction; user = [audio, stable evidence, state].
    ``pad`` single-letter words at the end of the state text make the token count exact without touching the
    shared evidence prefix."""

    state = f"Step {contact + 1} of {contacts}. Decide the next action from the notes above." + " x" * pad
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": "Notes: " + " ".join(words[:n_words])},
            {"type": "text", "text": state},
        ]},
    ]


# ---------------------------------------------------------------------------
# HTTP.
# ---------------------------------------------------------------------------


def _http_json(method: str, url: str, body: object = None, timeout: float = 60.0) -> object:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def tokenize_count(base_url: str, model: str, messages: list[dict[str, object]]) -> int:
    """The server's chat-form ``/tokenize`` count; refused unless ``count`` equals ``len(tokens)``."""

    response = _http_json("POST", base_url + "/tokenize", {
        "model": model, "messages": messages, "chat_template_kwargs": {"enable_thinking": True},
        "add_generation_prompt": True, "add_special_tokens": False,
    })
    count = response.get("count") if isinstance(response, dict) else None
    tokens = response.get("tokens") if isinstance(response, dict) else None
    if isinstance(count, bool) or not isinstance(count, int) or not isinstance(tokens, list) or len(tokens) != count:
        raise SweepError(f"{base_url}/tokenize returned no consistent token count")
    return count


@dataclass
class FittedContact:
    n_words: int
    pad: int
    target: int
    tokenized: int
    tokenize_calls: int

    @property
    def exact(self) -> bool:
        return self.tokenized == self.target


def fit_contact(count: Callable[[int, int], int], target: int, *, guess_tokens_per_word: float = 1.3,
                max_calls: int = 24) -> FittedContact:
    """Find ``(n_words, pad)`` whose prompt is exactly ``target`` tokens. ``count(n_words, pad)`` is the server's
    tokenize, assumed non-decreasing in ``n_words``.

    Secant steps on the word count inside a shrinking bracket ``(lo, hi)`` -- ``count(lo) <= target < count(hi)`` --
    until a word count hits the target or the bracket closes (one word too many); the remainder is then made up with
    single-letter pad words in the state suffix, which leaves the shared evidence prefix untouched."""

    cache: dict[tuple[int, int], int] = {}

    def measure(n_words: int, pad: int = 0) -> int:
        if (n_words, pad) not in cache:
            cache[(n_words, pad)] = count(n_words, pad)
        return cache[(n_words, pad)]

    overhead = measure(0)
    if overhead > target:
        raise SweepError(f"prompt target {target} is below the fixed overhead of {overhead} tokens")
    if overhead == target:
        return FittedContact(0, 0, target, overhead, len(cache))
    lo, hi = 0, None
    n = max(1, int((target - overhead) / guess_tokens_per_word))
    while len(cache) < max_calls:
        got = measure(n)
        if got == target:
            return FittedContact(n, 0, target, got, len(cache))
        if got < target:
            lo = max(lo, n)
        else:
            hi = n if hi is None else min(hi, n)
        if hi is not None and hi - lo <= 1:
            break
        slope = max((got - overhead) / n, 0.1)
        n = max(lo + 1, n + int(round((target - got) / slope)))
        if hi is not None:
            n = min(hi - 1, n)
    got = measure(lo)
    pad = target - got
    while len(cache) < max_calls + 8 and pad >= 0:
        got = measure(lo, pad)
        if got == target:
            break
        pad += target - got
    return FittedContact(lo, max(pad, 0), target, got, len(cache))


@dataclass
class RequestResult:
    level: int
    worker: int
    instance: int
    trajectory: int
    contact: int
    request_id: str
    prompt_tokens_target: int
    prompt_tokens_tokenized: int
    started_s: float = 0.0
    latency_s: float | None = None
    ttft_s: float | None = None
    status: str = "pending"
    error: str | None = None
    http_status: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    reasoning_chars: int = 0
    content_chars: int = 0

    @property
    def e2e_tok_s(self) -> float | None:
        if self.status != "ok" or not self.latency_s or self.completion_tokens is None:
            return None
        return self.completion_tokens / self.latency_s

    @property
    def decode_tok_s(self) -> float | None:
        if self.status != "ok" or self.ttft_s is None or self.latency_s is None or not self.completion_tokens:
            return None
        span = self.latency_s - self.ttft_s
        return (self.completion_tokens - 1) / span if span > 0 else None


def _delta_text(delta: Mapping[str, object], key: str) -> str:
    value = delta.get(key)
    return value if isinstance(value, str) else ""


def send_chat(base_url: str, body: dict[str, object], result: RequestResult, *, timeout: float, stream: bool) -> None:
    """One chat completion; fills ``result`` in place. A total deadline of ``timeout`` seconds covers connect, prefill
    and every streamed read (the per-request 420 s the flight enforces)."""

    parts = urllib.parse.urlsplit(base_url)
    payload = dict(body, stream=stream)
    if stream:
        payload["stream_options"] = {"include_usage": True}
    data = json.dumps(payload).encode("utf-8")
    start = time.monotonic()
    deadline = start + timeout
    connection = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    try:
        connection.request("POST", "/v1/chat/completions", body=data, headers={
            "Content-Type": "application/json", "X-Request-Id": result.request_id,
        })
        sock = connection.sock
        response = connection.getresponse()
        result.http_status = response.status
        if response.status != 200:
            result.status, result.error = "error", response.read(2000).decode("utf-8", errors="replace")
            return
        usage: Mapping[str, object] = {}
        if not stream:
            document = json.loads(response.read().decode("utf-8"))
            choice = (document.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            result.reasoning_chars = len(_delta_text(message, "reasoning") or _delta_text(message, "reasoning_content"))
            result.content_chars = len(_delta_text(message, "content"))
            result.finish_reason = choice.get("finish_reason")
            usage = document.get("usage") or {}
        else:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                if sock is not None:
                    sock.settimeout(remaining)
                line = response.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == b"[DONE]":
                    break
                event = json.loads(chunk.decode("utf-8"))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    reasoning = _delta_text(delta, "reasoning") or _delta_text(delta, "reasoning_content")
                    content = _delta_text(delta, "content")
                    if (reasoning or content or delta.get("tool_calls")) and result.ttft_s is None:
                        result.ttft_s = time.monotonic() - start
                    result.reasoning_chars += len(reasoning)
                    result.content_chars += len(content)
                    if choice.get("finish_reason"):
                        result.finish_reason = choice["finish_reason"]
        result.latency_s = time.monotonic() - start
        if result.latency_s > timeout:
            raise TimeoutError
        result.prompt_tokens = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None
        completion = usage.get("completion_tokens")
        result.completion_tokens = completion if isinstance(completion, int) else None
        if result.completion_tokens is None:
            result.status, result.error = "error", "response carries no usage.completion_tokens"
        else:
            result.status = "ok"
    except (TimeoutError, socket.timeout):
        result.status, result.latency_s, result.error = "timeout", time.monotonic() - start, f"exceeded {timeout} s"
    except (OSError, http.client.HTTPException, ValueError) as error:
        result.status, result.latency_s, result.error = "error", time.monotonic() - start, repr(error)[:500]
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def parse_prometheus(text: str) -> dict[str, float]:
    """``{metric name: sum over its label sets}`` from Prometheus text exposition (comments skipped)."""

    totals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)[0], line.rsplit("}", 1)[-1]
        else:
            name, _, rest = line.partition(" ")
        fields = rest.split()
        if not fields:
            continue
        try:
            value = float(fields[0])
        except ValueError:
            continue
        totals[name.strip()] = totals.get(name.strip(), 0.0) + value
    return totals


def read_metrics(base_url: str) -> dict[str, float | None] | None:
    try:
        totals = parse_prometheus(_http_text(base_url + "/metrics"))
    except (OSError, ValueError):
        return None
    out: dict[str, float | None] = {}
    for key, names in METRIC_COUNTERS.items():
        out[key] = next((totals[name] for name in names if name in totals), None)
    for key, name in METRIC_GAUGES.items():
        out[key] = totals.get(name)
    return out


class MetricsSampler(threading.Thread):
    """Samples the gauges of every instance every ``interval`` seconds until stopped."""

    def __init__(self, base_urls: Sequence[str], interval: float) -> None:
        super().__init__(daemon=True)
        self.base_urls, self.interval = list(base_urls), interval
        self.samples: list[list[dict[str, float | None] | None]] = [[] for _ in base_urls]
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            for index, url in enumerate(self.base_urls):
                self.samples[index].append(read_metrics(url))
            self.stop_event.wait(self.interval)

    def summary(self) -> list[dict[str, object]]:
        out = []
        for samples in self.samples:
            readable = [sample for sample in samples if sample]
            entry: dict[str, object] = {"samples": len(samples), "readable": len(readable)}
            for key in METRIC_GAUGES:
                values = [sample[key] for sample in readable if sample.get(key) is not None]
                entry[key] = {"max": max(values), "mean": sum(values) / len(values)} if values else None
            out.append(entry)
        return out


def probe_endpoint(base_url: str, model: str, timeout: float = 10.0) -> str | None:
    """``None`` when ``GET /v1/models`` answers 200 and lists ``model``; otherwise a short reason string."""

    try:
        document = _http_json("GET", base_url + "/v1/models", timeout=timeout)
    except (OSError, ValueError) as error:  # URLError, HTTPError (non-200), timeouts, refused, bad JSON
        return f"{type(error).__name__}: {error}"[:200]
    entries = document.get("data") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not any(isinstance(e, dict) and e.get("id") == model for e in entries):
        return f"/v1/models does not list {model!r}"
    return None


class LivenessPoller(threading.Thread):
    """Polls every instance's ``/v1/models`` every ``interval`` seconds (first poll immediately) until stopped.

    An endpoint is DEAD when ``attempts`` probes in a row (``retry_gap`` seconds apart) all fail, so one slow answer
    under load is not a death. The first death is recorded in :attr:`died` and :attr:`dead_event` is set; the thread
    then stops (the level is aborted by its owner). ``probe`` is injectable for tests."""

    def __init__(self, base_urls: Sequence[str], model: str, interval: float, *, timeout: float = 10.0,
                 attempts: int = 2, retry_gap: float | None = None,
                 probe: Callable[[str, str, float], str | None] | None = None) -> None:
        super().__init__(daemon=True)
        self.base_urls, self.model, self.interval = list(base_urls), model, interval
        self.timeout, self.attempts = timeout, max(1, attempts)
        self.retry_gap = LIVENESS_RETRY_GAP_S if retry_gap is None else retry_gap
        self.probe = probe or probe_endpoint
        self.polls = 0
        self.died: dict[str, object] | None = None
        self.dead_event = threading.Event()
        self.stop_event = threading.Event()
        self._t0 = time.monotonic()

    def _check(self, index: int) -> dict[str, object] | None:
        error: str | None = None
        for attempt in range(self.attempts):
            error = self.probe(self.base_urls[index], self.model, self.timeout)
            if error is None:
                return None
            if attempt + 1 < self.attempts:
                time.sleep(self.retry_gap)
        return {"instance": index, "base_url": self.base_urls[index], "error": error}

    def poll_once(self, detected_by: str = "poll") -> dict[str, object] | None:
        """One round over all instances (in parallel); the ``endpoint_died`` record if any is dead, else ``None``."""

        with ThreadPoolExecutor(max_workers=len(self.base_urls)) as pool:
            verdicts = list(pool.map(self._check, range(len(self.base_urls))))
        self.polls += 1
        dead = [verdict for verdict in verdicts if verdict]
        if not dead:
            return None
        self.died = {"detected_by": detected_by, "detected_at_s": round(time.monotonic() - self._t0, 3),
                     "endpoints": dead}
        self.dead_event.set()
        return self.died

    def run(self) -> None:
        while not self.stop_event.is_set():
            if self.poll_once():
                return
            self.stop_event.wait(self.interval)


# ---------------------------------------------------------------------------
# Statistics and the knee.
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in [0, 100]); None for no values."""

    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q / 100.0
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _spread(values: Sequence[float]) -> dict[str, float | None]:
    return {"p10": percentile(values, 10), "p50": percentile(values, 50), "p95": percentile(values, 95),
            "max": max(values) if values else None, "n": len(values)}


def summarize_level(level: int, results: Sequence[RequestResult], wall_s: float,
                    before: Sequence[Mapping[str, float | None] | None],
                    after: Sequence[Mapping[str, float | None] | None],
                    gauges: list[dict[str, object]]) -> dict[str, object]:
    ok = [result for result in results if result.status == "ok"]
    output_tokens = sum(result.completion_tokens or 0 for result in ok)
    deltas: dict[str, float | None] = {}
    for key in METRIC_COUNTERS:
        pairs = [(b.get(key) if b else None, a.get(key) if a else None) for b, a in zip(before, after)]
        deltas[key] = None if any(b is None or a is None for b, a in pairs) else sum(a - b for b, a in pairs)
    targets = [result.prompt_tokens_target for result in results]
    return {
        "concurrency_per_instance": level,
        "workers": len({result.worker for result in results}),
        "requests": len(results),
        "ok": len(ok),
        "errors": sum(result.status == "error" for result in results),
        "timeouts": sum(result.status == "timeout" for result in results),
        "wall_s": wall_s,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / wall_s if wall_s > 0 else None,
        "prompt_tokens_target": {"mean": sum(targets) / len(targets) if targets else None,
                                 "min": min(targets, default=None), "max": max(targets, default=None)},
        "prompt_tokens_exact": all(result.prompt_tokens_target == result.prompt_tokens_tokenized for result in results),
        "prompt_tokens_server_matches_tokenize": all(
            result.prompt_tokens == result.prompt_tokens_tokenized for result in ok if result.prompt_tokens is not None
        ),
        "completion_tokens": _spread([float(result.completion_tokens) for result in ok if result.completion_tokens is not None]),
        "e2e_tok_s": _spread([value for value in (result.e2e_tok_s for result in ok) if value is not None]),
        "decode_tok_s": _spread([value for value in (result.decode_tok_s for result in ok) if value is not None]),
        "latency_s": _spread([result.latency_s for result in ok if result.latency_s is not None]),
        "ttft_s": _spread([result.ttft_s for result in ok if result.ttft_s is not None]),
        "metrics_before": list(before),
        "metrics_after": list(after),
        "metrics_delta": deltas,
        "preemptions_delta": deltas["preemptions"],
        "gauges": gauges,
    }


def recommend_knee(levels: Sequence[Mapping[str, object]], *, min_seq_tok_s: float) -> dict[str, object]:
    """The highest concurrency that passes every gate (module docstring); each failing level says why."""

    verdicts: dict[str, list[str]] = {}
    passing: list[Mapping[str, object]] = []
    for level in levels:
        reasons = []
        if level.get("endpoint_died"):
            reasons.append("endpoint died")
        if level["errors"]:
            reasons.append(f"{level['errors']} error(s)")
        if level["timeouts"]:
            reasons.append(f"{level['timeouts']} timeout(s)")
        if level["preemptions_delta"] is None:
            reasons.append("preemption counter unreadable")
        elif level["preemptions_delta"] > 0:
            reasons.append(f"{level['preemptions_delta']:g} preemption(s)")
        p10 = level["e2e_tok_s"]["p10"]  # type: ignore[index]
        if p10 is None or p10 < min_seq_tok_s:
            reasons.append(f"per-sequence e2e tok/s p10 {p10} < {min_seq_tok_s}")
        verdicts[str(level["concurrency_per_instance"])] = reasons
        if not reasons:
            passing.append(level)
    knee = max(passing, key=lambda level: level["concurrency_per_instance"], default=None)
    best = max(passing, key=lambda level: level["output_tok_s"] or 0.0, default=None)
    return {
        "rule": (f"highest concurrency per instance with zero errors, zero timeouts, a readable zero preemption delta "
                 f"and per-sequence end-to-end tok/s p10 >= {min_seq_tok_s}"),
        "recommended_concurrency_per_instance": knee["concurrency_per_instance"] if knee else None,
        "recommended_output_tok_s": knee["output_tok_s"] if knee else None,
        "highest_throughput_passing_concurrency": best["concurrency_per_instance"] if best else None,
        "failing_reasons": verdicts,
    }


# ---------------------------------------------------------------------------
# The sweep.
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    base_urls: list[str]
    model: str
    levels: list[int]
    seed: str = "20261028"
    requests_per_level: int | None = None
    contacts_per_trajectory: int = 3
    prompt_mean: int = 8500
    prompt_min: int = 3000
    prompt_max: int = 20000
    prompt_sigma: float = 0.5
    audio_min_seconds: float = 2.0
    audio_max_seconds: float = 8.0
    max_tokens: int = 4096
    thinking_budget: int = 3072
    min_tokens: int | None = None
    decoding_seed: int = 20260913
    timeout_s: float = 420.0
    min_seq_tok_s: float = 10.0
    stream: bool = True
    metrics_interval_s: float = 5.0
    warmup: int = 1
    fit_workers: int = 16
    idle_wait_s: float = 120.0
    liveness_interval_s: float = 30.0
    liveness_timeout_s: float = 10.0
    label: str | None = None
    deployment_profile: dict[str, object] | None = field(default=None, repr=False)

    def validate(self) -> None:
        if not self.base_urls or len(set(self.base_urls)) != len(self.base_urls):
            raise SweepError("give one or more distinct --base-url")
        if not self.levels or any(level <= 0 for level in self.levels) or sorted(set(self.levels)) != self.levels:
            raise SweepError(f"levels must be strictly increasing positive integers: {self.levels}")
        if self.contacts_per_trajectory <= 0 or self.max_tokens <= 0 or not 0 <= self.thinking_budget <= self.max_tokens:
            raise SweepError("contacts, max_tokens and thinking budget are inconsistent")
        if not 0 < self.audio_min_seconds <= self.audio_max_seconds:
            raise SweepError("audio seconds range is invalid")
        if self.liveness_interval_s < 0 or self.liveness_timeout_s <= 0:
            raise SweepError("liveness interval must be >= 0 (0 disables) and the liveness timeout > 0")
        lognormal_mu_for_mean(self.prompt_mean, self.prompt_sigma, self.prompt_min, self.prompt_max)


def _decoding(config: SweepConfig, max_tokens: int | None = None) -> dict[str, object]:
    body: dict[str, object] = {
        "model": config.model, "temperature": 0, "seed": config.decoding_seed,
        "max_tokens": max_tokens or config.max_tokens, "chat_template_kwargs": {"enable_thinking": True},
    }
    if config.thinking_budget:
        body["thinking_token_budget"] = min(config.thinking_budget, int(body["max_tokens"]))  # type: ignore[arg-type]
    if config.min_tokens is not None and max_tokens is None:
        body["min_tokens"] = config.min_tokens
    return body


def preflight(config: SweepConfig) -> list[dict[str, object]]:
    """Every base URL must serve ``config.model``; its ``max_model_len`` (when reported) must hold the largest prompt
    plus ``max_tokens``."""

    served = []
    for url in config.base_urls:
        try:
            document = _http_json("GET", url + "/v1/models", timeout=10)
        except (OSError, ValueError) as error:
            raise SweepError(f"{url}/v1/models unreachable: {error}") from error
        entries = [entry for entry in (document.get("data") or []) if entry.get("id") == config.model]  # type: ignore[union-attr]
        if not entries:
            raise SweepError(f"{url} does not serve model {config.model!r}")
        window = entries[0].get("max_model_len")
        if isinstance(window, int) and config.prompt_max + config.max_tokens > window:
            raise SweepError(f"{url}: prompt max {config.prompt_max} + max_tokens {config.max_tokens} > window {window}")
        served.append({"base_url": url, "max_model_len": window})
    return served


def build_level(config: SweepConfig, level: int) -> tuple[int, list[Trajectory]]:
    workers = level * len(config.base_urls)
    contacts = config.contacts_per_trajectory
    requests = config.requests_per_level or workers * contacts
    if requests < workers * contacts:
        raise SweepError(f"level {level}: {requests} requests cannot keep {workers} workers busy "
                         f"({contacts} contact(s) per trajectory); need >= {workers * contacts}")
    rng = random.Random(f"level:{config.seed}:{level}")
    mu = lognormal_mu_for_mean(config.prompt_mean, config.prompt_sigma, config.prompt_min, config.prompt_max)
    trajectories = []
    for index in range(math.ceil(requests / contacts)):
        targets = sample_prompt_targets(rng, contacts, mu=mu, sigma=config.prompt_sigma,
                                        low=config.prompt_min, high=config.prompt_max)
        seconds = round(rng.uniform(config.audio_min_seconds, config.audio_max_seconds), 3)
        trajectories.append(Trajectory(index, f"{config.seed}:{level}:{index}", seconds, tuple(sorted(targets))))
    return workers, trajectories


def fit_trajectory(config: SweepConfig, trajectory: Trajectory, base_url: str) -> list[FittedContact]:
    audio = trajectory.audio_b64()
    words = trajectory.words(max(trajectory.targets) + 64)
    contacts = len(trajectory.targets)
    fitted = []
    for contact, target in enumerate(trajectory.targets):
        def count(n_words: int, pad: int) -> int:
            if n_words > len(words):
                words.extend(trajectory.words(n_words * 2)[len(words):])
            return tokenize_count(base_url, config.model,
                                  contact_messages(audio, words, n_words, contact, contacts, pad))
        fitted.append(fit_contact(count, target))
    return fitted


def _wait_idle(config: SweepConfig) -> bool:
    deadline = time.monotonic() + config.idle_wait_s
    while True:
        samples = [read_metrics(url) for url in config.base_urls]
        busy = [s for s in samples if s and ((s.get("num_requests_running") or 0) + (s.get("num_requests_waiting") or 0)) > 0]
        if not busy or time.monotonic() > deadline:
            return not busy
        time.sleep(2)


def warmup(config: SweepConfig) -> None:
    for index, url in enumerate(config.base_urls):
        for k in range(config.warmup):
            trajectory = Trajectory(k, f"warmup:{config.seed}:{index}:{k}", config.audio_min_seconds, (0,))
            messages = contact_messages(trajectory.audio_b64(), trajectory.words(64), 64, 0, 1, 0)
            result = RequestResult(-1, index, index, k, 0, f"sweep-warmup-i{index}-{k}", 0, 0)
            send_chat(url, dict(_decoding(config, max_tokens=16), messages=messages), result,
                      timeout=config.timeout_s, stream=config.stream)
            if result.status != "ok":
                raise SweepError(f"warmup on {url} failed: {result.status} {result.error}")


def run_level(config: SweepConfig, level: int) -> tuple[dict[str, object], list[RequestResult]]:
    instances = len(config.base_urls)
    workers, trajectories = build_level(config, level)
    owner = {t.index: t.index % workers for t in trajectories}
    with ThreadPoolExecutor(max_workers=config.fit_workers) as pool:
        fitted = list(pool.map(
            lambda t: fit_trajectory(config, t, config.base_urls[owner[t.index] % instances]), trajectories
        ))
    idle = _wait_idle(config)
    results: list[RequestResult] = []
    lock = threading.Lock()
    barrier = threading.Barrier(workers + 1)
    clock: dict[str, float] = {}
    abort = threading.Event()

    def worker(w: int) -> None:
        instance = w % instances
        url = config.base_urls[instance]
        mine = [t for t in trajectories if owner[t.index] == w]
        barrier.wait()
        for trajectory in mine:
            audio = trajectory.audio_b64()
            fits = fitted[trajectory.index]
            words = trajectory.words(max(fit.n_words for fit in fits))
            for contact, fit in enumerate(fits):
                if abort.is_set():
                    return
                result = RequestResult(level, w, instance, trajectory.index, contact,
                                       f"sweep-c{level}-w{w}-t{trajectory.index}-k{contact}", fit.target, fit.tokenized)
                result.started_s = time.monotonic() - clock["start"]
                body = dict(_decoding(config), messages=contact_messages(
                    audio, words, fit.n_words, contact, len(fits), fit.pad))
                send_chat(url, body, result, timeout=config.timeout_s, stream=config.stream)
                with lock:
                    results.append(result)

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(workers)]
    for thread in threads:
        thread.start()
    before = [read_metrics(url) for url in config.base_urls]
    sampler = MetricsSampler(config.base_urls, config.metrics_interval_s)
    sampler.start()
    poller: LivenessPoller | None = None
    if config.liveness_interval_s > 0:
        poller = LivenessPoller(config.base_urls, config.model, config.liveness_interval_s,
                                timeout=config.liveness_timeout_s)
        poller.start()
    clock["start"] = time.monotonic()
    barrier.wait()
    died: dict[str, object] | None = None
    while any(thread.is_alive() for thread in threads):
        if poller is not None and poller.dead_event.wait(0.25):
            died = poller.died
            abort.set()  # blocked requests to the dead endpoint are abandoned (daemon threads), not awaited
            break
        if poller is None:
            time.sleep(0.25)
    wall = time.monotonic() - clock["start"]
    if poller is not None:
        if died is None:  # a death between two polls, or after the last poll, is still a death
            died = poller.poll_once("level-end-check")
        poller.stop_event.set()
    sampler.stop_event.set()
    sampler.join()
    after = [read_metrics(url) for url in config.base_urls]
    with lock:
        finished = list(results)  # snapshot: abandoned workers of an aborted level may still append to ``results``
    finished.sort(key=lambda r: (r.worker, r.trajectory, r.contact))
    summary = summarize_level(level, finished, wall, before, after, sampler.summary())
    summary["endpoint_died"] = died
    summary["liveness"] = {"interval_s": config.liveness_interval_s, "polls": poller.polls if poller else 0}
    summary["aborted"] = abort.is_set()
    summary["instances_idle_at_start"] = idle
    summary["fit"] = {
        "contacts": sum(len(f) for f in fitted),
        "exact": sum(fit.exact for fits in fitted for fit in fits),
        "tokenize_calls": sum(fit.tokenize_calls for fits in fitted for fit in fits),
    }
    return summary, finished


def run_sweep(config: SweepConfig, *, log: Callable[[str], None] = lambda line: None) -> dict[str, object]:
    config.validate()
    served = preflight(config)
    if config.warmup:
        warmup(config)
    levels, requests = [], []
    endpoint_died: dict[str, object] | None = None
    for level in config.levels:
        log(f"level {level}/instance x {len(config.base_urls)} instance(s): fitting and running")
        summary, results = run_level(config, level)
        levels.append(summary)
        requests.extend(asdict(result) | {"e2e_tok_s": result.e2e_tok_s, "decode_tok_s": result.decode_tok_s}
                        for result in results)
        log(f"level {level}: {summary['ok']}/{summary['requests']} ok, {summary['output_tok_s']} out tok/s, "
            f"p10 e2e {summary['e2e_tok_s']['p10']} tok/s, preemptions {summary['preemptions_delta']}")
        if summary["endpoint_died"]:
            endpoint_died = dict(summary["endpoint_died"], level=level)  # type: ignore[arg-type]
            log(f"level {level}: ENDPOINT DIED {json.dumps(endpoint_died, sort_keys=True)}; sweep stopped")
            break
    settings = {key: value for key, value in asdict(config).items() if key != "deployment_profile"}
    profile = config.deployment_profile
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": config.label,
        "synthetic": True,
        "dataset_rows": 0,
        "settings": settings,
        "served": served,
        "deployment_profile": None if profile is None else {
            "sha256": hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "document": profile,
        },
        "endpoint_died": endpoint_died,
        "levels_requested": list(config.levels),
        "levels": levels,
        "knee": recommend_knee(levels, min_seq_tok_s=config.min_seq_tok_s),
        "requests": requests,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", action="append", required=True, help="one instance; repeat or comma-separate")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--levels", default="8,16,24,32,48", help="client concurrency PER INSTANCE")
    parser.add_argument("--requests-per-level", type=int)
    parser.add_argument("--contacts-per-trajectory", type=int, default=3)
    parser.add_argument("--seed", default="20261028")
    parser.add_argument("--prompt-mean", type=int, default=8500)
    parser.add_argument("--prompt-min", type=int, default=3000)
    parser.add_argument("--prompt-max", type=int, default=20000)
    parser.add_argument("--prompt-sigma", type=float, default=0.5)
    parser.add_argument("--audio-min-seconds", type=float, default=2.0)
    parser.add_argument("--audio-max-seconds", type=float, default=8.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking-budget", type=int, default=3072)
    parser.add_argument("--min-tokens", type=int, help="optional vLLM min_tokens floor to force long outputs")
    parser.add_argument("--decoding-seed", type=int, default=20260913)
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--min-seq-tok-s", type=float, default=10.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--metrics-interval", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=1, help="short unmeasured requests per instance first")
    parser.add_argument("--fit-workers", type=int, default=16)
    parser.add_argument("--liveness-interval", type=float, default=30.0,
                        help="seconds between /v1/models polls of every instance during a level (0 disables)")
    parser.add_argument("--liveness-timeout", type=float, default=10.0, help="timeout of one liveness probe")
    parser.add_argument("--deployment-profile", type=Path, help="embed this deployment profile JSON in the report")
    parser.add_argument("--label")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    urls = [url.rstrip("/") for chunk in args.base_url for url in chunk.split(",") if url.strip()]
    config = SweepConfig(
        base_urls=urls, model=args.model, levels=[int(level) for level in args.levels.split(",")], seed=args.seed,
        requests_per_level=args.requests_per_level, contacts_per_trajectory=args.contacts_per_trajectory,
        prompt_mean=args.prompt_mean, prompt_min=args.prompt_min, prompt_max=args.prompt_max,
        prompt_sigma=args.prompt_sigma, audio_min_seconds=args.audio_min_seconds,
        audio_max_seconds=args.audio_max_seconds, max_tokens=args.max_tokens, thinking_budget=args.thinking_budget,
        min_tokens=args.min_tokens, decoding_seed=args.decoding_seed, timeout_s=args.timeout,
        min_seq_tok_s=args.min_seq_tok_s, stream=not args.no_stream, metrics_interval_s=args.metrics_interval,
        warmup=args.warmup, fit_workers=args.fit_workers, label=args.label,
        liveness_interval_s=args.liveness_interval, liveness_timeout_s=args.liveness_timeout,
        deployment_profile=(json.loads(args.deployment_profile.read_text(encoding="utf-8"))
                            if args.deployment_profile else None),
    )
    report = run_sweep(config, log=lambda line: print(line, file=sys.stderr, flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    knee = report["knee"]
    print(json.dumps({"out": str(args.out), "knee": knee, "endpoint_died": report["endpoint_died"]},
                     indent=2, sort_keys=True))
    return EXIT_ENDPOINT_DIED if report["endpoint_died"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SweepError as error:
        sys.exit(f"refused: {error}")
