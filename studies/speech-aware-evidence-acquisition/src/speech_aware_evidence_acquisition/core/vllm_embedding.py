"""The study's speech/passage embedding recipe, computed by a vLLM pooling server over a serving view.

Port note (npu-vllm-port, 2026-10-28). ``core.speech_embedding.encode`` runs the pinned encoder in process: render the
sentence-transformers chat template, one forward pass, pool the last hidden state over every prompt token, L2
normalize. On the NPU stack the same recipe runs inside ``vllm serve --runner pooling --convert embed`` over a serving
view (``core.vllm_serving_view``), and this module is the client:

* the request is a chat-form ``/v1/embeddings`` body whose content parts are the recipe's own parts in the same order
  (``[text prompt, audio]`` for a speech query, ``[text prompt + passage]`` for a passage); the server renders them
  with the view's pinned template (``--chat-template``) and no generation prompt;
* the audio is the exact waveform the in-process recipe would feed its processor (the study's
  ``read_audio_span`` output, mono float32 at 16 kHz), sent as a 32-bit float WAV, which is lossless;
* the response is refused unless it carries exactly one finite vector of the recipe's dimension whose L2 norm is 1
  within ``NORM_TOLERANCE`` (the server-side normalization actually happened), and, when a bound is given, a prompt no
  longer than the registered sequence-token bound;
* before the first request the client verifies the serving view against its pinned receipt hash, requires every
  source file the view was built from to equal the pinned encoder inventory, and requires the server's
  ``/v1/models`` to list exactly the served model name rooted at that view directory.

Known numerical difference, declared rather than hidden: attention is causal (see ``core.vllm_serving_view``), and
vLLM's kernels are not bit-identical to the HF ``sdpa`` forward, so vectors are compared by ranking agreement, never
by equality, with the historical sealed vectors.
"""

from __future__ import annotations

import base64
import io
import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .vllm_serving_view import verify_serving_view

SAMPLE_RATE_HZ = 16000
NORM_TOLERANCE = 1e-3
EMBEDDINGS_PATH = "/v1/embeddings"
MODELS_PATH = "/v1/models"


class VllmEmbeddingError(ValueError):
    """Refuse an unbound server, an unverifiable view or a malformed embedding."""


def wav_bytes(wave: Sequence[float], sample_rate_hz: int = SAMPLE_RATE_HZ) -> bytes:
    """``wave`` (mono) as a 32-bit float WAV: every sample survives the round trip unchanged."""

    import numpy as np
    import soundfile

    array = np.asarray(wave, dtype=np.float32)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise VllmEmbeddingError("speech waveform must be finite, non-empty mono audio")
    buffer = io.BytesIO()
    soundfile.write(buffer, array, sample_rate_hz, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def speech_content(prompt: str, wave: Sequence[float]) -> list[dict[str, object]]:
    """The speech-query parts, in the recipe's order: the prompt text, then the audio."""

    data = base64.b64encode(wav_bytes(wave)).decode("ascii")
    return [
        {"type": "text", "text": prompt},
        {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
    ]


def text_content(text: str) -> list[dict[str, object]]:
    return [{"type": "text", "text": text}]


def embedding_request(served_model_name: str, content: list[dict[str, object]]) -> dict[str, object]:
    return {
        "model": served_model_name,
        "messages": [{"role": "user", "content": content}],
        "encoding_format": "float",
        "add_generation_prompt": False,
        "add_special_tokens": False,
    }


def parse_embedding_response(
    raw: bytes, *, dimensions: int, normalized: bool, max_prompt_tokens: int | None = None
) -> tuple[float, ...]:
    """The one vector of a ``/v1/embeddings`` response, or a refusal (module docstring)."""

    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise VllmEmbeddingError(f"embedding response is not JSON: {error}") from error
    data = document.get("data") if isinstance(document, dict) else None
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise VllmEmbeddingError("embedding response must carry exactly one embedding")
    vector = data[0].get("embedding")
    if not isinstance(vector, list) or len(vector) != dimensions:
        raise VllmEmbeddingError(
            f"embedding has {len(vector) if isinstance(vector, list) else 'no'} dimensions, expected {dimensions}"
        )
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector):
        raise VllmEmbeddingError("embedding values must be numbers")
    values = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in values):
        raise VllmEmbeddingError("embedding is not finite")
    if normalized and abs(math.sqrt(sum(value * value for value in values)) - 1.0) > NORM_TOLERANCE:
        raise VllmEmbeddingError("embedding is not L2-normalized; the server did not apply the recipe's activation")
    if max_prompt_tokens is not None:
        usage = document.get("usage")
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if type(prompt_tokens) is not int:
            raise VllmEmbeddingError("embedding response does not report prompt_tokens")
        if prompt_tokens > max_prompt_tokens:
            raise VllmEmbeddingError(
                f"embedding prompt has {prompt_tokens} tokens, above the registered bound {max_prompt_tokens}"
            )
    return values


def _default_post(timeout_seconds: float) -> Callable[[str, bytes], bytes]:
    def post(url: str, body: bytes) -> bytes:
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 (loopback only)
            return response.read()

    return post


def _default_get(timeout_seconds: float) -> Callable[[str], bytes]:
    def get(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310 (loopback only)
            return response.read()

    return get


def assert_view_matches_inventory(receipt: Mapping[str, object], inventory: Mapping[str, object]) -> None:
    """Every source file the view was built from must be a file of the pinned encoder inventory, with its hash."""

    pinned = inventory.get("files")
    if not isinstance(pinned, Mapping):
        raise VllmEmbeddingError("encoder inventory lists no files")
    for view_name, entry in receipt["files"].items():
        source_name = entry.get("source", view_name)
        expected = pinned.get(source_name)
        if expected is None:
            raise VllmEmbeddingError(f"serving view source file {source_name!r} is not in the pinned inventory")
        if (expected.get("sha256"), expected.get("bytes")) != (entry["source_sha256"], entry["source_bytes"]):
            raise VllmEmbeddingError(f"serving view source file {source_name!r} differs from the pinned inventory")


class VllmEmbeddingClient:
    """One verified vLLM pooling server serving one verified serving view."""

    def __init__(
        self,
        base_url: str,
        *,
        served_model_name: str,
        view: Path,
        expected_receipt_sha256: str,
        inventory: Mapping[str, object] | None = None,
        timeout_seconds: float = 120.0,
        post: Callable[[str, bytes], bytes] | None = None,
        get: Callable[[str], bytes] | None = None,
    ) -> None:
        if not served_model_name:
            raise VllmEmbeddingError("served_model_name must be non-empty")
        self.base_url = base_url.rstrip("/")
        self.served_model_name = served_model_name
        self.view = Path(view)
        self.receipt = verify_serving_view(self.view, expected_receipt_sha256=expected_receipt_sha256)
        if inventory is not None:
            assert_view_matches_inventory(self.receipt, inventory)
        self.recipe = self.receipt["recipe"]
        self._post = post or _default_post(timeout_seconds)
        self._get = get or _default_get(timeout_seconds)
        self._assert_server_serves_view()

    def _assert_server_serves_view(self) -> None:
        try:
            document = json.loads(self._get(self.base_url + MODELS_PATH).decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise VllmEmbeddingError(f"cannot read the server's model list: {error}") from error
        models = document.get("data") if isinstance(document, dict) else None
        if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
            raise VllmEmbeddingError("the embedding server must serve exactly one model")
        model = models[0]
        if model.get("id") != self.served_model_name:
            raise VllmEmbeddingError(f"server serves {model.get('id')!r}, not {self.served_model_name!r}")
        if model.get("root") != str(self.view):
            raise VllmEmbeddingError(f"server model root {model.get('root')!r} is not the serving view {self.view}")

    def embed(self, content: list[dict[str, object]], *, max_prompt_tokens: int | None = None) -> tuple[float, ...]:
        body = json.dumps(embedding_request(self.served_model_name, content)).encode("utf-8")
        try:
            raw = self._post(self.base_url + EMBEDDINGS_PATH, body)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise VllmEmbeddingError(f"embedding server answered HTTP {error.code}: {detail}") from error
        return parse_embedding_response(
            raw, dimensions=int(self.recipe["dimensions"]), normalized=bool(self.recipe["normalize"]),
            max_prompt_tokens=max_prompt_tokens,
        )

    def embed_speech(
        self, prompt: str, wave: Sequence[float], *, max_audio_samples: int, max_prompt_tokens: int | None = None
    ) -> tuple[float, ...]:
        if type(max_audio_samples) is not int or max_audio_samples <= 0:
            raise VllmEmbeddingError("a speech embedding requires an explicit positive sample bound")
        if len(wave) > max_audio_samples:
            raise VllmEmbeddingError("speech waveform exceeds the registered input bound")
        return self.embed(speech_content(prompt, wave), max_prompt_tokens=max_prompt_tokens)

    def embed_text(self, text: str, *, max_prompt_tokens: int | None = None) -> tuple[float, ...]:
        return self.embed(text_content(text), max_prompt_tokens=max_prompt_tokens)


class VllmSpeechEncoder:
    """Drop-in for ``speech_preparation_runtime.NemotronSpeechEncoder`` backed by a :class:`VllmEmbeddingClient`."""

    def __init__(self, client: VllmEmbeddingClient, inventory, protocol, data_root, event):
        from .speech_preparation_runtime import BoundedAudioPrefetch

        assert_view_matches_inventory(client.receipt, inventory)
        self.client, self.protocol, self.event = client, protocol, event
        self.prefetch = BoundedAudioPrefetch(protocol["cpu_workers"])

    def prepare_sources(self, sources):
        self.prefetch.prepare_sources(sources)

    def __call__(self, source):
        self.event("data_wait_started")
        wave = self.prefetch.take(source)
        self.event("data_wait_finished")
        self.event("encoder_forward_started")
        vector = self.client.embed_speech(
            self.protocol["query_prompt"], wave,
            max_audio_samples=self.protocol["max_audio_samples"],
            max_prompt_tokens=self.protocol["max_sequence_tokens"],
        )
        self.event("encoder_forward_finished")
        return vector

    def close(self):
        self.prefetch.close()


class VllmPassageEncoder:
    """Drop-in for ``speech_preparation_runtime.NemotronPassageEncoder`` backed by a :class:`VllmEmbeddingClient`."""

    def __init__(self, client: VllmEmbeddingClient, inventory, protocol, data_root, event):
        assert_view_matches_inventory(client.receipt, inventory)
        self.client, self.protocol, self.event = client, protocol, event

    def __call__(self, text):
        self.event("encoder_forward_started")
        vector = self.client.embed_text(
            self.protocol["passage_prompt"] + text, max_prompt_tokens=self.protocol["max_sequence_tokens"]
        )
        self.event("encoder_forward_finished")
        return vector

    def close(self):
        pass


def vllm_backend_factory(client: VllmEmbeddingClient, encoder_cls=VllmSpeechEncoder):
    """A ``backend_factory`` for ``speech_preparation_runtime.run_preparation`` / the passage run."""

    def factory(inventory, protocol, data_root, event):
        return encoder_cls(client, inventory, protocol, data_root, event)

    return factory
