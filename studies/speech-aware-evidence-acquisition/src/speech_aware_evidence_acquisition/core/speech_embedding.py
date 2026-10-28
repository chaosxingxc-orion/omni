"""Shared pinned Omni-Embed recipe and exact source-frame audio observation."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from .preparation import hash_stable_file

DIMS = 2048
SAMPLE_RATE_HZ = 16000


class SpeechEmbeddingError(ValueError):
    """Refuse an unbound, truncated or malformed embedding input."""


def read_audio_span(source):
    """Read exact original frames and apply the established mono/resample recipe."""
    import soundfile

    path = Path(source["media_path"])
    before = path.stat()
    fingerprint = (before.st_ino, before.st_size, before.st_mtime_ns)
    bound = source.get("verified_fingerprint")
    if bound is not None and tuple(bound) != fingerprint:
        raise SpeechEmbeddingError("speech source changed after hash verification")
    if bound is None and source.get("audio_sha256") is not None:
        _size, sha, _fingerprint = hash_stable_file(path, "speech embedding input")
        if sha != source["audio_sha256"]:
            raise SpeechEmbeddingError("speech source hash changed")
    start, end = source["start_frame"], source["end_frame"]
    with soundfile.SoundFile(path) as audio:
        if (
            audio.frames != source["source_frames"]
            or audio.samplerate != source["source_sample_rate_hz"]
            or not 0 <= start < end <= audio.frames
        ):
            raise SpeechEmbeddingError("speech source frame geometry changed")
        audio.seek(start)
        wave = audio.read(end - start, dtype="float32", always_2d=False)
        rate = audio.samplerate
    after = path.stat()
    if (after.st_ino, after.st_size, after.st_mtime_ns) != fingerprint:
        raise SpeechEmbeddingError("speech source changed during decoding")
    array = np.asarray(wave, dtype=np.float32)
    if array.shape[0] != end - start:
        raise SpeechEmbeddingError("speech decoder omitted source frames")
    if array.ndim > 1:
        array = array.mean(axis=1).astype(np.float32)
    if rate != SAMPLE_RATE_HZ:
        import librosa

        array = librosa.resample(array, orig_sr=rate, target_sr=SAMPLE_RATE_HZ).astype(
            np.float32
        )
    if not np.isfinite(array).all():
        raise SpeechEmbeddingError("speech waveform is not finite")
    return array


#: Encoder runtimes: the historical recipe (CUDA, bfloat16) and the port's CPU runtime. The CPU runtime is
#: float32 because bfloat16 is software-emulated on CPUs without bf16 instructions (Kunpeng-920: 5-10x slower).
ENCODER_RUNTIMES = {"cuda-bfloat16": ("cuda", "bfloat16"), "cpu-float32": ("cpu", "float32")}

#: Attention geometry of the text tower. As shipped, ``modeling_nv_omni_embed.py`` sets ``is_causal=False`` on
#: every text layer and overrides ``_update_causal_mask``; on transformers 5.x the override is never called, and
#: the flag reaches sdpa only when the mask is skipped. Measured 2026-10-28 (CPU float32, transformers 5.14.1):
#: batch one -> bidirectional; a padded batch -> causal. ``None`` keeps that as-shipped behaviour (the historical
#: recipe); ``"causal"`` forces ``is_causal=True`` on every text layer, which is causal under both mask paths.
ATTENTION_GEOMETRIES = (None, "causal")


def _text_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise SystemExit("the encoder has no text tower layers to set an attention geometry on")
    return layers


def load_stack(
    model_dir: Path,
    *,
    local_files_only: bool = False,
    runtime: str = "cuda-bfloat16",
    attention_geometry: str | None = None,
):
    """The three shipped modules, reproduced through plain ``transformers``.

    ``runtime`` names one entry of :data:`ENCODER_RUNTIMES`; the default is the historical recipe.
    ``attention_geometry`` is one of :data:`ATTENTION_GEOMETRIES`; the default is as shipped.
    """

    import torch
    from transformers import AutoModel, AutoProcessor

    if "sentence_transformers" in sys.modules:  # pragma: no cover - defensive
        raise SystemExit("sentence_transformers must not be imported in this process")
    template = (
        model_dir / "additional_chat_templates/sentence_transformers.jinja"
    ).read_text(encoding="utf-8")
    pooling = json.loads(
        (model_dir / "1_Pooling/config.json").read_text(encoding="utf-8")
    )
    if pooling.get("pooling_mode") != "mean" or not pooling.get("include_prompt"):
        raise SystemExit(f"unexpected pooling config {pooling!r}")
    if int(pooling.get("embedding_dimension", 0)) != DIMS:
        raise SystemExit(f"unexpected embedding dimension {pooling!r}")
    if runtime not in ENCODER_RUNTIMES:
        raise SystemExit(f"unknown encoder runtime {runtime!r}")
    if attention_geometry not in ATTENTION_GEOMETRIES:
        raise SystemExit(f"unknown attention geometry {attention_geometry!r}")
    device, dtype = ENCODER_RUNTIMES[runtime]
    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=local_files_only
    )
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=local_files_only,
        dtype=getattr(torch, dtype),
        attn_implementation="sdpa",
    )
    model = model.to(device).eval()
    if attention_geometry == "causal":
        for layer in _text_layers(model):
            layer.self_attn.is_causal = True
    return processor, model, template


#: Two synthetic texts, the first a strict token prefix of the second (checked at measurement time).
_GEOMETRY_PROBE = ("passage: switch the kitchen lights", "passage: switch the kitchen lights to blue please")


def measure_attention_geometry(processor, model, *, device: str = "cpu") -> str:
    """``"causal"`` or ``"bidirectional"``: the text tower's effective geometry at batch one.

    In a causal tower the hidden state at a position cannot depend on later tokens, so the last-layer states
    over a shared token prefix are identical between a text and its extension; a bidirectional tower changes
    them. Two batch-one text forwards on the loaded weights; no dataset content.
    """

    import torch

    tokenizer = processor.tokenizer
    short, long = (tokenizer(text, return_tensors="pt") for text in _GEOMETRY_PROBE)
    width = short["input_ids"].shape[1]
    if not torch.equal(long["input_ids"][0, :width], short["input_ids"][0]):
        raise SpeechEmbeddingError("geometry probe texts are not a strict token prefix pair")
    text_model = model.model
    with torch.inference_mode():
        states = [
            text_model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            .last_hidden_state[0, :width]
            .float()
            for batch in (short, long)
        ]
    difference = (states[0] - states[1]).abs().max().item()
    return "causal" if difference < 1e-3 else "bidirectional"


def render(processor, template: str, content: list[dict]) -> str:
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        chat_template=template,
        tokenize=False,
        add_generation_prompt=False,
    )


def encode(
    processor,
    model,
    texts: list[str],
    audio: list | None,
    *,
    device: str = "cuda",
    max_audio_samples: int | None = None,
    require_complete_audio: bool = False,
    max_sequence_tokens: int | None = None,
    forward_event=None,
):
    """One forward pass -> L2-normalised mean-pooled vectors."""

    import torch

    if audio is not None:
        if max_audio_samples is not None and (len(audio) != 1 or len(texts) != 1):
            raise SpeechEmbeddingError(
                "bounded speech embedding requires one audio per request"
            )
        for wave in audio:
            array = np.asarray(wave)
            if array.ndim != 1 or not array.size or not np.isfinite(array).all():
                raise SpeechEmbeddingError("speech waveform must be finite mono audio")
            if max_audio_samples is not None and len(array) > max_audio_samples:
                raise SpeechEmbeddingError(
                    "speech waveform exceeds the registered input bound"
                )
    kwargs: dict[str, object] = {"text": texts, "return_tensors": "pt", "padding": True}
    if audio is not None:
        kwargs["audio"] = audio
        kwargs["sampling_rate"] = SAMPLE_RATE_HZ
        if require_complete_audio:
            if type(max_audio_samples) is not int or max_audio_samples <= 0:
                raise SpeechEmbeddingError(
                    "complete audio requires an explicit sample bound"
                )
            # Processor defaults can propagate a text-token max_length into
            # audio samples. Bind this modality explicitly; never truncate it.
            kwargs["audio_kwargs"] = {
                "max_length": max_audio_samples,
                "truncation": False,
            }
    inputs = processor(**kwargs)
    # Coverage is checked on EVERY audio call, bounded or not: without the explicit audio bound the processor hands
    # the tokenizer's max_length (32,768 for nemotron) to the feature extractor, which silently cuts each clip to
    # its first 2.048 s. An unbounded call that would truncate is refused rather than returning a vector of a
    # different clip (npu-vllm-port, 2026-10-29); bounded calls -- every sealed path -- are unchanged.
    if audio is not None:
        mask = inputs.get("feature_attention_mask")
        hop = int(processor.feature_extractor.hop_length)
        expected = [math.ceil(len(wave) / hop) for wave in audio]
        if mask is None or mask.sum(-1).tolist() != expected:
            raise SpeechEmbeddingError(
                "audio processor did not preserve complete frame coverage"
            )
    if (
        max_sequence_tokens is not None
        and inputs["attention_mask"].shape[-1] > max_sequence_tokens
    ):
        raise SpeechEmbeddingError(
            "embedding sequence exceeds its registered token bound"
        )
    inputs = {
        key: (value.to(device) if hasattr(value, "to") else value)
        for key, value in inputs.items()
    }
    if forward_event is not None:
        forward_event("encoder_forward_started")
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1].float()
    mask = inputs["attention_mask"].to(hidden.device).unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    result = [tuple(float(v) for v in row) for row in pooled.cpu()]
    if forward_event is not None:
        forward_event("encoder_forward_finished")
    return result
