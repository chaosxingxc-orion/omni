"""Inference-engine / checkpoint adapter for the SAEA retrieval encoders.

What this absorbs, and why it is an adapter rather than a patch
---------------------------------------------------------------
The study's own encoder path (``core/speech_embedding.py``) is written for
exactly one checkpoint on exactly one stack. Four things are nailed down there:

  line  14  ``DIMS = 2048``                      -- one dimension, module-level
  line  81  ``pooling_mode != "mean"`` -> exit   -- one pooling recipe
  line  94  ``model.to("cuda")``                 -- one device
  line 113  ``device: str = "cuda"``             -- again

The 3-arm ladder (owner decision 2026-10-27) breaks all four:

  arm                      dim    pooling     attention tower   remote code
  omni-embed-nemotron-3b   2048   mean        bidirectional     yes (49 lines)
  LCO-Embedding-Omni-3B    2048   lasttoken   causal (native)   no
  LCO-Embedding-Omni-7B    3584   lasttoken   causal (native)   no

We do not edit the study tree. Three reasons: its receipts hash their own
source, its "D2 zero-dependency gate" is a property of that package, and the
control arm's whole value is being the *unmodified* recipe. Instead this module
re-implements ``load_stack``/``encode`` generically and is injected through the
seam the study already provides -- ``core.speech_preparation_runtime.run_preparation``
takes ``backend_factory=NemotronSpeechEncoder``, so a drop-in class is all that
is needed and the preparation runtime, its leases, monitors, receipts and
file-pinning are reused untouched.

Equivalence is verified, not assumed: ``tests/test_mean_equivalence.py`` asserts
this module's ``mean`` path reproduces the study's ``encode`` bit-for-bit on the
same inputs. Without that test the control arm would no longer be the control.

Everything else about the recipe is deliberately inherited verbatim from the
study: ``read_audio_span`` (hash + fingerprint + frame-geometry checks, mono
mixdown, the dead 16 kHz resample branch), ``render`` (the model's own
``additional_chat_templates/sentence_transformers.jinja``), the audio-completeness
assertion, the token and sample bounds, and L2 normalisation.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import attention_gate, device as device_mod

# The study is imported lazily, inside the functions that need it, for two
# reasons. Its ``core/__init__`` performs registration side effects and pulls in
# the whisper normaliser chain (more_itertools, regex), so importing it at
# adapter-import time would make this module unusable for anything smaller than
# a full environment. And the attention gate and the pooling functions are
# exactly the parts that must be unit-testable on their own -- a fail-closed
# gate whose own tests cannot run is not a gate.


def _study():
    from speech_aware_evidence_acquisition.core import speech_embedding

    return speech_embedding


def render(processor, template: str, content: list[dict]) -> str:
    """Re-export of the study's renderer, so callers have one import site."""
    return _study().render(processor, template, content)


def read_audio_span(source):
    """Re-export of the study's audio reader (hash, fingerprint, geometry)."""
    return _study().read_audio_span(source)

POOLING_MODES = ("mean", "lasttoken")


class EncoderConfigError(RuntimeError):
    """Refuse a checkpoint whose shipped contract is not the one declared."""


def load_stack(
    model_dir: Path,
    *,
    pooling_mode: str,
    embedding_dimension: int,
    attention_geometry: str,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
    attn_implementation: str = "sdpa",
    requested_device: str | None = None,
):
    """Load one encoder arm and prove it is the thing that was declared.

    Every keyword is a declared expectation, checked against the checkpoint's
    own shipped files. Nothing is inferred from the model's name or path: a
    directory name is not evidence, and the sealed vectors that come out of here
    are only as trustworthy as the weakest thing we took on faith.
    """

    import torch

    if pooling_mode not in POOLING_MODES:
        raise EncoderConfigError(f"unsupported pooling_mode {pooling_mode!r}")
    # The study's own guard, kept: sentence_transformers pulls torchcodec, which
    # cannot load libavutil because ffmpeg is deliberately absent and must stay
    # absent. We reproduce the sentence-transformers pipeline by hand instead.
    import sys

    if "sentence_transformers" in sys.modules:
        raise EncoderConfigError("sentence_transformers must not be imported in this process")

    template = (model_dir / "additional_chat_templates/sentence_transformers.jinja").read_text(
        encoding="utf-8"
    )
    pooling = json.loads((model_dir / "1_Pooling/config.json").read_text(encoding="utf-8"))
    if pooling.get("pooling_mode") != pooling_mode:
        raise EncoderConfigError(
            f"checkpoint ships pooling_mode {pooling.get('pooling_mode')!r}, "
            f"config declares {pooling_mode!r}"
        )
    if not pooling.get("include_prompt"):
        # include_prompt False would mean the prompt tokens are excluded from the
        # pool, i.e. a different vector for the same input. All three arms ship
        # true; a checkpoint that does not is a different recipe.
        raise EncoderConfigError(f"unexpected pooling config {pooling!r}")
    if int(pooling.get("embedding_dimension", 0)) != embedding_dimension:
        raise EncoderConfigError(
            f"checkpoint ships dimension {pooling.get('embedding_dimension')}, "
            f"config declares {embedding_dimension}"
        )

    # The prompt-leak guard (selection-ticket risk (b), 2026-08-20): both LCO
    # checkpoints carry "default_prompt_name": "default" whose body is the Qwen
    # persona string, and P-EMB-1's binding decision was no instruct prefix on
    # either side. We never construct a SentenceTransformer, so that default is
    # structurally unreachable -- the only prompts applied are the explicit
    # protocol["query_prompt"] / ["passage_prompt"]. Recorded rather than
    # trusted, so the receipt shows the string we declined to inherit.
    st_config_path = model_dir / "config_sentence_transformers.json"
    declined_prompt = None
    if st_config_path.exists():
        st_config = json.loads(st_config_path.read_text(encoding="utf-8"))
        name = st_config.get("default_prompt_name")
        if name:
            declined_prompt = (st_config.get("prompts") or {}).get(name)

    from transformers import AutoModel, AutoProcessor

    resolved = device_mod.resolve(requested_device)
    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=trust_remote_code, local_files_only=local_files_only
    )
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    model = model.to(resolved).eval()

    # The gate. See attention_gate.py: for nemotron on transformers 5.x this is
    # the only thing standing between us and a sealed vector set computed by a
    # silently causal tower.
    geometry = attention_gate.require(
        attention_geometry, processor, model, device=resolved
    )

    facts = {
        "model_dir": str(model_dir),
        "pooling_mode": pooling_mode,
        "embedding_dimension": embedding_dimension,
        "attn_implementation": attn_implementation,
        "trust_remote_code": trust_remote_code,
        "attention_geometry": geometry,
        "declined_default_prompt": declined_prompt,
        **device_mod.describe(resolved),
    }
    return Stack(processor, model, template, resolved, pooling_mode, embedding_dimension, facts)


class Stack:
    """The loaded arm plus the facts a receipt has to carry about it."""

    __slots__ = ("processor", "model", "template", "device", "pooling_mode", "dimension", "facts")

    def __init__(self, processor, model, template, device, pooling_mode, dimension, facts):
        self.processor, self.model, self.template = processor, model, template
        self.device, self.pooling_mode, self.dimension = device, pooling_mode, dimension
        self.facts = facts


def encode(
    stack: Stack,
    texts: list[str],
    audio: list | None,
    *,
    max_audio_samples: int | None = None,
    require_complete_audio: bool = False,
    max_sequence_tokens: int | None = None,
    forward_event=None,
):
    """One forward pass -> L2-normalised pooled vectors.

    Input handling is the study's, line for line; only the pooling step is
    parameterised. Keeping the checks here rather than delegating is deliberate:
    they are the measurement's bounds, and a reader comparing this function
    against the study's must be able to do it without following a call chain.
    """

    import math

    import numpy as np
    import torch

    study = _study()
    SpeechEmbeddingError = study.SpeechEmbeddingError
    SAMPLE_RATE_HZ = study.SAMPLE_RATE_HZ

    if audio is not None:
        if max_audio_samples is not None and (len(audio) != 1 or len(texts) != 1):
            raise SpeechEmbeddingError("bounded speech embedding requires one audio per request")
        for wave in audio:
            array = np.asarray(wave)
            if array.ndim != 1 or not array.size or not np.isfinite(array).all():
                raise SpeechEmbeddingError("speech waveform must be finite mono audio")
            if max_audio_samples is not None and len(array) > max_audio_samples:
                raise SpeechEmbeddingError("speech waveform exceeds the registered input bound")

    kwargs: dict[str, object] = {"text": texts, "return_tensors": "pt", "padding": True}
    if audio is not None:
        kwargs["audio"] = audio
        kwargs["sampling_rate"] = SAMPLE_RATE_HZ
        if require_complete_audio:
            if type(max_audio_samples) is not int or max_audio_samples <= 0:
                raise SpeechEmbeddingError("complete audio requires an explicit sample bound")
            # Processor defaults can propagate a text-token max_length into audio
            # samples. Bind this modality explicitly; never truncate it.
            kwargs["audio_kwargs"] = {"max_length": max_audio_samples, "truncation": False}

    inputs = stack.processor(**kwargs)
    if require_complete_audio and audio is not None:
        mask = inputs.get("feature_attention_mask")
        hop = int(stack.processor.feature_extractor.hop_length)
        expected = [math.ceil(len(wave) / hop) for wave in audio]
        if mask is None or mask.sum(-1).tolist() != expected:
            raise SpeechEmbeddingError("audio processor did not preserve complete frame coverage")
    if max_sequence_tokens is not None and inputs["attention_mask"].shape[-1] > max_sequence_tokens:
        raise SpeechEmbeddingError("embedding sequence exceeds its registered token bound")

    inputs = {
        key: (value.to(stack.device) if hasattr(value, "to") else value)
        for key, value in inputs.items()
    }
    if forward_event is not None:
        forward_event("encoder_forward_started")
    with torch.inference_mode():
        outputs = stack.model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1].float()
    attention = inputs["attention_mask"].to(hidden.device)

    pooled = _pool(stack.pooling_mode, hidden, attention)
    if pooled.shape[-1] != stack.dimension:
        raise SpeechEmbeddingError(
            f"pooled width {pooled.shape[-1]} is not the declared {stack.dimension}"
        )
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    result = [tuple(float(v) for v in row) for row in pooled.cpu()]
    if forward_event is not None:
        forward_event("encoder_forward_finished")
    return result


def _pool(mode: str, hidden, attention):
    import torch

    if mode == "mean":
        # Byte-identical to core/speech_embedding.py lines 176-177, which is
        # what makes the nemotron arm the study's own recipe and not a lookalike.
        mask = attention.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    if mode == "lasttoken":
        # sentence_transformers.models.Pooling's LASTTOKEN branch, reproduced.
        # The index arithmetic is the part worth stating: argmin over a 1/0
        # right-padded mask returns the FIRST pad position, so the last real
        # token is that minus one; when a row has no padding at all the min is 1
        # and the expression falls back to seq_len - 1. Getting this wrong does
        # not raise -- it silently pools a pad position -- which is the same
        # class of bug as the attention gate's, so it is written out longhand.
        batch, seq_len, width = hidden.shape
        values, indices = torch.min(attention, 1, keepdim=False)
        gather = torch.where(values == 0, indices, torch.full_like(indices, seq_len)) - 1
        gather = torch.clamp(gather, min=0)
        gather = gather.unsqueeze(-1).repeat(1, width).unsqueeze(1)
        masked = hidden * attention.unsqueeze(-1).expand(hidden.size()).float()
        return torch.gather(masked, 1, gather).squeeze(dim=1)

    raise EncoderConfigError(f"unsupported pooling_mode {mode!r}")
