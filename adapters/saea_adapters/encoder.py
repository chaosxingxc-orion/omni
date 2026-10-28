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
assertion (here applied to every audio call, not only bounded ones), the token
and sample bounds, and L2 normalisation.
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

    from transformers import AutoConfig, AutoProcessor

    resolved = device_mod.resolve(requested_device)
    processor = AutoProcessor.from_pretrained(
        str(model_dir), trust_remote_code=trust_remote_code, local_files_only=local_files_only
    )
    config = AutoConfig.from_pretrained(
        str(model_dir), trust_remote_code=trust_remote_code, local_files_only=local_files_only
    )
    model_class, model_class_source = resolve_model_class(
        model_dir, config, trust_remote_code=trust_remote_code, local_files_only=local_files_only
    )
    # ``config=`` is what AutoModel itself passes on: the checkpoint's config is
    # parsed once, by AutoConfig, for both the class resolution and the load.
    model = model_class.from_pretrained(
        str(model_dir),
        config=config,
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
        "model_class": f"{model_class.__module__}.{model_class.__qualname__}",
        "model_class_source": model_class_source,
        "attention_geometry": geometry,
        "declined_default_prompt": declined_prompt,
        **device_mod.describe(resolved),
    }
    return Stack(processor, model, template, resolved, pooling_mode, embedding_dimension, facts)


def resolve_model_class(
    model_dir: Path, config, *, trust_remote_code: bool, local_files_only: bool = True
):
    """The class the checkpoint's own ``config.json`` names -> ``(class, source)``.

    ``AutoModel`` maps a config class to a model class through a table inside
    transformers, and that table is not the checkpoint's statement of what it
    is. On transformers 5.14.1 it has no entry at all for
    ``Qwen2_5OmniThinkerConfig``, the native config of both LCO checkpoints, so
    ``AutoModel.from_pretrained`` refuses them outright. The checkpoint itself
    says exactly which class saved it: ``architectures`` (one name), and, for a
    remote-code checkpoint, ``auto_map`` (where that name's code lives). Both
    are read here and nothing else is consulted:

      * an ``auto_map`` entry whose class is ``architectures[0]`` -> that class
        from the checkpoint's own module (requires ``trust_remote_code``);
      * otherwise -> the transformers class of exactly that name.

    nemotron ships ``architectures: ["NVOmniEmbedModel"]`` with
    ``auto_map.AutoModel: "modeling_nv_omni_embed.NVOmniEmbedModel"`` -- the
    same class AutoModel resolved before, so the control arm is unchanged. The
    LCO arms ship ``["Qwen2_5OmniThinkerForConditionalGeneration"]`` and no
    ``auto_map``.
    """

    architectures = list(getattr(config, "architectures", None) or ())
    if len(architectures) != 1 or not isinstance(architectures[0], str):
        raise EncoderConfigError(
            f"{model_dir}: config.json must name exactly one architecture, got {architectures!r}"
        )
    name = architectures[0]
    auto_map = getattr(config, "auto_map", None) or {}
    references = sorted(
        {
            reference
            for reference in auto_map.values()
            if isinstance(reference, str) and reference.rsplit(".", 1)[-1] == name
        }
    )
    if len(references) > 1:
        raise EncoderConfigError(f"{model_dir}: auto_map names {name!r} in {references!r}")
    if references:
        if not trust_remote_code:
            raise EncoderConfigError(
                f"{model_dir}: architecture {name!r} is remote code ({references[0]!r}) "
                f"but trust_remote_code is False"
            )
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        model_class = get_class_from_dynamic_module(
            references[0], str(model_dir), local_files_only=local_files_only
        )
        source = f"auto_map:{references[0]}"
    else:
        import transformers

        model_class = getattr(transformers, name, None)
        source = f"transformers:{name}"
        if model_class is None:
            raise EncoderConfigError(
                f"{model_dir}: architecture {name!r} is neither in auto_map nor a transformers class"
            )
    if getattr(model_class, "__name__", None) != name:
        raise EncoderConfigError(
            f"{model_dir}: resolved {model_class!r} for architecture {name!r}"
        )
    return model_class, source


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
    if audio is not None:
        # Checked on EVERY audio call, not only bounded ones (the study checks
        # only under require_complete_audio). Without the explicit bound the
        # processor forwards nemotron's tokenizer max_length (32,768) to the
        # audio feature extractor, which cuts every clip to its first 2.048 s --
        # silently, at any batch size (measured 2026-10-29: batch-one cosine
        # 0.914 on an 8 s clip). That is what made the encbench "batched" audio
        # rows disagree: they were the only unbounded calls. A truncated clip is
        # refused here; it never becomes a vector.
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
        # The last REAL token of each row, read off the attention mask: the
        # highest position whose mask is 1. That is correct whichever side the
        # processor pads on, and it has to be, because the side is not the one
        # the tokenizer reports: ``Qwen2_5OmniProcessor`` pins ``padding_side:
        # "left"`` in its own text defaults, so all three checkpoints pad on the
        # LEFT although the LCO tokenizers say ``right``.
        # sentence_transformers' LASTTOKEN branch (argmin over the mask, minus
        # one) assumes right padding; under left padding it gathers position 0,
        # a masked pad, and returns a zero vector for every row but the longest
        # -- silently. The two agree exactly on right-padded and unpadded rows.
        # A row with no real token at all has nothing to pool and is refused.
        batch, seq_len = hidden.shape[:2]
        real = attention.to(hidden.device) != 0
        if not bool(real.any(dim=1).all()):
            raise EncoderConfigError("lasttoken pooling over a row with no unmasked token")
        positions = torch.arange(seq_len, device=hidden.device).expand(batch, seq_len)
        last = torch.where(real, positions, torch.full_like(positions, -1)).max(dim=1).values
        return hidden[torch.arange(batch, device=hidden.device), last]

    raise EncoderConfigError(f"unsupported pooling_mode {mode!r}")
