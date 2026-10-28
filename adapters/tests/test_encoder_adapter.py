"""Tests that make the adapter's two substitutions falsifiable.

Two things in this package could be wrong in a way that produces plausible
numbers instead of an error, so both are pinned here:

1. **Pooling.** ``encoder._pool`` replaces the study's inlined pooling
   expression. If the ``mean`` branch is not bit-identical to the study's, the
   nemotron arm stops being the study's own recipe and the whole ladder loses
   its control. Tested against the study's expression verbatim on synthetic
   tensors -- no weights needed, and exactness is checkable because both sides
   are the same arithmetic on the same float32 inputs.

2. **The attention gate.** A gate that cannot fail is worse than no gate: it
   converts an unverified assumption into a false assurance. So it is run
   against a deliberately causal stub and a deliberately bidirectional one, and
   required to tell them apart in both directions.

3. **The model class.** ``load_stack`` resolves the class from the
   checkpoint's own ``config.json`` (``architectures`` / ``auto_map``), because
   ``AutoModel`` refuses the LCO checkpoints' native config on transformers
   5.14.1. Tested with stub classes and with a tiny real Qwen2.5-Omni thinker
   built from a config (no checkpoint weights are read).

4. **Audio coverage.** Without the explicit sample bound the processor
   truncates nemotron audio to 32,768 samples; ``encode`` must refuse that on
   every audio call, bounded or not. Tested with a stub processor.

Run inside the image:  /opt/saea-venv/bin/python -m pytest adapters/tests -q
(or with the system python, which is where torch lives).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import json  # noqa: E402
import sys  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from saea_adapters import attention_gate, encoder  # noqa: E402
from saea_adapters.encoder import EncoderConfigError, _pool, resolve_model_class  # noqa: E402


def study_mean(hidden, attention):
    """core/speech_embedding.py lines 176-177, copied character for character."""
    mask = attention.to(hidden.device).unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mean_pooling_is_bit_identical_to_the_study(seed):
    torch.manual_seed(seed)
    hidden = torch.randn(4, 7, 2048)
    attention = torch.ones(4, 7, dtype=torch.long)
    attention[1, 5:] = 0  # right padding
    attention[2, :4] = 0  # left padding, the shape the Qwen2.5-Omni processor produces
    ours, theirs = _pool("mean", hidden, attention), study_mean(hidden, attention)
    assert torch.equal(ours, theirs), (ours - theirs).abs().max()


def test_mean_pooling_ignores_padded_positions():
    hidden = torch.zeros(1, 4, 3)
    hidden[0, 0] = torch.tensor([1.0, 2.0, 3.0])
    hidden[0, 1] = torch.tensor([3.0, 2.0, 1.0])
    hidden[0, 2] = torch.tensor([99.0, 99.0, 99.0])  # padded: must not contribute
    attention = torch.tensor([[1, 1, 0, 0]])
    assert torch.equal(_pool("mean", hidden, attention), torch.tensor([[2.0, 2.0, 2.0]]))


def test_lasttoken_pooling_picks_the_last_real_token_when_padded():
    hidden = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
    attention = torch.tensor([[1, 1, 1, 0, 0]])
    # index 2 is the last unpadded position -> row [4., 5.]
    assert torch.equal(_pool("lasttoken", hidden, attention), torch.tensor([[4.0, 5.0]]))


def test_lasttoken_pooling_picks_the_final_token_when_unpadded():
    hidden = torch.arange(1 * 3 * 2, dtype=torch.float32).reshape(1, 3, 2)
    attention = torch.tensor([[1, 1, 1]])
    # no padding at all: min(mask) == 1, so the fallback seq_len - 1 must apply
    assert torch.equal(_pool("lasttoken", hidden, attention), torch.tensor([[4.0, 5.0]]))


def test_lasttoken_pooling_handles_a_mixed_batch():
    hidden = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
    attention = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    out = _pool("lasttoken", hidden, attention)
    assert torch.equal(out, torch.stack([hidden[0, 1], hidden[1, 3]]))


def st_lasttoken(hidden, attention):
    """The pre-fix branch (sentence_transformers' LASTTOKEN), kept as the right-padding reference."""
    batch, seq_len, width = hidden.shape
    values, indices = torch.min(attention, 1, keepdim=False)
    gather = torch.where(values == 0, indices, torch.full_like(indices, seq_len)) - 1
    gather = torch.clamp(gather, min=0)
    gather = gather.unsqueeze(-1).repeat(1, width).unsqueeze(1)
    masked = hidden * attention.unsqueeze(-1).expand(hidden.size()).float()
    return torch.gather(masked, 1, gather).squeeze(dim=1)


def test_lasttoken_pooling_picks_the_final_token_when_left_padded():
    hidden = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
    attention = torch.tensor([[0, 0, 1, 1, 1]])
    # the last real token is the last position; the pre-fix branch pooled the pad at 0
    assert torch.equal(_pool("lasttoken", hidden, attention), torch.tensor([[8.0, 9.0]]))


def test_lasttoken_pooling_handles_a_left_padded_mixed_batch():
    """The processor's real shape: every row but the longest is left-padded.
    Before the fix every such row came back as a zero vector."""
    hidden = torch.randn(3, 6, 4) + 10.0  # no state is zero
    attention = torch.tensor([[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    out = _pool("lasttoken", hidden, attention)
    assert torch.equal(out, hidden[:, -1])
    assert bool((out != 0).all())
    assert not torch.equal(st_lasttoken(hidden, attention), out)  # the pre-fix branch is wrong here


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lasttoken_pooling_equals_the_reference_on_right_padding(seed):
    torch.manual_seed(seed)
    hidden = torch.randn(4, 7, 16)
    attention = torch.ones(4, 7, dtype=torch.long)
    attention[0, 2:] = 0
    attention[2, 6:] = 0
    attention[3, 1:] = 0
    assert torch.equal(_pool("lasttoken", hidden, attention), st_lasttoken(hidden, attention))


@pytest.mark.parametrize("side", ["left", "right"])
def test_lasttoken_pooling_is_padding_invariant(side):
    """One row's real states, pooled alone and inside a padded batch, give the same vector."""
    torch.manual_seed(3)
    real = torch.randn(1, 4, 8)
    alone = _pool("lasttoken", real, torch.ones(1, 4, dtype=torch.long))
    pad = torch.full((1, 3, 8), 7.0)
    padded = torch.cat([pad, real] if side == "left" else [real, pad], dim=1)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]] if side == "left" else [[1, 1, 1, 1, 0, 0, 0]])
    other = torch.randn(1, 7, 8)
    batch = torch.cat([padded, other])
    masks = torch.cat([mask, torch.ones(1, 7, dtype=torch.long)])
    assert torch.equal(_pool("lasttoken", batch, masks)[0], alone[0])


def test_lasttoken_pooling_refuses_a_row_with_no_real_token():
    with pytest.raises(EncoderConfigError):
        _pool("lasttoken", torch.ones(2, 3, 2), torch.tensor([[1, 1, 1], [0, 0, 0]]))


def test_unknown_pooling_mode_is_refused():
    with pytest.raises(Exception):
        _pool("cls", torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.long))


# --- the attention gate -----------------------------------------------------


class _Processor:
    """Minimal stand-in: whitespace tokens -> ids, no padding, no template."""

    def __call__(self, text, return_tensors=None, padding=None):
        ids = [[abs(hash(tok)) % 1000 + 1 for tok in text[0].split()]]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.ones(1, len(ids[0]), dtype=torch.long),
        }


class _Out:
    def __init__(self, hidden):
        self.hidden_states = [hidden]


class _CausalModel:
    """Position i sees only tokens 0..i -- a running mean, so prefix states
    cannot move when a suffix is appended."""

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        x = input_ids.float().unsqueeze(-1).repeat(1, 1, 8)
        running = torch.cumsum(x, dim=1) / torch.arange(
            1, x.shape[1] + 1, dtype=torch.float32
        ).view(1, -1, 1)
        return _Out(running)


class _BidirectionalModel:
    """Every position sees the whole sequence -- the sequence mean broadcast, so
    a longer suffix changes the prefix states."""

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        x = input_ids.float().unsqueeze(-1).repeat(1, 1, 8)
        return _Out(x.mean(dim=1, keepdim=True).expand_as(x).contiguous())


def test_gate_detects_a_causal_tower():
    measured = attention_gate.measure_text_tower(
        _Processor(), _CausalModel(), device="cpu"
    )
    assert measured["bidirectional"] is False
    assert measured["compared_tokens"] >= 2


def test_gate_detects_a_bidirectional_tower():
    measured = attention_gate.measure_text_tower(
        _Processor(), _BidirectionalModel(), device="cpu"
    )
    assert measured["bidirectional"] is True


def test_require_passes_when_geometry_matches():
    facts = attention_gate.require(
        "bidirectional", _Processor(), _BidirectionalModel(), device="cpu"
    )
    assert facts["required"] == "bidirectional"
    facts = attention_gate.require("causal", _Processor(), _CausalModel(), device="cpu")
    assert facts["required"] == "causal"


def test_require_fails_closed_on_the_nemotron_regression():
    """The real failure: the arm declares bidirectional, the loaded tower is
    causal because the model file's override no longer binds."""
    with pytest.raises(attention_gate.AttentionGeometryError) as caught:
        attention_gate.require(
            "bidirectional", _Processor(), _CausalModel(), device="cpu"
        )
    message = str(caught.value)
    assert "_update_causal_mask" in message
    assert "content fingerprint" in message  # do not edit the pinned asset


def test_require_rejects_an_unknown_geometry_name():
    with pytest.raises(ValueError):
        attention_gate.require("acausal", _Processor(), _CausalModel(), device="cpu")


def test_measure_only_asserts_nothing_but_records_everything():
    """The control arm's default: no verdict, but the measurement is in the
    facts, so it reaches the receipt and cannot pass unnoticed."""
    for model in (_CausalModel(), _BidirectionalModel()):
        facts = attention_gate.require("measure-only", _Processor(), model, device="cpu")
        assert facts["required"] == "measure-only"
        assert facts["pinned"] is False
        assert "bidirectional" in facts and "max_abs_delta" in facts


def test_pinned_geometries_are_marked_pinned():
    facts = attention_gate.require("causal", _Processor(), _CausalModel(), device="cpu")
    assert facts["pinned"] is True


# --- model class resolution -------------------------------------------------


class _StubArchitecture:
    """Stands in for a transformers model class; only its name is consulted."""


def _config(**fields):
    return SimpleNamespace(**{"architectures": None, "auto_map": None, **fields})


def test_class_comes_from_architectures_not_automodel(monkeypatch, tmp_path):
    import transformers

    monkeypatch.setattr(transformers, "_StubArchitecture", _StubArchitecture, raising=False)
    model_class, source = resolve_model_class(
        tmp_path, _config(architectures=["_StubArchitecture"]), trust_remote_code=False
    )
    assert model_class is _StubArchitecture
    assert source == "transformers:_StubArchitecture"


def test_auto_map_entry_for_the_architecture_loads_the_checkpoint_code(monkeypatch, tmp_path):
    import transformers.dynamic_module_utils as dynamic

    calls = []

    def fake_loader(reference, path, **kwargs):
        calls.append((reference, path, kwargs))
        return type("NVOmniEmbedModel", (), {})

    monkeypatch.setattr(dynamic, "get_class_from_dynamic_module", fake_loader)
    config = _config(
        architectures=["NVOmniEmbedModel"],
        auto_map={
            "AutoConfig": "modeling_nv_omni_embed.NVOmniEmbedConfig",
            "AutoModel": "modeling_nv_omni_embed.NVOmniEmbedModel",
        },
    )
    model_class, source = resolve_model_class(tmp_path, config, trust_remote_code=True)
    assert model_class.__name__ == "NVOmniEmbedModel"
    assert source == "auto_map:modeling_nv_omni_embed.NVOmniEmbedModel"
    assert calls == [
        ("modeling_nv_omni_embed.NVOmniEmbedModel", str(tmp_path), {"local_files_only": True})
    ]


def test_remote_architecture_without_trust_is_refused(tmp_path):
    config = _config(architectures=["NVOmniEmbedModel"],
                     auto_map={"AutoModel": "modeling_nv_omni_embed.NVOmniEmbedModel"})
    with pytest.raises(EncoderConfigError, match="trust_remote_code"):
        resolve_model_class(tmp_path, config, trust_remote_code=False)


@pytest.mark.parametrize("architectures", [None, [], ["A", "B"]])
def test_config_must_name_exactly_one_architecture(tmp_path, architectures):
    with pytest.raises(EncoderConfigError, match="exactly one architecture"):
        resolve_model_class(tmp_path, _config(architectures=architectures), trust_remote_code=True)


def test_unknown_architecture_is_refused(tmp_path):
    with pytest.raises(EncoderConfigError, match="neither in auto_map"):
        resolve_model_class(
            tmp_path, _config(architectures=["NoSuchArchitectureAnywhere"]), trust_remote_code=True
        )


def test_resolved_class_must_carry_the_architecture_name(monkeypatch, tmp_path):
    import transformers.dynamic_module_utils as dynamic

    monkeypatch.setattr(dynamic, "get_class_from_dynamic_module",
                        lambda *args, **kwargs: type("SomethingElse", (), {}))
    config = _config(architectures=["Wanted"], auto_map={"AutoModel": "modeling_x.Wanted"})
    with pytest.raises(EncoderConfigError, match="resolved"):
        resolve_model_class(tmp_path, config, trust_remote_code=True)


def _hide_torch_npu(monkeypatch):
    # The image's torch_npu needs the Ascend driver, which a CPU test never has;
    # transformers imports it while loading any modeling module.
    if "torch_npu" not in sys.modules:
        monkeypatch.setitem(sys.modules, "torch_npu", None)


def _tiny_thinker_dir(tmp_path):
    """A tiny, randomly initialised Qwen2.5-Omni thinker saved to disk: the LCO arms'
    native architecture and config class, with nothing read from a real checkpoint."""
    from transformers import (
        Qwen2_5OmniThinkerConfig,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    config = Qwen2_5OmniThinkerConfig(
        audio_config={"d_model": 16, "encoder_layers": 1, "encoder_attention_heads": 2,
                      "encoder_ffn_dim": 32, "num_mel_bins": 8, "output_dim": 32,
                      "max_source_positions": 64, "n_window": 4},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 32, "fullatt_block_indexes": [0]},
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                     "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 128,
                     "rope_scaling": {"type": "default", "mrope_section": [2, 3, 3]}},
    )
    torch.manual_seed(0)
    Qwen2_5OmniThinkerForConditionalGeneration(config).save_pretrained(tmp_path)
    return tmp_path


def test_a_native_thinker_checkpoint_loads_where_automodel_refuses(monkeypatch, tmp_path):
    """The LCO failure, reproduced on a tiny config: AutoModel has no mapping for
    Qwen2_5OmniThinkerConfig; the checkpoint's own architecture loads it."""
    _hide_torch_npu(monkeypatch)
    from transformers import AutoConfig, AutoModel

    model_dir = _tiny_thinker_dir(tmp_path)
    assert json.loads((model_dir / "config.json").read_text())["architectures"] == [
        "Qwen2_5OmniThinkerForConditionalGeneration"
    ]
    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        AutoModel.from_pretrained(str(model_dir), local_files_only=True)
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    model_class, source = resolve_model_class(model_dir, config, trust_remote_code=False)
    assert source == "transformers:Qwen2_5OmniThinkerForConditionalGeneration"
    model = model_class.from_pretrained(str(model_dir), config=config, local_files_only=True)
    assert type(model).__name__ == "Qwen2_5OmniThinkerForConditionalGeneration"


def test_load_stack_runs_its_checks_on_the_resolved_class(monkeypatch, tmp_path):
    """End to end through load_stack on the tiny thinker: the pooling-config checks, the
    class resolution and the attention gate all run, and the facts name the class."""
    _hide_torch_npu(monkeypatch)
    from transformers import AutoProcessor

    model_dir = _tiny_thinker_dir(tmp_path)
    (model_dir / "additional_chat_templates").mkdir()
    (model_dir / "additional_chat_templates/sentence_transformers.jinja").write_text("{{ x }}")
    (model_dir / "1_Pooling").mkdir()
    (model_dir / "1_Pooling/config.json").write_text(json.dumps(
        {"pooling_mode": "lasttoken", "include_prompt": True, "embedding_dimension": 32}))
    seen = {}

    def fake_require(geometry, processor, model, *, device):
        seen["model"] = model
        return {"required": geometry}

    monkeypatch.setattr(AutoProcessor, "from_pretrained", classmethod(lambda cls, *a, **k: "processor"))
    monkeypatch.setattr(encoder.attention_gate, "require", fake_require)
    stack = encoder.load_stack(model_dir, pooling_mode="lasttoken", embedding_dimension=32,
                               attention_geometry="causal", trust_remote_code=False,
                               requested_device="cpu")
    assert type(stack.model).__name__ == "Qwen2_5OmniThinkerForConditionalGeneration"
    assert seen["model"] is stack.model
    assert next(stack.model.parameters()).dtype == torch.bfloat16
    assert stack.facts["model_class_source"] == "transformers:Qwen2_5OmniThinkerForConditionalGeneration"
    assert stack.facts["model_class"].endswith(".Qwen2_5OmniThinkerForConditionalGeneration")
    with pytest.raises(EncoderConfigError, match="config declares"):
        encoder.load_stack(model_dir, pooling_mode="mean", embedding_dimension=32,
                           attention_geometry="causal", requested_device="cpu")


# --- audio frame coverage -----------------------------------------------------


class _AudioProcessor:
    """Stands in for the Qwen2.5-Omni processor's audio side: reports a feature
    mask covering at most ``cap`` samples, as nemotron's does without the bound."""

    feature_extractor = SimpleNamespace(hop_length=160)

    def __init__(self, cap=None):
        self.cap, self.calls = cap, []

    def __call__(self, text, audio=None, **kwargs):
        self.calls.append(kwargs)
        frames = [-(-min(len(w), self.cap or len(w)) // 160) for w in audio]
        width = max(frames)
        return {
            "input_ids": torch.ones(len(text), 3, dtype=torch.long),
            "attention_mask": torch.ones(len(text), 3, dtype=torch.long),
            "feature_attention_mask": torch.tensor([[1] * n + [0] * (width - n) for n in frames]),
        }


class _HiddenModel:
    def __call__(self, **kwargs):
        return SimpleNamespace(hidden_states=[torch.ones(kwargs["input_ids"].shape[0], 3, 4)])


def _audio_stack(processor):
    return encoder.Stack(processor, _HiddenModel(), "", "cpu", "mean", 4, {})


def test_unbounded_truncated_audio_is_refused():
    pytest.importorskip("speech_aware_evidence_acquisition.core.speech_embedding")
    np = pytest.importorskip("numpy")
    stack = _audio_stack(_AudioProcessor(cap=32768))
    with pytest.raises(ValueError, match="complete frame coverage"):
        encoder.encode(stack, ["x", "y"], [np.ones(16000, np.float32), np.ones(128000, np.float32)])


def test_complete_audio_still_encodes_unbounded_and_bounded():
    pytest.importorskip("speech_aware_evidence_acquisition.core.speech_embedding")
    np = pytest.importorskip("numpy")
    processor = _AudioProcessor()
    stack = _audio_stack(processor)
    waves = [np.ones(16000, np.float32), np.ones(128000, np.float32)]
    assert len(encoder.encode(stack, ["x", "y"], waves)) == 2
    assert len(encoder.encode(stack, ["x"], waves[1:], max_audio_samples=1920000,
                              require_complete_audio=True)) == 1
    assert processor.calls[-1]["audio_kwargs"] == {"max_length": 1920000, "truncation": False}
