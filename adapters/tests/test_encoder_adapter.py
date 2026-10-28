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

Run inside the image:  /opt/saea-venv/bin/python -m pytest adapters/tests -q
(or with the system python, which is where torch lives).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from saea_adapters import attention_gate  # noqa: E402
from saea_adapters.encoder import _pool  # noqa: E402


def study_mean(hidden, attention):
    """core/speech_embedding.py lines 176-177, copied character for character."""
    mask = attention.to(hidden.device).unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mean_pooling_is_bit_identical_to_the_study(seed):
    torch.manual_seed(seed)
    hidden = torch.randn(4, 7, 2048)
    attention = torch.ones(4, 7, dtype=torch.long)
    attention[1, 5:] = 0  # right padding, the shape the processor produces
    attention[2, 3:] = 0
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
