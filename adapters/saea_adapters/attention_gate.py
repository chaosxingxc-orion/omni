"""The bidirectionality gate: prove the text tower is not causal before sealing.

Why this module exists
----------------------
``omni-embed-nemotron-3b`` is not really a model, it is a 49-line patch. Its
whole ``modeling_nv_omni_embed.py`` takes ``Qwen2_5OmniThinkerForConditionalGeneration``
and makes the text tower bidirectional, two ways:

    for layer in self.layers:
        layer.self_attn.is_causal = False          # (1) per-layer flag
    def _update_causal_mask(self, ...):            # (2) mask override
        return _prepare_4d_attention_mask(attention_mask, dtype=...)

Mechanism (2) is an override of a transformers-internal hook. Measured inside
``quay.io/ascend/vllm-ascend:v0.30.0rc2`` (transformers 5.14.1) on 2026-10-27:

    Qwen2_5OmniThinkerTextModel has _update_causal_mask: False

transformers 5.x moved mask construction to ``masking_utils.create_causal_mask``,
so the parent class no longer has the hook. All five of the model file's imports
still resolve, the config still loads, the weights still load, the forward still
runs, and it still returns 2,048-dimensional L2-normalised vectors. **Nothing
raises.** The override is simply never called, and whether (1) alone still
reaches the attention kernels is version- and backend-dependent -- on Ascend it
is a third unknown on top of the two.

That is the worst failure shape available: an encoder that silently produces
embeddings from a tower running in a mode it was not trained as. Those vectors
would be sealed with a sha256, pass every identity check the study has, and
quietly mis-rank every retrieval built on them. No downstream gate in the study
can catch it, because the study's gates verify *file identity*, not *attention
geometry*.

And it is not a deviation we would be introducing -- it is a property of the
published vectors
--------------------------------------------------------------------------------
``_update_causal_mask`` disappeared from ``modeling_qwen2_5_omni.py`` at
transformers **v4.55.0**, replaced by ``masking_utils``. Measured by reading the
file at each tag (2026-10-27):

    v4.52.4   _update_causal_mask=True   masking_utils=False
    v4.55.0   _update_causal_mask=False  masking_utils=True
    v4.57.3   _update_causal_mask=False  masking_utils=True
    v5.0.0 / v5.6.0 / v5.14.1   same as above

And the ledger records what the study itself ran
(``SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-V1 COMPLETED``, exposure-ledger line 2563):

    "loaded through plain transformers ... attn_implementation='sdpa'),
     torch 2.9.1+cu128, transformers 5.12.1"

**transformers 5.12.1.** So the override was already dead when the study produced
the sealed vectors the whole Fold B baseline depends on. Two consequences:

  * Reproduction fidelity points at the NEW transformers, not the old one. Our
    container's 5.14.1 is the same line as the study's 5.12.1. Running nemotron
    on 4.52.4 would make the override bind, produce genuinely bidirectional
    vectors, and thereby disagree with the artifact we are trying to reproduce.
  * The reference work's own retrieval numbers were produced by a tower whose
    one distinguishing feature -- the bidirectional text tower that is the entire
    content of ``modeling_nv_omni_embed.py`` -- was inert.

Whether ``layer.self_attn.is_causal = False`` alone still reaches the attention
kernels on transformers 5.x, and on Ascend, is NOT settled by reading the source:
some attention paths consult the module flag, some only the mask. That is exactly
the question this gate answers empirically, against the real weights, which is
why the control arm ships as ``measure-only`` until it has been measured once.

So we test the geometry directly, at prep time, against the real weights, and
record it in the receipt; arms with an established geometry assert it and fail
closed.

The test
--------
Causality is a statement about information flow: in a causal tower the hidden
state at position *i* cannot depend on any token after *i*. So embed two texts
where one is a strict token-prefix of the other and compare the last-layer
hidden states over the shared prefix.

  * causal          -> prefix states are **identical** (bit-for-bit, modulo the
                       kernel's own nondeterminism, hence ``tol``)
  * bidirectional   -> prefix states **change**, because the extra suffix tokens
                       are now visible to them

Two separate batch-1 forwards, so padding never enters the comparison. Text
only: the thinker's text tower is what the patch touches, and keeping audio out
removes the audio-frame padding path as a confound.

This is a property test on the loaded object, not a version check. It therefore
stays correct across transformers upgrades, across Ascend/CUDA, and for any
future encoder -- including the LCO arms, whose native
``Qwen2_5OmniThinkerForConditionalGeneration`` is *expected* to be causal and
must be asserted as such rather than assumed.
"""

from __future__ import annotations

PREFIX_PROBE = "alpha bravo charlie delta"
SUFFIX_PROBE = " echo foxtrot golf hotel india juliet"


class AttentionGeometryError(RuntimeError):
    """The loaded tower's information flow is not the one the encoder needs."""


def measure_text_tower(processor, model, *, device: str, tol: float = 1e-3) -> dict[str, object]:
    """Return the measured attention geometry of the loaded text tower."""

    import torch

    def hidden(text: str):
        inputs = processor(text=[text], return_tensors="pt", padding=False)
        inputs = {
            key: (value.to(device) if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        # float() before any comparison: bf16's ~3 decimal digits would put the
        # causal/bidirectional separation inside its own rounding noise.
        return inputs["input_ids"][0], out.hidden_states[-1][0].float()

    short_ids, short_hidden = hidden(PREFIX_PROBE)
    long_ids, long_hidden = hidden(PREFIX_PROBE + SUFFIX_PROBE)

    shared = _shared_prefix_length(short_ids.tolist(), long_ids.tolist())
    if shared < 2:
        # Without a shared prefix there is nothing to compare, so a "pass" here
        # would be vacuous. Surface it instead of reporting a verdict.
        raise AttentionGeometryError(
            "the probe pair shares no token prefix on this tokenizer; "
            f"short={short_ids.tolist()[:8]} long={long_ids.tolist()[:8]}"
        )
    # Drop the final shared position: for a strict prefix pair it is the last
    # token the short sequence has, and in several chat tokenizers it is where a
    # terminator would attach, which makes it the one position whose difference
    # is not evidence about attention.
    compare = shared - 1
    delta = (short_hidden[:compare] - long_hidden[:compare]).abs()
    max_delta = float(delta.max())
    return {
        "shared_prefix_tokens": int(shared),
        "compared_tokens": int(compare),
        "max_abs_delta": max_delta,
        "tolerance": float(tol),
        # "The suffix changed the prefix" == the prefix could see it.
        "bidirectional": bool(max_delta > tol),
    }


def require(geometry: str, processor, model, *, device: str, tol: float = 1e-3) -> dict[str, object]:
    """Assert the tower's geometry is ``"bidirectional"`` or ``"causal"``.

    ``geometry`` comes from the encoder's own config entry, never from a model
    name: it is a claim the operator makes about the checkpoint, and this is the
    function that makes the claim falsifiable.
    """

    if geometry not in ("bidirectional", "causal", "measure-only"):
        raise ValueError(f"unknown required attention geometry {geometry!r}")
    measured = measure_text_tower(processor, model, device=device, tol=tol)
    if geometry == "measure-only":
        # Deliberate escape hatch for an arm whose true geometry is not yet
        # established against real weights. It asserts nothing, but it cannot
        # hide: the measurement goes into the arm's facts and from there into
        # the vector receipt, so a reader sees what the tower actually did.
        measured["required"] = geometry
        measured["pinned"] = False
        return measured
    expected_bidirectional = geometry == "bidirectional"
    if measured["bidirectional"] is not expected_bidirectional:
        got = "bidirectional" if measured["bidirectional"] else "causal"
        raise AttentionGeometryError(
            f"text tower is {got}, the encoder requires {geometry}. "
            f"max|delta| over {measured['compared_tokens']} shared prefix tokens = "
            f"{measured['max_abs_delta']:.3e} against tolerance {measured['tolerance']:.1e}. "
            "For omni-embed-nemotron-3b this is the known transformers-5.x failure: "
            "modeling_nv_omni_embed.py overrides _update_causal_mask, which "
            "Qwen2_5OmniThinkerTextModel no longer defines, so the override is dead code "
            "and the tower silently stays causal. Pin transformers to a version whose "
            "Qwen2_5OmniThinkerTextModel still defines _update_causal_mask; do not edit the "
            "pinned model asset, that would break its content fingerprint in the umbrella lock."
        )
    measured["required"] = geometry
    measured["pinned"] = True
    return measured


def _shared_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count
