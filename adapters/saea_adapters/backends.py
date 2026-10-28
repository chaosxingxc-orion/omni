"""``backend_factory`` implementations for ``run_preparation``.

``core.speech_preparation_runtime.run_preparation`` is parameterised:

    run_preparation(config, *, source_root, data_root, backend_factory=NemotronSpeechEncoder)

so the whole preparation runtime -- generation lease, attempt store, pinned-file
verification, the monitor thread, the receipt, the output-path confinement -- is
reused exactly as the study wrote it, and the only thing we substitute is the
object that turns one source into one vector. These two classes mirror
``NemotronSpeechEncoder`` and ``NemotronPassageEncoder`` method for method,
including the lazy first-call load (the ``model_load_started`` /
``model_load_finished`` events the monitor and receipt expect) and the
``prepare_sources`` / ``close`` lifecycle.

The arm's contract -- pooling recipe, width, and required attention geometry --
is read from the inventory's ``encoder_contract`` block. When an inventory
predates that key the historical nemotron contract applies, which is the
fail-closed direction: it demands ``bidirectional``, so an inventory written by
the study's own tooling still trips the attention gate on a transformers version
where the model's override no longer binds.
"""

from __future__ import annotations

from speech_aware_evidence_acquisition.core.speech_preparation_runtime import (
    BoundedAudioPrefetch,
    confined,
)

from . import encoder as encoder_mod

# omni-embed-nemotron-3b, i.e. what the sealed vectors' identity is anchored to.
#
# attention_geometry is "measure-only" on purpose, and it is the one default here
# that is a judgement rather than a transcription. The naive choice would be
# "bidirectional", because that is the entire point of the checkpoint's 49-line
# modeling_nv_omni_embed.py. But the study produced its sealed vectors on
# transformers 5.12.1 (exposure-ledger line 2563), and the override those 49
# lines install has not bound since transformers 4.55.0. Asserting
# "bidirectional" would therefore fail-closed against REPRODUCING THE STUDY,
# which is the opposite of what this ladder is for; asserting "causal" would
# prejudge a question that reading source cannot settle (the per-layer
# is_causal=False flag may or may not reach the sdpa kernel on 5.x, and on
# Ascend). So: measure it against the real weights, land it in the receipt, and
# pin it explicitly once measured. See attention_gate for the full record.
HISTORICAL_CONTRACT = {
    "pooling_mode": "mean",
    "embedding_dimension": 2048,
    "attention_geometry": "measure-only",
    "trust_remote_code": True,
    "attn_implementation": "sdpa",
}

# The LCO arms, for reference when writing an inventory's encoder_contract.
# Both are native Qwen2_5OmniThinkerForConditionalGeneration with no auto_map
# and no modeling_*.py at all, so trust_remote_code is not needed and there is
# no override to lose; their towers are expected causal and that IS assertable.
LCO_3B_CONTRACT = {
    "pooling_mode": "lasttoken",
    "embedding_dimension": 2048,
    "attention_geometry": "causal",
    "trust_remote_code": False,
    "attn_implementation": "sdpa",
}
LCO_7B_CONTRACT = dict(LCO_3B_CONTRACT, embedding_dimension=3584)


def contract_of(inventory) -> dict:
    declared = dict(HISTORICAL_CONTRACT)
    declared.update(inventory.get("encoder_contract") or {})
    unknown = set(declared) - set(HISTORICAL_CONTRACT)
    if unknown:
        raise encoder_mod.EncoderConfigError(f"unknown encoder_contract keys: {sorted(unknown)}")
    return declared


class _Base:
    def __init__(self, inventory, protocol, data_root, event):
        self.inventory, self.protocol, self.data_root = inventory, protocol, data_root
        self.event, self.stack = event, None
        self.contract = contract_of(inventory)

    def _load(self):
        if self.stack is None:
            self.event("model_load_started")
            self.stack = encoder_mod.load_stack(
                confined(self.data_root, self.inventory["model_dir_relpath"]),
                local_files_only=True,
                **self.contract,
            )
            self.event("model_load_finished")
        return self.stack

    @property
    def facts(self):
        """The loaded arm's measured facts, for the receipt. None before load."""
        return None if self.stack is None else self.stack.facts

    def close(self):
        self.stack = None


class LadderSpeechEncoder(_Base):
    """Speech QUERY vectors: audio alone, through the arm's own chat template."""

    def __init__(self, inventory, protocol, data_root, event):
        super().__init__(inventory, protocol, data_root, event)
        self.prefetch = BoundedAudioPrefetch(protocol["cpu_workers"])

    def prepare_sources(self, sources):
        self.prefetch.prepare_sources(sources)

    def __call__(self, source):
        stack = self._load()
        self.event("data_wait_started")
        wave = self.prefetch.take(source)
        self.event("data_wait_finished")
        # The owner's 2026-09-06 rule lives here: the content list carries the
        # audio and the protocol's query prompt, and never a transcript, draft,
        # answer or label. A text-only query vector cannot be produced by this
        # code path at all.
        text = encoder_mod.render(
            stack.processor,
            stack.template,
            [
                {"type": "text", "text": self.protocol["query_prompt"]},
                {"type": "audio"},
            ],
        )
        self.event("processor_started")
        return encoder_mod.encode(
            stack,
            [text],
            [wave],
            max_audio_samples=self.protocol["max_audio_samples"],
            require_complete_audio=True,
            max_sequence_tokens=self.protocol["max_sequence_tokens"],
            forward_event=self.event,
        )[0]

    def close(self):
        self.prefetch.close()
        super().close()


class LadderPassageEncoder(_Base):
    """Text KEY (facet) vectors: batch-one passage keys, same loaded arm."""

    def __call__(self, text):
        stack = self._load()
        self.event("processor_started")
        rendered = encoder_mod.render(
            stack.processor,
            stack.template,
            [{"type": "text", "text": self.protocol["passage_prompt"] + text}],
        )
        return encoder_mod.encode(
            stack,
            [rendered],
            None,
            max_sequence_tokens=self.protocol["max_sequence_tokens"],
            forward_event=self.event,
        )[0]
