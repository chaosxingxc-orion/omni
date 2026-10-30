#!/usr/bin/env python3
"""tfrl_v2_text.py -- the item-text resolver and the token measures of the TFRL-v2 distiller (task T3b, plan 2026-10-30 section 11 ruling 4).

Import-only and model-free. This module constructs no network client of any kind and calls no model. The vLLM tokenize measure takes an
INJECTED client (an object with ``request_json(method, path, payload)``, e.g. the guarded tokenizer client of a registered fit gate); a test
injects a fake one.

ITEM TEXT. The rollout trace carries the supplied item ids of a Solver contact, not their text. :class:`ItemTextResolver` maps an id to the exact
text the Solver read:

* source of truth: the knowledge stores of the registered port cell, ``knowledge_stores`` of ``scripts/flights/nx/l338_port_pins.json``, loaded by
  ``rails.knowledge_stores.load_knowledge_store_binding`` (every pin and hash verified) from the data root (``knowledge/slurp/...``). The binding's
  composite snapshot qualifies every value id as ``<store id>::<local id>`` (for example ``train-exemplars::value:1``); that is the id the trace
  records, because the runner renders each visible value as an evidence part ``{"evidence_id": value.value_id, "content": value.content}``
  (``rails/runner.py`` ``_agent_messages``);
* renderer: the transport renders each such part as ``[evidence <evidence_id>] <content>``
  (``core/model/transport.py``, the ``supplied_evidence`` loop). :func:`render_evidence_line` is that line; a test pins the format string in the
  transport source. The rank-visible label of cell RANKVIS (``retrieval rank N: ``) is NOT added: it depends on the row's retrieval order and is
  armed only by ``native_episode.knowledge_value_rendering``; the resolver's identity records ``rank_label: none``;
* every resolved item carries the sha256 of its content; an id the stores do not hold is refused.

TOKEN MEASURES. Two measures behind one small interface (``name``, ``count(text)``):

* ``estimate``       ``ceil(chars / 3.5)`` (offline upper-bound estimate);
* ``vllm-tokenize``  the vLLM chat-form ``POST /tokenize`` that the FIT gate uses (``core.preparation.measure_vllm_multimodal_request``), through an
  injected client, one user text message per measured text. The count INCLUDES the fixed chat-template wrapper (a constant of a few tokens per
  measured text), so the measure is conservative. Results are cached by text.

A bank receipt records the measure that produced its counts; the sampler's plan builder accepts only ``vllm-tokenize`` banks.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import tfrl_v2_common as cm

MEASURE_ESTIMATE = cm.MEASURE_ESTIMATE
MEASURE_VLLM = cm.MEASURE_VLLM
MEASURES = (MEASURE_ESTIMATE, MEASURE_VLLM)
RESOLVER_ID = "tfrl-v2-item-text-v1"
RENDERER_FORMAT = "[evidence <evidence_id>] <content>"
RENDERER_SOURCE = "core/model/transport.py (supplied_evidence loop); content from rails/runner.py _agent_messages (value.content)"
PINS_RELPATH = "scripts/flights/nx/l338_port_pins.json"
EXEMPLAR_TEXT_RESOLVED = cm.EXEMPLAR_TEXT_RESOLVED
EXEMPLAR_TEXT_IDS_ONLY = cm.EXEMPLAR_TEXT_IDS_ONLY


class ItemTextError(ValueError):
    """A refusal of the resolver: an id the stores do not hold, a malformed store, a missing pin."""


class MeasureError(ValueError):
    """A refusal of a token measure (a tokenizer reply that is not a consistent count)."""


# ----------------------------------------------------------------------------------------------------------------------------------------
# token measures
# ----------------------------------------------------------------------------------------------------------------------------------------


class TokenMeasure(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def detail(self) -> Mapping[str, Any]: ...


class EstimateMeasure:
    """``ceil(chars / 3.5)``: offline, an upper-leaning estimate. Banks counted with it are NOT accepted by the sampler's plan builder."""

    name = MEASURE_ESTIMATE

    def count(self, text: str) -> int:
        return cm.estimate_tokens(text)

    def detail(self) -> Mapping[str, Any]:
        return {"method": f"ceil(chars / {cm.CHARS_PER_TOKEN})"}


class VllmTokenizeMeasure:
    """The FIT gate's vLLM tokenize measure over an injected client.

    ``client.request_json("POST", "/tokenize", body)`` must answer ``{"count": n, "tokens": [... n ids ...]}`` (the contract of
    ``measure_vllm_multimodal_request``). The measured request is one user message holding one text part; the count includes the chat-template
    wrapper of that single message (a small constant), which makes the measure conservative for a block that is itself a part of a larger prompt.
    """

    name = MEASURE_VLLM

    def __init__(self, client: Any, *, model: str | None = None, chat_template_kwargs: Mapping[str, Any] | None = None) -> None:
        if not callable(getattr(client, "request_json", None)):
            raise MeasureError("the vLLM tokenize measure needs a client with request_json(method, path, payload)")
        self._client = client
        self._model = model
        self._kwargs = dict(chat_template_kwargs) if chat_template_kwargs else None
        self._cache: dict[str, int] = {}
        self.calls = 0

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise MeasureError("only text can be measured")
        if text == "":
            return 0
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        from speech_aware_evidence_acquisition.core.preparation import PreparationError, measure_vllm_multimodal_request  # noqa: PLC0415

        body: dict[str, Any] = {"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}], "max_tokens": 0}
        if self._model:
            body["model"] = self._model
        if self._kwargs:
            body["chat_template_kwargs"] = dict(self._kwargs)
        self.calls += 1
        try:
            measured = measure_vllm_multimodal_request(self._client, body, slot_tokens=2**31 - 1, fit_multiple=1.0, special_tokens=0)
        except PreparationError as error:
            raise MeasureError(str(error)) from error
        count = measured["measured_prompt_tokens"]
        self._cache[text] = count
        return count

    def detail(self) -> Mapping[str, Any]:
        return {"method": "vLLM chat-form POST /tokenize via core.preparation.measure_vllm_multimodal_request", "wrapper_tokens_included": True,
                "model": self._model, "chat_template_kwargs": self._kwargs}


def default_measure() -> TokenMeasure:
    return EstimateMeasure()


# ----------------------------------------------------------------------------------------------------------------------------------------
# item text
# ----------------------------------------------------------------------------------------------------------------------------------------


def render_evidence_line(evidence_id: str, content: str) -> str:
    """The line the transport renders for one supplied evidence part: ``[evidence <evidence_id>] <content>``."""

    return f"[evidence {evidence_id}] {content}"


def content_sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolvedItem:
    item_id: str
    content: str
    content_sha256: str
    rendered: str


def _fold(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def pipe_item_fields(content: str) -> tuple[str | None, list[str]]:
    """``(intent, slot value strings)`` of an exemplar store item in the pipe form ``<utterance> | <intent> | <slot>=<value>; ...``, else ``(None, [])``."""

    parts = [p.strip() for p in content.split("|")]
    if len(parts) != 3 or not parts[1]:
        return None, []
    values: list[str] = []
    for chunk in parts[2].split(";"):
        _, sep, value = chunk.partition("=")
        if sep and value.strip():
            values.append(value.strip())
    return parts[1], values


def pipe_item_used(content: str, answer: Mapping[str, Any]) -> bool | None:
    """The repo's usage rule (``core.rollout_trace.usage_flags``) applied to the pipe form: intent equal to the answer's intent, or a slot value
    occurring (word boundaries, case- and punctuation-folded) in one of the answer's fillers. ``None`` when the content is not in the pipe form."""

    intent, values = pipe_item_fields(content)
    if intent is None:
        return None
    if intent == f"{answer['scenario']}_{answer['action']}":
        return True
    fillers = [_fold(e["filler"]) for e in answer["entities"]]
    return any(needle and any(f" {needle} " in f" {filler} " for filler in fillers) for needle in (_fold(v) for v in values))


class ItemTextResolver:
    """Maps a supplied item id (``<store>::<local id>``) to the text the Solver read. Unknown ids are refused."""

    def __init__(self, texts: Mapping[str, str], *, source: Mapping[str, Any] | None = None) -> None:
        if not texts:
            raise ItemTextError("the resolver holds no item text")
        bad = [i for i, t in texts.items() if not isinstance(i, str) or not i or not isinstance(t, str) or not t]
        if bad:
            raise ItemTextError(f"{len(bad)} items have an empty id or empty content")
        self._texts = dict(texts)
        self._source = dict(source) if source else {"kind": "in-memory"}
        index = hashlib.sha256()
        for item_id in sorted(self._texts):
            index.update(f"{item_id}\x00{content_sha256_text(self._texts[item_id])}\n".encode("utf-8"))
        self._index_sha256 = index.hexdigest()

    def __len__(self) -> int:
        return len(self._texts)

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._texts

    def resolve(self, item_id: str) -> ResolvedItem:
        content = self._texts.get(item_id)
        if content is None:
            raise ItemTextError(f"unknown supplied item id {item_id!r}: it is not a value of the knowledge stores this resolver was built from")
        return ResolvedItem(item_id=item_id, content=content, content_sha256=content_sha256_text(content),
                            rendered=render_evidence_line(item_id, content))

    def identity(self) -> dict[str, Any]:
        """What a bank receipt records about the text source (hashes and counts only, no item text)."""

        return {"resolver": RESOLVER_ID, "renderer": RENDERER_FORMAT, "renderer_source": RENDERER_SOURCE, "rank_label": "none",
                "values": len(self._texts), "values_index_sha256": self._index_sha256, "source": dict(self._source)}

    @classmethod
    def from_binding(cls, binding: Any, *, source: Mapping[str, Any] | None = None) -> "ItemTextResolver":
        """From a ``KnowledgeStoreBinding``: the composite snapshot's values, whose ids are the qualified ids the trace records."""

        texts = {value.value_id: value.content for value in binding.composite_snapshot.values}
        info = {"kind": "knowledge-store-binding", "binding_sha256": binding.binding_sha256, "stores": list(binding.store_ids)}
        info.update(source or {})
        return cls(texts, source=info)

    @classmethod
    def from_pins(cls, data_root: Path, pins_path: Path | None = None) -> "ItemTextResolver":
        """From the registered port pins and the data root (``rails.knowledge_stores.load_knowledge_store_binding``: every pin and hash verified)."""

        from speech_aware_evidence_acquisition.rails.knowledge_stores import KnowledgeStoreError, load_knowledge_store_binding  # noqa: PLC0415

        path = Path(pins_path) if pins_path else Path(__file__).resolve().parents[3] / PINS_RELPATH
        try:
            pins = json.loads(path.read_text(encoding="utf-8"))
            stores = pins["knowledge_stores"]
            binding = load_knowledge_store_binding({"knowledge_stores": stores}, data_root=Path(data_root))
        except (OSError, ValueError, KeyError, KnowledgeStoreError) as error:
            raise ItemTextError(f"cannot load the knowledge stores of {path}: {error}") from error
        if binding is None:
            raise ItemTextError("the pins declare no knowledge stores")
        return cls.from_binding(binding, source={"pins": PINS_RELPATH, "pins_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def used_items(contact_items: Sequence[Mapping[str, Any]], answer: Mapping[str, Any], resolver: ItemTextResolver | None, limit: int) -> list[dict[str, Any]]:
    """The supplied items an exemplar's answer used, in supplied order, at most ``limit``.

    The trace writer's judgement is preferred: an item with ``used`` true is used, basis ``trace`` (the form the writer read is
    its own ``used_basis`` field and is not repeated here); an item with ``used`` false is
    not used and is never re-judged. Only an item the writer could not judge (``used`` null, ``used_basis`` ``none`` or absent: a trace written
    before the writer read the pipe form) is judged by :func:`pipe_item_used` on its resolved content, basis ``content-pipe-form``.
    Returns ``[{"item_id", "basis"}]``.
    """

    out: list[dict[str, Any]] = []
    for item in contact_items:
        if item["used"] is True:
            out.append({"item_id": item["item_id"], "basis": "trace"})
        elif item["used"] is None and resolver is not None and item["item_id"] in resolver:
            if pipe_item_used(resolver.resolve(item["item_id"]).content, answer) is True:
                out.append({"item_id": item["item_id"], "basis": "content-pipe-form"})
        if len(out) >= limit:
            break
    return out
