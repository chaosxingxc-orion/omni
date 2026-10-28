"""The vLLM port's speech layout and token fit.

``native-raw-speech-query-prefix-v1`` must render EXACTLY the ``speech-query-prefix-v2`` layout with the original audio
in the query-packet slot, and ``measure_vllm_multimodal_request`` must count through vLLM's chat-form ``/tokenize``.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from speech_aware_evidence_acquisition.core.model import FrozenCoreError
from speech_aware_evidence_acquisition.core.model.transport import (
    _RAW_QUERY_PREFIX_PROMPT_ORDER,
    _ordered_speech_feature_content,
    _raw_query_prefix_content,
)
from speech_aware_evidence_acquisition.core.preparation import (
    PreparationError,
    measure_vllm_multimodal_request,
)
from test_raw_native_transport import _payload, _transport

_AUDIO = {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}
_PACKET = {"type": "input_speech_features", "speech_features": {"packet_id": "p", "sha256": "0" * 64, "role": "query"}}


@pytest.mark.parametrize(
    "evidence, roles, state",
    [
        ([], (), []),
        (["[evidence e1] a"], (), []),
        (["[evidence e1] a", "[evidence e2] b", "[evidence e3] c"], (), []),
        (["[evidence e1] a", "[evidence e2] b"], (), ["state one", "state two"]),
        (["[evidence k1] key", "[evidence v1] value", "[evidence u1] untyped"], ("key", "value", None), ["state"]),
        (["[evidence v1] value", "[evidence k1] key"], ("value", "key"), []),
    ],
)
def test_the_raw_prefix_layout_is_prefix_v2_with_the_audio_in_the_query_slot(evidence, roles, state):
    feature = _ordered_speech_feature_content(
        prompt_order="speech-query-prefix-v2", evidence_texts=list(evidence), feature_parts=[_PACKET],
        state_texts=state, knowledge_roles=roles,
    )
    raw = _raw_query_prefix_content(
        evidence_texts=list(evidence), audio_part=_AUDIO, state_texts=state, knowledge_roles=roles,
    )
    assert raw == [_AUDIO if part is _PACKET else part for part in feature]
    assert raw[0] is _AUDIO


def _render(tmp_path, prompt_order, **updates):
    captured = []
    transport = _transport(tmp_path, prompt_order=prompt_order, captured=captured)
    payload, raw = _payload(tmp_path, **updates)
    transport(json.dumps(payload).encode())
    (body,) = captured
    return body, raw


_EVIDENCE = [
    {"evidence_id": "e1", "content": "first", "source_uri": "kb://1"},
    {"evidence_id": "e2", "content": "second", "source_uri": "kb://2"},
]


def test_the_wire_puts_the_original_audio_first_where_raw_v1_puts_it_last(tmp_path):
    prefix, raw = _render(tmp_path, _RAW_QUERY_PREFIX_PROMPT_ORDER, supplied_evidence=_EVIDENCE)
    legacy, _ = _render(tmp_path / "legacy", "native-raw-speech-v1", supplied_evidence=_EVIDENCE)
    prefix_user, legacy_user = prefix["messages"][1]["content"], legacy["messages"][1]["content"]
    assert [part["type"] for part in prefix_user] == ["input_audio", "text", "text"]
    assert [part["type"] for part in legacy_user] == ["text", "text", "input_audio"]
    assert prefix_user[1:] == legacy_user[:2]
    assert prefix["messages"][0] == legacy["messages"][0]
    import base64

    assert base64.b64decode(prefix_user[0]["input_audio"]["data"]) == raw


def test_the_raw_query_identity_is_still_bound_under_the_prefix_order(tmp_path):
    with pytest.raises(FrozenCoreError, match="differs from the loaded original audio"):
        _render(
            tmp_path, _RAW_QUERY_PREFIX_PROMPT_ORDER,
            raw_speech_query={"media_relpath": "media/window.wav", "sha256": hashlib.sha256(b"other").hexdigest()},
        )


def test_feature_descriptors_are_refused_under_the_prefix_order(tmp_path):
    with pytest.raises(FrozenCoreError, match="mixed raw and feature"):
        _render(
            tmp_path, _RAW_QUERY_PREFIX_PROMPT_ORDER,
            speech_feature_parts=[{"packet_id": "p", "sha256": "0" * 64, "role": "query"}],
        )


# --- token fit -------------------------------------------------------------------------------------------------------


class _Client:
    def __init__(self, response):
        self.response, self.calls = response, []

    def request_json(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _body():
    return {
        "model": "qwen3-omni-30b-a3b-thinking",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": [_AUDIO]}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": 4096,
        "thinking_budget_tokens": 3072,
        "temperature": 0,
        "seed": 20260913,
    }


def test_vllm_fit_counts_the_rendered_prompt_through_chat_tokenize():
    client = _Client({"count": 1500, "tokens": list(range(1500)), "max_model_len": 24576})
    row = measure_vllm_multimodal_request(client, _body(), slot_tokens=24576, fit_multiple=1.0, special_tokens=2)
    ((method, path, request),) = client.calls
    assert (method, path) == ("POST", "/tokenize")
    assert request == {
        "messages": _body()["messages"], "tools": _body()["tools"], "chat_template_kwargs": {"enable_thinking": True},
        "model": "qwen3-omni-30b-a3b-thinking", "add_generation_prompt": True, "add_special_tokens": False,
    }
    assert row["measured_prompt_tokens"] == 1500 and row["measurement"] == "vllm-chat-tokenize"
    assert row["required_tokens"] == 1500 + 4096 + 2 and row["margin_tokens"] == 24576 - 5598 and row["fits"]


@pytest.mark.parametrize(
    "response",
    [
        {"count": 3, "tokens": [1, 2]},
        {"count": True, "tokens": [1]},
        {"tokens": [1]},
        RuntimeError("404"),
    ],
)
def test_vllm_fit_refuses_anything_but_a_consistent_count(response):
    with pytest.raises(PreparationError):
        measure_vllm_multimodal_request(_Client(response), _body(), slot_tokens=24576)


def test_vllm_fit_reports_a_request_that_does_not_fit():
    client = _Client({"count": 21000, "tokens": [0] * 21000})
    row = measure_vllm_multimodal_request(client, _body(), slot_tokens=24576)
    assert row["fits"] is False and row["margin_tokens"] < 0
