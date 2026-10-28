"""``core.vllm_embedding``: the study's embedding recipe as a client of a vLLM pooling server.

The server is canned (``post``/``get`` callables); the serving view is a real one built by ``core.vllm_serving_view``
over a tiny synthetic checkpoint, so the view verification and inventory binding run for real.
"""

from __future__ import annotations

import base64
import io
import json
import math
import urllib.error

import pytest

from speech_aware_evidence_acquisition.core import vllm_embedding as ve
from speech_aware_evidence_acquisition.core import vllm_serving_view as sv

from test_vllm_serving_view import _checkpoint

_SERVED = "omni-embed-nemotron-3b-causal"


@pytest.fixture()
def view(tmp_path):
    source = _checkpoint(tmp_path / "src")
    sv.build_serving_view(source, tmp_path / "view", view_id="nemotron-3b-causal-v1")
    return tmp_path / "view"


def _inventory(view):
    receipt = json.loads((view / sv.RECEIPT_NAME).read_text())
    files = {}
    for name, entry in receipt["files"].items():
        files[entry.get("source", name)] = {"sha256": entry["source_sha256"], "bytes": entry["source_bytes"]}
    return {"schema": "saea-pinned-speech-query-encoder-inventory-v1", "files": files}


def _unit(dimensions=2048):
    return [1.0] + [0.0] * (dimensions - 1)


def _response(vector, prompt_tokens=40):
    return json.dumps(
        {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": vector}],
         "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}}
    ).encode()


class _Server:
    def __init__(self, view, *, models=None, response=None):
        self.models = models if models is not None else [{"id": _SERVED, "object": "model", "root": str(view)}]
        self.response = response if response is not None else _response(_unit())
        self.posts = []

    def get(self, url):
        assert url.endswith("/v1/models")
        return json.dumps({"object": "list", "data": self.models}).encode()

    def post(self, url, body):
        self.posts.append((url, json.loads(body)))
        return self.response


def _client(view, server, **kwargs):
    return ve.VllmEmbeddingClient(
        "http://127.0.0.1:8100/", served_model_name=_SERVED, view=view,
        expected_receipt_sha256=sv.receipt_sha256(view), post=server.post, get=server.get, **kwargs,
    )


# --- pure pieces -----------------------------------------------------------------------------------------------------


def test_wav_bytes_round_trip_every_sample_exactly():
    soundfile = pytest.importorskip("soundfile")
    import numpy as np

    wave = np.array([0.0, 0.123456789, -0.987654321, 1e-8, 0.5], dtype=np.float32)
    decoded, rate = soundfile.read(io.BytesIO(ve.wav_bytes(wave)), dtype="float32")
    assert rate == 16000 and decoded.tobytes() == wave.tobytes()


@pytest.mark.parametrize("wave", [[], [[0.1, 0.2]], [float("nan")]])
def test_wav_bytes_refuses_empty_stereo_or_non_finite_audio(wave):
    with pytest.raises(ve.VllmEmbeddingError):
        ve.wav_bytes(wave)


def test_speech_content_keeps_the_recipe_order_prompt_then_audio():
    content = ve.speech_content("query: ", [0.1, 0.2])
    assert [part["type"] for part in content] == ["text", "input_audio"]
    assert content[0]["text"] == "query: " and content[1]["input_audio"]["format"] == "wav"
    assert base64.b64decode(content[1]["input_audio"]["data"])[:4] == b"RIFF"


def test_request_renders_without_a_generation_prompt():
    body = ve.embedding_request(_SERVED, ve.text_content("passage: x"))
    assert body["model"] == _SERVED and body["add_generation_prompt"] is False
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "passage: x"}]}]


def test_parse_accepts_one_unit_vector_within_the_token_bound():
    assert ve.parse_embedding_response(_response(_unit(4)), dimensions=4, normalized=True, max_prompt_tokens=40) == (1.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "raw, match",
    [
        (_response([1.0, 0.0, 0.0]), "dimensions"),
        (_response([0.5, 0.5, 0.0, 0.0]), "L2-normalized"),
        (_response([float("nan"), 0.0, 0.0, 0.0]).replace(b"NaN", b"NaN"), "finite"),
        (json.dumps({"data": [{"embedding": _unit(4)}, {"embedding": _unit(4)}]}).encode(), "exactly one"),
        (_response(_unit(4), prompt_tokens=41), "above the registered bound"),
        (json.dumps({"data": [{"embedding": _unit(4)}]}).encode(), "prompt_tokens"),
        (b"<html>", "not JSON"),
    ],
)
def test_parse_refuses_anything_but_the_recipe_output(raw, match):
    with pytest.raises(ve.VllmEmbeddingError, match=match):
        ve.parse_embedding_response(raw, dimensions=4, normalized=True, max_prompt_tokens=40)


# --- the client ------------------------------------------------------------------------------------------------------


def test_client_posts_the_recipe_body_and_returns_the_vector(view):
    server = _Server(view)
    client = _client(view, server, inventory=_inventory(view))
    vector = client.embed_speech("query: ", [0.1, -0.2, 0.3], max_audio_samples=16000, max_prompt_tokens=8192)
    assert len(vector) == 2048 and math.isclose(sum(v * v for v in vector), 1.0)
    (url, body), = server.posts
    assert url == "http://127.0.0.1:8100/v1/embeddings"
    assert body["model"] == _SERVED and body["messages"][0]["content"][0] == {"type": "text", "text": "query: "}


@pytest.mark.parametrize(
    "models, match",
    [
        ([], "exactly one model"),
        ([{"id": "other", "root": "x"}], "not 'omni-embed"),
        ([{"id": _SERVED, "root": "/elsewhere"}], "not the serving view"),
    ],
)
def test_client_refuses_a_server_that_does_not_serve_the_view(view, models, match):
    with pytest.raises(ve.VllmEmbeddingError, match=match):
        _client(view, _Server(view, models=models))


def test_client_refuses_a_drifted_view_before_contacting_the_server(view):
    (view / "chat_template.jinja").write_text("changed")
    server = _Server(view)
    with pytest.raises(sv.ServingViewError):
        _client(view, server)
    assert server.posts == []


def test_client_refuses_audio_beyond_the_registered_bound(view):
    server = _Server(view)
    with pytest.raises(ve.VllmEmbeddingError, match="input bound"):
        _client(view, server).embed_speech("query: ", [0.1] * 11, max_audio_samples=10)
    assert server.posts == []


def test_http_errors_are_refusals_not_vectors(view):
    server = _Server(view)

    def post(url, body):
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b'{"error": "too long"}'))

    client = _client(view, server)
    client._post = post
    with pytest.raises(ve.VllmEmbeddingError, match="HTTP 400"):
        client.embed_text("passage: x")


def test_the_view_must_derive_from_the_pinned_inventory(view):
    inventory = _inventory(view)
    ve.assert_view_matches_inventory(json.loads((view / sv.RECEIPT_NAME).read_text()), inventory)
    changed = json.loads(json.dumps(inventory))
    changed["files"]["tokenizer.json"]["sha256"] = "0" * 64
    with pytest.raises(ve.VllmEmbeddingError, match="differs from the pinned inventory"):
        _client(view, _Server(view), inventory=changed)
    missing = json.loads(json.dumps(inventory))
    del missing["files"]["vocab.json"]
    with pytest.raises(ve.VllmEmbeddingError, match="not in the pinned inventory"):
        _client(view, _Server(view), inventory=missing)


def test_backend_factory_builds_drop_in_encoders(view):
    server = _Server(view)
    client = _client(view, server)
    events = []
    protocol = {"passage_prompt": "passage: ", "query_prompt": "query: ", "max_sequence_tokens": 8192,
                "max_audio_samples": 16000, "cpu_workers": 1}
    passage = ve.vllm_backend_factory(client, ve.VllmPassageEncoder)(_inventory(view), protocol, None, events.append)
    assert len(passage("hello")) == 2048
    assert server.posts[-1][1]["messages"][0]["content"] == [{"type": "text", "text": "passage: hello"}]
    speech = ve.vllm_backend_factory(client)(_inventory(view), protocol, None, events.append)
    speech.prefetch.take = lambda source: [0.25, -0.25]
    assert len(speech({"source": "s"})) == 2048
    speech.close()
    assert events == ["encoder_forward_started", "encoder_forward_finished", "data_wait_started", "data_wait_finished",
                      "encoder_forward_started", "encoder_forward_finished"]
