"""The ``vllm`` transport: the llama-server request path, translated to vLLM's wire dialect and nothing else.

Every guard the base transport applies must still apply, so the full-path tests below go through a REAL
``VllmServerTransport`` whose session is verified for real (a live process running a pinned binary copy and owning the
listening socket, exactly as ``tests/unit/test_llama_server_transport.py`` builds it). Only the HTTP exchange is canned.
The canned vLLM bodies use the field names of vLLM 0.30's ``ChatCompletionResponse`` (``message.reasoning``, ``usage``
``*_tokens_details: null``) and its ``ErrorResponse`` (``{"error": {"message", "type", "param", "code"}}``).
"""

import hashlib
import io
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error

import pytest

from speech_aware_evidence_acquisition.contracts import STUDY_SLUG
from speech_aware_evidence_acquisition.core.base import TRANSPORTS
from speech_aware_evidence_acquisition.core.model import (
    FrozenCoreContextSizeError,
    FrozenCoreError,
    VllmServerTransport,
)
from speech_aware_evidence_acquisition.core.model.vllm_transport import (
    vllm_context_error_body,
    vllm_request_body,
    vllm_response_bytes,
)
from speech_aware_evidence_acquisition.core.reasoning_knobs import (
    ReasoningKnobError,
    message_sha256,
    validate_fragment_knobs,
    vllm_forced_close_message_from_server_args,
)
from speech_aware_evidence_acquisition.core.session import (
    SESSION_RECEIPT_SCHEMA,
    ModelFileRef,
    SessionError,
    SessionReceipt,
)

_BINARY_PATH = "vllm/bin/python3"
_MODEL_FILES = (ModelFileRef(path="models/qwen3-omni-thinking/model-00001-of-00016.safetensors", sha256="a" * 64),)
_SERVED = "qwen3-omni-30b-a3b-thinking"
_FORCED = "Considering the limited time by the user, I have to give the solution based on the thinking directly now."
_CONTEXT_400 = (
    "The decoder prompt (length 24601) is longer than the maximum model length of 24576. Make sure that "
    "`max_model_len` is no smaller than the number of text tokens plus multimodal tokens. For image inputs, the number "
    "of image tokens depends on the number of images, and possibly their aspect ratios as well."
)

_live: dict[str, object] = {}


@pytest.fixture(scope="module", autouse=True)
def _live_listener(tmp_path_factory):
    """A real process, running a pinned binary copy, that owns a real listening socket (see the module docstring)."""

    runtime_root = tmp_path_factory.mktemp("vllm-session") / "runtime"
    binary = runtime_root / _BINARY_PATH
    binary.parent.mkdir(parents=True)
    shutil.copy(sys.executable, binary)
    if os.name != "nt":
        binary.chmod(0o755)
    code = (
        "import socket, time\n"
        "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "s.bind(('127.0.0.1', 0)); s.listen(1)\n"
        "print(s.getsockname()[1], flush=True); time.sleep(300)\n"
    )
    process = subprocess.Popen([str(binary), "-c", code], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    box: "queue.Queue[str]" = queue.Queue(maxsize=1)
    threading.Thread(target=lambda: box.put(process.stdout.readline()), daemon=True).start()
    port = int(box.get(timeout=15.0).strip())
    deadline = time.monotonic() + 5.0
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    _live.update(
        runtime_root=runtime_root,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        pid=process.pid,
        port=port,
    )
    yield
    process.terminate()
    process.wait(timeout=5)


def _session():
    return SessionReceipt(
        schema=SESSION_RECEIPT_SCHEMA,
        study=STUDY_SLUG,
        session_id="test-vllm-session",
        binary_path=_BINARY_PATH,
        binary_sha256=_live["binary_sha256"],
        argv=(_BINARY_PATH, "--host", "127.0.0.1", "--port", str(_live["port"])),
        pid=_live["pid"],
        host="127.0.0.1",
        port=_live["port"],
        model_files=_MODEL_FILES,
        started_at="2026-10-28T00:00:00+00:00",
        startup_stdout_sha256=hashlib.sha256(b"").hexdigest(),
    )


def _runtime_receipt():
    return {
        "llama_cpp": {"binary_path": _BINARY_PATH, "binary_sha256": _live["binary_sha256"]},
        "gguf_files": [{"path": f.path, "sha256": f.sha256} for f in _MODEL_FILES],
    }


def _transport(data_root, post, **kwargs):
    return VllmServerTransport(
        f"http://127.0.0.1:{_live['port']}",
        data_root,
        _session(),
        _runtime_receipt(),
        _live["runtime_root"],
        post=post,
        served_model_name=_SERVED,
        **kwargs,
    )


def _payload(tmp_path, **decoding):
    media = tmp_path / "media" / "s1.flac"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"fake-flac-bytes")
    return {
        "request_id": "req-1",
        "carrier_lock_key": "slurp",
        "sample_id": "s1",
        "speech_ref": "slurp/s1",
        "media_relpath": "media/s1.flac",
        "audio_seconds": 2.0,
        "sample_rate_hz": 16000,
        "task_instruction": "Answer with the intent label.",
        "history": [],
        "decoding_params": {"temperature": 0, "seed": 20260913, **decoding},
    }


def _vllm_response(content, *, reasoning=None, details=None):
    message = {"role": "assistant", "content": content, "refusal": None, "tool_calls": [], "reasoning": reasoning}
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": _SERVED,
            "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 5,
                "total_tokens": 16,
                "prompt_tokens_details": details,
            },
        }
    ).encode()


class _Recorder:
    def __init__(self, response):
        self.response, self.calls = response, []

    def __call__(self, url, body):
        self.calls.append((url, json.loads(body)))
        return self.response


# --- pure translations -----------------------------------------------------------------------------------------------


def test_request_carries_the_served_model_and_vllms_budget_field():
    body = vllm_request_body({"messages": [], "thinking_budget_tokens": 3072, "max_tokens": 4096}, _SERVED)
    assert body == {"messages": [], "thinking_token_budget": 3072, "max_tokens": 4096, "model": _SERVED}


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [], "id_slot": 0},
        {"messages": [], "thinking_token_budget": 10},
        {"messages": [], "model": "some-other-model"},
    ],
)
def test_request_refuses_llama_controls_direct_budget_and_a_foreign_model(body):
    with pytest.raises(SessionError):
        vllm_request_body(body, _SERVED)


def test_response_moves_vllms_reasoning_channel_to_the_name_the_projection_strips():
    normalized = json.loads(vllm_response_bytes(_vllm_response("PlayMusic", reasoning="the user wants music")))
    message = normalized["choices"][0]["message"]
    assert "reasoning" not in message
    assert message["reasoning_content"] == "the user wants music"
    assert "prompt_tokens_details" not in normalized["usage"]


def test_response_with_nothing_to_translate_is_passed_through_byte_for_byte():
    raw = json.dumps({"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": 1}}).encode()
    assert vllm_response_bytes(raw) is raw
    assert vllm_response_bytes(b"not json") == b"not json"


def test_response_with_two_different_reasoning_channels_is_refused():
    raw = json.dumps({"choices": [{"message": {"content": "x", "reasoning": "a", "reasoning_content": "b"}}]}).encode()
    with pytest.raises(SessionError):
        vllm_response_bytes(raw)


def test_only_vllms_exact_context_overflow_messages_are_relabelled():
    def body(message, error_type="BadRequestError"):
        return json.dumps({"error": {"message": message, "type": error_type, "param": None, "code": 400}}).encode()

    relabelled = json.loads(vllm_context_error_body(400, body(_CONTEXT_400)))
    assert relabelled["error"]["type"] == FrozenCoreContextSizeError.WIRE_ERROR_TYPE
    assert relabelled["error"]["message"] == _CONTEXT_400
    plus_output = (
        "The decoder prompt (length 24576) plus the number of requested output tokens (at least 1) is longer than the "
        "maximum model length of 24576. Make sure that `max_model_len` is no smaller than the number of text tokens "
        "(prompt + requested output tokens)."
    )
    assert vllm_context_error_body(400, body(plus_output)) is not None
    # reworded, prefixed, wrong type, wrong status, unparseable: never guessed at
    assert vllm_context_error_body(400, body(_CONTEXT_400.replace("longer", "larger"))) is None
    assert vllm_context_error_body(400, body("note: " + _CONTEXT_400)) is None
    assert vllm_context_error_body(400, body(_CONTEXT_400, "NotFoundError")) is None
    assert vllm_context_error_body(500, body(_CONTEXT_400)) is None
    assert vllm_context_error_body(400, b"<html>") is None


# The serving-layer wording of vLLM 0.30.0 (raw evidence: the smoke check "overflow", HTTP 400, param input_tokens).
_SERVING_CONTEXT_400 = (
    "This model's maximum context length is 24576 tokens. However, you requested 16 output tokens and your prompt "
    "contains at least 24561 input tokens, for a total of at least 24577 tokens. Please reduce the length of the "
    "input prompt or the number of requested output tokens. (parameter=input_tokens, value=24561)"
)


def _error_body(message, error_type="BadRequestError", param=None):
    return json.dumps({"error": {"message": message, "type": error_type, "param": param, "code": 400}}).encode()


def test_serving_layer_context_overflow_wording_is_relabelled():
    relabelled = json.loads(vllm_context_error_body(400, _error_body(_SERVING_CONTEXT_400, param="input_tokens")))
    assert relabelled["error"]["type"] == FrozenCoreContextSizeError.WIRE_ERROR_TYPE
    assert relabelled["error"]["message"] == _SERVING_CONTEXT_400
    assert relabelled["error"]["code"] == 400
    # the trailing parameter clause is optional
    without_clause = _SERVING_CONTEXT_400.split(" (parameter=")[0]
    assert vllm_context_error_body(400, _error_body(without_clause)) is not None


def test_engine_level_context_overflow_wording_is_still_relabelled():
    relabelled = json.loads(vllm_context_error_body(400, _error_body(_CONTEXT_400)))
    assert relabelled["error"]["type"] == FrozenCoreContextSizeError.WIRE_ERROR_TYPE


def test_serving_layer_near_misses_are_not_relabelled():
    # one changed word, a prefix, a different parameter clause, a foreign error type, a foreign status
    assert vllm_context_error_body(400, _error_body(_SERVING_CONTEXT_400.replace("reduce", "shorten"))) is None
    assert vllm_context_error_body(400, _error_body(_SERVING_CONTEXT_400.replace("maximum", "minimum"))) is None
    assert vllm_context_error_body(400, _error_body("note: " + _SERVING_CONTEXT_400)) is None
    assert vllm_context_error_body(400, _error_body(_SERVING_CONTEXT_400 + " extra")) is None
    assert vllm_context_error_body(400, _error_body(_SERVING_CONTEXT_400.replace("value=24561", "value=x"))) is None
    assert vllm_context_error_body(400, _error_body(_SERVING_CONTEXT_400, "NotFoundError")) is None
    assert vllm_context_error_body(500, _error_body(_SERVING_CONTEXT_400)) is None


# --- the full transport path ----------------------------------------------------------------------------------------


def test_full_path_sends_vllms_dialect_and_seals_no_reasoning_bytes(tmp_path):
    post = _Recorder(_vllm_response("PlayMusic", reasoning="the user wants music"))
    transport = _transport(
        tmp_path, post, reasoning_content_guard="refuse-think-marker-in-content", reasoning_telemetry=True,
        forced_close_message=_FORCED, preserve_wire_responses=True,
    )
    payload = _payload(
        tmp_path, max_tokens=4096, thinking_budget_tokens=3072, chat_template_kwargs={"enable_thinking": True}
    )
    envelope = json.loads(transport(json.dumps(payload).encode()))
    (url, sent), = post.calls
    assert url.endswith("/v1/chat/completions")
    assert sent["model"] == _SERVED and sent["thinking_token_budget"] == 3072
    assert "thinking_budget_tokens" not in sent and "id_slot" not in sent
    assert envelope["text"] == "PlayMusic"
    assert envelope["usage"] == {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    assert envelope["reasoning"]["reasoning_content_sha256"] == hashlib.sha256(b"the user wants music").hexdigest()
    import base64

    assert b"the user wants music" not in base64.b64decode(envelope["wire_response_b64"])


def test_full_path_degrades_vllms_context_overflow_like_llama_servers(tmp_path):
    def post(url, body):
        detail = json.dumps({"error": {"message": _CONTEXT_400, "type": "BadRequestError", "param": None, "code": 400}})
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(detail.encode()))

    with pytest.raises(FrozenCoreContextSizeError):
        _transport(tmp_path, post)(json.dumps(_payload(tmp_path)).encode())


def test_full_path_other_400s_still_raise_hard(tmp_path):
    def post(url, body):
        detail = json.dumps({"error": {"message": "bad tool schema", "type": "BadRequestError", "param": None, "code": 400}})
        raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(detail.encode()))

    with pytest.raises(FrozenCoreError) as caught:
        _transport(tmp_path, post)(json.dumps(_payload(tmp_path)).encode())
    assert not isinstance(caught.value, FrozenCoreContextSizeError)


def test_full_path_keeps_the_reasoning_guard(tmp_path):
    post = _Recorder(_vllm_response("<think>leaked</think>PlayMusic"))
    transport = _transport(tmp_path, post, reasoning_content_guard="refuse-think-marker-in-content")
    with pytest.raises(FrozenCoreError, match="reasoning marker"):
        transport(json.dumps(_payload(tmp_path)).encode())


def test_full_path_refuses_a_fragment_that_still_declares_slot_affinity(tmp_path):
    post = _Recorder(_vllm_response("PlayMusic"))
    with pytest.raises(SessionError, match="id_slot"):
        _transport(tmp_path, post)(json.dumps(_payload(tmp_path, id_slot=2)).encode())
    assert post.calls == []


def test_speech_feature_packet_orders_are_refused(tmp_path):
    with pytest.raises(SessionError, match="raw"):
        _transport(tmp_path, _Recorder(b"{}"), prompt_order="speech-query-prefix-v2")


def test_session_is_still_verified_for_real(tmp_path):
    session = _session()
    forged = SessionReceipt(**{**session.__dict__, "pid": 2**22 - 7})
    with pytest.raises(SessionError):
        VllmServerTransport(
            f"http://127.0.0.1:{_live['port']}", tmp_path, forged, _runtime_receipt(), _live["runtime_root"],
            post=_Recorder(b"{}"), served_model_name=_SERVED,
        )


# --- the fragment surface -------------------------------------------------------------------------------------------


def _fragment(**overrides):
    values = {
        "transport": "vllm",
        "base_url": f"http://127.0.0.1:{_live['port']}",
        "served_model_name": _SERVED,
        "server_args": ["--reasoning-parser", "qwen3", "--reasoning-config",
                        json.dumps({"reasoning_end_str": _FORCED + "\n</think>\n\n"})],
        "decoding_params": {"temperature": 0, "max_tokens": 4096},
        "reasoning_content_guard": "refuse-think-marker-in-content",
        "reasoning_budget_tokens": 3072,
        "answer_reserve_tokens": 1024,
        "enable_thinking": True,
        "forced_close_message_sha256": message_sha256(_FORCED),
    }
    values.update(overrides)
    return values


def test_forced_close_message_is_read_from_vllms_reasoning_config():
    assert vllm_forced_close_message_from_server_args(_fragment()["server_args"]) == _FORCED
    validate_fragment_knobs(_fragment())
    with pytest.raises(ReasoningKnobError, match="differs"):
        validate_fragment_knobs(_fragment(forced_close_message_sha256="0" * 64))


@pytest.mark.parametrize(
    "end",
    [_FORCED, "  \n</think>", 7],
)
def test_a_reasoning_end_string_that_cannot_close_or_be_detected_is_refused(end):
    with pytest.raises(ReasoningKnobError):
        vllm_forced_close_message_from_server_args(["--reasoning-config", json.dumps({"reasoning_end_str": end})])


def test_the_vllm_factory_refuses_llama_server_flags_and_wires_the_forced_close(tmp_path):
    factory = TRANSPORTS.resolve("vllm")
    common = dict(session=_session(), runtime_receipt=_runtime_receipt(), runtime_root=_live["runtime_root"])
    with pytest.raises(SessionError, match="llama-server flags"):
        factory(_fragment(server_args=["-np", "4"]), tmp_path, **common)
    transport = factory(_fragment(), tmp_path, **common)
    assert isinstance(transport, VllmServerTransport)
    assert transport._forced_close_message == _FORCED
    assert transport.served_model_name == _SERVED
