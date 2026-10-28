"""The ``vllm`` transport: the same frozen-core request path, served by vLLM (vllm-ascend >= 0.30) instead of llama-server.

Port note (npu-vllm-port, 2026-10-28). Everything this study verifies about a model contact -- payload binding, media
confinement, prompt ordering, the reasoning-content guard, the decision-286 reasoning projection, the envelope, the session
re-verification on every send -- lives in ``LlamaServerTransport`` and is reused UNCHANGED by subclassing. What differs
between the two servers is only the OpenAI-compatible wire dialect, so that is the whole of this module, applied at the one
seam ``LlamaServerTransport`` already has for it (its injected ``post`` callable):

request   ``thinking_budget_tokens`` (llama-server budget sampler) is sent as ``thinking_token_budget`` (vLLM's per-request
          field, same meaning); ``model`` is always set to the fragment's ``served_model_name``, so a server serving any
          other model answers 404 instead of silently answering; ``id_slot`` is REFUSED -- it is llama-server slot
          affinity, vLLM has no slots (prefix caching is automatic), and a fragment that declares it was written for the
          other server.
response  vLLM names the reasoning channel ``message.reasoning`` (it renames the deprecated ``reasoning_content``); it is
          moved to ``reasoning_content``, the only name the decision-286 projection strips, so reasoning bytes can never
          ride past it under a name it does not read. ``usage`` entries vLLM sends as ``null`` (``*_tokens_details``) are
          dropped, as the base transport already drops nested detail objects. A response with neither difference is
          passed through byte for byte.
error     a context overflow is the one 400 the base transport degrades (E20, ``compact-or-abstain``). vLLM reports it as a
          plain ``BadRequestError`` whose only distinguishing signal is its message, so the message is matched against the
          two exact templates of vLLM 0.30 (``v1/engine/input_processor.py``) -- full match, anchored, never a substring
          -- and only then re-labelled with the llama-server error type the base transport classifies. Every other 400
          raises exactly as before. A vLLM upgrade that rewords the message therefore fails closed to a hard error.

Speech-feature packet prompt orders are refused: their packets are llama.cpp decoder inputs, and vLLM computes audio
features itself from raw ``input_audio`` parts.
"""

from __future__ import annotations

import io
import json
import re
import urllib.error
from pathlib import Path
from typing import Callable, Mapping

from ...contracts.errors import GateClosed
from ...contracts.gate_runtime_vllm import classify_server_flags
from ..base import TRANSPORTS
from ..reasoning_knobs import (
    BUDGET_MESSAGE_FLAG,
    WIRE_BUDGET_KEY,
    vllm_forced_close_message_from_server_args,
)
from ..session import SessionError
from ..session.runtime_identity import runtime_identity_for_receipt
from . import FrozenCoreContextSizeError
from .transport import (
    _SPEECH_FEATURE_PROMPT_ORDERS,
    LlamaServerTransport,
    _default_post,
    _llama_server_factory,
)

VLLM_TRANSPORT = "vllm"
#: vLLM's per-request spelling of the reasoning budget.
VLLM_WIRE_BUDGET_KEY = "thinking_token_budget"
#: llama-server request fields that have no meaning on vLLM and are refused rather than silently ignored.
VLLM_REFUSED_REQUEST_KEYS = frozenset({"id_slot"})
#: llama-server flags a vLLM fragment must not carry (they would be read as if a llama-server were answering).
VLLM_REFUSED_SERVER_FLAGS = frozenset({BUDGET_MESSAGE_FLAG, "-np", "--parallel", "-c", "--ctx-size", "--reasoning-format"})

# The two prompt-length refusals of vLLM 0.30 (vllm/v1/engine/input_processor.py), matched whole.
_VLLM_CONTEXT_MESSAGES = (
    re.compile(
        r"The (?:decoder |encoder )?prompt \(length \d+\) is longer than the maximum model length of \d+\. "
        r"Make sure that `max_model_len` is no smaller than the number of text tokens(?: plus multimodal tokens\. "
        r"For image inputs, the number of image tokens depends on the number of images, and possibly their aspect "
        r"ratios as well\.|\.)"
    ),
    re.compile(
        r"The (?:decoder |encoder )?prompt \(length \d+\) plus the number of requested output tokens \(at least 1\) is "
        r"longer than the maximum model length of \d+\. Make sure that `max_model_len` is no smaller than the number "
        r"of text tokens \(prompt \+ requested output tokens\)\."
    ),
)


def vllm_request_body(body: Mapping[str, object], served_model_name: str) -> dict[str, object]:
    """Translate one rendered chat-completions body into vLLM's dialect (module docstring, "request")."""

    refused = sorted(VLLM_REFUSED_REQUEST_KEYS & set(body))
    if refused:
        raise SessionError(
            f"request fields {refused} are llama-server controls with no vLLM meaning; a vllm fragment must not declare them"
        )
    if VLLM_WIRE_BUDGET_KEY in body:
        raise SessionError(f"{VLLM_WIRE_BUDGET_KEY} must come from the fragment's reasoning knobs, never directly")
    if "model" in body and body["model"] != served_model_name:
        raise SessionError(f"request model {body['model']!r} is not the served model {served_model_name!r}")
    translated: dict[str, object] = {}
    for key, value in body.items():
        translated[VLLM_WIRE_BUDGET_KEY if key == WIRE_BUDGET_KEY else key] = value
    translated["model"] = served_model_name
    return translated


def vllm_response_bytes(raw: bytes) -> bytes:
    """Normalize one vLLM chat-completions response to the shape the base transport reads (module docstring, "response").

    Bytes that need no change are returned unchanged; bytes that are not a JSON object are returned unchanged too, so
    the base transport's own malformed-response handling sees exactly what the server sent.
    """

    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        return raw
    if not isinstance(document, dict):
        return raw
    changed = False
    choices = document.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict) or "reasoning" not in message:
                continue
            reasoning = message.pop("reasoning")
            changed = True
            if reasoning is None:
                continue
            held = message.get("reasoning_content")
            if held is not None and held != reasoning:
                raise SessionError("vLLM response carries two different reasoning channels; refused")
            message["reasoning_content"] = reasoning
    usage = document.get("usage")
    if isinstance(usage, dict):
        for key in [key for key, value in usage.items() if value is None]:
            del usage[key]
            changed = True
    if not changed:
        return raw
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def vllm_context_error_body(status: int, detail: bytes) -> bytes | None:
    """The llama-server-shaped error body for vLLM's context-overflow 400, else None (module docstring, "error")."""

    if status != 400:
        return None
    try:
        document = json.loads(detail.decode("utf-8"))
    except (UnicodeError, ValueError):
        return None
    error = document.get("error") if isinstance(document, dict) else None
    if not isinstance(error, dict) or error.get("type") != "BadRequestError":
        return None
    message = error.get("message")
    if not isinstance(message, str) or not any(pattern.fullmatch(message) for pattern in _VLLM_CONTEXT_MESSAGES):
        return None
    return json.dumps(
        {"error": {"code": 400, "message": message, "type": FrozenCoreContextSizeError.WIRE_ERROR_TYPE}}
    ).encode("utf-8")


def vllm_post(
    post: Callable[[str, bytes], bytes], served_model_name: str
) -> Callable[[str, bytes], bytes]:
    """Wrap a raw ``post(url, body) -> bytes`` with the three translations above."""

    def send(url: str, body: bytes) -> bytes:
        translated = vllm_request_body(json.loads(body), served_model_name)
        try:
            raw = post(url, json.dumps(translated).encode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read()
            relabelled = vllm_context_error_body(error.code, detail)
            raise urllib.error.HTTPError(
                error.url, error.code, error.reason, error.headers,
                io.BytesIO(relabelled if relabelled is not None else detail),
            ) from error
        return vllm_response_bytes(raw)

    return send


class VllmServerTransport(LlamaServerTransport):
    """``LlamaServerTransport`` speaking vLLM's wire dialect; see the module docstring for exactly what differs."""

    def __init__(
        self,
        base_url: str,
        data_root: Path,
        session,
        runtime_receipt: Mapping[str, object],
        runtime_root: Path,
        timeout_seconds: float = 300.0,
        post: Callable[[str, bytes], bytes] | None = None,
        *,
        served_model_name: str,
        **kwargs,
    ) -> None:
        if not isinstance(served_model_name, str) or not served_model_name:
            raise SessionError("a vllm transport requires a non-empty served_model_name")
        if kwargs.get("prompt_order") in _SPEECH_FEATURE_PROMPT_ORDERS:
            raise SessionError(
                f"prompt_order {kwargs['prompt_order']!r} sends llama.cpp speech-feature packets; vLLM takes raw "
                "input_audio and computes audio features itself"
            )
        self.served_model_name = served_model_name
        inner = post or _default_post(timeout_seconds)
        super().__init__(
            base_url, data_root, session, runtime_receipt, runtime_root, timeout_seconds,
            vllm_post(inner, served_model_name), **kwargs,
        )

    def _assert_session_still_valid(self) -> None:
        """The per-send session re-verification, asked of the runtime-identity seam (decision 9.2/9.3).

        Called at construction and before every single send, exactly where and as often as the base class calls it --
        this override changes only WHICH identity answers, never how strictly or how often. A vLLM runtime receipt is
        answered by the vLLM adapter (interpreter image re-hashed, source trees, pinned environment, namespaces, the
        instance's own device group, and this endpoint's listening socket -- ``core.session.vllm_identity``); anything
        else is the base class's own call, byte for byte, so a fragment that names this transport while still pinning a
        llama.cpp runtime receipt is verified exactly as it was before this seam existed.
        """

        identity = runtime_identity_for_receipt(self._runtime_receipt)
        if identity.transport != VLLM_TRANSPORT:
            super()._assert_session_still_valid()
            return
        identity.assert_session_alive(self._session, self._runtime_receipt, self._runtime_root)


def _assert_fragment_matches_receipt_protocol(
    values: Mapping[str, object], runtime_receipt: Mapping[str, object]
) -> None:
    """Refuse a fragment whose served model name or protocol flags differ from its vLLM runtime receipt's.

    The receipt is written FROM a running server (``scripts/write_vllm_runtime_receipt.py``), so this is the rail that
    catches a fragment edited after the receipt was taken: the request path would otherwise send
    ``model: served_model_name`` and a decoding configuration the pinned server never agreed to. Only the receipt's
    PROTOCOL half is compared -- the deployment half (ports, device groups, ``--max-num-seqs``,
    ``--gpu-memory-utilization``) is per instance and is checked per endpoint when a session is started or verified.
    A fragment pinning a llama.cpp receipt is left to the historical checks (module docstring of
    ``core.session.runtime_identity``).
    """

    identity = runtime_identity_for_receipt(runtime_receipt)
    if identity.transport != VLLM_TRANSPORT:
        return
    protocol = runtime_receipt["protocol"]
    if values.get("served_model_name") != protocol["served_model_name"]:
        raise SessionError(
            f"the fragment serves {values.get('served_model_name')!r} but the vllm runtime receipt pins "
            f"{protocol['served_model_name']!r}"
        )
    try:
        served, protocol_tokens, _deployment = classify_server_flags(values.get("server_args") or ())
    except GateClosed as error:
        raise SessionError(f"fragment server_args: {error}") from error
    if served is not None:
        raise SessionError(
            "a vllm fragment must not set --served-model-name in server_args; it declares served_model_name"
        )
    expected = [str(token) for token in protocol["server_args"]]
    if list(protocol_tokens) != expected:
        raise SessionError(
            f"the fragment's protocol flags {list(protocol_tokens)} differ from the vllm runtime receipt's {expected}"
        )


def _vllm_factory(
    values: Mapping[str, object],
    data_root: Path,
    *,
    session,
    runtime_receipt: Mapping[str, object],
    runtime_root: Path,
    feature_cache_guard=None,
    reasoning_retention=None,
):
    server_args = values.get("server_args") or ()
    flags = {str(token).split("=", 1)[0] for token in server_args}
    refused = sorted(VLLM_REFUSED_SERVER_FLAGS & flags)
    if refused:
        raise SessionError(f"server flags {refused} are llama-server flags; a vllm fragment must not carry them")
    if feature_cache_guard is not None:
        raise SessionError("a vllm fragment has no llama.cpp feature cache to guard")
    _assert_fragment_matches_receipt_protocol(values, runtime_receipt)
    return _llama_server_factory(
        values,
        data_root,
        session=session,
        runtime_receipt=runtime_receipt,
        runtime_root=runtime_root,
        feature_cache_guard=None,
        reasoning_retention=reasoning_retention,
        transport_cls=VllmServerTransport,
        forced_close_message=vllm_forced_close_message_from_server_args(server_args),
        backend_kwargs={"served_model_name": values.get("served_model_name")},
    )


TRANSPORTS.register(VLLM_TRANSPORT, _vllm_factory)
