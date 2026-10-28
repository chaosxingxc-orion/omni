"""The runtime-identity seam, live half: one interface for "prove the process answering this endpoint is the pinned
runtime" (owner decision 9.3, 2026-10-28: "preferably compatible via an adapter").

``contracts.runtime_identity`` holds the receipt half (what a committed runtime receipt must say, re-verified from
bytes when the gate opens) and the rule that ``contracts`` never imports ``core``. This module completes each adapter
with the half that needs a live process, and is the ONE thing the transport and the session lifecycle call:

``validate_runtime_receipt``  gate time: the receipt half, inherited unchanged from ``contracts.runtime_identity``.
``verify_session``            declared agreement between a session receipt and the runtime receipt (no live process).
``assert_session_alive``      the full per-send re-check for ONE endpoint: declared agreement, liveness, process
                              identity and listening-socket ownership.
``assert_sessions_alive``     the same for a LIST of endpoints (multi-instance serving; see below).
``start_session`` / ``attach_session`` / ``stop_session``  the lifecycle, keyword-for-keyword as the llama path's.

Adapters
========

``llama-server``  :class:`LlamaServerRuntimeIdentity`. Every method delegates to the existing functions, called exactly
                  as their existing callers call them. Nothing in the llama path changed: ``LlamaServerTransport``
                  still calls the module-level ``assert_session_alive`` it always called, and
                  ``core.session.lifecycle``'s ``start_session``/``attach_session``/``stop_session`` still hold the
                  llama bodies themselves (they gained only a dispatch line that is a no-op for a llama receipt), so
                  the monkeypatch seams the existing tests use are untouched.
``vllm``          :class:`VllmRuntimeIdentity`, implemented in ``core.session.vllm_identity`` (see that module for what
                  a Python interpreter + two editable source trees + a container image can prove that an ~18 KB
                  llama-server binary proves by itself).

Choosing one
============

A caller that knows the model fragment's ``transport`` asks :func:`runtime_identity`. A caller that holds only a
runtime receipt asks :func:`runtime_identity_for_receipt`, which selects on the receipt's ``schema``: the vLLM schema
selects the vLLM adapter, and EVERYTHING else -- including an empty mapping and a receipt that is not a mapping at all
-- selects the llama adapter, whose own validation then has the last word. That default is deliberate: it is what
keeps every existing llama call site, and the hand-built receipts in the existing tests, behaving byte for byte as
before. A new backend is registered, not defaulted into.

Multi-instance
==============

Serving is N instances, each with its own port and device group (coordinator design input, 2026-10-28). The interface
is therefore list-of-instances ready: ``assert_session_alive`` is defined per ENDPOINT and never assumes a single
server or a single socket, and ``assert_sessions_alive`` takes the list. The multi-endpoint transport and the card
allocator are deliberately NOT built here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...contracts.runtime_identity import (
    LLAMA_SERVER_TRANSPORT,
    VLLM_TRANSPORT,
    LlamaServerReceiptIdentity,
    VllmReceiptIdentity,
)
from ...contracts.identifiers import VLLM_RUNTIME_RECEIPT_SCHEMA
from . import SessionError
from .receipt import SessionReceipt


class RuntimeIdentity:
    """One serving backend's full runtime identity: its receipt half plus its live-process half."""

    transport: str = ""

    # -- declared agreement ------------------------------------------------
    def verify_session(
        self, session: SessionReceipt, runtime_receipt: Mapping[str, object]
    ) -> SessionReceipt:
        raise NotImplementedError

    # -- live identity -----------------------------------------------------
    def assert_session_alive(
        self,
        session: SessionReceipt,
        runtime_receipt: Mapping[str, object],
        runtime_root: Path,
    ) -> None:
        raise NotImplementedError

    def assert_sessions_alive(
        self,
        sessions: Sequence[SessionReceipt],
        runtime_receipt: Mapping[str, object],
        runtime_root: Path,
    ) -> None:
        """Every endpoint of a serving set, each against its own instance.

        The default implementation is the single-endpoint check applied to each session after refusing a set that
        claims one endpoint, or one pid, twice -- either would mean the caller cannot know which instance it is
        talking to. A backend with a stronger cross-instance rule overrides this.
        """

        if not sessions:
            raise SessionError("a serving set must hold at least one session; refused (I5)")
        endpoints: set[tuple[str, int]] = set()
        pids: set[int] = set()
        for session in sessions:
            session.validate()
            endpoint = (session.host, session.port)
            if endpoint in endpoints:
                raise SessionError(
                    f"two sessions claim the endpoint {session.host}:{session.port}; refused (I5)"
                )
            if session.pid in pids:
                raise SessionError(f"two sessions claim pid {session.pid}; refused (I5)")
            endpoints.add(endpoint)
            pids.add(session.pid)
        for session in sessions:
            self.assert_session_alive(session, runtime_receipt, runtime_root)

    # -- lifecycle ---------------------------------------------------------
    def start_session(self, **kwargs) -> SessionReceipt:
        raise NotImplementedError

    def attach_session(
        self, receipt_path: Path, *, runtime_receipt: Mapping[str, object], runtime_root: Path
    ) -> SessionReceipt:
        raise NotImplementedError

    def stop_session(self, session: SessionReceipt, **kwargs) -> dict[str, object]:
        raise NotImplementedError


class LlamaServerRuntimeIdentity(LlamaServerReceiptIdentity, RuntimeIdentity):
    """The historical llama.cpp identity, reached through the seam. Every method is a delegation, nothing more.

    Each delegation resolves its target through the MODULE OBJECT at call time (``core.session`` for the liveness
    check, ``core.session.lifecycle`` for the lifecycle), never through a name bound at import time, so a test that
    monkeypatches ``core.session.assert_session_alive`` or a lifecycle-level name still observes the call it always
    observed.
    """

    def _session_package(self):
        from .. import session

        return session

    def _lifecycle(self):
        from . import lifecycle

        return lifecycle

    def verify_session(self, session, runtime_receipt):
        return self._session_package().verify_against_runtime_receipt(session, runtime_receipt)

    def assert_session_alive(self, session, runtime_receipt, runtime_root):
        self._session_package().assert_session_alive(session, runtime_receipt, Path(runtime_root))

    def start_session(self, **kwargs):
        kwargs.pop("proc_root", None)
        return self._lifecycle().start_session(**kwargs)

    def attach_session(self, receipt_path, *, runtime_receipt, runtime_root):
        return self._lifecycle().attach_session(
            receipt_path, runtime_receipt=runtime_receipt, runtime_root=runtime_root
        )

    def stop_session(self, session, **kwargs):
        kwargs.pop("proc_root", None)
        return self._lifecycle().stop_session(session, **kwargs)


class VllmRuntimeIdentity(VllmReceiptIdentity, RuntimeIdentity):
    """vLLM in the same container as the harness (decision 9.2); implemented in ``core.session.vllm_identity``."""

    def _module(self):
        from . import vllm_identity

        return vllm_identity

    def verify_session(self, session, runtime_receipt):
        return self._module().verify_vllm_session(session, runtime_receipt)

    def assert_session_alive(self, session, runtime_receipt, runtime_root):
        self._module().assert_vllm_session_alive(session, runtime_receipt, runtime_root)

    def assert_sessions_alive(self, sessions, runtime_receipt, runtime_root):
        self._module().assert_vllm_sessions_alive(sessions, runtime_receipt, runtime_root)

    def start_session(self, **kwargs):
        return self._module().start_vllm_session(**kwargs)

    def attach_session(self, receipt_path, *, runtime_receipt, runtime_root):
        return self._module().attach_vllm_session(
            receipt_path, runtime_receipt=runtime_receipt, runtime_root=runtime_root
        )

    def stop_session(self, session, **kwargs):
        return self._module().stop_vllm_session(session, **kwargs)


RUNTIME_IDENTITIES: Mapping[str, RuntimeIdentity] = {
    LLAMA_SERVER_TRANSPORT: LlamaServerRuntimeIdentity(),
    VLLM_TRANSPORT: VllmRuntimeIdentity(),
}


def runtime_identity(transport: object) -> RuntimeIdentity:
    """The adapter for one model fragment ``transport``; an unregistered transport is refused, never guessed."""

    identity = RUNTIME_IDENTITIES.get(transport) if isinstance(transport, str) else None
    if identity is None:
        raise SessionError(
            f"no runtime identity is registered for transport {transport!r} "
            f"(registered: {sorted(RUNTIME_IDENTITIES)}); refused (I5)"
        )
    return identity


def runtime_identity_for_receipt(runtime_receipt: object) -> RuntimeIdentity:
    """The adapter one runtime receipt belongs to; anything that is not the vLLM schema is the llama path.

    See the module docstring ("Choosing one") for why the llama path is the default rather than a refusal.
    """

    if isinstance(runtime_receipt, Mapping) and runtime_receipt.get("schema") == VLLM_RUNTIME_RECEIPT_SCHEMA:
        return RUNTIME_IDENTITIES[VLLM_TRANSPORT]
    return RUNTIME_IDENTITIES[LLAMA_SERVER_TRANSPORT]
