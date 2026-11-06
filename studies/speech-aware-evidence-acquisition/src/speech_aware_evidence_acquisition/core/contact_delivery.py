"""Durable generation dispatch boundary, scoped to one adapter invocation.

Only the instrumented transport can establish that rendering began. Unknown
transports and failures after POST admission remain unknown. No retries occur.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .contact_journal import digest

_current = ContextVar("generation_delivery", default=None)


@dataclass
class _Delivery:
    journal: object
    request_id: str
    request_sha256: str
    rendering: bool = False
    dispatching: bool = False


@contextmanager
def delivery_scope(journal, request_id, request_sha256):
    state = _Delivery(journal, request_id, request_sha256)
    token = _current.set(state)
    try:
        yield
    except Exception as error:
        if journal is not None and state.rendering and not state.dispatching:
            journal.delivery_event(request_id, request_sha256, "not_sent",
                                   error_type=type(error).__name__)
        raise
    finally:
        _current.reset(token)


def begin_render(request_bytes):
    state = _current.get()
    if state is None:
        return
    if state.rendering or digest(request_bytes) != state.request_sha256:
        raise ValueError("generation delivery scope differs from transport request")
    state.rendering = True


def before_post(request_bytes):
    state = _current.get()
    if state is None:
        return
    if not state.rendering or state.dispatching or digest(request_bytes) != state.request_sha256:
        raise ValueError("generation dispatch boundary differs from rendering")
    # Set first: failure while persisting admission is conservatively UNKNOWN.
    state.dispatching = True
    if state.journal is not None:
        state.journal.delivery_event(state.request_id, state.request_sha256, "dispatch")
