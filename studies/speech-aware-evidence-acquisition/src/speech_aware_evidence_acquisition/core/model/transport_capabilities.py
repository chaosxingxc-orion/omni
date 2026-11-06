"""Inspect scheduling compatibility without changing the dispatch guard chain."""

from __future__ import annotations


def supports_single_vllm_ready_request(transport: object) -> bool:
    """Allow only known guards around one concrete vLLM endpoint.

    Ready requests may acquire their slot after FIT because vLLM removes slot
    metadata from the wire body. Arbitrary proxies, multiple endpoints and
    cycles fail closed. This inspection never returns an unwrapped transport:
    dispatch must still run the original lease and resource guards.
    """
    from ..session.resource_guard import ResourceCheckedTransport
    from ..session.worker import LeaseCheckedTransport
    from .vllm_router_transport import VllmRouterTransport
    from .vllm_transport import VllmServerTransport

    seen: set[int] = set()
    while id(transport) not in seen:
        seen.add(id(transport))
        if type(transport) is LeaseCheckedTransport:
            transport = transport._transport
        elif type(transport) is ResourceCheckedTransport:
            transport = transport.transport
        elif type(transport) is VllmRouterTransport:
            if len(transport._transports) != 1:
                return False
            transport = transport._transports[0]
        else:
            return type(transport) is VllmServerTransport
    return False
