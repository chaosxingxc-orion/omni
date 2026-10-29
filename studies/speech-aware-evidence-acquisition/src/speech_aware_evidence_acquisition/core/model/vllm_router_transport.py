"""``VllmRouterTransport``: client-side dispatch across N resident vLLM instances by slot affinity (Gap B of the
2026-10-30 Fold B plan, section 4; Decision 2 there).

Why this exists
================

The hashed model fragment carries exactly one ``base_url`` -- the PROTOCOL fact, instance 0's endpoint. A real
deployment is N resident ``vllm serve`` processes, one per tensor-parallel group (e.g. 4 x TP2 on 8 cards), each its
own port and device group -- a DEPLOYMENT fact (``core.deployment.DeploymentProfile`` / ``deployment_record``), never
hashed. Something has to turn one flight's stream of contacts into N endpoints' worth of HTTP without that split ever
looking like a protocol change. This module is that something, and it is deliberately thin:

* Routing is ``decoding_params.id_slot mod instance_count`` -- the SAME ``id_slot`` the runner already sets on every
  contact for llama-server slot affinity (``rails/runner.py:2128,4353,4978,5195,7367``), read here BEFORE the chosen
  instance's own :class:`~.vllm_transport.VllmServerTransport` strips it (Gap A, ``vllm_transport.py``). One runner,
  one slot-affinity scheme, two consumers: the deployment-time instance count, and (via that same transport's own
  translation) nothing the server ever sees.
* Each instance keeps its OWN :class:`~.vllm_transport.VllmServerTransport`, its own :class:`~..session.SessionReceipt`
  and its own runtime-identity re-verification on every send (``core.session.vllm_identity``, reached through the
  seam at ``runtime_identity_for_receipt`` exactly as a single-instance flight reaches it) -- this class adds no
  identity logic of its own, only the one-line choice of WHICH transport answers a given contact.
* A dead or mismatched instance's :class:`~..session.SessionError` is re-raised naming the instance and propagates
  unchanged -- there is no retry, no failover, and no silent re-routing to a different instance. One instance's
  session failure ends the whole flight exactly as a single-instance flight's session failure always has; the
  instance name is added purely so the failure is legible, never to soften it.
* Instance 0 is the protocol fact: :func:`build_vllm_router_transport` refuses to build a router whose instance 0
  endpoint differs from the fragment's own ``base_url`` -- a deployment record that disagreed there would mean the
  hashed protocol and the live deployment are not describing the same server.

Deliberately NOT here: card allocation, profile validation, argv construction (all ``core.deployment``, imported
read-only), and session spawn/attach (``core.flight.runtime``, which builds the N sessions and N
``VllmServerTransport`` instances this module only combines).
"""

from __future__ import annotations

import json
from typing import Callable, Mapping, Sequence

from ..session import SessionError
from .vllm_transport import VllmServerTransport

#: The one decoding-params field the router reads for dispatch, before the chosen instance's own transport strips
#: it (``vllm_transport.vllm_request_body``, Gap A). Kept as a named constant so both halves of the seam -- this
#: module's read and that module's drop -- name the same field, not a repeated string literal.
ROUTING_DECODING_KEY = "id_slot"


class VllmRouterTransportError(SessionError):
    """A router-level refusal: no instances, a routing key that is not a usable slot, or a dead instance.

    Subclasses :class:`~..session.SessionError` so every existing ``except SessionError`` call site (the adapter,
    the flight runtime's retained-row handling) catches this exactly as it catches a single-instance session
    failure; nothing downstream needs to know a router is involved.
    """


def _instance_label(index: int, instance_ids: Sequence[str]) -> str:
    return f"instance {index} ({instance_ids[index]!r})" if index < len(instance_ids) else f"instance {index}"


class VllmRouterTransport:
    """Dispatch one runner contact to exactly one of ``len(transports)`` resident vLLM instances.

    ``transports`` is already built: one :class:`~.vllm_transport.VllmServerTransport` per deployment instance, in
    deployment order (index 0 is the protocol instance). ``instance_ids`` names each one for error messages and the
    run manifest; it defaults to the string index when not given (every existing single-instance caller that never
    names an instance).
    """

    def __init__(
        self,
        transports: Sequence[VllmServerTransport],
        *,
        instance_ids: Sequence[str] | None = None,
    ) -> None:
        transports = tuple(transports)
        if not transports:
            raise VllmRouterTransportError("a vllm router transport requires at least one instance")
        self._transports = transports
        self._instance_ids = (
            tuple(str(value) for value in instance_ids)
            if instance_ids is not None
            else tuple(str(index) for index in range(len(transports)))
        )
        if len(self._instance_ids) != len(self._transports):
            raise VllmRouterTransportError(
                "instance_ids must have exactly one entry per transport "
                f"(got {len(self._instance_ids)} ids for {len(self._transports)} transports)"
            )

    @property
    def instance_count(self) -> int:
        return len(self._transports)

    @property
    def instance_ids(self) -> tuple[str, ...]:
        return self._instance_ids

    def instance_for_slot(self, id_slot: object) -> int:
        """The instance index a given ``id_slot`` routes to; raises on anything that is not a usable slot."""

        if isinstance(id_slot, bool) or not isinstance(id_slot, int) or id_slot < 0:
            raise VllmRouterTransportError(
                f"{ROUTING_DECODING_KEY} must be a non-negative integer for routing, got {id_slot!r}"
            )
        return id_slot % len(self._transports)

    def _route(self, request_bytes: bytes) -> int:
        try:
            payload = json.loads(request_bytes)
        except (UnicodeError, ValueError) as error:
            raise VllmRouterTransportError(f"a vllm router transport requires a JSON request body: {error}") from error
        decoding_params = payload.get("decoding_params") if isinstance(payload, Mapping) else None
        if not isinstance(decoding_params, Mapping) or ROUTING_DECODING_KEY not in decoding_params:
            raise VllmRouterTransportError(
                f"every contact must carry decoding_params.{ROUTING_DECODING_KEY} to route across "
                f"{len(self._transports)} vllm instances; the runner sets it on every contact "
                "(rails/runner.py) -- a request without it never reached the runner's own dispatch"
            )
        return self.instance_for_slot(decoding_params[ROUTING_DECODING_KEY])

    def __call__(self, request_bytes: bytes) -> bytes:
        index = self._route(request_bytes)
        try:
            return self._transports[index](request_bytes)
        except SessionError as error:
            # Fail closed for the WHOLE flight: no retry against the same instance, no failover to another one.
            # The instance is named so the failure is legible; the flight still ends here, exactly as a
            # single-instance session failure always has.
            raise VllmRouterTransportError(
                f"{_instance_label(index, self._instance_ids)} failed; the flight fails closed with no failover "
                f"to another instance: {error}"
            ) from error

    def assert_session_alive(self) -> None:
        """Re-verify every instance's own session (mirrors ``LlamaServerTransport.assert_session_alive``'s public
        entry point, one call per instance since each instance's identity check is scoped to its own endpoint,
        ``core.session.vllm_identity``)."""

        for index, transport in enumerate(self._transports):
            try:
                transport.assert_session_alive()
            except SessionError as error:
                raise VllmRouterTransportError(
                    f"{_instance_label(index, self._instance_ids)} is not alive: {error}"
                ) from error


def build_vllm_router_transport(
    base_url: str,
    instance_endpoints: Sequence[str],
    build_transport: Callable[[int, str], VllmServerTransport],
    *,
    instance_ids: Sequence[str] | None = None,
) -> VllmRouterTransport:
    """Build one router transport from N endpoints in deployment order, refusing a deployment whose instance 0
    endpoint disagrees with the fragment's own ``base_url`` (module docstring: "Instance 0 is the protocol fact").

    ``build_transport(index, endpoint)`` returns that one instance's fully session-bound
    :class:`VllmServerTransport` -- the actual construction (session lookup/spawn, runtime receipt, prepared-cache
    wrapping) stays wherever a single instance already does it
    (:func:`~..configured_transport.build_configured_transport`, called once per instance by
    :mod:`~..flight.runtime`); this function only validates the endpoint list and assembles the result, which is
    what makes the instance-0 refusal unit-testable without any of that machinery.
    """

    if not instance_endpoints:
        raise VllmRouterTransportError("a vllm deployment record must declare at least one instance")
    if instance_endpoints[0] != base_url:
        raise VllmRouterTransportError(
            f"deployment instance 0 endpoint {instance_endpoints[0]!r} differs from the fragment's base_url "
            f"{base_url!r}; base_url is the protocol fact and must be instance 0's endpoint "
            "(core.deployment.deployment_record's instances[0].base_url)"
        )
    transports = [build_transport(index, endpoint) for index, endpoint in enumerate(instance_endpoints)]
    return VllmRouterTransport(transports, instance_ids=instance_ids)
