"""Ready scheduling understands production guards without bypassing them."""

import pytest

from speech_aware_evidence_acquisition.core.model.transport_capabilities import (
    supports_single_vllm_ready_request,
)
from speech_aware_evidence_acquisition.core.model.vllm_router_transport import VllmRouterTransport
from speech_aware_evidence_acquisition.core.model.vllm_transport import VllmServerTransport
from speech_aware_evidence_acquisition.core.session.resource_guard import ResourceCheckedTransport
from speech_aware_evidence_acquisition.core.session.worker import LeaseCheckedTransport


@pytest.mark.parametrize("routed", [False, True])
def test_known_guard_chain_is_inspected_without_running_or_removing_guards(routed):
    def unexpected(*args, **kwargs):
        pytest.fail("capability inspection must not contact or run admission")

    endpoint = object.__new__(VllmServerTransport)
    resource = ResourceCheckedTransport(endpoint, None, None, None, now=unexpected)
    inner = VllmRouterTransport([resource]) if routed else resource
    guarded = LeaseCheckedTransport(inner, unexpected)
    assert supports_single_vllm_ready_request(guarded)
    assert guarded._transport is inner
    assert resource.transport is endpoint


def test_unknown_multiple_and_cyclic_transports_fail_closed():
    endpoint = object.__new__(VllmServerTransport)
    class Proxy:
        def __getattr__(self, name):
            return getattr(endpoint, name)

    class CustomVllm(VllmServerTransport):
        pass

    cyclic = LeaseCheckedTransport(endpoint, lambda: None)
    cyclic._transport = cyclic
    for value in (None, Proxy(), object.__new__(CustomVllm), cyclic,
                  VllmRouterTransport([endpoint, endpoint])):
        assert not supports_single_vllm_ready_request(value)
