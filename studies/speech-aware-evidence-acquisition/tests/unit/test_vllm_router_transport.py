"""``VllmRouterTransport`` (Gap B of the 2026-10-30 Fold B plan, section 4; design doc section 9.3): one flight, N
resident vLLM instances, routed by ``decoding_params.id_slot mod instance_count`` -- the SAME field the runner
already sets on every contact for llama-server slot affinity (``rails/runner.py``).

Built on the S12 ``_vllm_world`` fixture (``tests/unit/_vllm_world.py``): a real multi-instance vLLM runtime receipt,
plus a fake ``/proc`` built from that receipt's own canonical argv, so every instance's per-send identity check
(``core.session.vllm_identity``) is exercised for real -- no NPU, no server, no network.
"""

from __future__ import annotations

import json

import pytest

from _vllm_world import SERVED, VllmWorld

from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import protocol_sha256
from speech_aware_evidence_acquisition.core.model import VllmServerTransport
from speech_aware_evidence_acquisition.core.model.vllm_router_transport import (
    ROUTING_DECODING_KEY,
    VllmRouterTransport,
    VllmRouterTransportError,
    build_vllm_router_transport,
)
from speech_aware_evidence_acquisition.core.session import vllm_identity as vllm_module

PID_BASE = 4200000


@pytest.fixture()
def world(tmp_path, monkeypatch):
    built = VllmWorld(tmp_path)
    monkeypatch.setattr(vllm_module, "DEFAULT_PROC_ROOT", built.proc_root)
    return built


class _Recorder:
    """A ``post`` callable that records every call it answers; one per instance."""

    def __init__(self, text: str = "ok") -> None:
        self.calls: list[tuple[str, dict]] = []
        self._text = text

    def __call__(self, url: str, body: bytes) -> bytes:
        self.calls.append((url, json.loads(body)))
        return json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": SERVED,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": self._text, "refusal": None, "tool_calls": []}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()


def _four_instance_setup(world):
    """Four spawned, live instances and their four per-instance ``VllmServerTransport``/recorder pairs."""

    media = world.root / "media" / "s1.flac"
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"fake-flac-bytes")

    receipt = world.receipt_with_instances(4)
    recorders, transports, endpoints = [], [], []
    for index in range(4):
        instance_id = f"tp2-{index}"
        pid = PID_BASE + index
        world.spawn(instance_id, pid=pid, receipt=receipt)
        session = world.session(instance_id, pid, receipt=receipt)
        recorder = _Recorder(f"instance-{index}")
        transport = VllmServerTransport(
            f"http://{session.host}:{session.port}",
            world.root,
            session,
            receipt,
            world.root / "runtime",
            post=recorder,
            served_model_name=SERVED,
        )
        recorders.append(recorder)
        transports.append(transport)
        endpoints.append(f"http://{session.host}:{session.port}")
    return receipt, recorders, transports, endpoints


def _payload(id_slot):
    return json.dumps(
        {
            "request_id": "req-1",
            "carrier_lock_key": "slurp",
            "sample_id": "s1",
            "speech_ref": "slurp/s1",
            "media_relpath": "media/s1.flac",
            "audio_seconds": 2.0,
            "sample_rate_hz": 16000,
            "task_instruction": "Answer with the intent label.",
            "history": [],
            "decoding_params": {"temperature": 0, "id_slot": id_slot},
        }
    ).encode()


def _kill_listener(world, port: int) -> None:
    """Remove exactly one instance's LISTEN row from the fake ``/proc/net/tcp`` (its process died, or something
    else now owns the port); every other instance's row -- and therefore its own identity check -- is untouched."""

    world._listeners = [entry for entry in world._listeners if entry[1] != port]
    world._write_tcp_table()


# ---------------------------------------------------------------------------
# protocol_sha256 is untouched by instance count (S12; re-asserted here as the Gap B precondition).
# ---------------------------------------------------------------------------


def test_protocol_sha256_is_unchanged_by_instance_count(world):
    one = world.receipt()
    four = world.receipt_with_instances(4)
    assert four["protocol_sha256"] == one["protocol_sha256"]
    assert protocol_sha256(four) == four["protocol_sha256"]


# ---------------------------------------------------------------------------
# Routing by slot.
# ---------------------------------------------------------------------------


def test_routes_each_contact_to_id_slot_mod_instance_count(world):
    _, recorders, transports, _ = _four_instance_setup(world)
    router = VllmRouterTransport(transports, instance_ids=[f"tp2-{i}" for i in range(4)])

    for id_slot, expected_index in ((0, 0), (1, 1), (4, 0), (5, 1), (9, 1), (11, 3)):
        envelope = json.loads(router(_payload(id_slot)))
        assert envelope["text"] == f"instance-{expected_index}"

    assert [len(r.calls) for r in recorders] == [2, 3, 0, 1]


def test_the_routed_contacts_id_slot_never_reaches_the_wire(world):
    _, recorders, transports, _ = _four_instance_setup(world)
    router = VllmRouterTransport(transports)
    router(_payload(7))
    (_, sent), = recorders[3].calls
    assert ROUTING_DECODING_KEY not in sent


def test_a_contact_without_id_slot_is_refused_before_any_instance_is_contacted(world):
    _, recorders, transports, _ = _four_instance_setup(world)
    router = VllmRouterTransport(transports)
    with pytest.raises(VllmRouterTransportError, match=ROUTING_DECODING_KEY):
        router(json.dumps({"messages": [], "decoding_params": {"temperature": 0}}).encode())
    assert all(r.calls == [] for r in recorders)


@pytest.mark.parametrize("bad_slot", ["per-sample", -1, True, 1.5])
def test_a_non_slot_id_slot_is_refused(world, bad_slot):
    _, _, transports, _ = _four_instance_setup(world)
    router = VllmRouterTransport(transports)
    with pytest.raises(VllmRouterTransportError):
        router(_payload(bad_slot))


# ---------------------------------------------------------------------------
# Instance 0 is the protocol fact.
# ---------------------------------------------------------------------------


def test_build_refuses_a_deployment_whose_instance_zero_endpoint_differs_from_base_url(world):
    receipt, _, transports, endpoints = _four_instance_setup(world)
    built_calls = []

    def build_transport(index, endpoint):
        built_calls.append((index, endpoint))
        return transports[index]

    with pytest.raises(VllmRouterTransportError, match="instance 0"):
        build_vllm_router_transport(
            "http://127.0.0.1:9999",  # the fragment's base_url; does not match endpoints[0]
            endpoints,
            build_transport,
        )
    # Refused before any instance transport was ever built.
    assert built_calls == []


def test_build_accepts_a_deployment_whose_instance_zero_endpoint_matches_base_url(world):
    receipt, recorders, transports, endpoints = _four_instance_setup(world)

    def build_transport(index, endpoint):
        assert endpoint == endpoints[index]
        return transports[index]

    router = build_vllm_router_transport(endpoints[0], endpoints, build_transport)
    assert router.instance_count == 4
    envelope = json.loads(router(_payload(2)))
    assert envelope["text"] == "instance-2"


# ---------------------------------------------------------------------------
# A dead instance fails closed for the whole flight; no failover.
# ---------------------------------------------------------------------------


def test_a_dead_instance_fails_closed_naming_the_instance_no_failover(world):
    receipt, recorders, transports, endpoints = _four_instance_setup(world)
    router = VllmRouterTransport(transports, instance_ids=["tp2-0", "tp2-1", "tp2-2", "tp2-3"])

    # Routing to instance 1 works before the socket disappears.
    json.loads(router(_payload(1)))
    assert len(recorders[1].calls) == 1

    # Instance 1's listening socket vanishes (process died, or something else now owns the port) -- the per-send
    # identity re-check (core.session.vllm_identity) must catch this, not the router.
    _kill_listener(world, receipt["deployment"]["instances"][1]["port"])

    with pytest.raises(VllmRouterTransportError, match=r"instance 1.*tp2-1") as error:
        router(_payload(1))
    assert "no failover to another instance" in str(error.value)
    # No retry against instance 1, and no silent dispatch to a surviving instance instead: the failed send never
    # reached ANY recorder (the identity re-check runs before the post call), and instance 1's earlier, successful
    # call is still the only call on record.
    assert len(recorders[1].calls) == 1
    assert sum(len(r.calls) for r in recorders) == 1

    # Every OTHER instance still answers normally: one dead instance does not take down the others' routing.
    envelope = json.loads(router(_payload(2)))
    assert envelope["text"] == "instance-2"


def test_assert_session_alive_names_the_first_dead_instance(world):
    receipt, _, transports, _ = _four_instance_setup(world)
    router = VllmRouterTransport(transports, instance_ids=["tp2-0", "tp2-1", "tp2-2", "tp2-3"])
    router.assert_session_alive()  # every instance alive: no error

    _kill_listener(world, receipt["deployment"]["instances"][0]["port"])
    with pytest.raises(VllmRouterTransportError, match="tp2-0"):
        router.assert_session_alive()


# ---------------------------------------------------------------------------
# Construction refusals.
# ---------------------------------------------------------------------------


def test_router_refuses_an_empty_instance_list():
    with pytest.raises(VllmRouterTransportError):
        VllmRouterTransport([])


def test_router_refuses_a_mismatched_instance_id_count(world):
    _, _, transports, _ = _four_instance_setup(world)
    with pytest.raises(VllmRouterTransportError):
        VllmRouterTransport(transports, instance_ids=["only-one"])
