"""Gap C / Decision 2: the flight width is a DEPLOYMENT fact (instances x admitted level); ``serving_slots`` stays the
hashed geometry fact. Width 128 on the 4xTP2 ``-v2`` profile, the fallback without ``admitted_level``, refusal above
the ceiling, and the runner path with a fake adapter."""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.core import deployment as dep
from speech_aware_evidence_acquisition.core.flight.width import FlightWidthError, flight_width
from speech_aware_evidence_acquisition.rails import AudioRef
from speech_aware_evidence_acquisition.rails.runner import RailsRunnerError, run_speech_native_agent_loop

from test_rails_runner import (
    _SHA_A, _SHA_C, _FakeAdapter, _FakeSample, _ImmediateStopController, _RecordingAgentClient, _agent_trace,
    _controller_registry, _knowledge_snapshot, _plan,
)

V2 = "qwen3-omni-thinking-bf16-181-4xtp2-v2"
V3 = "qwen3-omni-thinking-bf16-181-4xtp2-v3"
V1 = "qwen3-omni-thinking-bf16-181-4xtp2-v1"


def _document(profile_id: str) -> dict:
    return json.loads(dep.resolve_profile_path(profile_id).read_text(encoding="utf-8"))


def test_width_is_instances_times_admitted_level_on_the_v2_and_v3_profiles():
    assert _document(V2)["admitted_level"] == 32 and len(_document(V2)["instances"]) == 4
    assert flight_width(V2, 4) == 128
    assert flight_width(V3, 4) == 128
    assert flight_width(dep.load_deployment_profile(V2), 4) == 128  # a loaded DeploymentProfile names its file
    assert flight_width(_document(V2), 4) == 128  # a raw mapping


def test_width_falls_back_to_serving_slots_without_the_field_or_the_profile():
    assert "admitted_level" not in _document(V1)
    assert flight_width(V1, 4) == 4
    assert flight_width(None, 4) == 4
    assert flight_width(_document(V1), 7) == 7


def test_an_admitted_level_above_the_ceiling_is_refused():
    document = _document(V2)
    document["admitted_level"] = 33
    with pytest.raises(FlightWidthError, match="above the profile ceiling"):
        flight_width(document, 4)
    document["admitted_level"] = 32  # equal to the ceiling is allowed
    assert flight_width(document, 4) == 128


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "32"])
def test_a_malformed_admitted_level_or_fallback_is_refused(bad):
    document = _document(V2)
    document["admitted_level"] = bad
    with pytest.raises(FlightWidthError):
        flight_width(document, 4)
    with pytest.raises(FlightWidthError):
        flight_width(V1, bad)


def test_a_profile_without_instances_is_refused():
    document = copy.deepcopy(_document(V2))
    document["instances"] = []
    with pytest.raises(FlightWidthError, match="no instances"):
        flight_width(document, 4)


def test_the_deployment_record_form_is_read_too():
    record = {"max_num_seqs_per_instance": 32, "admitted_level": 16, "instances": [{}, {}]}
    assert flight_width(record, 4) == 32


class _PeakClient(_RecordingAgentClient):
    """Holds every request until ``width`` are in flight at once, then records the peak."""

    def __init__(self, width: int):
        super().__init__()
        self.barrier = threading.Barrier(width)
        self.active = 0
        self.peak = 0

    def complete(self, messages, **options):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            self.barrier.wait(timeout=20)
            return super().complete(messages, **options)
        finally:
            with self._lock:
                self.active -= 1


def _run(tmp_path: Path, *, samples: int, width: int, slot_count: int, queue_depth: int, profile):
    rows = tuple(
        _FakeSample(sample_id=f"s{index}", audio_ref=AudioRef(path=f"a-{index}.wav", sha256=_SHA_A), audio_seconds=2.0)
        for index in range(samples)
    )
    client = _PeakClient(width)
    summary = run_speech_native_agent_loop(
        data_root=tmp_path, plan=_plan(execution_profile="model-free-check", planned_model_calls=1000), adapter=_FakeAdapter(samples=rows),
        core_client=client, split_id="all", knowledge_snapshot=_knowledge_snapshot(),
        speech_packet_bindings={s.sample_id: {"packet_id": f"fake/dev/{s.sample_id}", "sha256": _SHA_C} for s in rows},
        controller_policy_id="immediate-stop-test-v1",
        controller_registry=_controller_registry("immediate-stop-test-v1", lambda _sample_id: _ImmediateStopController()),
        max_contacts=1, slot_count=slot_count, queue_depth=queue_depth, trace_sink=_agent_trace(),
        agent_executor_id="reference-trajectory-v1", deployment_profile=profile,
    )
    return client, summary


def test_the_runner_puts_128_rows_in_flight_on_the_v2_profile(tmp_path):
    client, summary = _run(tmp_path, samples=128, width=128, slot_count=4, queue_depth=4, profile=V2)
    assert len(summary.results) == 128 and client.peak == 128
    assert {options["decoding_params"]["id_slot"] for _, options in client.calls} == set(range(128))


def test_the_runner_keeps_serving_slots_without_admitted_level_or_profile(tmp_path):
    for profile in (V1, None):
        client, summary = _run(tmp_path, samples=8, width=4, slot_count=4, queue_depth=4, profile=profile)
        assert len(summary.results) == 8 and client.peak == 4
        assert {options["decoding_params"]["id_slot"] for _, options in client.calls} == {0, 1, 2, 3}


def test_the_runner_refuses_an_admitted_level_above_the_ceiling(tmp_path):
    document = _document(V2)
    document["admitted_level"] = 64
    with pytest.raises(RailsRunnerError, match="above the profile ceiling"):
        _run(tmp_path, samples=2, width=2, slot_count=2, queue_depth=2, profile=document)
