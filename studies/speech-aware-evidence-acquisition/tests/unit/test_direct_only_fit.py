"""Model-free tests for the direct-only raw-audio FIT path (one contact per row, no knowledge, no program).

The path hangs off ``build_live_phase_fit_gate`` behind the program id ``phase-direct-only-v1`` on the raw-audio arm; every other program and arm keeps its
existing refusal or behaviour (covered by test_live_phase_fit.py and test_native_contact_fit.py, and by the byte snapshot in
test_native_contact_fit_port_snapshot.py).
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import wave
from types import SimpleNamespace

import pytest

from speech_aware_evidence_acquisition.contracts.plan import ExecutionPlan
from speech_aware_evidence_acquisition.core import preparation_fit as fitting
from speech_aware_evidence_acquisition.core.preparation import PreparationError, canonical_bytes
from speech_aware_evidence_acquisition.core.preparation_fit import (
    DIRECT_ONLY_PROGRAM_ID,
    build_direct_only_raw_fit_gate,
    build_live_phase_fit_gate,
    direct_only_fit_contact_cap,
)
from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
from speech_aware_evidence_acquisition.rails.core_client import PreparedCoreRequest

SHA = "a" * 64
ROWS = ("s1", "s2", "s3", "s4")
RATE, FRAMES = 1000, 1500  # 1.5 s per row, 6.0 s for the four-row fixture
FIT_RUN, FIT_ATTEMPT = "SAEA-DIRECT-ONLY-FIT-TEST-V1", "attempt-direct-only-fit-test-v1"


def _wav(seed: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(bytes([seed % 256, 0]) * FRAMES)
    return buffer.getvalue()


AUDIO = {row: _wav(index) for index, row in enumerate(ROWS)}


class _Transport:
    def __init__(self):
        self.calls = []

    def render_request(self, raw):
        self.calls.append(raw)
        payload = json.loads(raw)
        data = base64.b64encode(AUDIO[payload["sample_id"]]).decode("ascii")
        return {
            "model": "direct-only-test-model",
            "messages": [
                {"role": "system", "content": payload["task_instruction"]},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}]},
            ],
            "max_tokens": payload["decoding_params"]["max_tokens"],
        }


class _Tokenizer:
    def __init__(self, **_kwargs):
        self.calls = []

    def request_json(self, method, path, payload):
        self.calls.append((method, path, payload))
        assert (method, path) == ("POST", "/tokenize")
        return {"count": 11, "tokens": list(range(11))}


class _Runtime:
    def __init__(self, tmp_path, *, cap=len(ROWS), arm="raw-speech-agent-native-tools", extra=None):
        values = {
            "transport": "vllm",
            "whole_roster_wave_barrier": True,
            "live_phase_fit": {"run_id": FIT_RUN, "attempt_id": FIT_ATTEMPT, "contact_cap": cap},
            **(extra or {}),
        }
        self._composed = SimpleNamespace(values=values)
        self._data_root = self._repo_root = self._runtime_root = tmp_path
        self._samples = tuple(SimpleNamespace(sample_id=row) for row in ROWS)
        self._transport = _Transport()
        self._session = SimpleNamespace(host="127.0.0.1", port=9999, session_id="direct-only-session", model_files=())
        self._adapter = SimpleNamespace(serving_geometry=lambda: {"slot_tokens": 24576})
        self._gate = object()
        self._arm_name = arm
        self.touches, self.unconsumed, self.lease_checks = [], [], 0

    def uses_speech_feature_packets(self):
        return False

    def _arm(self):
        return self._arm_name

    def assert_generation_lease(self):
        self.lease_checks += 1

    def _gate_touch(self, gate, fit_plan):
        assert gate is self._gate
        self.touches.append(fit_plan)

    def _require_exposure_row_unconsumed(self, run_id):
        self.unconsumed.append(run_id)

    def session_required_identity(self):
        return {"session": "direct-only-session"}

    def request_fit_inputs(self, payloads):
        return {"config_sha256": "d" * 64, "split_identity_hash": SHA, "sources": {"payloads": payloads}}


def _plan(**overrides):
    values = dict(
        run_id="SAEA-DIRECT-ONLY-GENERATION-TEST-V1", attempt_id="attempt-generation", execution_profile="bounded-discovery-probe",
        carrier_lock_key="slurp", split_role="discovery", split_identity_hash=SHA, planned_model_calls=8, planned_gpu_hours=0.5,
        planned_speech_audio_seconds=12, protocol_hash="b" * 64)
    values.update(overrides)
    return ExecutionPlan(**values).validate()


def _request(row, **changes):
    payload = {
        "request_id": "saea-prepared-request", "sample_id": row, "media_relpath": f"media/{row}.wav", "audio_seconds": FRAMES / RATE,
        "raw_speech_query": {"media_relpath": f"media/{row}.wav", "sha256": hashlib.sha256(AUDIO[row]).hexdigest()},
        "task_instruction": "Pick the label from the closed list.", "history": [], "decoding_params": {"max_tokens": 4},
    }
    payload.update(changes)
    return PreparedCoreRequest(canonical_bytes(payload), object())


def _batch(**changes):
    return tuple(_request(row, **changes) for row in ROWS)


def _program():
    return default_phase_program(DIRECT_ONLY_PROGRAM_ID)


def _build(runtime, plan=None, program=None, tokenizer=None):
    tokenizer = tokenizer or _Tokenizer()
    return build_live_phase_fit_gate(runtime, program or _program(), plan or _plan(), client_factory=lambda **_k: tokenizer), tokenizer


# ---- accept ----------------------------------------------------------------------------------------------------------------------------


def test_accepts_a_four_row_raw_direct_only_wave_through_the_live_phase_entry(tmp_path):
    runtime = _Runtime(tmp_path)
    gate, tokenizer = _build(runtime)
    fit_plan = gate.execution_plan
    assert (fit_plan.run_id, fit_plan.attempt_id, fit_plan.planned_model_calls) == (FIT_RUN, FIT_ATTEMPT, 4)
    assert fit_plan.planned_speech_audio_seconds == 12 and fit_plan.protocol_hash == _plan().protocol_hash
    assert runtime.touches == [fit_plan] and runtime.unconsumed == [FIT_RUN]

    gate("direct", _batch())

    assert [receipt["phase"] for receipt in gate.receipts] == ["direct"]
    assert [(method, path) for method, path, _ in tokenizer.calls] == [("POST", "/tokenize")] * 4
    artifact = tmp_path / "prepared" / FIT_RUN / FIT_ATTEMPT / "direct.request-fit.json"
    document = json.loads(artifact.read_text(encoding="utf-8"))
    output = document["output"]
    # The live phase receipt shape: the fit_requests fields plus the phase, session and contact accounting.
    assert set(output) == {
        "sample_count", "rows", "boundary_sha256", "every_request_fits", "worst_sample_id", "minimum_margin_tokens", "phase", "session_id",
        "session_identity_sha256", "contacts", "cumulative_contacts", "generation_contacts"}
    assert output["every_request_fits"] is True and output["sample_count"] == 4 and output["contacts"] == 4 and output["generation_contacts"] == 0
    assert {row["measurement"] for row in output["rows"]} == {"vllm-chat-tokenize"}
    assert all(row["slot_tokens"] == 24576 and row["maximum_completion_tokens"] == 4 for row in output["rows"])
    assert document["inputs"]["live_phase"] == "direct"
    assert document["inputs"]["direct_only"] == {
        "program_id": DIRECT_ONLY_PROGRAM_ID, "contacts_per_row": 1, "rows": 4, "audio_seconds_ceiled": 6, "audio_seconds_ceiling": 12}
    journal = artifact.with_name(artifact.name + ".contacts.jsonl")
    assert len(journal.read_text().splitlines()) == 4
    with pytest.raises(PreparationError, match="order changed or repeated"):
        gate("direct", _batch())
    assert len(tokenizer.calls) == 4


def test_direct_entry_point_matches_the_dispatch(tmp_path):
    runtime = _Runtime(tmp_path)
    tokenizer = _Tokenizer()
    gate = build_direct_only_raw_fit_gate(runtime, _program(), _plan(), client_factory=lambda **_k: tokenizer)
    gate("direct", _batch())
    assert gate.receipts[0]["phase"] == "direct" and len(tokenizer.calls) == 4


def test_the_registered_cap_is_rows_times_one_and_the_packet_formula_is_refused(tmp_path):
    assert [direct_only_fit_contact_cap(n) for n in (4, 1080)] == [4, 1080]
    with pytest.raises(PreparationError, match="positive row count"):
        direct_only_fit_contact_cap(0)
    runtime = _Runtime(tmp_path, cap=4 * len(ROWS))  # the packet-path four endpoints per contact
    with pytest.raises(PreparationError, match="cap must equal rows x one contact"):
        _build(runtime)
    assert not runtime.touches and not runtime.unconsumed and runtime.lease_checks == 0


# ---- call caps ---------------------------------------------------------------------------------------------------------------------------


def test_refuses_a_cap_above_rows_before_lease_or_gate(tmp_path):
    runtime = _Runtime(tmp_path, cap=len(ROWS) + 1)
    with pytest.raises(PreparationError, match="cap must equal rows x one contact"):
        _build(runtime)
    assert not runtime.touches and not runtime.unconsumed and runtime.lease_checks == 0


def test_refuses_a_generation_plan_that_cannot_cover_one_contact_per_row(tmp_path):
    runtime = _Runtime(tmp_path)
    with pytest.raises(PreparationError, match="cannot cover one contact per row"):
        _build(runtime, plan=_plan(planned_model_calls=3))
    assert not runtime.touches and runtime.lease_checks == 0


def test_refuses_more_than_one_contact_per_row_before_any_tokenizer_contact(tmp_path):
    runtime = _Runtime(tmp_path)
    gate, tokenizer = _build(runtime)
    with pytest.raises(PreparationError, match="exceeds one contact per row"):
        gate("direct", (*_batch(), _request("s1")))
    with pytest.raises(PreparationError, match="exceeds one contact per row"):
        gate("direct", (_request("s1"), _request("s1"), _request("s2"), _request("s3")))
    with pytest.raises(PreparationError, match="full registered roster"):
        gate("direct", _batch()[:3])
    assert tokenizer.calls == [] and not list(tmp_path.rglob("*.contacts.jsonl"))


def test_the_wave_spends_exactly_the_registered_cap_and_no_generation_contact(tmp_path):
    runtime = _Runtime(tmp_path)
    gate, _ = _build(runtime)
    gate("direct", _batch())
    document = json.loads((tmp_path / "prepared" / FIT_RUN / FIT_ATTEMPT / "direct.request-fit.json").read_text())
    assert document["output"]["cumulative_contacts"] == document["output"]["contacts"] == gate.execution_plan.planned_model_calls == 4
    assert document["output"]["generation_contacts"] == 0


# ---- audio ceiling -----------------------------------------------------------------------------------------------------------------------


def test_refuses_audio_over_the_ceiling_before_any_contact(tmp_path):
    runtime = _Runtime(tmp_path)
    gate, tokenizer = _build(runtime, plan=_plan(planned_speech_audio_seconds=5))  # 4 x 1.5 s = 6 s
    with pytest.raises(PreparationError, match="audio exceeds the registered audio ceiling: 6 > 5"):
        gate("direct", _batch())
    assert tokenizer.calls == [] and not list(tmp_path.rglob("*.contacts.jsonl")) and not list(tmp_path.rglob("*.request-fit.json"))


def test_accepts_audio_exactly_at_the_ceiling(tmp_path):
    runtime = _Runtime(tmp_path)
    gate, tokenizer = _build(runtime, plan=_plan(planned_speech_audio_seconds=6))
    gate("direct", _batch())
    assert len(tokenizer.calls) == 4


def test_refuses_rendered_audio_that_is_not_the_named_recording(tmp_path):
    runtime = _Runtime(tmp_path)
    gate, tokenizer = _build(runtime)
    changed = (_request("s1", raw_speech_query={"media_relpath": "media/s1.wav", "sha256": "f" * 64}), *_batch()[1:])
    with pytest.raises(PreparationError, match="rendered audio differs"):
        gate("direct", changed)
    assert tokenizer.calls == []


# ---- knowledge-bearing and other-program refusals ------------------------------------------------------------------------------------------


def test_refuses_a_knowledge_bearing_program(tmp_path):
    base = _program()
    supplied = dataclasses.replace(base, phases=(dataclasses.replace(base.phases[0], knowledge="supplied"),))
    runtime = _Runtime(tmp_path)
    with pytest.raises(PreparationError, match="knowledge-bearing program"):
        _build(runtime, program=supplied)
    assert not runtime.touches and not runtime.unconsumed and runtime.lease_checks == 0
    two_contacts = dataclasses.replace(base, max_contacts=2)
    with pytest.raises(PreparationError, match="one-phase, one-contact"):
        build_direct_only_raw_fit_gate(runtime, two_contacts, _plan())


def test_refuses_a_request_that_carries_supplied_knowledge(tmp_path):
    runtime = _Runtime(tmp_path)
    gate, tokenizer = _build(runtime)
    with pytest.raises(PreparationError, match="supplied knowledge"):
        gate("direct", tuple(_request(row, supplied_evidence=[{"evidence_id": "e", "content": "x"}]) for row in ROWS))
    assert tokenizer.calls == []


def test_refuses_a_native_episode_and_a_feature_packet_arm(tmp_path):
    with pytest.raises(PreparationError, match="native episode"):
        _build(_Runtime(tmp_path, extra={"native_episode": {"roles": {}}}))
    with pytest.raises(PreparationError, match="raw-speech-agent-native-tools"):
        build_direct_only_raw_fit_gate(_Runtime(tmp_path, arm="speech-feature-query"), _program(), _plan())


def test_other_programs_on_the_raw_arm_keep_the_existing_refusal(tmp_path):
    runtime = _Runtime(tmp_path)
    with pytest.raises(PreparationError, match="validated feature packets or prepared text"):
        build_live_phase_fit_gate(runtime, default_phase_program("phase-direct-then-slots-v1"), _plan())
    assert not runtime.touches
