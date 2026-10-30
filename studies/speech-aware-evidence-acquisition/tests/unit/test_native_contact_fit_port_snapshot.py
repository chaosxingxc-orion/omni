"""Snapshot: the native-contact FIT receipt of the port L338 smoke fixture is byte-identical.

The direct-only FIT path (``build_direct_only_raw_fit_gate``) and the shared endpoint-count helpers were added to
``core/preparation_fit.py`` after these digests were taken from the unchanged module. The fixture is the port smoke cell's FIT shape: the port smoke FIT run and
attempt ids, four rows, the raw-audio arm, the vLLM tokenize measurement, the native-role-episode program, and one contact per (row, role) pair. Every request-fit artifact the
gate writes is hashed; any change to the artifact bytes (inputs, output, plan, journal-free fields) moves a digest and fails this test.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

from speech_aware_evidence_acquisition.contracts.plan import ExecutionPlan
from speech_aware_evidence_acquisition.core.preparation import canonical_bytes
from speech_aware_evidence_acquisition.core.preparation_fit import build_native_contact_fit_gate
from speech_aware_evidence_acquisition.rails.core_client import PreparedCoreRequest

SHA = "a" * 64
FIT_RUN = "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-SMOKE-4-FIT-NPU-PORT-V1"
FIT_ATTEMPT = "attempt-slurp-native-stage3-l338-lever-column-fold-b-smoke-4-fit-npu-port-v1"
ROWS = ("s1", "s2", "s3", "s4")
CONTACTS_PER_ROW = 15
#: Four rows x 15 contacts x one resident multimodal counting endpoint (the raw-audio rule), the value the gate demands.
CONTACT_CAP = len(ROWS) * CONTACTS_PER_ROW * 1
ROLES = ("perception", "solver")


@dataclass(frozen=True)
class _Sample:
    sample_id: str


class _Transport:
    def render_request(self, raw):
        payload = json.loads(raw)
        audio = f"RIFF synthetic port smoke audio {payload['sample_id']}".encode()
        return {
            "model": "port-smoke-model",
            "messages": [
                {"role": "system", "content": payload["task_instruction"]},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": "wav"}}]},
            ],
            "max_tokens": payload["decoding_params"]["max_tokens"],
        }


class _Tokenizer:
    def __init__(self, **_kwargs):
        self.calls = []

    def request_json(self, method, path, payload):
        self.calls.append((method, path))
        assert (method, path) == ("POST", "/tokenize")
        return {"count": 11, "tokens": list(range(11))}


class _Runtime:
    def __init__(self, tmp_path):
        self._composed = SimpleNamespace(values={
            "transport": "vllm",
            "native_contact_fit": {"run_id": FIT_RUN, "attempt_id": FIT_ATTEMPT, "contact_cap": CONTACT_CAP, "max_speech_packets": 1},
        })
        self._data_root = self._repo_root = self._runtime_root = tmp_path
        self._samples = tuple(_Sample(row) for row in ROWS)
        self._transport = _Transport()
        self._session = SimpleNamespace(host="127.0.0.1", port=9999, session_id="port-smoke-session", model_files=())
        self._adapter = SimpleNamespace(serving_geometry=lambda: {"slot_tokens": 24576})
        self._gate = object()
        self.touches = []

    def uses_speech_feature_packets(self):
        return False

    def _arm(self):
        return "raw-speech-agent-native-tools"

    def assert_generation_lease(self):
        pass

    def _gate_touch(self, gate, fit_plan):
        self.touches.append(fit_plan)

    def _require_exposure_row_unconsumed(self, run_id):
        pass

    def session_required_identity(self):
        return {"session": "port-smoke-session"}

    def request_fit_inputs(self, payloads):
        return {"config_sha256": "d" * 64, "split_identity_hash": SHA, "sources": {"payloads": payloads}}


def _plan():
    return ExecutionPlan(
        run_id="SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-SMOKE-4-NPU-PORT-V1", attempt_id="attempt-generation", execution_profile="bounded-discovery-probe",
        carrier_lock_key="slurp", split_role="discovery", split_identity_hash=SHA, planned_model_calls=272, planned_gpu_hours=0.5,
        planned_speech_audio_seconds=347, protocol_hash="b" * 64).validate()


def _request(row, role):
    audio = f"RIFF synthetic port smoke audio {row}".encode()
    return PreparedCoreRequest(canonical_bytes({
        "request_id": "saea-prepared-request", "sample_id": row, "media_relpath": f"media/{row}.wav",
        "raw_speech_query": {"media_relpath": f"media/{row}.wav", "sha256": hashlib.sha256(audio).hexdigest()},
        "task_instruction": f"Answer the spoken request as {role}.",
        "history": [{"turn": 1, "role": "control", "contact_id": f"{row}-{role}-1", "content": "Act in the assigned role.", "content_sha256": "e" * 64}],
        "decoding_params": {"max_tokens": 4}}), object())


def _program():
    return SimpleNamespace(max_contacts=CONTACTS_PER_ROW, stop_rule="native-role-episode", phase_names=lambda: ("native-episode",))


#: sha256 of each request-fit artifact's bytes, keyed by the contact (sample, role), taken from the module before the direct-only path was added.
GOLDEN = {
    "s1/s1-perception-1": "e0a193c4c61d0ed57f56e2c78216c8c768df8f8f20094fa44a24ad42ce309528",
    "s1/s1-solver-1": "dbfa4b12760d136d5a850258d785135ef1783fd18089eb8765be28d9542bace8",
    "s2/s2-perception-1": "d9896741e6857b9eb676dfdc8f647b8b2e6b846d84c8cf38f08e766ede079ffe",
    "s2/s2-solver-1": "07f43699f78d474c353a058157d52e3fba90e97198e9c2fad689559765764d19",
    "s3/s3-perception-1": "f3c7f61cf86b4d93c0d4de0a52196c3b50dcda17db2b09ad5e3c9c05d27ed283",
    "s3/s3-solver-1": "ca1cf7229c4b33aac9925ef4a2c8b1748ab2cdcb7896f58215186da05f97f9fa",
    "s4/s4-perception-1": "950ba6ffd31e7a9eec89fda789f215446f34a07ecbba6ddc4ac369e1f9b5cc29",
    "s4/s4-solver-1": "697883953b3d6147d3fd234ddfed9d12bb704ff187befcfca2b5cbe5ae27c8ef",
}
GOLDEN_PLAN_SHA256 = "0368ccbb8c6b97291a8cc7668618efd2fb2f2c5932dd1eca4e5ed0382542724a"


def _run(tmp_path):
    runtime = _Runtime(tmp_path)
    gate = build_native_contact_fit_gate(runtime, _program(), _plan(), client_factory=lambda **_k: _Tokenizer())
    digests = {}
    for row in ROWS:
        for role in ROLES:
            gate(_request(row, role))
    for path in sorted((tmp_path / "prepared" / FIT_RUN / FIT_ATTEMPT).glob("*.request-fit.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        key = f"{document['inputs']['native_contact']['sample_id']}/{document['inputs']['native_contact']['agent_contact_id']}"
        digests[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan_sha = hashlib.sha256(canonical_bytes(gate.execution_plan.__dict__)).hexdigest()
    return gate, digests, plan_sha


def test_port_smoke_native_contact_fit_receipts_are_byte_identical(tmp_path):
    gate, digests, plan_sha = _run(tmp_path)
    assert len(gate.receipts) == len(ROWS) * len(ROLES)
    assert gate.execution_plan.planned_model_calls == CONTACT_CAP == 60
    assert digests == GOLDEN
    assert plan_sha == GOLDEN_PLAN_SHA256
