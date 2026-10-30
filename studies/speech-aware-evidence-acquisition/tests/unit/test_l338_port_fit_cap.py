"""The port L338 cells' ``native_contact_fit.contact_cap`` is the value the native contact FIT gate demands.

Defect (plan 2026-10-30 section 9, ruling 1): the registered port cell carried the packet-arm cap (rows x contacts x 4 endpoints: 64,800 flight / 240 smoke at
the historical 15 contacts per row) although the port is a RAW-audio arm, whose gate (``core.preparation_fit.build_native_contact_fit_gate``) demands rows x
contacts x 1 resident counting endpoint (16,200 / 60 at 15 contacts) and refused the cell at the FIT step with "native contact fit cap must equal the registered
maximum endpoint count". The writer's derivation (``l338_port_family.fit_calls``) now takes the gate's own helper. The tests compute the writer's value and the
gate's requirement independently and assert equality; they follow ``l338_port_family.CONTACTS_PER_ROW`` (15 before the replicate, 17 with it).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "flights" / "direct", REPO / "scripts" / "flights" / "tfrl", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import l338_family as fam  # noqa: E402
import l338_port_derive_cells as dc  # noqa: E402
import l338_port_family as pf  # noqa: E402

from speech_aware_evidence_acquisition.contracts.plan import ExecutionPlan  # noqa: E402
from speech_aware_evidence_acquisition.core.preparation import PreparationError  # noqa: E402
from speech_aware_evidence_acquisition.core.preparation_fit import (  # noqa: E402
    build_native_contact_fit_gate,
    native_contact_endpoints_per_contact,
    native_contact_required_cap,
)

SHA = "a" * 64
KEYS = list(pf.ALL_KEYS)


@dataclass(frozen=True)
class _Sample:
    sample_id: str


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        run_id="generation-run", attempt_id="generation-attempt", execution_profile="bounded-discovery-probe", carrier_lock_key="slurp", split_role="discovery",
        split_identity_hash=SHA, planned_model_calls=10, planned_gpu_hours=0.5, planned_speech_audio_seconds=10, protocol_hash="b" * 64).validate()


def _runtime(tmp_path, *, rows: int, cap: int, feature_packets: bool):
    runtime = SimpleNamespace(
        _composed=SimpleNamespace(values={"native_contact_fit": {"run_id": "fit-run", "attempt_id": "fit-attempt", "contact_cap": cap, "max_speech_packets": 1},
                                          "speech_feature_root": "/synthetic-feature-root"}),
        _data_root=tmp_path, _repo_root=tmp_path, _runtime_root=tmp_path, _samples=tuple(_Sample(f"s{i}") for i in range(rows)), _transport=object(),
        _session=SimpleNamespace(host="127.0.0.1", port=9999, session_id="synthetic", model_files=(SimpleNamespace(path="decoder.gguf", sha256="c" * 64),)),
        _adapter=SimpleNamespace(serving_geometry=lambda: {"slot_tokens": 128}), _gate=object(), touches=[], uses_speech_feature_packets=lambda: feature_packets,
        _arm=lambda: "raw-speech-agent-native-tools", assert_generation_lease=lambda: None, _require_exposure_row_unconsumed=lambda run_id: None,
        session_required_identity=lambda: {"session": "synthetic"})
    runtime._gate_touch = lambda gate, fit_plan: runtime.touches.append(fit_plan)
    return runtime


def _program(contacts: int):
    return SimpleNamespace(max_contacts=contacts, stop_rule="native-role-episode", phase_names=lambda: ("native-episode",))


def _gate_accepts(tmp_path, *, rows: int, contacts: int, cap: int, feature_packets: bool) -> bool:
    try:
        build_native_contact_fit_gate(_runtime(tmp_path, rows=rows, cap=cap, feature_packets=feature_packets), _program(contacts), _plan())
    except PreparationError as error:
        assert "native contact fit cap must equal the registered maximum endpoint count" in str(error)
        return False
    return True


@pytest.mark.parametrize("key", KEYS)
def test_writer_cap_equals_the_cap_the_raw_arm_gate_demands(tmp_path, key):
    rows = pf.CELLS[key]["rows"]
    written = pf.fit_calls(key)
    # the writer's value, and the gate's requirement computed two independent ways
    assert written == rows * pf.CONTACTS_PER_ROW * 1 == native_contact_required_cap(rows * pf.CONTACTS_PER_ROW, feature_packets=False)
    assert pf.RAW_FIT_ENDPOINTS_PER_CONTACT == native_contact_endpoints_per_contact(feature_packets=False) == 1
    assert _gate_accepts(tmp_path, rows=rows, contacts=pf.CONTACTS_PER_ROW, cap=written, feature_packets=False)
    # the gate demands exactly this value: one more, one fewer and the packet-arm value are all refused
    assert not _gate_accepts(tmp_path, rows=rows, contacts=pf.CONTACTS_PER_ROW, cap=written + 1, feature_packets=False)
    assert not _gate_accepts(tmp_path, rows=rows, contacts=pf.CONTACTS_PER_ROW, cap=written - 1, feature_packets=False)
    assert not _gate_accepts(tmp_path, rows=rows, contacts=pf.CONTACTS_PER_ROW, cap=fam.tf.fit_contact_cap(rows, pf.CONTACTS_PER_ROW), feature_packets=False)


def test_the_defect_numbers_at_the_historical_arm_table(tmp_path):
    """At the 15-contact arm table the plan's numbers hold: the packet formula gave 64,800 / 240, the raw gate demands 16,200 / 60."""

    for key, packet_cap, raw_cap in (("arm", 64800, 16200), ("smoke", 240, 60)):
        rows = fam.CELLS[key]["rows"]
        assert fam.fit_contact_cap(rows) == packet_cap == rows * fam.CONTACTS_PER_ROW * 4
        assert native_contact_required_cap(rows * fam.CONTACTS_PER_ROW, feature_packets=False) == raw_cap
        assert _gate_accepts(tmp_path, rows=rows, contacts=fam.CONTACTS_PER_ROW, cap=raw_cap, feature_packets=False)
        assert not _gate_accepts(tmp_path, rows=rows, contacts=fam.CONTACTS_PER_ROW, cap=packet_cap, feature_packets=False)


@pytest.mark.parametrize("key", KEYS)
def test_derived_port_cells_carry_the_gate_cap(key):
    rows, cap = pf.CELLS[key]["rows"], pf.fit_calls(key)
    base, exp = dc.port_baseline(key), dc.port_experiment(key)
    assert base["native_contact_fit"]["contact_cap"] == cap
    assert exp["family_budget"]["fit_endpoint_calls_cap"] == cap
    assert f"makes {rows} x {pf.CONTACTS_PER_ROW} x 1 = {cap:,} tokenizer endpoint calls" in exp["generation_budget"]["definition"]


# ---- the historical (packet) derivation is unchanged -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key,expected", [("arm", 64800), ("smoke", 240)])
def test_historical_packet_derivation_is_unchanged(tmp_path, key, expected):
    rows = fam.CELLS[key]["rows"]
    assert fam.tf.FIT_ENDPOINTS_PER_CONTACT == 4 and fam.CONTACTS_PER_ROW == 15
    assert fam.fit_contact_cap(rows) == fam.tf.fit_contact_cap(rows, 15) == pf.HISTORICAL_CAPS[key][1] == expected
    # the packet-arm gate still demands exactly this value (max_speech_packets 1 + 3 endpoints per contact)
    assert native_contact_endpoints_per_contact(feature_packets=True, max_speech_packets=1) == 4
    assert native_contact_required_cap(rows * 15, feature_packets=True) == expected
    assert _gate_accepts(tmp_path, rows=rows, contacts=15, cap=expected, feature_packets=True)
    assert not _gate_accepts(tmp_path, rows=rows, contacts=15, cap=rows * 15, feature_packets=True)
    # the historical sentence the derive replaces keeps its 4 endpoints
    old_sentence, new_sentence = dc.program_sentences(key)
    assert f"makes {rows} x 15 x 4 = {expected:,} tokenizer endpoint calls" in old_sentence
    assert f"x {pf.RAW_FIT_ENDPOINTS_PER_CONTACT} = {pf.fit_calls(key):,} tokenizer endpoint calls" in new_sentence
