"""Model-free tests for the D cell (direct-only on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold B): ids, caps, derive, writer, package.

The cells are derived from the committed 266-E Thinking cell by ``scripts/flights/direct/direct_port_derive_cells.py`` and written up by
``direct_port_write_registration.py``; the port settings come from ``l338_port_family``. Nothing here contacts a model, a server or an encoder.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "direct", REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "flights" / "tfrl", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import direct_port_derive_cells as dc  # noqa: E402
import direct_port_family as fam  # noqa: E402
import direct_port_write_registration as wr  # noqa: E402
import l338_port_family as pf  # noqa: E402

from speech_aware_evidence_acquisition.core.config import compose, protocol_hash  # noqa: E402
from speech_aware_evidence_acquisition.core.preparation_fit import direct_only_fit_contact_cap  # noqa: E402

CJK = re.compile("[\u4e00-\u9fff]")
KEYS = list(fam.ALL_KEYS)
PKG_DIR = REPO / fam.PACKAGE
NEW_FILES = [
    REPO / "scripts/flights/direct/direct_port_family.py", REPO / "scripts/flights/direct/direct_port_derive_cells.py",
    REPO / "scripts/flights/direct/direct_port_write_registration.py", REPO / "tests/unit/test_direct_port_cell.py", REPO / fam.PREREG,
    *[REPO / f"configs/baseline/{fam.CELLS[k]['baseline']}.json" for k in fam.ALL_KEYS],
    *[REPO / f"configs/experiment/{fam.CELLS[k]['cell']}.json" for k in fam.ALL_KEYS],
    REPO / f"configs/campaign/{fam.CELLS['lock']['cell']}.json",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline(key: str) -> dict:
    return json.loads((REPO / "configs/baseline" / f"{fam.CELLS[key]['baseline']}.json").read_text(encoding="utf-8"))


def _experiment(key: str) -> dict:
    return json.loads((REPO / "configs/experiment" / f"{fam.CELLS[key]['cell']}.json").read_text(encoding="utf-8"))


def _compose(key: str):
    spec = fam.CELLS[key]
    cell = _experiment(key)["cells"][0]
    return compose(REPO, cell["model_fragment"], cell["dataset_fragment"], cell["baseline_fragment"], spec["cell"])


# ---- ids and caps -------------------------------------------------------------------------------------------------------------------------


def test_ids_follow_the_brief_and_chain_order():
    assert fam.ALL_KEYS == ("smoke", "lock", "aa", "ab")
    assert fam.CELLS["lock"]["run"] == "SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1"
    assert fam.CELLS["lock"]["fit_run"] == "SAEA-SLURP-DIRECT-ONLY-FOLD-B-FIT-NPU-PORT-V1"
    assert fam.CELLS["smoke"]["run"] == "SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-NPU-PORT-V1"
    assert fam.CELLS["smoke"]["fit_run"] == "SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-FIT-NPU-PORT-V1"
    assert fam.CELLS["aa"]["fit_attempt"] == "attempt-slurp-direct-only-fold-b-aa-fit-npu-port-v1"
    assert fam.CELLS["ab"]["attempt"] == "attempt-slurp-direct-only-fold-b-ab-generation-npu-port-v1"
    assert fam.PACKAGE == "docs/readiness/2026-10-30-slurp-direct-only-fold-b-npu-port-v1"
    assert fam.DECISION_ID == "slurp-direct-only-fold-b-npu-port-admission-v1"
    assert fam.DECISION_RELPATH == "coordination/slurp-direct-only-fold-b-npu-port-v1/root-admission-v1.json"
    assert fam.eligible_run_ids() == [fam.CELLS[k][n] for k in KEYS for n in ("run", "fit_run")]
    assert len(set(fam.eligible_run_ids())) == len(set(fam.eligible_attempt_ids())) == 8
    assert fam.MODEL_FRAGMENT == pf.MODEL_FRAGMENT and fam.SERVICE_SESSION == pf.SERVICE_SESSION


@pytest.mark.parametrize("key,caps", [("smoke", (8, 4, 20)), ("lock", (2160, 1080, 6310)), ("aa", (2160, 1080, 6310)), ("ab", (2160, 1080, 6310))])
def test_caps_per_cell(key, caps):
    assert (fam.generation_calls(key), fam.fit_calls(key), fam.audio_seconds(key)) == caps
    assert fam.fit_calls(key) == direct_only_fit_contact_cap(fam.CELLS[key]["rows"]) == fam.CELLS[key]["rows"]
    family = _experiment(key)["family_budget"]
    assert (family["generation_calls_cap"], family["fit_endpoint_calls_cap"], family["repeated_speech_audio_seconds_cap"]) == caps
    plan = _experiment(key)["execution_plan"]
    assert (plan["planned_model_calls"], plan["planned_speech_audio_seconds"]) == (caps[0], caps[2])


def test_cell_total_and_fallback_factor():
    assert fam.CELL_TOTAL == (6488, 3244, 18950)
    assert fam.FALLBACK_CONTACTS_PER_ROW == 1 and fam.GENERATION_CONTACTS_PER_ROW == 2 and fam.CONTACTS_PER_ROW == 1
    assert fam.ARM_MIX_ROWS == 0.52


def test_provisional_card_hours_follow_the_l338_form():
    low = fam.provisional_card_hours(1080, lock_rows_per_minute_per_instance=1.95, tensor_parallel_size=1)
    assert low == pytest.approx(6.3, abs=0.05)                              # the draft's illustration: 6.3 x T per pass
    assert fam.provisional_card_hours(1080, lock_rows_per_minute_per_instance=1.95, tensor_parallel_size=2) == pytest.approx(2 * low, abs=0.1)
    for key in KEYS:
        assert fam.ledger_gpu_cell(key).startswith("PROVISIONAL")
        assert fam.CARD_HOUR_FORMULA[key].startswith("PROVISIONAL")


def test_declared_differences_are_the_d_subset_plus_d1():
    assert [d.split()[0] for d in fam.DECLARED_DIFFERENCES] == ["8.1", "8.2", "8.3", "8.5", "D1"]
    assert "3072" in fam.D1_DIFFERENCE and "8,192" in fam.D1_DIFFERENCE


# ---- the derived configs -----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", KEYS)
def test_baselines_are_raw_arm_direct_only_without_packet_keys(key):
    base = _baseline(key)
    assert base["arm"] == "raw-speech-agent-native-tools" and base["policy"] == "phase-direct-only-v1"
    for name in ("speech_packet_promotion_receipt", "speech_packet_promotion_receipt_sha256", "speech_feature_key_representation_id",
                 "speech_feature_matcher_policy_id", "native_episode", "native_contact_fit", "prepared_phase_supply"):
        assert name not in base
    assert base["whole_roster_wave_barrier"] is True
    fit = base["live_phase_fit"]
    assert fit == {"attempt_id": fam.CELLS[key]["fit_attempt"], "contact_cap": fam.fit_calls(key), "run_id": fam.CELLS[key]["fit_run"]}
    assert base["live_phase_fit_audio_seconds_cap"] == fam.audio_seconds(key)


@pytest.mark.parametrize("key", KEYS)
def test_each_cell_composes_and_the_composed_fit_cap_is_the_gates(key):
    composed = _compose(key)
    cell = _experiment(key)["cells"][0]
    assert cell["model_fragment"] == pf.MODEL_FRAGMENT and cell["readiness"]["state"] == "held"
    assert cell["readiness"]["requirements"][0]["sha256"] == "0" * 64
    assert composed.values["live_phase_fit"]["contact_cap"] == direct_only_fit_contact_cap(fam.CELLS[key]["rows"])
    assert composed.values["arm"] == fam.ARM and composed.values["preserve_wire_responses"] is False
    split = _experiment(key)["population"]["split_identity_hash"]
    assert re.fullmatch(r"[0-9a-f]{64}", protocol_hash(composed.config_hash, composed.values["arm"], split))
    assert _experiment(key)["service_session"] == pf.SERVICE_SESSION


def test_passes_are_request_identical_to_the_lock():
    lock = _compose("lock").values
    prose = {"live_phase_fit", "what_this_is", "cells", "purpose", "experiment_id", "generation_budget", "core_read_design", "readiness"}
    hashes = {_compose(k).config_hash for k in fam.PASS_KEYS}
    assert len(hashes) == 3                                                  # the FIT identity makes the three composed hashes distinct
    for key in fam.REPLICATE_KEYS:
        values = _compose(key).values
        assert sorted(n for n in set(values) | set(lock) if n not in prose and values.get(n) != lock.get(n)) == []
        assert dc.moved_leaves(_baseline("lock"), _baseline(key)) == ["live_phase_fit.attempt_id", "live_phase_fit.run_id", "what_this_is"]


def test_smoke_is_the_first_four_rows_of_the_arm_roster():
    smoke, arm = _experiment("smoke"), _experiment("lock")
    assert smoke["populations"][0]["ids"] == arm["populations"][0]["ids"][:4]
    assert smoke["population"]["roster_receipt"] == fam.ROSTER["smoke"] and arm["population"]["roster_receipt"] == fam.ROSTER["arm"]
    assert smoke["population"]["split_identity_hash"].startswith("26b306e5")
    assert arm["population"]["split_identity_hash"].startswith("4407794c") and arm["population"]["count"] == 1080


def test_campaign_lists_the_three_passes_only():
    doc = json.loads((REPO / f"configs/campaign/{fam.CELLS['lock']['cell']}.json").read_text(encoding="utf-8"))
    assert [c["cell"] for c in doc["cells"]] == [fam.CELLS[k]["cell"] for k in fam.PASS_KEYS]
    assert "campaign" not in _experiment("smoke")["cells"][0]["allowed_execution_stages"]       # the smoke flies by worker only
    for key in fam.PASS_KEYS:
        assert "campaign" in _experiment(key)["cells"][0]["allowed_execution_stages"]


def test_derive_dry_run_reproduces_the_committed_configs_and_writes_nothing(tmp_path):
    result = dc.derive(REPO, dry_run=True)
    for key in KEYS:
        moved = result[key]["versus_committed_266e"]["moved"]
        assert set(moved) <= {"arm", "live_phase_fit.attempt_id", "live_phase_fit.contact_cap", "live_phase_fit.run_id", "live_phase_fit_audio_seconds_cap",
                              "speech_feature_key_representation_id", "speech_feature_matcher_policy_id", "speech_packet_promotion_receipt",
                              "speech_packet_promotion_receipt_sha256", "what_this_is"}
    for relpath in result["outputs"]:
        assert (REPO / relpath).is_file()
    # a derive into an empty root writes the same bytes the committed files carry
    root = tmp_path / "root"
    for kind in ("baseline", "experiment"):
        (root / "configs" / kind).mkdir(parents=True)
        for name in {fam.CELLS[k]["historical_baseline" if kind == "baseline" else "historical_cell"] for k in KEYS}:
            (root / "configs" / kind / f"{name}.json").write_bytes((REPO / "configs" / kind / f"{name}.json").read_bytes())
    dc.derive(root, dry_run=False)
    for relpath in result["outputs"]:
        assert (root / relpath).read_bytes() == (REPO / relpath).read_bytes(), relpath


def test_derive_refuses_an_undeclared_moved_leaf():
    base = dc.port_baseline("lock")
    moved = copy.deepcopy(base)
    moved["policy"] = "phase-direct-only-v2"
    with pytest.raises(ValueError, match="undeclared leaves"):
        dc.verify_against_committed("lock", moved, REPO)
    moved = copy.deepcopy(base)
    moved["task_instruction_template"] = "probes/other.txt"
    with pytest.raises(ValueError, match="undeclared leaves"):
        dc.verify_against_committed("lock", moved, REPO)


def test_derive_refuses_a_pass_that_is_not_request_identical(monkeypatch):
    real = dc.port_baseline

    def tampered(key, root=REPO):
        doc = real(key, root)
        if key == "aa":
            doc["knowledge_snapshot_sha256"] = "f" * 64
        return doc

    monkeypatch.setattr(dc, "port_baseline", tampered)
    with pytest.raises(ValueError, match="undeclared leaves|not request-identical"):
        dc.derive(REPO, dry_run=True)


def test_derive_refuses_a_llama_geometry_text(monkeypatch):
    real = dc.what_this_is

    monkeypatch.setattr(dc, "what_this_is", lambda key: real(key) + " resident np4 session")
    with pytest.raises(ValueError, match="llama.cpp geometry"):
        dc.derive(REPO, dry_run=True)


def test_derive_never_writes_a_266e_file():
    result = dc.derive(REPO, dry_run=True)
    for relpath in result["outputs"]:
        assert not Path(relpath).name.startswith(("slurp-direct-only-fold-b-thinking", "slurp-direct-only-fold-b-instruct"))
    assert all(relpath.endswith(("-raw-port-v1.json", "-npu-port-v1.json")) for relpath in result["outputs"])


def test_no_llama_geometry_in_any_config_text():
    for key in KEYS:
        assert dc.scan_geometry(_baseline(key)) == [] and dc.scan_geometry(_experiment(key)) == []


# ---- the writer and the package -------------------------------------------------------------------------------------------------------


def _tmp_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "docs").mkdir(parents=True)
    os.symlink(REPO / "configs", root / "configs")
    return root


def test_writer_is_deterministic_and_matches_the_committed_package(tmp_path):
    facts = json.loads((PKG_DIR / "registered-facts-v1.json").read_text(encoding="utf-8"))
    commit, at = facts["study_commit_at_registration"], facts["registered_at_utc"]
    outs = []
    for name in ("a", "b"):
        root = _tmp_root(tmp_path / name)
        wr.write_package(root, commit, at, Path("/nonexistent"), derive=False, verify_audio=False)
        outs.append(root)
    pkg = fam.PACKAGE
    files = sorted(p.name for p in (outs[0] / pkg).iterdir())
    assert files == sorted(p.name for p in PKG_DIR.iterdir())
    assert len(files) == 15
    for name in files:
        assert (outs[0] / pkg / name).read_bytes() == (outs[1] / pkg / name).read_bytes(), name
    # facts and the admission draft carry the audio-pin check, which this run skips; every other file equals the committed package
    for name in files:
        if name not in {"registered-facts-v1.json", "root-admission-draft-v1.json"}:
            assert (outs[0] / pkg / name).read_bytes() == (PKG_DIR / name).read_bytes(), name
    assert (outs[0] / fam.PREREG).read_bytes() == (REPO / fam.PREREG).read_bytes()


def test_package_documents_say_draft_held_and_name_the_fit_path():
    for path in PKG_DIR.glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        status = doc.get("status")
        if status is not None:
            assert "DRAFT" in status or "HELD" in status, path.name
    readme = (PKG_DIR / "README.md").read_text(encoding="utf-8")
    assert "DRAFT / HELD" in readme and "build_direct_only_raw_fit_gate" in readme
    for k in KEYS:
        assert fam.CELLS[k]["run"] in readme


def test_plan_files_carry_the_caps_and_the_composed_hashes():
    facts = json.loads((PKG_DIR / "registered-facts-v1.json").read_text(encoding="utf-8"))
    for key in KEYS:
        gen = json.loads((PKG_DIR / fam.plan_file_name(key, "generation")).read_text(encoding="utf-8"))
        fit = json.loads((PKG_DIR / fam.plan_file_name(key, "fit")).read_text(encoding="utf-8"))
        assert gen["execution_plan"]["planned_model_calls"] == fam.generation_calls(key)
        assert fit["execution_plan"]["planned_model_calls"] == fam.fit_calls(key) == fit["contact_cap"]
        assert gen["execution_plan"]["planned_speech_audio_seconds"] == fam.audio_seconds(key)
        assert gen["composed_config_hash"] == facts["composed_config_hashes"][key] == _compose(key).config_hash
        assert gen["execution_plan"]["protocol_hash"] == facts["identity"][key]["protocol_hash"]
        assert fit["execution_plan"]["run_id"] == fam.CELLS[key]["fit_run"] and fit["execution_plan"]["attempt_id"] == fam.CELLS[key]["fit_attempt"]
    assert facts["audio_pins"]["checked"] == facts["audio_pins"]["matching"] == 1080
    assert facts["accounting"]["cell_total"] == {"generation_calls": 6488, "fit_endpoint_calls": 3244, "speech_audio_seconds": 18950}


def test_stage3_measurement_blocks_are_keyed_by_the_port_run_ids():
    facts = json.loads((PKG_DIR / "registered-facts-v1.json").read_text(encoding="utf-8"))
    blocks = facts["stage3_measurement"]
    assert set(blocks) == {fam.CELLS[k]["run"] for k in KEYS}
    assert blocks[fam.CELLS["smoke"]["run"]]["measurement_design"] == "form-proof"
    for key in fam.PASS_KEYS:
        block = blocks[fam.CELLS[key]["run"]]
        assert block["measurement_design"] == "cross-flight-with-in-chain-replicate" and block["claims_benefit"] is False and block["rows"] == 1080
        assert block["replicate_run_id"] in blocks and block["replicate_run_id"] != fam.CELLS[key]["run"]


def test_admission_draft_is_not_minted_and_lists_the_eight_ids():
    draft = json.loads((PKG_DIR / "root-admission-draft-v1.json").read_text(encoding="utf-8"))
    assert draft["status"] == "DRAFT-NOT-MINTED" and draft["recorded_at_utc"] is None and draft["owner"] == "root"
    assert draft["eligible_run_ids"] == fam.eligible_run_ids() and draft["eligible_attempt_ids"] == fam.eligible_attempt_ids()
    assert draft["decision_id"] == fam.DECISION_ID
    assert draft["population"]["fold_a_contact_authorized"] is False and draft["population"]["test_contact_authorized"] is False


def _ledger_rows() -> list[list[str]]:
    text = (PKG_DIR / "draft-ledger-rows.md").read_text(encoding="utf-8")
    return [[c.strip() for c in line.strip().strip("|").split(" | ")] for line in text.splitlines() if line.startswith("| 2026-")]


def test_ledger_rows_are_eight_unconsumed_provisional_drafts():
    rows = _ledger_rows()
    assert len(rows) == 8
    text = (PKG_DIR / "draft-ledger-rows.md").read_text(encoding="utf-8")
    assert text.startswith(fam.LEDGER_MARK) and "build_direct_only_raw_fit_gate" in text
    by_run = {r[1]: r for r in rows}
    assert set(by_run) == set(fam.eligible_run_ids())
    for key in KEYS:
        gen, fit = by_run[fam.CELLS[key]["run"]], by_run[fam.CELLS[key]["fit_run"]]
        for row in (gen, fit):
            assert len(row) == 15 and row[8] == "no" and row[11].startswith("PROVISIONAL") and row[0] == "2026-10-30"
            assert row[3].startswith(("DRAFT -- REGISTERED", "DRAFT"))
            for diff in ("8.1", "8.2", "8.3", "8.5", "D1"):
                assert diff in row[3] or "FIT spend identity" in row[3]
        assert int(gen[9]) == fam.generation_calls(key) and int(fit[9]) == fam.fit_calls(key)
        assert int(gen[10]) == int(fit[10]) == fam.audio_seconds(key)
        assert fam.CELLS[key]["run"] in gen[13] or "plan-registered" in gen[13]


def test_append_ledger_is_idempotent_and_only_touches_the_given_root(tmp_path):
    facts = json.loads((PKG_DIR / "registered-facts-v1.json").read_text(encoding="utf-8"))
    root = _tmp_root(tmp_path)
    (root / "docs" / "exposure-ledger.md").write_text("# ledger\n\nrow\n", encoding="utf-8")
    first = wr.write_package(root, facts["study_commit_at_registration"], facts["registered_at_utc"], Path("/nonexistent"),
                             append_ledger=True, derive=False, verify_audio=False)
    second = wr.write_package(root, facts["study_commit_at_registration"], facts["registered_at_utc"], Path("/nonexistent"),
                              append_ledger=True, derive=False, verify_audio=False)
    assert (first["ledger_appended"], second["ledger_appended"]) == (True, False)
    text = (root / "docs" / "exposure-ledger.md").read_text(encoding="utf-8")
    assert text.count(fam.LEDGER_MARK) == 1 and text.startswith("# ledger\n\nrow\n")
    assert len([ln for ln in text.splitlines() if ln.startswith("| 2026-10-30 | SAEA-SLURP-DIRECT-ONLY")]) == 8


def test_signoff_conditions_validate_with_the_machine_signoff_tool():
    sys.path.insert(0, str(REPO / "scripts" / "flights" / "batch"))
    import smoke_signoff  # noqa: PLC0415

    doc = json.loads((PKG_DIR / "signoff-conditions-v1.json").read_text(encoding="utf-8"))
    smoke_signoff.validate_conditions(doc)
    assert doc == fam.signoff_conditions() and doc["expected_arms"] == ["direct"]
    ok = {"schema": doc["table_schema"], "arms": {"direct": {"solver": {"contacts": 4, "cap_hits": 0, "empty_content": 0, "finish_reason_length": 0, "no_json_object": 0}}},
          "throughput": {"arm_mix_measured_rows": 0.5, "arm_mix_registered_rows": 0.52}, "unattributed_contacts": 0}
    assert all(c["ok"] for c in smoke_signoff.evaluate(doc, ok))
    bad = copy.deepcopy(ok)
    bad["arms"]["direct"]["solver"]["cap_hits"] = 1
    assert not all(c["ok"] for c in smoke_signoff.evaluate(doc, bad))


def test_no_cjk_in_new_or_changed_files():
    files = NEW_FILES + sorted(PKG_DIR.iterdir())
    assert all(p.is_file() for p in files)
    assert [str(p) for p in files if CJK.search(p.read_text(encoding="utf-8"))] == []


def test_the_266e_package_and_writer_are_untouched():
    import subprocess

    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "scripts/flights/direct/direct_fb_family.py",
                          "scripts/flights/direct/direct_fb_write_registration.py", "scripts/flights/direct/direct_fb_derive_cells.py",
                          fam.HISTORICAL_PACKAGE, "configs/baseline/slurp-direct-only-fold-b-thinking-baseline-v1.json",
                          "configs/experiment/slurp-direct-only-fold-b-thinking-v1.json"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == ""
