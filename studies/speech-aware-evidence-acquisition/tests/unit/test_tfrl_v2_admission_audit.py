import json

from tfrl_v2_admission_audit import audit


def test_draft_and_zero_bank_are_not_admitted(tmp_path):
    (tmp_path / "root-admission-draft-v1.json").write_text(
        json.dumps({"status": "DRAFT-NOT-MINTED"}), encoding="utf-8"
    )
    (tmp_path / "bank-release-v1.json").write_text(
        json.dumps({"state": "PENDING", "sha": "0" * 64}), encoding="utf-8"
    )
    result = audit(tmp_path)
    assert result["ready"] is False
    assert {item["id"] for item in result["blockers"]} >= {"root-admission", "deployment-profile", "bank-release"}
    assert result["model_calls"] == 0


def test_decided_complete_minimal_root_can_pass(tmp_path):
    (tmp_path / "root-admission-draft-v1.json").write_text(
        json.dumps({"status": "DECIDED", "deployment_profile": {"cards": 8}}), encoding="utf-8"
    )
    (tmp_path / "bank-release-v1.json").write_text(
        json.dumps({"state": "RELEASED", "sha": "a" * 64}), encoding="utf-8"
    )
    assert audit(tmp_path)["ready"] is True
