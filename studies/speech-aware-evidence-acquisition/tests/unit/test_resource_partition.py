import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("partition", ROOT / "scripts/environment/validate_resource_partition.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def manifest():
    return {"schema": "saea-resource-partition-v1", "card_count": 8, "lanes": [
        {"name": name, "profile": profile, "state_root": "/var/lib/omni-local-state/resource-partition-v1/lanes/" + name}
        for name, profile in [("fulltest", "qwen3-omni-fulltest-1xtp4-s128-ctx48k-v1"),
            ("explore-a", "qwen3-omni-explore-a-1xtp2-s48-ctx48k-v1"),
            ("explore-b", "qwen3-omni-explore-b-1xtp2-s48-ctx48k-v1")]]}


def test_real_profiles_and_model_endpoints_are_disjoint():
    result = m.validate(manifest(), ROOT)
    assert result["used_cards"] == list(range(8))
    assert [x["ports"] for x in result["lanes"]] == [[8200], [8202], [8203]]


@pytest.mark.parametrize("fault", ["duplicate", "ancestor", "same_root", "relative", "too_few_cards", "traversal"])
def test_refuses_resource_overlap_and_invalid_roots(fault):
    doc = manifest()
    if fault == "duplicate": doc["lanes"][1]["profile"] = doc["lanes"][0]["profile"]
    elif fault == "ancestor": doc["lanes"][1]["state_root"] = "/var/lib/omni-local-state/resource-partition-v1"
    elif fault == "same_root": doc["lanes"][1]["state_root"] = doc["lanes"][0]["state_root"]
    elif fault == "relative": doc["lanes"][0]["state_root"] = "relative"
    elif fault == "too_few_cards": doc["card_count"] = 4
    else: doc["lanes"][0]["profile"] = "../../other"
    with pytest.raises(ValueError): m.validate(doc, ROOT)


@pytest.mark.parametrize("fault", ["port", "endpoint", "duplicate_card", "bool_card"])
def test_checks_actual_profile_values(tmp_path, fault):
    doc = manifest()
    for lane in doc["lanes"]:
        p = ROOT / "configs/deployment" / (lane["profile"] + ".json")
        profile = json.loads(p.read_bytes())
        target = tmp_path / p.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(profile))
        p = ROOT / "configs/model" / (profile["model_fragment"] + ".json")
        target = tmp_path / p.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    path = tmp_path / "configs/deployment" / (doc["lanes"][1]["profile"] + ".json")
    profile = json.loads(path.read_bytes())
    if fault == "port": profile["instances"][0]["port"] = 8200
    elif fault == "duplicate_card": profile["instances"][0]["cards"] = [4, 4]
    elif fault == "bool_card": profile["instances"][0]["cards"] = [True, 5]
    else:
        model_path = tmp_path / "configs/model" / (profile["model_fragment"] + ".json")
        model = json.loads(model_path.read_bytes()); model["base_url"] = "http://127.0.0.1:8200"
        model_path.write_text(json.dumps(model))
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError): m.validate(doc, tmp_path)
