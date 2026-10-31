import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("compat", Path(__file__).resolve().parents[2] / "scripts/analysis/audit_real500_a0_inputs.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_operational_changes_do_not_hide_decoding_or_prompt_changes():
    model = {"runtime_receipt": "old", "timeout_seconds": 420, "temperature": .7}
    base = {"native_contact_fit": {"run_id": "r", "attempt_id": "a", "contact_cap": 2000}, "prompt": "same"}
    target = {**model, "runtime_receipt": "new", "timeout_seconds": 1800}
    assert len(m.compare_method(model, target, base, base)) == 2
    with pytest.raises(ValueError, match="decoding"):
        m.compare_method(model, {**target, "temperature": .8}, base, base)
    with pytest.raises(ValueError, match="scientific"):
        m.compare_method(model, target, base, {**base, "prompt": "different"})


def test_exact_subset_evidence_and_missing_rows():
    full = {"facets": {}, "values": {"perception": {"a": [1], "b": [2]}}}
    subset = {"facets": {}, "values": {"perception": {"b": [2]}}}
    assert m.compare_supply(full, subset, ["b"]) == 1
    with pytest.raises(ValueError, match="exactly"):
        m.compare_supply(full, subset, ["a", "b"])
    changed = copy.deepcopy(subset); changed["values"]["perception"]["b"] = [3]
    with pytest.raises(ValueError, match="differs"):
        m.compare_supply(full, changed, ["b"])
    with pytest.raises(ValueError, match="no prepared"):
        m.compare_supply({"values": {}}, {"values": {}}, ["b"])


def test_timing_normalization_does_not_hide_body_changes():
    plain = "def f(x):\n    return x + 1\n"
    timed = "from .stage_timing import measure\n@measure('request_total')\n" + plain
    assert m.without_timing(plain) == m.without_timing(timed)
    assert m.without_timing(plain) != m.without_timing(timed.replace("x + 1", "x + 2"))


def test_runtime_rejects_weight_or_image_change():
    old = {"model": {"sha": "a"}, "container": {"image_id": "id", "image_ref": "tag"}, "deployment": {"pid": 1}, "protocol_sha256": "old"}
    new = copy.deepcopy(old); new["deployment"] = {"pid": 2}
    assert m.compare_runtime(old, new)["same_image_digest"]
    new["model"]["sha"] = "b"
    with pytest.raises(ValueError, match="receipt differs"):
        m.compare_runtime(old, new)
