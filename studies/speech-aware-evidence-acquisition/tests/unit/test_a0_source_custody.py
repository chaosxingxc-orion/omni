import importlib.util
from pathlib import Path
import pytest
spec = importlib.util.spec_from_file_location("custody", Path(__file__).resolve().parents[2] / "scripts/analysis/prepare_a0_source_custody.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_historical_precedence_keeps_original_output_and_disjoint_remainder():
    old = {"a": {"text": "old"}}
    candidate = {"a": {"text": "new better answer"}, "b": {"text": "other"}}
    selected, remaining = m.partition(["a", "b", "c"], old, candidate)
    assert selected["a"][1] is old["a"]
    assert selected["b"][1] is candidate["b"]
    assert remaining == ["c"]


def test_foreign_and_duplicate_population_refused():
    with pytest.raises(ValueError, match="outside"):
        m.partition(["a"], {}, {"b": {}})
    with pytest.raises(ValueError, match="duplicate"):
        m.partition(["a", "a"], {}, {})
