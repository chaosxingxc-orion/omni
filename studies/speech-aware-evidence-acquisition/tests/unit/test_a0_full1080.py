import importlib.util
import json
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
spec = importlib.util.spec_from_file_location("full_a0", ROOT / "scripts/analysis/score_a0_full1080.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_exact_union_preserves_answer_and_refuses_missing_or_overlap():
    old = {"a": {"sample_id": "a", "text": "original"}}
    new = {"b": {"sample_id": "b", "text": "fresh"}}
    rows = m.complete_union(["b", "a"], old, ["b"], new)
    assert rows == [old["a"], new["b"]]
    with pytest.raises(ValueError, match="exactly"):
        m.complete_union(["a", "b"], old, ["b"], {})
    with pytest.raises(ValueError, match="partition"):
        m.complete_union(["a", "b"], old, ["a", "b"], new)
    with pytest.raises(ValueError, match="exactly"):
        m.complete_union(["a", "b"], old, ["b"], {**old, **new})


def test_duplicate_output_refused():
    with pytest.raises(ValueError, match="duplicate"):
        m.unique_rows([{"sample_id": "a"}, {"sample_id": "a"}])


@pytest.mark.parametrize("field,value", [
    ("scorer", "slurp-slu-official-test"),
    ("rows", 727),
    ("selection", "best answer per sample"),
    ("continuation_run_id", "another-run"),
    ("new_model_calls", 1),
    ("new_official_scoring_jobs", 2),
])
def test_registered_scope_rejects_different_scorer_or_selection(field, value):
    recipe = json.loads((ROOT / "docs/readiness/2026-10-31-a0-full1080-source-read/analysis-recipe-v1.json").read_bytes())
    m.validate_recipe(recipe)
    recipe[field] = value
    with pytest.raises(ValueError, match="scope or scoring recipe"):
        m.validate_recipe(recipe)
