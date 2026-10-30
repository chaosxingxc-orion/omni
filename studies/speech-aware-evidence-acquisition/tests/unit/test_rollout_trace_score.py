"""Plan T1 / M4: the scorer emits one ``score`` line per (row, arm) once the official per-row scores are known (model-free)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.contracts import rollout_trace as schema
from speech_aware_evidence_acquisition.core import rollout_trace as trace

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("stage3_rollout_score_for_trace", ROOT / "scripts/analysis/stage3_rollout_score.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SCORER = "slurp-slu-official-projected-v3-train"
PER_ROW = {
    "lock": {"r1": (1.0, 0.75), "r2": (0.0, 0.0)},
    "aa": {"r1": (1.0, 1.0), "r2": (1.0, 0.5)},
}


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def fake_score_complete_derived_copy(*, outputs_copy, scorer_id, result_path, column_binding, **_):
    arm = column_binding["arm"]
    rows = [
        {"sample_id": sample, "scenario_action_exact": exact, "slu_f1_f1": f1}
        for sample, (exact, f1) in PER_ROW[arm].items()
    ]
    result = {"aggregate": {"official": {"slu_f1": 0.5, "intent_accuracy": 0.5}, "identity_check": {"passed": True}}, "per_sample": rows}
    Path(result_path).write_text(json.dumps(result))
    return result


@pytest.fixture
def supply_variant_world(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.core import scoring_batch

    monkeypatch.setattr(scoring_batch, "score_complete_derived_copy", fake_score_complete_derived_copy)
    out_dir = tmp_path / "analysis" / "READ-T1"
    out_dir.mkdir(parents=True)
    (out_dir / "render-manifest.json").write_text("{}")
    copies = {}
    for arm in PER_ROW:
        path = out_dir / f"{arm}.jsonl"
        path.write_text("".join(json.dumps({"sample_id": sample}) + "\n" for sample in PER_ROW[arm]))
        copies[arm] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        "schema": mod.SUPPLY_RENDER_SCHEMA, "rows": 2, "arm_order": list(PER_ROW), "copies": copies,
        "arm_declarations": [{"arm_id": arm} for arm in PER_ROW],
        "source": {"manifest_path": "m", "roster_sha256": "r", "sealed_scores_sha256": "s"},
    }
    delivered = {"r1": {"scenario_action_exact": 1.0, "slu_f1_f1": 0.75}, "r2": {"scenario_action_exact": 0.0, "slu_f1_f1": 0.25}}
    return out_dir, manifest, delivered


def test_supply_variant_scoring_emits_one_score_line_per_row_and_arm(supply_variant_world):
    out_dir, manifest, delivered = supply_variant_world
    writer = mod.open_score_trace(out_dir, "RUN-1", "attempt-1")
    mod.score_supply_variant_copies(SCORER, manifest, out_dir, {}, out_dir, delivered, trace_writer=writer)
    lines = read_lines(out_dir / "rollout-trace.jsonl")
    assert all(schema.validate_line(line) == line for line in lines)
    assert {(line["row_id"], line["arm"]) for line in lines} == {
        (row, arm) for arm in ("lock", "aa", "delivered") for row in ("r1", "r2")
    }
    by_key = {(line["row_id"], line["arm"]): line for line in lines}
    assert by_key[("r1", "lock")]["slu_f1"] == 0.75 and by_key[("r1", "lock")]["scenario_action_exact"] is True
    assert by_key[("r2", "lock")]["scenario_action_exact"] is False
    assert by_key[("r2", "delivered")]["slu_f1"] == 0.25
    assert {line["scorer_id"] for line in lines} == {SCORER} and {line["run_id"] for line in lines} == {"RUN-1"}


def test_a_scorer_rerun_appends_nothing_twice_and_scoring_works_without_a_trace(supply_variant_world, tmp_path):
    out_dir, manifest, delivered = supply_variant_world
    for _ in range(2):
        mod.score_supply_variant_copies(
            SCORER, manifest, out_dir, {}, out_dir, delivered, trace_writer=mod.open_score_trace(out_dir, "RUN-1", "attempt-1"),
        )
    assert len(read_lines(out_dir / "rollout-trace.jsonl")) == 6
    # the untouched signature still works: no writer, no trace file
    other = tmp_path / "other"
    other.mkdir()
    (other / "render-manifest.json").write_text("{}")
    manifest = {**manifest, "copies": {arm: dict(entry) for arm, entry in manifest["copies"].items()}}
    summary = mod.score_supply_variant_copies(SCORER, manifest, other, {}, other, delivered)
    assert summary["arms"].keys() == PER_ROW.keys() and not (other / "rollout-trace.jsonl").exists()


def test_an_unwritable_trace_never_fails_the_scoring(supply_variant_world):
    out_dir, manifest, delivered = supply_variant_world
    (out_dir / "rollout-trace.jsonl").mkdir()  # a directory where the file should be
    writer = mod.open_score_trace(out_dir, "RUN-1", "attempt-1")
    summary = mod.score_supply_variant_copies(SCORER, manifest, out_dir, {}, out_dir, delivered, trace_writer=writer)
    assert summary["sealed_delivered_rows"] == 2


def test_grouped_copies_scoring_emits_score_lines_keyed_by_copy_name(tmp_path, monkeypatch):
    out_dir = tmp_path / "analysis" / "READ-T1"
    out_dir.mkdir(parents=True)
    copy = out_dir / "sample-0.jsonl"
    copy.write_text("{}\n")

    def fake_score_copy(scorer_id, outputs_copy, result_path, lock, root):
        rows = [{"sample_id": "r1", "scenario_action_exact": 1.0, "slu_f1_f1": 0.5}]
        return {"aggregate": {"official": {"slu_f1": 0.5, "intent_accuracy": 1.0}}, "per_sample": rows}, 0.0

    monkeypatch.setattr(mod, "score_copy", fake_score_copy)
    monkeypatch.setattr(mod, "map_by_key", lambda tasks, workers: {name: task() for name, task in tasks.items()})
    manifest = {"copies": {"sample-0": {"path": str(copy), "sha256": hashlib.sha256(copy.read_bytes()).hexdigest()}}}
    mod.score(SCORER, manifest, out_dir, {}, out_dir, workers=1, trace_writer=mod.open_score_trace(out_dir, "RUN-1", "attempt-1"))
    (line,) = read_lines(out_dir / "rollout-trace.jsonl")
    assert (line["kind"], line["row_id"], line["arm"], line["slu_f1"]) == ("score", "r1", "sample-0", 0.5)


def test_score_lines_and_flight_lines_share_one_file(tmp_path):
    flight = trace.RolloutTraceWriter(tmp_path / "rollout-trace.jsonl", run_id="RUN-1", attempt_id="attempt-1")
    assert flight.tfrl(row_id="r1", arm=None, trajectory_id="t", reward=0.1, selected=False)
    scorer = mod.open_score_trace(tmp_path, "RUN-1", "attempt-1")
    assert trace.emit_arm_scores(scorer, arm="lock", scorer_id=SCORER, rows=[
        {"sample_id": "r1", "scenario_action_exact": 1.0, "slu_f1_f1": 1.0},
    ]) == 1
    assert [line["kind"] for line in read_lines(tmp_path / "rollout-trace.jsonl")] == ["tfrl", "score"]
