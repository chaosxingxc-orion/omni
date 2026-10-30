"""The vote scorer driver (``scripts/analysis/slurp_tfrl_v2_port_vote_score.py``, plan section 13 ruling 3 (b)): the derived `vote` answers become a scored column and the
construction gain becomes a number, so the reader's stop word is CONTINUE (gate pass) or STOP instead of UNDETERMINED.

Model-free, fake answers, a FAKE official scorer injected through ``score_fn_for`` (the production path is ``core.scoring_batch.score_complete_derived_copy``, whose arguments are proved
with a monkeypatched stand-in). The proofs: (1) the vote copy is the complete half roster in canonical rendering with a placeholder for unusable rows, and is write-once; (2) the
vote answers file is validated; (3) end to end over the reader's own synthetic data root: construction stage (gain null, UNDETERMINED) -> lock + vote columns -> gain document
(CONTINUE at a gain over +1 pp, STOP at none) -> the Fold B read takes the gain from the scored document; (4) a missing lock column or a single half leaves the gain null and writes no
gain document; (5) the gain document is write-once; (6) the bound scorer hands the verified parent manifest and the column binding to the derived-copy scorer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "analysis", REPO / "src", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import slurp_direct_fold_b_core_read as base  # noqa: E402
import slurp_tfrl_v2_port_read as rd  # noqa: E402
import slurp_tfrl_v2_port_vote_score as vs  # noqa: E402
import stage3_rollout_score as s3  # noqa: E402
import test_tfrl_v2_port_read as fx  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_port_family as tf  # noqa: E402
import tfrl_v2_port_rails as rails  # noqa: E402

GOLD = s3.render_answer({"scenario": "alarm", "action": "set", "entities": [{"type": "date", "filler": "monday"}]})


def fake_scorer(win_text=GOLD, win=(4, 0, 1), lose=(3, 1, 1)):
    """A stand-in for the official scorer: a row whose text is ``win_text`` gets the winning count row, any other the losing one."""

    def score_fn_for(half, entry, scorer_id):
        def score(outputs_copy, result_path):
            rows = [json.loads(line) for line in Path(outputs_copy).read_text(encoding="utf-8").splitlines()]
            per_sample = [{"sample_id": r["sample_id"], **fx.row(*(win if r["text"] == win_text else lose))} for r in rows]
            return {"per_sample": per_sample}
        return score
    return score_fn_for


def args(root, **over):
    base_args = dict(iteration=1, half="both", data_root=root, repo_root=REPO, analysis_root=None, scorer=rd.SCORER, lock={})
    base_args.update(over)
    return argparse.Namespace(**base_args)


def construction(root, tmp_path, capsys):
    recipe = tmp_path / "recipe.json"
    recipe.write_bytes(rails.recipe_bytes())
    assert rd.main(["--stage", "construction", "--iteration", "1", "--data-root", str(root), "--recipe", str(recipe)]) == 0
    capsys.readouterr()
    return recipe


def lock_columns(root, halves=cm.HALVES):
    for half in halves:
        roster = json.loads((REPO / cm.HALF_ROSTER_RELPATH[half]).read_text(encoding="utf-8"))
        columns = rd.columns_dir(root, 1, half)
        columns.mkdir(parents=True, exist_ok=True)
        (columns / rd.COLUMN_SCORE_NAME.format(arm="lock", scorer=rd.SCORER)).write_text(
            json.dumps(fx.scores_file({i["sample_id"]: fx.row(3, 1, 1) for i in roster["items"]})), encoding="utf-8")


# ---- (1) and (2) the copy --------------------------------------------------------------------------------------------------------------------------------


def vote_line(row_id, scenario="alarm", action="set", entities=()):
    return {"schema": rd.VOTE_SCHEMA, "row_id": row_id, "iteration": 1, "half": "R01C-H1", "scenario": scenario, "action": action, "entities": [{"type": t, "filler": f} for t, f in entities],
            "intent_votes": 3, "n": 4}


def write_votes(path, lines):
    Path(path).write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), encoding="utf-8")


def test_the_vote_copy_is_the_complete_half_roster_in_canonical_rendering_with_a_placeholder_for_an_unusable_row(tmp_path):
    roster = {"items": [{"sample_id": "r1"}, {"sample_id": "r2"}, {"sample_id": "r3"}]}
    votes = tmp_path / "votes.jsonl"
    write_votes(votes, [vote_line("r1", entities=[("date", "monday")]), vote_line("r3", "music", "play")])
    entry = vs.render_vote_copy(votes, roster, tmp_path / "columns", half="R01C-H1", iteration=1)
    rows = [json.loads(line) for line in Path(entry["path"]).read_text(encoding="utf-8").splitlines()]
    assert [r["sample_id"] for r in rows] == ["r1", "r2", "r3"] and all(set(r) == {"sample_id", "text"} for r in rows)
    assert rows[0]["text"] == GOLD and rows[1]["text"] == s3.EMPTY_ANSWER and json.loads(rows[2]["text"])["scenario"] == "music"
    assert (entry["rows"], entry["voted_rows"], entry["placeholder_rows"]) == (3, 2, 1) and len(entry["sha256"]) == 64
    assert vs.render_vote_copy(votes, roster, tmp_path / "columns", half="R01C-H1", iteration=1) == entry                      # idempotent
    write_votes(votes, [vote_line("r1", "music", "play")])
    with pytest.raises(rd.ReadError, match="different bytes"):
        vs.render_vote_copy(votes, roster, tmp_path / "columns", half="R01C-H1", iteration=1)


def test_the_vote_answers_file_is_validated(tmp_path):
    path = tmp_path / "votes.jsonl"
    write_votes(path, [vote_line("r1"), vote_line("r1")])
    with pytest.raises(rd.ReadError, match="malformed or repeats a row"):
        vs.load_votes(path)
    write_votes(path, [{**vote_line("r1"), "schema": "other"}])
    with pytest.raises(rd.ReadError, match="is not saea-tfrl-v2-vote-answers-v1"):
        vs.load_votes(path)
    write_votes(path, [{**vote_line("r1"), "entities": [{"type": "date"}]}])
    with pytest.raises(rd.ReadError, match="malformed"):
        vs.load_votes(path)
    with pytest.raises(rd.ReadError, match="not roster rows"):
        vs.vote_copy_rows({"zz": vote_line("zz")}, ["r1"])


# ---- (3) end to end ------------------------------------------------------------------------------------------------------------------------------------


def test_scoring_the_vote_column_turns_the_stop_word_from_undetermined_into_continue_and_the_fold_b_read_takes_it(tmp_path, capsys):
    root = tmp_path / "data"
    fx.build_data_root(root, with_columns=False)
    recipe = construction(root, tmp_path, capsys)
    first = json.loads((root / rails.ANALYSIS_ROOT / "construction-i1.json").read_text(encoding="utf-8"))
    assert first["construction_gain"]["gain_pp"] is None and first["stop"]["gate"]["word"] == rd.WORD_UNDETERMINED
    lock_columns(root)
    summary = vs.run(args(root), score_fn_for=fake_scorer())
    assert summary["gate"] == rd.WORD_CONTINUE and summary["gain_pp"] > 1.0
    assert {h: v["voted_rows"] for h, v in summary["halves"].items()} == {"R01C-H1": 313, "R01C-H2": 312}
    gain = json.loads((root / rails.ANALYSIS_ROOT / "construction-gain-i1.json").read_text(encoding="utf-8"))
    assert gain["schema"] == vs.GAIN_SCHEMA and gain["construction_gain"]["rows"] == 625 and gain["construction_gain"]["vote_slu_f1_pct"] > gain["construction_gain"]["lock_slu_f1_pct"]
    assert gain["stop"]["gate"]["word"] == "CONTINUE" and gain["stop"]["gate"]["continue"] is True
    for name in ("vote_answers:R01C-H1", "vote_column:R01C-H1", "vote_column:R01C-H2", "lock_column:R01C-H2"):
        assert len(gain["inputs"][name]["sha256"]) == 64
    # the column file is exactly what the reader loads for any arm
    columns = rd.columns_dir(root, 1, "R01C-H1")
    assert rd._load_scores(columns, rd.VOTE_ARM, rd.SCORER) is not None
    # the Fold B read prefers the scored gain over the null gain of the write-once construction read
    cell = tf.CELLS["fb"]
    paths = base.run_paths(root, cell["run"], cell["attempt"])
    paths["dir"].mkdir(parents=True)
    paths["trace"].write_text("{}\n", encoding="utf-8")
    paths["outputs"].write_text("{}\n", encoding="utf-8")
    paths["manifest"].write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    ids, _, _ = base.roster_maps(REPO / tf.ROSTERS["fb"])
    foldb_columns = root / rails.ANALYSIS_ROOT / "columns"
    foldb_columns.mkdir(parents=True)
    lock = {i: fx.row(3, 1, 1) for i in ids}
    arms = {tf.DELIVERING: lock, tf.TWIN: dict(lock), tf.REPLICATE: dict(lock), tf.T_ARM: {i: (fx.row(4, 0, 1) if k % 3 == 0 else fx.row(3, 1, 1)) for k, i in enumerate(ids)}}
    for arm, rows in arms.items():
        (foldb_columns / rd.COLUMN_SCORE_NAME.format(arm=arm, scorer=rd.SCORER)).write_text(json.dumps(fx.scores_file(rows)), encoding="utf-8")
    assert rd.main(["--stage", "read", "--data-root", str(root), "--recipe", str(recipe), "--draws", "200"]) == 0
    read_summary = json.loads(capsys.readouterr().out)
    assert read_summary["verdict"] == rd.WORD_ACCEPTED and read_summary["next_iteration_allowed"] is True
    verdicts = json.loads((root / rails.ANALYSIS_ROOT / "tfrl-v2-verdicts-i1.json").read_text(encoding="utf-8"))
    assert verdicts["construction"]["gain"]["gain_pp"] == gain["construction_gain"]["gain_pp"] and len(verdicts["inputs"]["construction_gain"]["sha256"]) == 64


def test_a_vote_that_does_not_beat_the_lock_stops_the_family(tmp_path, capsys):
    root = tmp_path / "data"
    fx.build_data_root(root, with_columns=False)
    construction(root, tmp_path, capsys)
    lock_columns(root)
    summary = vs.run(args(root), score_fn_for=fake_scorer(win=(3, 1, 1)))                   # the vote scores exactly like the lock: gain 0.0
    assert summary["gain_pp"] == pytest.approx(0.0) and summary["gate"] == rd.WORD_STOP
    gain = json.loads((root / rails.ANALYSIS_ROOT / "construction-gain-i1.json").read_text(encoding="utf-8"))
    assert gain["stop"]["gate"]["stop"] is True and gain["stop"]["next_iteration_allowed"] is False


# ---- (4) incomplete inputs, (5) write-once -------------------------------------------------------------------------------------------------------------


def test_a_missing_lock_column_or_a_single_half_leaves_the_gain_null_and_writes_no_gain_document(tmp_path, capsys):
    root = tmp_path / "data"
    fx.build_data_root(root, with_columns=False)
    construction(root, tmp_path, capsys)
    summary = vs.run(args(root), score_fn_for=fake_scorer())
    assert summary["gain_pp"] is None and summary["gate"] == rd.WORD_UNDETERMINED and summary["written"] is None and "lock column" in summary["note"]
    assert not (root / rails.ANALYSIS_ROOT / "construction-gain-i1.json").exists()
    assert rd._load_scores(rd.columns_dir(root, 1, "R01C-H1"), rd.VOTE_ARM, rd.SCORER) is not None                          # the vote columns ARE scored and kept
    lock_columns(root)
    one = vs.run(args(root, half="R01C-H1"), score_fn_for=fake_scorer())
    assert one["gain_pp"] is None and one["written"] is None and "both halves" in one["note"]
    assert not (root / rails.ANALYSIS_ROOT / "construction-gain-i1.json").exists()


def test_a_scored_column_is_read_back_not_scored_again_and_the_gain_document_is_write_once(tmp_path, capsys):
    root = tmp_path / "data"
    fx.build_data_root(root, with_columns=False)
    construction(root, tmp_path, capsys)
    lock_columns(root)
    vs.run(args(root), score_fn_for=fake_scorer())
    with pytest.raises(rd.ReadError, match="write-once"):
        vs.run(args(root), score_fn_for=fake_scorer())

    (root / rails.ANALYSIS_ROOT / "construction-gain-i1.json").unlink()
    again = vs.run(args(root), score_fn_for=lambda *a: (lambda *b: pytest.fail("scored twice")))
    assert again["gate"] == rd.WORD_CONTINUE
    code = vs.main(["--data-root", str(root), "--repo-root", str(REPO)])
    assert code == 2 and "write-once" in capsys.readouterr().err


def test_the_cli_refuses_when_the_construction_stage_has_not_written_the_vote_answers(tmp_path, capsys):
    assert vs.main(["--data-root", str(tmp_path / "empty"), "--repo-root", str(REPO)]) == 2
    assert "run the construction stage" in capsys.readouterr().err


# ---- (6) the production scorer's wiring ------------------------------------------------------------------------------------------------------------------


def test_the_bound_scorer_hands_the_verified_parent_manifest_and_the_column_binding_to_the_derived_copy_scorer(tmp_path, monkeypatch):
    seen = {}

    def fake_derived(**kwargs):
        seen.update(kwargs)
        return {"per_sample": []}

    import speech_aware_evidence_acquisition.core.scoring_batch as sb
    monkeypatch.setattr(sb, "score_complete_derived_copy", fake_derived)
    entry = {"half": "R01C-H1", "iteration": 1, "sha256": "a" * 64, "vote_answers_sha256": "b" * 64, "path": str(tmp_path / "copy.jsonl")}
    parent = tmp_path / "run" / "attempt" / "RUN.run-manifest.json"
    score = vs.bound_scorer(parent_manifest=parent, entry=entry, scorer_id=rd.SCORER, lock={"k": 1}, data_root=tmp_path, roster_sha256="c" * 64)
    score(Path(entry["path"]), tmp_path / "arm-vote.json")
    assert seen["parent_manifest"] == parent and seen["expected_copy_sha256"] == "a" * 64 and seen["scorer_id"] == rd.SCORER and seen["data_root"] == tmp_path
    assert seen["column_binding"] == {"column": "vote", "half": "R01C-H1", "iteration": 1, "vote_answers_sha256": "b" * 64, "roster_sha256": "c" * 64,
                                      "visit": "derived-vote-over-the-sampled-trajectories"}
