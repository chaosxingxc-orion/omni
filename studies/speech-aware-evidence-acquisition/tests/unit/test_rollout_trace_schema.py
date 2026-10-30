"""Plan T1 / M4: the ``saea-rollout-trace-v1`` schema, its writer and its value helpers (model-free, no server)."""

from __future__ import annotations

import json
import os
import threading

import pytest

from speech_aware_evidence_acquisition.contracts import rollout_trace as schema
from speech_aware_evidence_acquisition.core import rollout_trace as trace

TS = "2026-10-30T10:00:00.000Z"
SHA = "a" * 64


def contact_line(**over):
    line = {
        "schema": schema.SCHEMA, "kind": "contact", "ts": TS, "run_id": "RUN-1", "attempt_id": "attempt-1",
        "row_id": "slurp-1", "arm": "lock", "node": "solver", "role_id": "solver", "contact_id": "c-1",
        "contact_index": 1, "sample_index": None, "request_id": "r-0001", "outcome": "ok", "error_type": None,
        "id_slot": 5, "instance": 1, "supply_variant_id": None,
        "supply_items": [
            {"item_id": "store::a", "rank": 1, "score": 0.91, "used": True},
            {"item_id": "store::b", "rank": 2, "score": None, "used": None},
        ],
        "prompt_tokens": 4100, "completion_tokens": 3300, "thinking_tokens": 3112,
        "thinking_tokens_basis": "server-usage-details", "reasoning_chars": 9000, "thinking_budget": 3112,
        "max_tokens": 4096, "truncated": True, "finish_reason": "stop", "forced_close": True, "answer_only_reask": False,
        "answer": {"parsed": True, "scenario": "alarm", "action": "set", "entities": [{"type": "time", "filler": "nine am"}]},
        "response_text_sha256": SHA, "latency_seconds": 41.5,
    }
    line.update(over)
    return line


def score_line(**over):
    line = {
        "schema": schema.SCHEMA, "kind": "score", "ts": TS, "run_id": "RUN-1", "attempt_id": "attempt-1",
        "row_id": "slurp-1", "arm": "lock", "scorer_id": "slurp-slu-official-projected-v3-train",
        "scenario_action_exact": True, "slu_f1": 0.75,
    }
    line.update(over)
    return line


def tfrl_line(**over):
    line = {
        "schema": schema.SCHEMA, "kind": "tfrl", "ts": TS, "run_id": "RUN-1", "attempt_id": "attempt-1",
        "row_id": "slurp-1", "arm": None, "iteration": 1, "trajectory_id": "traj-1-2", "reward": 0.5, "selected": True,
        "reason": None,
    }
    line.update(over)
    return line


# ---------------------------------------------------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("line", [contact_line(), score_line(), tfrl_line()])
def test_valid_lines_validate_and_round_trip(line):
    assert schema.validate_line(line) == line
    assert schema.loads_line(schema.dumps_line(line)) == line
    assert schema.dumps_line(line).endswith("\n")


def test_field_names_are_fixed_per_kind():
    assert set(schema.field_names("contact")) == set(contact_line())
    assert set(schema.field_names("score")) == set(score_line())
    assert set(schema.field_names("tfrl")) == set(tfrl_line())
    with pytest.raises(schema.RolloutTraceSchemaError):
        schema.field_names("trajectory")


@pytest.mark.parametrize("kind", ["trajectory", "", None, 7, "Contact"])
def test_unknown_kinds_are_refused(kind):
    with pytest.raises(schema.RolloutTraceSchemaError, match="unknown rollout-trace kind"):
        schema.validate_line({**contact_line(), "kind": kind})


def test_non_objects_are_refused():
    for bad in (None, [], "x", 3):
        with pytest.raises(schema.RolloutTraceSchemaError):
            schema.validate_line(bad)


@pytest.mark.parametrize("maker", [contact_line, score_line, tfrl_line])
def test_missing_and_extra_fields_are_refused(maker):
    line = maker()
    for name in list(line):
        if name == "kind":
            continue
        broken = {k: v for k, v in line.items() if k != name}
        with pytest.raises(schema.RolloutTraceSchemaError, match="missing"):
            schema.validate_line(broken)
    with pytest.raises(schema.RolloutTraceSchemaError, match="unknown fields"):
        schema.validate_line({**line, "extra": 1})


@pytest.mark.parametrize(
    "over",
    [
        {"schema": "saea-rollout-trace-v2"},
        {"ts": "yesterday"},
        {"run_id": ""},
        {"node": "judge"},
        {"outcome": "maybe"},
        {"contact_index": -1},
        {"contact_index": True},
        {"id_slot": 1.5},
        {"truncated": 1},
        {"prompt_tokens": "4100"},
        {"latency_seconds": -0.1},
        {"latency_seconds": float("nan")},
        {"response_text_sha256": "abc"},
        {"supply_items": "none"},
        {"supply_items": [{"item_id": "a", "rank": 2, "score": None, "used": None}]},
        {"supply_items": [{"item_id": "a", "rank": 1, "score": "high", "used": None}]},
        {"supply_items": [{"item_id": "a", "rank": 1, "score": None, "used": "yes"}]},
        {"supply_items": [{"item_id": "a", "rank": 1, "score": None}]},
        {"supply_items": [
            {"item_id": "a", "rank": 1, "score": None, "used": None}, {"item_id": "a", "rank": 2, "score": None, "used": None},
        ]},
        {"answer": {"parsed": True, "scenario": "a", "action": "b"}},
        {"answer": {"parsed": True, "scenario": "a", "action": "b", "entities": [{"type": "t"}]}},
        {"answer": {"parsed": "yes", "scenario": "a", "action": "b", "entities": []}},
    ],
)
def test_malformed_contact_fields_are_refused(over):
    with pytest.raises(schema.RolloutTraceSchemaError):
        schema.validate_line(contact_line(**over))


def test_contact_cross_field_rules():
    with pytest.raises(schema.RolloutTraceSchemaError, match="thinking_tokens is null exactly when"):
        schema.validate_line(contact_line(thinking_tokens=None))
    with pytest.raises(schema.RolloutTraceSchemaError, match="thinking_tokens is null exactly when"):
        schema.validate_line(contact_line(thinking_tokens_basis="not-reported-by-server"))
    schema.validate_line(contact_line(thinking_tokens=None, thinking_tokens_basis="not-reported-by-server"))
    with pytest.raises(schema.RolloutTraceSchemaError, match="ok.*error_type"):
        schema.validate_line(contact_line(error_type="X"))
    with pytest.raises(schema.RolloutTraceSchemaError, match="names its error_type"):
        schema.validate_line(contact_line(outcome="error"))
    schema.validate_line(contact_line(outcome="refused", error_type="RailsRunnerError", answer=None))


def test_score_and_tfrl_field_rules():
    for over in ({"slu_f1": "0.7"}, {"scenario_action_exact": 1}, {"scorer_id": ""}):
        with pytest.raises(schema.RolloutTraceSchemaError):
            schema.validate_line(score_line(**over))
    for over in ({"reward": None}, {"selected": "yes"}, {"trajectory_id": ""}, {"iteration": -1}):
        with pytest.raises(schema.RolloutTraceSchemaError):
            schema.validate_line(tfrl_line(**over))


def test_loads_refuses_non_json_and_dumps_refuses_invalid():
    with pytest.raises(schema.RolloutTraceSchemaError, match="not JSON"):
        schema.loads_line("{nope")
    with pytest.raises(schema.RolloutTraceSchemaError):
        schema.dumps_line({**score_line(), "slu_f1": float("inf")})


def test_truncation_rule():
    assert schema.is_truncated(finish_reason="length", forced_close=None, thinking_tokens=None, thinking_budget=None)
    assert schema.is_truncated(finish_reason="stop", forced_close=True, thinking_tokens=None, thinking_budget=None)
    assert schema.is_truncated(finish_reason="stop", forced_close=False, thinking_tokens=3112, thinking_budget=3112)
    assert not schema.is_truncated(finish_reason="stop", forced_close=False, thinking_tokens=900, thinking_budget=3112)
    assert not schema.is_truncated(finish_reason=None, forced_close=None, thinking_tokens=None, thinking_budget=3112)
    assert not schema.is_truncated(finish_reason="stop", forced_close=None, thinking_tokens=5, thinking_budget=0)


# ---------------------------------------------------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------------------------------------------------

def make_writer(tmp_path, name="rollout-trace.jsonl"):
    return trace.RolloutTraceWriter(tmp_path / "analysis" / "READ-1" / name, run_id="RUN-1", attempt_id="attempt-1",
                                    clock=lambda: TS)


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writer_appends_validated_lines_and_fsyncs(tmp_path, monkeypatch):
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    writer = make_writer(tmp_path)
    assert writer.score(row_id="r1", arm="lock", scorer_id="s", scenario_action_exact=True, slu_f1=1.0)
    assert writer.tfrl(row_id="r1", arm=None, trajectory_id="t1", reward=-0.25, selected=False, iteration=2, reason="breaks lock")
    assert len(synced) == 2  # one fsync per line
    lines = read_lines(writer.path)
    assert [line["kind"] for line in lines] == ["score", "tfrl"]
    assert all(schema.validate_line(line) for line in lines)
    assert (writer.written, writer.failures) == (2, 0)


def test_writer_failure_logs_and_never_raises(tmp_path, caplog):
    writer = make_writer(tmp_path)
    assert not writer.emit("score", row_id="r1")  # missing fields
    assert not writer.emit("trajectory")  # unknown kind
    assert writer.failures == 2 and writer.written == 0
    blocked = trace.RolloutTraceWriter(tmp_path / "file" / "x.jsonl", run_id="RUN-1", attempt_id="a")
    (tmp_path / "file").write_text("not a directory")
    with caplog.at_level("WARNING"):
        assert not blocked.score(row_id="r", arm=None, scorer_id="s", scenario_action_exact=False, slu_f1=0.0)
    assert blocked.failures == 1
    assert "rollout trace line dropped" in caplog.text


def test_writer_survives_a_closed_descriptor(tmp_path):
    writer = make_writer(tmp_path)
    assert writer.score(row_id="r1", arm=None, scorer_id="s", scenario_action_exact=True, slu_f1=1.0)
    os.close(writer._appender.fd)
    assert not writer.score(row_id="r2", arm=None, scorer_id="s", scenario_action_exact=True, slu_f1=1.0)
    assert writer.failures == 1
    writer._appender.fd = None  # reopen
    assert writer.score(row_id="r3", arm=None, scorer_id="s", scenario_action_exact=True, slu_f1=1.0)
    assert [line["row_id"] for line in read_lines(writer.path)] == ["r1", "r3"]


def test_two_writers_on_one_path_share_one_file_and_threads_do_not_interleave(tmp_path):
    first, second = make_writer(tmp_path), make_writer(tmp_path)

    def work(writer, tag):
        for index in range(40):
            assert writer.tfrl(row_id=f"{tag}{index}", arm=None, trajectory_id=f"{tag}-{index}", reward=0.1, selected=False)

    threads = [threading.Thread(target=work, args=(w, t)) for w, t in ((first, "a"), (second, "b"), (first, "c"))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    lines = read_lines(first.path)
    assert len(lines) == 120 and len({line["trajectory_id"] for line in lines}) == 120


def test_open_writer_is_off_without_the_launcher_switch(tmp_path):
    assert trace.open_writer(data_root=tmp_path, run_id="RUN-1", attempt_id="a", environ={}) is None
    writer = trace.open_writer(
        data_root=tmp_path, run_id="RUN-1", attempt_id="a", environ={trace.READ_ID_ENV: "READ-1"},
    )
    assert writer.path == tmp_path / "analysis" / "READ-1" / "rollout-trace.jsonl"
    # a READ id that is not a single safe path segment turns tracing off instead of escaping the analysis directory
    for bad in ("../x", "a/b", ".hidden"):
        assert trace.open_writer(data_root=tmp_path, run_id="RUN-1", attempt_id="a", environ={trace.READ_ID_ENV: bad}) is None
    # no data root anywhere: off
    assert trace.open_writer(data_root=None, run_id="R", attempt_id="a", environ={trace.READ_ID_ENV: "READ-1"}) is None


def test_trace_path_refuses_unsafe_read_ids(tmp_path):
    for bad in ("", "..", "a/b", "a b"):
        with pytest.raises(ValueError):
            trace.trace_path(tmp_path, bad)


def test_emit_arm_scores_is_idempotent(tmp_path):
    writer = make_writer(tmp_path)
    rows = [
        {"sample_id": "r1", "scenario_action_exact": 1.0, "slu_f1_f1": 0.5},
        {"sample_id": "r2", "scenario_action_exact": 0.0, "slu_f1_f1": 0.0},
    ]
    assert trace.emit_arm_scores(writer, arm="lock", scorer_id="s", rows=rows) == 2
    assert trace.emit_arm_scores(make_writer(tmp_path), arm="lock", scorer_id="s", rows=rows) == 0
    assert trace.emit_arm_scores(writer, arm="aa", scorer_id="s", rows=rows) == 2
    assert trace.emit_arm_scores(None, arm="aa", scorer_id="s", rows=rows) == 0
    lines = read_lines(writer.path)
    assert len(lines) == 4
    assert lines[0]["scenario_action_exact"] is True and lines[1]["scenario_action_exact"] is False
    # a malformed row is skipped without raising
    assert trace.emit_arm_scores(writer, arm="bad", scorer_id="s", rows=[{"sample_id": "x"}]) == 0


# ---------------------------------------------------------------------------------------------------------------------
# value helpers
# ---------------------------------------------------------------------------------------------------------------------

def cost(**over):
    record = {"calls": 1, "audio_seconds": 3.0, "latency_seconds": 12.5, "prompt_tokens": 4000, "completion_tokens": 3200,
              "finish_reason": "stop", "reasoning_tokens": 3112, "reasoning_chars": 9100, "forced_close": True}
    record.update(over)
    return record


def test_cost_facts_thinking_and_truncation():
    facts = trace.cost_facts([cost()], thinking_budget=3112)
    assert facts["thinking_tokens"] == 3112 and facts["thinking_tokens_basis"] == "server-usage-details"
    assert facts["truncated"] is True and facts["forced_close"] is True and facts["answer_only_reask"] is False
    assert (facts["prompt_tokens"], facts["completion_tokens"], facts["latency_seconds"]) == (4000, 3200, 12.5)


def test_cost_facts_server_without_reasoning_tokens_says_so():
    record = cost()
    del record["reasoning_tokens"]
    facts = trace.cost_facts([record], thinking_budget=3112)
    assert facts["thinking_tokens"] is None and facts["thinking_tokens_basis"] == "not-reported-by-server"
    assert facts["reasoning_chars"] == 9100
    assert facts["truncated"] is True  # forced close still marks the cap hit
    schema.validate_line(contact_line(**{k: facts[k] for k in ("thinking_tokens", "thinking_tokens_basis")}))


def test_cost_facts_reask_sums_completion_and_latency_but_keeps_first_request_facts():
    first = cost(finish_reason="length", forced_close=None, reasoning_tokens=3112, completion_tokens=4096, latency_seconds=30.0)
    second = cost(finish_reason="stop", reasoning_tokens=0, completion_tokens=200, latency_seconds=2.0, prompt_tokens=4100)
    facts = trace.cost_facts([first, second], thinking_budget=3112)
    assert facts["answer_only_reask"] is True and facts["truncated"] is True and facts["finish_reason"] == "length"
    assert facts["completion_tokens"] == 4296 and facts["latency_seconds"] == 32.0 and facts["prompt_tokens"] == 4000


def test_cost_facts_without_cost_records_uses_the_completion_finish_reason():
    facts = trace.cost_facts([], finish_reason_fallback="length")
    assert facts["finish_reason"] == "length" and facts["truncated"] is True
    assert facts["prompt_tokens"] is None and facts["latency_seconds"] is None


def test_parse_answer_as_emitted():
    text = json.dumps({"scenario": "alarm", "action": "set", "entities": [{"type": "time", "filler": "nine am"}]})
    assert trace.parse_answer(text) == {
        "parsed": True, "scenario": "alarm", "action": "set", "entities": [{"type": "time", "filler": "nine am"}],
    }
    assert trace.parse_answer("not json") == {"parsed": False, "scenario": "", "action": "", "entities": []}
    assert trace.parse_answer(None) is None


def test_supply_items_ranks_scores_and_dedupes():
    items = trace.supply_items(["a", "b", "a", "", "c"], {"a": 0.9, "c": 1}, {"a": True, "b": False})
    assert [(i["item_id"], i["rank"], i["score"], i["used"]) for i in items] == [
        ("a", 1, 0.9, True), ("b", 2, None, False), ("c", 3, 1.0, None),
    ]
    schema.validate_line(contact_line(supply_items=items))


def test_supplied_values_skip_keys_proposals_and_request_derived_ids():
    payload = {"supplied_evidence": [
        {"evidence_id": "key:facet:1", "content": "k"},
        {"evidence_id": "store::v1", "content": "v1", "knowledge_role": "value"},
        {"evidence_id": "store::k", "content": "kk", "knowledge_role": "key"},
        {"evidence_id": "candidate-proposal:abc", "content": "p"},
        {"evidence_id": "rails-0007-supply-3", "content": "positional"},
        {"evidence_id": "store::v2", "content": "v2"},
    ]}
    assert trace.supplied_values_from_payload(payload) == [("store::v1", "v1"), ("store::v2", "v2")]
    assert trace.supplied_values_from_payload({}) == []


def test_usage_flags_intent_and_slot_strings():
    exemplar = json.dumps({"observed_intent": "alarm_set", "observed_slot_examples": {"time": ["nine am"], "date": ["friday"]}})
    other = json.dumps({"observed_intent": "iot_wemo_off", "observed_slot_examples": {"device_type": ["smart plug"]}})
    prose = "Slot contracts: time is a clock time."
    answer = {"parsed": True, "scenario": "alarm", "action": "set", "entities": [{"type": "time", "filler": "at Nine, AM"}]}
    flags = trace.usage_flags({"a": exemplar, "b": other, "c": prose}, answer)
    assert flags == {"a": True, "b": False, "c": None}
    assert trace.usage_flags({"a": exemplar}, {"parsed": False, "scenario": "", "action": "", "entities": []}) == {"a": None}
    assert trace.usage_flags({"a": exemplar}, None) == {"a": None}
    only_string = {"parsed": True, "scenario": "x", "action": "y", "entities": [{"type": "date", "filler": "next friday"}]}
    assert trace.usage_flags({"a": exemplar}, only_string) == {"a": True}


def test_node_and_instance_helpers():
    assert trace.node_of("solver") == "solver" and trace.node_of("judge") == "other" and trace.node_of(None) == "other"

    class Router:
        instance_count = 4

    assert trace.instance_for_slot(Router(), 10) == 2
    assert trace.instance_for_slot(object(), 10) == 0
    assert trace.instance_for_slot(Router(), -1) is None and trace.instance_for_slot(Router(), True) is None


def test_record_contact_writes_one_valid_line_and_never_raises(tmp_path):
    writer = make_writer(tmp_path)
    answer = json.dumps({"scenario": "alarm", "action": "set", "entities": [{"type": "time", "filler": "nine am"}]})
    assert trace.record_contact(
        writer, row_id="slurp-1", arm="lock", node="solver", role_id="solver", contact_id="c-1", contact_index=1,
        sample_index=None, request_id="r-1", outcome="ok", error_type=None, id_slot=5, instance=1, supply_variant_id=None,
        supply_values=[("s::a", json.dumps({"observed_intent": "alarm_set", "observed_slot_examples": {}}))],
        supply_scores={"s::a": 0.8}, costs=[cost()], decoding={"max_tokens": 4096, "thinking_budget_tokens": 3112},
        finish_reason_fallback=None, answer_text=answer, response_text_sha256=SHA,
    )
    (line,) = read_lines(writer.path)
    assert line["supply_items"] == [{"item_id": "s::a", "rank": 1, "score": 0.8, "used": True}]
    assert line["thinking_tokens"] == 3112 and line["truncated"] is True and line["max_tokens"] == 4096
    assert line["answer"]["entities"] == [{"type": "time", "filler": "nine am"}]
    # nothing to write to: a no-op; a bad field: logged, no raise
    assert not trace.record_contact(
        None, row_id="r", arm=None, node="solver", role_id=None, contact_id="c", contact_index=0, sample_index=None,
        request_id=None, outcome="ok", error_type=None, id_slot=0, instance=0, supply_variant_id=None, supply_values=(),
        supply_scores=None, costs=[], decoding=None, finish_reason_fallback=None, answer_text=None, response_text_sha256=None,
    )
    assert not trace.record_contact(
        writer, row_id=None, arm=None, node="solver", role_id=None, contact_id="c", contact_index=0, sample_index=None,
        request_id=None, outcome="ok", error_type=None, id_slot=0, instance=0, supply_variant_id=None, supply_values=(),
        supply_scores=None, costs=[], decoding=None, finish_reason_fallback=None, answer_text=None, response_text_sha256=None,
    )
    assert writer.failures == 1
