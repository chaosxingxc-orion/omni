"""Plan T1 / M4: the rollout trace at the runner's dispatch seams (no model, no server).

Two seams are exercised: the controller-driven dispatch (``rails.runner._complete_agent_request``) with a fake client, and the native
role graph (``rails.runner._invoke_native_role_episode``, the contact scheduler's ``metered`` closure) through the same actual-episode
harness the native runner proofs use. The second is the proof that tracing moves no prompt byte, no hash and no decision.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.contracts import rollout_trace as schema
from speech_aware_evidence_acquisition.core import rollout_trace as trace
from speech_aware_evidence_acquisition.core.tracing import TraceBuffer
from speech_aware_evidence_acquisition.rails import runner
from speech_aware_evidence_acquisition.reproduction.ojw import openjiuwen_available

# The native-graph proof uses the same live-process fixture as the native runner proofs (its module-scoped autouse fixture must be
# in this module's namespace to apply).
from test_llama_server_transport import _spawn_live_matching_process  # noqa: F401

TS = "2026-10-30T10:00:00.000Z"
ANSWER = json.dumps({"scenario": "alarm", "action": "set", "entities": [{"type": "time", "filler": "nine am"}]})


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class FakeClient:
    """The slice of ``FrozenCoreClient`` the dispatch seam touches."""

    def __init__(self, buffer, writer, *, text=ANSWER, router=4):
        self.rollout_trace = writer
        self._buffer, self._text, self._router = buffer, text, router
        self.calls = []

    def rollout_instance_for_slot(self, id_slot):
        return trace.instance_for_slot(type("R", (), {"instance_count": self._router})(), id_slot)

    def complete(self, messages, **arguments):
        self.calls.append((messages, arguments))
        self._buffer.emit("COST", "rails-0001/cost", {
            "calls": 1, "audio_seconds": 2.0, "latency_seconds": 7.25, "prompt_tokens": 1500, "completion_tokens": 3300,
            "finish_reason": "stop", "reasoning_tokens": 3112, "reasoning_chars": 9000, "forced_close": True,
        })
        return self._text


def arguments(**over):
    base = {
        "decoding_params": {"max_tokens": 4096, "thinking_budget_tokens": 3112, "id_slot": 6},
        "raw_agent_sample_id": "slurp-1", "agent_contact_id": "c-1",
    }
    base.update(over)
    return base


def test_dispatch_seam_writes_one_line_and_returns_the_untraced_result(tmp_path):
    buffer = TraceBuffer("RUN-1")
    writer = trace.RolloutTraceWriter(tmp_path / "t.jsonl", run_id="RUN-1", attempt_id="attempt-1", clock=lambda: TS)
    client = FakeClient(buffer, writer)
    result = runner._complete_agent_request(core_client=client, messages=[{"role": "user"}], arguments=arguments(), trace_buffer=buffer)
    assert result == (ANSWER, None)
    (line,) = read_lines(tmp_path / "t.jsonl")
    assert schema.validate_line(line) == line
    assert (line["kind"], line["row_id"], line["contact_id"], line["node"]) == ("contact", "slurp-1", "c-1", "other")
    assert (line["id_slot"], line["instance"]) == (6, 2)  # id_slot mod 4
    assert (line["prompt_tokens"], line["completion_tokens"], line["thinking_tokens"]) == (1500, 3300, 3112)
    assert (line["thinking_budget"], line["max_tokens"], line["truncated"], line["latency_seconds"]) == (3112, 4096, True, 7.25)
    assert line["outcome"] == "ok" and line["thinking_tokens_basis"] == "server-usage-details"


class BareClient:
    """A client that predates the trace: no ``rollout_trace`` attribute at all."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, **arguments):
        self.calls += 1
        return ANSWER


def test_dispatch_seam_without_a_writer_is_the_untraced_path(tmp_path):
    buffer = TraceBuffer("RUN-1")
    assert runner._complete_agent_request(
        core_client=FakeClient(buffer, None), messages=[], arguments=arguments(), trace_buffer=buffer,
    ) == (ANSWER, None)
    bare = BareClient()
    assert runner._complete_agent_request(core_client=bare, messages=[], arguments=arguments(), trace_buffer=buffer) == (ANSWER, None)
    assert bare.calls == 1
    assert not list(tmp_path.iterdir())


def test_a_broken_writer_never_fails_the_contact(tmp_path):
    (tmp_path / "blocked").write_text("a file where the trace directory should be")
    writer = trace.RolloutTraceWriter(tmp_path / "blocked" / "t.jsonl", run_id="RUN-1", attempt_id="attempt-1")
    buffer = TraceBuffer("RUN-1")
    client = FakeClient(buffer, writer)
    assert runner._complete_agent_request(core_client=client, messages=[], arguments=arguments(), trace_buffer=buffer) == (ANSWER, None)
    assert writer.failures == 1 and writer.written == 0


def test_a_failed_contact_is_traced_as_an_error_and_still_raises(tmp_path):
    buffer = TraceBuffer("RUN-1")
    writer = trace.RolloutTraceWriter(tmp_path / "t.jsonl", run_id="RUN-1", attempt_id="attempt-1", clock=lambda: TS)

    class Exploding(FakeClient):
        def complete(self, messages, **arguments):
            raise ValueError("server said no")

    with pytest.raises(ValueError, match="server said no"):
        runner._complete_agent_request(core_client=Exploding(buffer, writer), messages=[], arguments=arguments(), trace_buffer=buffer)
    (line,) = read_lines(tmp_path / "t.jsonl")
    assert (line["outcome"], line["error_type"], line["request_id"]) == ("error", "ValueError", None)
    assert line["prompt_tokens"] is None and line["truncated"] is False


# ---------------------------------------------------------------------------------------------------------------------
# the native role graph
# ---------------------------------------------------------------------------------------------------------------------

def _run_native(tmp_path, monkeypatch, *, traced):
    from test_native_episode_runner import _ActualEpisodeHarness

    if traced:
        monkeypatch.setenv(trace.READ_ID_ENV, "READ-T1")
    else:
        monkeypatch.delenv(trace.READ_ID_ENV, raising=False)
    harness = _ActualEpisodeHarness(
        tmp_path, sample_count=1, max_revisions=0, max_contacts=4, planner_actions=("ACCEPT",),
        arm="speech-feature-query-native-tools",
    )
    try:
        summary = harness.run()
    finally:
        close_summary = harness.close()
    return harness, summary, close_summary


@pytest.mark.skipif(not openjiuwen_available(), reason="OpenJiuwen unavailable")
def test_native_graph_tracing_moves_no_prompt_byte_hash_or_decision(tmp_path, monkeypatch):
    off, off_summary, off_close = _run_native(tmp_path / "off", monkeypatch, traced=False)
    on, on_summary, on_close = _run_native(tmp_path / "on", monkeypatch, traced=True)

    # prompt bytes: every rendered request body is identical, key by key
    assert set(off.prepared) == set(on.prepared)
    for key in off.prepared:
        assert json.dumps(off.prepared[key]["rendered"], sort_keys=True) == json.dumps(on.prepared[key]["rendered"], sort_keys=True)
        assert off.prepared[key]["slot_id"] == on.prepared[key]["slot_id"]
    # composed hashes and decisions: request/response hashes of every contact, the receipts and the delivered answers
    for off_result, on_result in zip(off_summary.results, on_summary.results, strict=True):
        assert off_result.final_response_text == on_result.final_response_text
        assert off_result.native_episode_receipt == on_result.native_episode_receipt
        assert [c.as_json() for c in off_result.contacts] == [c.as_json() for c in on_result.contacts]
    for key in off.prepared:
        contact = key  # (sample, phase, visit, turn)
        assert off.dispatched[contact] == on.dispatched[contact]
    assert off_close["request_count"] == on_close["request_count"] == 4
    for field in ("calls_used", "prompt_tokens_total", "completion_tokens_total", "audio_seconds_used", "tool_calls"):
        assert off_close["cost"][field] == on_close["cost"][field]
    assert off_close["tool_schemas_sha256"] == on_close["tool_schemas_sha256"]

    # the traced run wrote exactly one valid contact line per dispatched contact (one row, so request ids are deterministic and
    # the receipts compare byte for byte); the untraced run wrote none
    assert not (off.world.data_root / "analysis").exists()
    lines = read_lines(on.world.data_root / "analysis" / "READ-T1" / "rollout-trace.jsonl")
    assert len(lines) == 4 and all(schema.validate_line(line) == line for line in lines)
    assert sorted(line["node"] for line in lines) == ["perception", "planner", "reviewer", "solver"]
    assert {line["row_id"] for line in lines} == {"s1"}
    for row in ("s1",):
        mine = sorted((line for line in lines if line["row_id"] == row), key=lambda line: line["contact_index"])
        assert [line["contact_index"] for line in mine] == [0, 1, 2, 3]
        assert [line["node"] for line in mine] == ["perception", "solver", "reviewer", "planner"]
        assert len({line["id_slot"] for line in mine}) == 1  # one slot per row
        assert all(line["instance"] == 0 and line["outcome"] == "ok" for line in mine)
        assert all((line["prompt_tokens"], line["completion_tokens"], line["finish_reason"]) == (7, 3, "stop") for line in mine)
        solver = mine[1]
        assert solver["answer"] == {"parsed": False, "scenario": "", "action": "", "entities": []}  # fake text is not an SLU answer
        assert all(line["answer"] is None for line in mine if line["node"] != "solver")
        assert all(line["thinking_tokens"] is None and line["thinking_tokens_basis"] == "not-reported-by-server" for line in mine)
    request_ids = {line["request_id"] for line in lines}
    assert len(request_ids) == 4 and None not in request_ids


# ---------------------------------------------------------------------------------------------------------------------
# the watchdog summariser reads what the writer writes
# ---------------------------------------------------------------------------------------------------------------------

WATCHDOG = Path("/mnt/cephfs/xingchao/omni/work/watchdog/watchdog.py")


@pytest.mark.skipif(not WATCHDOG.is_file(), reason="host watchdog script not present")
def test_watchdog_summariser_accepts_the_real_writer_output(tmp_path):
    spec = importlib.util.spec_from_file_location("saea_watchdog_for_trace_test", WATCHDOG)
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    assert watchdog.ROLLOUT_SCHEMA == schema.SCHEMA

    path = tmp_path / "t.jsonl"
    writer = trace.RolloutTraceWriter(path, run_id="RUN-1", attempt_id="attempt-1", clock=lambda: TS)
    buffer = TraceBuffer("RUN-1")
    client = FakeClient(buffer, writer)
    runner._complete_agent_request(core_client=client, messages=[], arguments=arguments(), trace_buffer=buffer)
    assert writer.score(row_id="slurp-1", arm="lock", scorer_id="s", scenario_action_exact=True, slu_f1=0.5)
    assert writer.tfrl(row_id="slurp-1", arm=None, trajectory_id="t1", reward=0.5, selected=True)
    agg = watchdog._rt_agg()
    for line in path.read_text().splitlines():
        watchdog.rt_apply(agg, json.loads(line))
    metrics = watchdog.rt_metrics(agg)
    assert metrics["contacts"] == 1 and metrics["nodes"]["other"]["trunc_pct"] == 100.0
    assert metrics["scored"]["lock"]["rows"] == 1 and metrics["tfrl"]["accepted"] == 1
    assert agg["foreign"] == 0
