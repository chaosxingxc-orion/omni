"""Real episode -> checkpoint -> FlightRuntime seal -> real bundle verifier.

The transport is scripted and never calls a model. The producer uses the real
native runner and canonical renderer; the successor uses actual checkpoint
recovery and seal. Scoring is a stub because this tests custody, not metrics.
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("openjiuwen")

from speech_aware_evidence_acquisition.core.bundle import build_manifest, verify_bundle
from speech_aware_evidence_acquisition.core.flight.resume import resume_declarations_from_document
from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter, read_checkpoint
from test_flight_resume_recovery import _harness, _values
from test_flight_runtime import _make_runtime, _FakeAdapter, _FakeSample
from test_llama_server_transport import _spawn_live_matching_process  # noqa: F401
import test_native_episode_runner as native_fixture
from speech_aware_evidence_acquisition.core.flight import runtime as flight_module
from types import SimpleNamespace

RUN = "SAEA-TEST-DERIVED-CHECKPOINT"


class FourSamples(_FakeAdapter):
    def __init__(self, lock, root):
        super().__init__(lock, root)
        self.samples = tuple(_FakeSample(s) for s in ("s1", "s2", "s3", "s4"))


def _produce(tmp_path, monkeypatch, sample, kind, plan):
    # Each source is an independent real client: their flight counters collide.
    harness = _harness(tmp_path, "producer-" + sample, (sample,))
    harness.config = replace(harness.config, finite_completion_policy="deliver-latest-or-abstain")
    renderer = object.__new__(flight_module.FlightRuntime)
    renderer._composed = SimpleNamespace(values={
        "native_episode": {
            "finite_completion_policy": harness.config.finite_completion_policy,
            "exhaustion_policy": "raise",
            "max_revisions": 0,
            "actions": dict(zip(("accept", "revise", "acquire"), harness.config.actions)),
        },
    })
    renderer._plan = plan
    renderer._result_kind = lambda: "answer"
    renderer._arm = lambda: harness.client.arm
    attempt = "history-" + sample
    writer = RowCheckpointWriter(tmp_path / "data" / "runs" / RUN / attempt, RUN, attempt, identity={
        "protocol_hash": plan.protocol_hash, "split_identity_hash": plan.split_identity_hash,
        "implementation": dict(flight_module.FlightRuntime.row_checkpoint_implementation()),
    })
    renderer._row_checkpoint_writer = writer
    harness.trace.row_checkpoint = lambda **kwargs: renderer._checkpoint_row(
        **kwargs, client=harness.client, program=harness.program, policy=harness.program.program_id)
    original_wire = native_fixture._wire_response
    original_output = harness._output
    if kind == "finite":
        def finite_wire(text):
            document = json.loads(original_wire(text))
            if text.startswith("perception-"):
                document["choices"][0]["finish_reason"] = "length"
            return json.dumps(document, separators=(",", ":")).encode()
        monkeypatch.setattr(native_fixture, "_wire_response", finite_wire)
    elif kind == "seal":
        harness._output = lambda key: "" if key[1] == "solver" else original_output(key)
    try:
        summary = harness.run()
        trace = list(harness.trace.records)
        costs = harness.close()["cost"]
    finally:
        writer.close()
        monkeypatch.setattr(native_fixture, "_wire_response", original_wire)
    checkpoint = read_checkpoint(writer.path)
    assert checkpoint.complete_rows == 1 and checkpoint.rows[0].keepable, checkpoint.rows[0].reason
    row, = checkpoint.rows[0].output_rows
    if kind in {"finite", "seal"}:
        assert row["text"] == ""
        assert row["native_episode_receipt"]["terminal_disposition"]["reason"] == (
            "finite_completion" if kind == "finite" else "seal_refusal")
        assert any(ch == "TOOL" and rid == row["request_id"] + "/request" for ch, rid, _ in trace)
    assert costs["calls_used"] > 0
    return row, trace, costs, checkpoint


@pytest.mark.parametrize("all_kept", [True, False], ids=["seal-only", "current-row"])
def test_native_checkpoint_seal_with_real_bundle_verifier(tmp_path, monkeypatch, all_kept):
    # _make_runtime fakes infrastructure only; manifest and verifier here are
    # production functions. No substitute may stand in for verify_bundle.
    source = _harness(tmp_path, "configuration", ("s1",))
    source.client.close()
    monkeypatch.setattr(flight_module, "default_phase_program", lambda _policy: source.program)
    native = _values(source)["native_episode"]
    native["finite_completion_policy"] = "deliver-latest-or-abstain"
    runtime = _make_runtime(tmp_path, run_id=RUN, happy_path=True, adapter_cls=FourSamples,
        config_kwargs={"arm": "speech-feature-query", "carrier_lock_key": "fake-carrier",
                       "policy": source.program.program_id,
                       "baseline_extra": {"native_episode": native, "scorers": ["asr-wer"]}},
        manifest_builder=build_manifest, bundle_verifier=verify_bundle)
    # Prompt estimation is unrelated to this model-free custody test.
    runtime._assess_prompt_budget = lambda _samples: ()
    # Preflight computes exactly the identity which is pinned in each checkpoint.
    values, carrier, _, identity = runtime._prepare()
    runtime._plan = runtime._build_execution_plan(values, carrier, identity)
    normal = _produce(tmp_path, monkeypatch, "s1", "normal", runtime.plan)
    finite = _produce(tmp_path, monkeypatch, "s2", "finite", runtime.plan)
    refusal = _produce(tmp_path, monkeypatch, "s3", "seal", runtime.plan)
    current = _produce(tmp_path, monkeypatch, "s4", "normal", runtime.plan)
    assert set(r for c, r, p in normal[1] if c == "MODEL_REQUEST") & set(
        r for c, r, p in finite[1] if c == "MODEL_REQUEST")
    documents = [item[3] for item in ((normal, finite, refusal, current) if all_kept else (normal, finite, refusal))]
    declarations = [{"attempt_id": doc.header["attempt_id"], "checkpoint_sha256": doc.sha256} for doc in documents]
    runtime._resume_declarations = resume_declarations_from_document(declarations)
    preflight = runtime.preflight()
    assert preflight["resume"]["counts"]["kept"] == (4 if all_kept else 3)
    assert runtime.is_seal_only is all_kept
    assert [x for x in runtime._resume_flown_sample_ids] == ([] if all_kept else ["s4"])

    if all_kept:
        # These production stages must not construct any model or FIT client.
        runtime._client_factory = lambda **kw: pytest.fail("unexpected model client")
        runtime._session_starter = lambda **kw: pytest.fail("unexpected session")
        assert runtime.preprocess()["seal_only"]
        assert runtime.serve()["seal_only"]
        assert runtime.run_phases()["seal_only"]
    else:
        # This is the original mixed seal shape. The scripted current contact
        # was produced above; import its exact trace into this attempt and
        # let FlightRuntime merge it with both immutable predecessors.
        from speech_aware_evidence_acquisition.core.tracing import TraceSink
        from speech_aware_evidence_acquisition.core.attempts import AttemptStore
        runtime.receipts.record("preprocess", {})
        runtime.receipts.record("serve", {})
        runtime.receipts.record("run_phases", {})
        runtime._output_rows = [current[0]]
        sink = TraceSink(runtime._run_dir(), RUN)
        for channel, record_id, payload in current[1]:
            sink.emit(channel, record_id, payload)
        class CapturedClient:
            timings = ()
            def close(self):
                return {"arm": "speech-feature-query", "trace_manifest_hash": sink.close(), "cost": current[2]}
        runtime._client = CapturedClient()
        AttemptStore(runtime._data_root).open_attempt(RUN, runtime._attempt_id,
            carrier_lock_key=carrier, reserved_calls=runtime.plan.planned_model_calls,
            reserved_audio_seconds=runtime.plan.planned_speech_audio_seconds)
    result = runtime.seal()
    bundle = verify_bundle(Path(result["manifest_path"]))
    rows = {row["sample_id"]: row for row in bundle.outputs_rows}
    assert rows == {item[0]["sample_id"]: item[0] for item in (normal, finite, refusal, current)}
    blocks = bundle.manifest["predecessor_attempts"]
    assert {block["attempt_id"] for block in blocks} == {d["attempt_id"] for d in declarations}
    assert sum(row.contacts for doc in documents for row in doc.rows) == sum(
        item[2]["calls_used"] for item in ((normal, finite, refusal, current) if all_kept else (normal, finite, refusal)))
    if all_kept:
        assert result["cost"]["calls_used"] == 0 and not bundle.trace_records
        generation = json.loads((Path(result["manifest_path"]).parent / (RUN + ".generation-manifest.json")).read_text())
        assert generation["cost"]["calls_used"] == sum(item[2]["calls_used"] for item in (normal, finite, refusal, current))
    else:
        assert result["cost"]["calls_used"] == current[2]["calls_used"]
        assert bundle.trace_records[current[0]["request_id"] + "/request"]["payload"]["sample_id"] == "s4"
