"""Owner decision 159: the dispatcher slot-fill key changes scheduling, never bytes.

Four layers:

* the key itself -- absent means None, a declared block is parsed strictly;
* the one computation the key moves (``recall_prefetch``): a prefetched
  recall yields the SAME ``PhaseSupply`` -- rank result, fusion decision,
  every hash -- as the live recall, on vector and on segmented queries, and a
  revisit after the prefetch was consumed recomputes the same answer;
* the ACTUAL native role episode through the ACTUAL ``FlightRuntime`` dispatch
  (``test_native_episode_runner._ActualEpisodeHarness``: the real OpenJiuwen
  graph, the real request composer, a scripted transport): with every switch
  on, every rendered request, every sealed row and every trace record equals
  the switch-off flight's, modulo the two arrival facts the dispatcher already
  assigns by timing (request-id counter, slot number);
* a crash and a resume-recovery with the key on: the kept row is byte
  identical to the uninterrupted switch-off seal and the re-flown row
  delivers the same answer (E10 / I7r semantics unchanged).

No model, server, tokenizer, encoder or network contact anywhere.
"""

from __future__ import annotations

import gc
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from speech_aware_evidence_acquisition.rails import phase_supply as ps
from speech_aware_evidence_acquisition.rails.dispatch_slot_fill import (
    DISPATCH_SLOT_FILL_SCHEMA,
    DispatchSlotFill,
    DispatchSlotFillError,
    parse_dispatch_slot_fill,
)
from speech_aware_evidence_acquisition.reproduction.ojw import openjiuwen_available

from test_phase_supply import (
    _context,
    _fragment,
    _program,
    _segment_source,
    _snapshot,
    _source,
    _write_segment_vectors,
    _write_vectors,
    _SHA_A,
)
# The module-scoped fixture the episode harness's transport identity needs (a
# copy of the interpreter holding an ephemeral socket, llama-server shaped);
# no request ever leaves this process.
from test_llama_server_transport import _spawn_live_matching_process  # noqa: F401

ALL_ON = {
    "schema": DISPATCH_SLOT_FILL_SCHEMA,
    "recall_prefetch": True,
    "role_input_offload": True,
    "gc_freeze": True,
}


@pytest.mark.parametrize("kind", ["prepare", "validation"])
def test_bounded_worker_repeated_cancel_keeps_capacity_until_thread_finishes(kind):
    import asyncio
    import threading
    from speech_aware_evidence_acquisition.rails.runner import _SlotFillScheduler

    entered, release, second = (threading.Event() for _ in range(3))
    scheduler = _SlotFillScheduler(DispatchSlotFill(
        role_input_offload=True, prepare_workers=1, validation_workers=1
    ))

    def first(*args):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test worker was not released")

    def next_work(*args):
        second.set()

    async def check():
        run = scheduler.role_input_runner if kind == "prepare" else scheduler.validation_runner
        args = ("phase", object()) if kind == "prepare" else (object(),)
        first_task = asyncio.create_task(run(first, *args))
        while not entered.is_set():
            await asyncio.sleep(.001)
        first_task.cancel()
        await asyncio.sleep(.01)
        first_task.cancel()
        second_task = asyncio.create_task(run(next_work, *args))
        await asyncio.sleep(.03)
        assert not first_task.done() and not second.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first_task
        await second_task
        assert second.is_set()

    try:
        asyncio.run(check())
    finally:
        release.set()
        scheduler.close()


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_an_absent_key_is_none_so_every_committed_config_is_untouched():
    assert parse_dispatch_slot_fill({}) is None
    assert parse_dispatch_slot_fill({"policy": "x"}) is None


def test_a_declared_block_parses_to_its_switches():
    parsed = parse_dispatch_slot_fill({"dispatch": ALL_ON})
    assert parsed == DispatchSlotFill(True, True, True)
    assert parsed.as_json() == ALL_ON
    only = parse_dispatch_slot_fill(
        {"dispatch": {"schema": DISPATCH_SLOT_FILL_SCHEMA, "gc_freeze": True}}
    )
    assert only == DispatchSlotFill(gc_freeze=True)


@pytest.mark.parametrize(
    ("block", "message"),
    (
        ("on", "must be a mapping"),
        ({**ALL_ON, "rows_in_flight": 8}, "unknown key"),
        ({**ALL_ON, "schema": "saea-dispatch-slot-fill-v0"}, "schema"),
        ({**ALL_ON, "gc_freeze": 1}, "boolean"),
        ({"schema": DISPATCH_SLOT_FILL_SCHEMA}, "no switch"),
        (
            {"schema": DISPATCH_SLOT_FILL_SCHEMA, "recall_prefetch": False},
            "no switch",
        ),
    ),
)
def test_a_malformed_block_is_refused_before_any_contact(block, message):
    with pytest.raises(DispatchSlotFillError, match=message):
        parse_dispatch_slot_fill({"dispatch": block})


def test_no_committed_configuration_declares_the_key():
    """Default = current behaviour: not one committed fragment arms it, so
    every committed config composes to the bytes it composed to before."""

    root = Path(__file__).resolve().parents[2] / "configs"
    offenders = []
    for path in sorted(root.rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        if DISPATCH_SLOT_FILL_SCHEMA in text and "slotfill" not in path.name:
            offenders.append(path.name)
        try:
            document = json.loads(text)
        except ValueError:
            continue
        if isinstance(document, dict) and "dispatch" in document and "slotfill" not in path.name:
            offenders.append(path.name)
    assert offenders == []


# ---------------------------------------------------------------------------
# recall_prefetch: the moved computation returns the same bytes
# ---------------------------------------------------------------------------


def _binding(tmp_path, *, segmented=False):
    snapshot = _snapshot()
    if segmented:
        query_digest = _write_segment_vectors(
            tmp_path / "v" / "q.jsonl",
            [{"id": "dev:s1", "audio_sha256": _SHA_A, "source_frames": 4,
              "source_sample_rate_hz": 16000,
              "segments": [
                  {"id": "early", "start_frame": 0, "end_frame": 2,
                   "vectors": {"content": [0.0, 1.0]}},
                  {"id": "late", "start_frame": 2, "end_frame": 4,
                   "vectors": {"content": [1.0, 0.2]}},
              ]}],
        )
        query = _segment_source("v/q.jsonl", query_digest)
    else:
        query_digest = _write_vectors(
            tmp_path / "v" / "q.jsonl", [("dev:s1", {"content": [0.8, 0.6]})]
        )
        query = _source("v/q.jsonl", query_digest)
    facet_digest = _write_vectors(
        tmp_path / "v" / "f.jsonl",
        [
            ("facet:content:u1", {"content": [1.0, 0.0]}),
            ("facet:content:u2", {"content": [0.0, 1.0]}),
        ],
    )
    specs = ps.load_phase_supply_specs(
        _fragment(query, _source("v/f.jsonl", facet_digest)), _program()
    )
    supply_for_phase = ps.build_supply_for_phase(
        specs, snapshot=snapshot, snapshot_sha256=snapshot.content_hash(),
        data_root=tmp_path,
    )
    return snapshot, supply_for_phase


def _supply_bytes(supply) -> str:
    return json.dumps(
        {
            "rank_result": supply.rank_result.as_json(),
            "fusion_decision": supply.fusion_decision.as_json(),
            "expose_facets": supply.expose_facets,
            "retain_prior": supply.retain_prior,
            "action_target_ref": supply.action_target_ref,
        },
        sort_keys=True,
    )


@pytest.mark.parametrize("segmented", (False, True), ids=("vectors", "segments"))
def test_a_prefetched_recall_yields_the_live_supply_byte_for_byte(tmp_path, segmented):
    snapshot, live = _binding(tmp_path / "live", segmented=segmented)
    query_source = "speech_audio" if segmented else "speech_features"
    expected = _supply_bytes(live("evidence", _context(snapshot, query_source=query_source)))

    snapshot, prefetched = _binding(tmp_path / "prefetched", segmented=segmented)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert prefetched.prefetch_recall("dev:s1", pool.submit) == 1
        # a second prefetch of the same row is a no-op, never a second job
        assert prefetched.prefetch_recall("dev:s1", pool.submit) == 0
        first = _supply_bytes(prefetched("evidence", _context(snapshot, query_source=query_source)))
        # the prefetch was consumed; a revisit recomputes live, identically
        again = _supply_bytes(prefetched("evidence", _context(snapshot, query_source=query_source)))
    assert first == expected
    assert again == expected


def test_prefetch_skips_rows_without_a_query_and_phases_that_are_seeded(tmp_path):
    snapshot, supply_for_phase = _binding(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert supply_for_phase.prefetch_recall("dev:absent", pool.submit) == 0
    specs = ps.load_phase_supply_specs(
        _fragment(
            _source("v/q.jsonl", _write_vectors(tmp_path / "v" / "q.jsonl", [("dev:s1", {"content": [1.0, 0.0]})])),
            _source("v/f.jsonl", _write_vectors(tmp_path / "v" / "f.jsonl", [("facet:content:u1", {"content": [1.0, 0.0]})])),
        ),
        _program(),
    )
    seeded = ps.build_supply_for_phase(
        specs, snapshot=snapshot, snapshot_sha256=snapshot.content_hash(),
        data_root=tmp_path, seed_phases=("evidence",),
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert seeded.prefetch_recall("dev:s1", pool.submit) == 0


# ---------------------------------------------------------------------------
# The actual episode through the actual dispatch
# ---------------------------------------------------------------------------

pytestmark_episode = pytest.mark.skipif(
    not openjiuwen_available(), reason="OpenJiuwen unavailable"
)

_ARRIVAL = re.compile(r'"(id_slot|slot_id)": ?\d+')


def _arrival_free(obj) -> str:
    """Mask the two facts the dispatcher assigns by timing (the slot number
    a row drew and the request-id counter), nothing else."""

    text = json.dumps(obj, sort_keys=True)
    text = _ARRIVAL.sub(r'"\1": "ARRIVAL"', text)
    text = re.sub(r'"slot_ids": \[[0-9, ]*\]', '"slot_ids": "ARRIVAL"', text)
    return text


def _fly_with(tmp_path, monkeypatch, name, sample_ids, dispatch, *, max_revisions=1, harness_hook=None):
    from test_flight_resume_recovery import _harness, _skeleton

    harness = _harness(tmp_path, name, sample_ids, max_revisions=max_revisions)
    if dispatch and dispatch.get("ready_request"):
        # The harness post is a deterministic stub; retain its rendering and
        # fake session while exercising the vLLM-only routing contract.
        from speech_aware_evidence_acquisition.core.model.vllm_transport import VllmServerTransport
        transport = harness.client._rollout_transport
        transport.__class__ = VllmServerTransport
        transport.served_model_name = "stub-model"
    if harness_hook is not None:
        harness_hook(harness)
    runtime = _skeleton(harness, monkeypatch)
    if dispatch is not None:
        runtime._composed = SimpleNamespace(
            values={**runtime._composed.values, "dispatch": dispatch}
        )
    try:
        rows, _contacts, _kind, receipt, _phases = runtime._dispatch_phase_program(
            harness.client, harness.plan, harness.program.program_id
        )
    finally:
        harness.close()
    runtime._output_rows = rows
    canonical = runtime._canonical_output_rows(harness.client)
    trace_path = harness.world.run_dir / f"{harness.plan.run_id}.trace.jsonl"
    return harness, runtime, canonical, receipt, trace_path


def _rendered(harness):
    bodies = {}
    for key, record in harness.prepared.items():
        body = dict(record["rendered"])
        body.pop("id_slot", None)
        bodies[json.dumps(key)] = json.dumps(body, sort_keys=True)
    return bodies


@pytestmark_episode
@pytest.mark.parametrize("max_revisions", (0, 1))
def test_every_switch_on_sends_the_same_requests_and_seals_the_same_rows(
    tmp_path, monkeypatch, max_revisions
):
    rows = ("s1", "s2")
    off, _rt, off_rows, off_receipt, _ = _fly_with(
        tmp_path, monkeypatch, "off", rows, None, max_revisions=max_revisions
    )
    on, _rt, on_rows, on_receipt, _ = _fly_with(
        tmp_path, monkeypatch, "on", rows, ALL_ON, max_revisions=max_revisions
    )
    assert _rendered(on) == _rendered(off)
    assert len(_rendered(off)) == len(rows) * (4 + 3 * max_revisions)
    assert [_arrival_free(row) for row in on_rows] == [_arrival_free(row) for row in off_rows]
    # the receipt says the switches were armed; the unarmed receipt does not
    # carry the key at all
    assert on_receipt["slot_fill"] == ALL_ON
    assert "slot_fill" not in off_receipt
    # the collector is handed back unfrozen
    assert gc.get_freeze_count() == 0


@pytestmark_episode
def test_the_trace_records_are_the_same_records_modulo_arrival(tmp_path, monkeypatch):
    rows = ("s1", "s2")
    *_, off_trace = _fly_with(tmp_path, monkeypatch, "off", rows, None)
    *_, on_trace = _fly_with(tmp_path, monkeypatch, "on", rows, ALL_ON)

    def normalised(path):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        # request ids are numbered by dispatch order; rename each to its own
        # (row, contact) identity before comparing
        names = {}
        for record in lines:
            if record["channel"] == "MODEL_REQUEST":
                head = record["record_id"].split("/")[0]
                payload = record["payload"]["payload"]
                names[head] = f"{payload['sample_id']}|{payload['history'][-1].get('contact_id')}"
        out = []
        for record in lines:
            record["payload"].pop("latency_seconds", None)
            timing = record["payload"].get("use_sample_execution_timing")
            if timing is not None:
                # Keep the sample, status, scope and admission identity. Only
                # measured monotonic boundaries differ between real executions.
                for boundary in ("batch_started_ns", "admitted_ns", "finished_ns"):
                    assert type(timing[boundary]) is int
                    timing.pop(boundary)
            text = _arrival_free(record)
            text = re.sub(r"flight-\d+", lambda m: names.get(m.group(0), "?"), text)
            out.append(text)
        return sorted(out)

    off_lines = normalised(off_trace)
    on_lines = normalised(on_trace)
    assert len(on_lines) == len(off_lines) > 0
    differing = [a for a, b in zip(off_lines, on_lines) if a != b]
    # the only digests left to differ are the ones computed OVER the arrival
    # fields themselves (a request's own sha256, a row's runtime-trace
    # sha256); strip exactly those and nothing else remains
    digest_keys = re.compile(
        r'"(payload_sha256|use_request_sha256|request_sha256|use_runtime_trace_sha256)": "[0-9a-f]{64}"'
    )
    assert [digest_keys.sub(r'"\1": "D"', a) for a in off_lines] == [
        digest_keys.sub(r'"\1": "D"', b) for b in on_lines
    ], f"{len(differing)} record(s) differ beyond arrival facts"


# ---------------------------------------------------------------------------
# Crash, then resume-recovery, with the key on
# ---------------------------------------------------------------------------


@pytestmark_episode
@pytest.mark.parametrize("started", (True, False), ids=("partial", "unstarted"))
def test_a_crashed_attempt_resumes_with_identical_semantics_under_the_key(
    tmp_path, monkeypatch, started
):
    from speech_aware_evidence_acquisition.core.flight.resume import (
        RESUME_PREDECESSOR_SCHEMA,
        MergedPredecessors,
    )
    from test_flight_resume_recovery import (
        _harness,
        _scan,
        _skeleton,
        _trace_lines,
        _write_predecessor,
    )

    # The uninterrupted switch-OFF flight is the reference seal.
    _h, _rt, reference, _receipt, _trace = _fly_with(
        tmp_path, monkeypatch, "reference", ("s1", "s2"), None, max_revisions=0
    )
    reference_rows = {row["sample_id"]: row for row in reference}

    # The attempt that "crashed" flew with the key ON; its trace is cut to
    # the rows it had reached.
    _h, _rt, _rows, _receipt, crashed_trace = _fly_with(
        tmp_path, monkeypatch, "crashed", ("s1", "s2"), ALL_ON, max_revisions=0
    )
    predecessor = _write_predecessor(
        _trace_lines(crashed_trace), tmp_path / "aborted", _h.plan.run_id,
        keep={"s1"}, truncate={"s2"} if started else frozenset(),
    )
    scan = _scan(predecessor, _h)
    assert scan.complete_sample_ids == ("s1",)

    # The recovery also flies with the key ON, over exactly the rows the
    # predecessor did not complete.
    recovery = _harness(tmp_path, "recovery", ("s2",))
    runtime = _skeleton(recovery, monkeypatch)
    runtime._composed = SimpleNamespace(values={**runtime._composed.values, "dispatch": ALL_ON})
    runtime._samples = tuple(SimpleNamespace(sample_id=s) for s in ("s1", "s2"))
    runtime._resume = MergedPredecessors((scan,))
    runtime._resume_flown_sample_ids = ("s2",)
    try:
        rows, *_ = runtime._dispatch_phase_program(
            recovery.client, recovery.plan, recovery.program.program_id
        )
    finally:
        recovery.close()
    runtime._output_rows = rows
    runtime._client = recovery.client
    merged, block, _seal_only_results = runtime._merged_canonical_rows()

    assert [row["sample_id"] for row in merged] == ["s1", "s2"]
    # The kept row, replayed from a key-ON trace, equals the key-OFF seal
    # modulo only the slot number the crashed attempt's row drew.
    assert _arrival_free(merged[0]) == _arrival_free(reference_rows["s1"])
    assert merged[1]["text"] == reference_rows["s2"]["text"]
    assert merged[1]["response_text_sha256"] == reference_rows["s2"]["response_text_sha256"]
    assert block["schema"] == RESUME_PREDECESSOR_SCHEMA
    assert block["kept_sample_ids"] == ["s1"]
    assert gc.get_freeze_count() == 0

@pytestmark_episode
def test_configured_prepare_and_validation_workers_preserve_real_native_outputs(tmp_path, monkeypatch):
    config={"schema":DISPATCH_SLOT_FILL_SCHEMA,"role_input_offload":True,
            "recall_prefetch":True,"prepare_workers":2,"validation_workers":2,
            "model_inflight":1,"admitted_trajectories":4,"ready_request":True}
    off,_,off_rows,_,_=_fly_with(tmp_path,monkeypatch,"thread-off",("s1","s2"),None,max_revisions=0)
    on,_,on_rows,receipt,_=_fly_with(tmp_path,monkeypatch,"thread-on",("s1","s2"),config,max_revisions=0)
    assert _rendered(on)==_rendered(off)
    assert [(x['sample_id'],x['text']) for x in on_rows]==[(x['sample_id'],x['text']) for x in off_rows]
    assert receipt['slot_fill']['validation_workers']==2


def test_new_dispatch_limits_are_strict():
    c=parse_dispatch_slot_fill({'dispatch':{'schema':DISPATCH_SLOT_FILL_SCHEMA,
        'role_input_offload':True,'prepare_workers':4,'validation_workers':4,
        'model_inflight':64,'admitted_trajectories':96}})
    assert c.model_inflight==64 and c.admitted_trajectories==96
    for bad in ({'prepare_workers':2},{'validation_workers':0},{'model_inflight':True},{'ready_request':True,'prepare_workers':2,'model_inflight':1}):
        with pytest.raises(DispatchSlotFillError):parse_dispatch_slot_fill({'dispatch':{'schema':DISPATCH_SLOT_FILL_SCHEMA,**bad}})


def test_validation_pool_drains_cancelled_work_and_closes():
    import asyncio,threading
    from speech_aware_evidence_acquisition.rails.runner import _SlotFillScheduler
    started=threading.Event();release=threading.Event();done=threading.Event()
    scheduler=_SlotFillScheduler.open(DispatchSlotFill(validation_workers=1))
    async def run():
        def work(_):started.set();release.wait(3);done.set();return 42
        task=asyncio.create_task(scheduler.validation_runner(work,{}))
        assert await asyncio.to_thread(started.wait,2)
        task.cancel();await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):await task
    try:asyncio.run(run())
    finally:release.set();scheduler.close()
    assert done.is_set()


@pytestmark_episode
def test_ready_request_fit_wait_does_not_hold_send_permit(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    blocked = threading.Event()
    release = threading.Event()
    other_sent = threading.Event()
    order = []

    def install(harness):
        original_gate = harness.request_gate
        original_post = harness.transport._post

        def gate(prepared, **kwargs):
            payload = prepared.as_payload()
            if payload["sample_id"] == "s1" and not blocked.is_set():
                blocked.set()
                if not release.wait(10):
                    raise TimeoutError("blocked FIT gate not released")
            return original_gate(prepared, **kwargs)

        def post(url, body):
            if blocked.is_set() and not release.is_set():
                order.append("send-during-fit")
                other_sent.set()
            return original_post(url, body)

        harness.request_gate = gate
        harness.transport._post = post

    config = {"schema": DISPATCH_SLOT_FILL_SCHEMA, "role_input_offload": True,
              "prepare_workers": 2, "validation_workers": 2, "ready_request": True,
              "model_inflight": 1, "admitted_trajectories": 2}
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_fly_with, tmp_path, monkeypatch, "ready-overlap",
                                 ("s1", "s2"), config, max_revisions=0, harness_hook=install)
        assert blocked.wait(5)
        assert other_sent.wait(5), "another sample must send while the first is blocked in FIT"
        release.set()
        future.result(timeout=15)
        assert order
    finally:
        release.set()
        executor.shutdown(wait=True)


@pytestmark_episode
def test_ready_fit_failure_discards_real_client_token_and_records_unsent(tmp_path, monkeypatch):
    from test_flight_resume_recovery import _harness, _skeleton
    from speech_aware_evidence_acquisition.core.model.vllm_transport import VllmServerTransport
    harness = _harness(tmp_path, "ready-fit-failure", ("s1",), max_revisions=0)
    transport = harness.client._rollout_transport
    transport.__class__ = VllmServerTransport
    transport.served_model_name = "stub-model"
    runtime = _skeleton(harness, monkeypatch)
    config = {"schema": DISPATCH_SLOT_FILL_SCHEMA, "role_input_offload": True,
              "prepare_workers": 1, "ready_request": True, "model_inflight": 1,
              "admitted_trajectories": 1}
    runtime._composed = SimpleNamespace(values={**runtime._composed.values, "dispatch": config})
    gate_calls = []
    def fail_gate(prepared, **kwargs):
        gate_calls.append(prepared.as_payload()["sample_id"])
        raise RuntimeError("intentional fit failure")
    harness.request_gate = fail_gate
    try:
        with pytest.raises(Exception, match="intentional fit failure"):
            runtime._dispatch_phase_program(harness.client, harness.plan, harness.program.program_id)
        assert len(gate_calls) == 1
        assert not harness.client._prepared_requests
        assert not harness.client._request_counter
        trace_path = harness.world.run_dir / f"{harness.plan.run_id}.trace.jsonl"
        assert 'ready-disposition' in trace_path.read_text()
        records = [json.loads(line) for line in trace_path.read_text().splitlines()]
        assert any(row.get("payload", {}).get("org_ready_state") == "unsent" for row in records)
    finally:
        harness.close()


@pytestmark_episode
@pytest.mark.parametrize("cancel_phase", ["preparing", "waiting_slot"])
def test_ready_real_client_cancel_discards_token_retains_fixture_fit_and_trace(tmp_path, monkeypatch, cancel_phase):
    import asyncio
    import threading
    from test_flight_resume_recovery import _harness, _skeleton
    from speech_aware_evidence_acquisition.core.model.vllm_transport import VllmServerTransport
    from speech_aware_evidence_acquisition.reproduction.ojw import runner as ojw_runner
    harness = _harness(tmp_path, "ready-cancel-" + cancel_phase, ("s1", "s2"), max_revisions=0)
    transport = harness.client._rollout_transport
    transport.__class__ = VllmServerTransport
    transport.served_model_name = "stub-model"
    runtime = _skeleton(harness, monkeypatch)
    config = {"schema": DISPATCH_SLOT_FILL_SCHEMA, "role_input_offload": True,
              "prepare_workers": 2, "ready_request": True, "model_inflight": 1,
              "admitted_trajectories": 2}
    runtime._composed = SimpleNamespace(values={**runtime._composed.values, "dispatch": config})
    first_send, second_fit, release_fit, release_send, cancel_observed = (threading.Event() for _ in range(5))
    original_post, original_gate = transport._post, harness.request_gate
    fit_samples = []
    def post(url, body):
        if not first_send.is_set():
            first_send.set()
            if not release_send.wait(10): raise TimeoutError("test send timeout")
        return original_post(url, body)
    def gate(request, **kwargs):
        sample = request.as_payload()["sample_id"]
        result = original_gate(request, **kwargs)
        fit_samples.append(sample)
        if sample == "s2":
            second_fit.set()
            if cancel_phase == "preparing" and not release_fit.wait(10):
                raise TimeoutError("test fit timeout")
        return result
    transport._post = post
    harness.request_gate = gate
    original_execute = ojw_runner.execute_speech_agent_workflows
    def intercept(states, *, episode_invoker, **kwargs):
        async def wrapped(state, contact, session, **extra):
            if state.prepared.sample.sample_id != "s2":
                return await episode_invoker(state, contact, session, **extra)
            while not first_send.is_set(): await asyncio.sleep(.001)
            async def marked_contact(callback, **opts):
                asyncio.current_task().set_name("ready-s2-slot-wait")
                return await contact(callback, **opts)
            task = asyncio.create_task(episode_invoker(state, marked_contact, session, **extra))
            while not second_fit.is_set(): await asyncio.sleep(.001)
            if cancel_phase == "waiting_slot":
                for _ in range(5000):
                    if any(t.get_name() == "ready-s2-slot-wait" for t in asyncio.all_tasks()):
                        break
                    await asyncio.sleep(.001)
                else:
                    raise AssertionError("s2 did not enter contact slot wait")
            task.cancel()
            await asyncio.sleep(.01)
            task.cancel()
            release_fit.set()
            try:
                try:
                    await task
                except asyncio.CancelledError:
                    cancel_observed.set()
                else:
                    raise AssertionError("episode did not propagate cancellation")
            finally:
                release_send.set()
        return original_execute(states, episode_invoker=wrapped, **kwargs)
    monkeypatch.setattr(ojw_runner, "execute_speech_agent_workflows", intercept)
    try:
        from speech_aware_evidence_acquisition.rails.runner import RailsRunnerError
        with pytest.raises(RailsRunnerError, match="native graph ended without an accepted Solver outcome"):
            runtime._dispatch_phase_program(harness.client, harness.plan, harness.program.program_id)
        assert cancel_observed.is_set()
        assert second_fit.is_set() and "s2" in fit_samples
        assert any(event == "fit" and key[0] == "s2" for event, key in harness.events)
        assert not harness.client._prepared_requests
        records = [json.loads(line) for line in (harness.world.run_dir / f"{harness.plan.run_id}.trace.jsonl").read_text().splitlines()]
        assert any(row.get("payload", {}).get("org_ready_state") == "unsent"
                   and ":s2" in row.get("record_id", "") for row in records)
        assert not any(x[0] == "dispatch" and x[1][0] == "s2" for x in harness.events)
    finally:
        release_fit.set(); release_send.set(); harness.close()
