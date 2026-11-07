"""Recorded transport through the production runner and durable row writer."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("openjiuwen")
from speech_aware_evidence_acquisition.core.flight.runtime import FlightRuntime, FlightRuntimeError
from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter, read_checkpoint
from speech_aware_evidence_acquisition.core.flight.resume import read_checkpoint_predecessor, ResumeRecoveryError
from test_llama_server_transport import _spawn_live_matching_process  # noqa: F401
from test_native_episode_runner import _ActualEpisodeHarness
import test_native_episode_runner as fixture
from speech_aware_evidence_acquisition.core.bundle.verify import _verify_node_results
from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_raw_trace, _verify_outputs_trace_binding
from speech_aware_evidence_acquisition.core.bundle import BundleError
from speech_aware_evidence_acquisition.core.flight.runtime import _node_scientific_identity


def test_node_mode_is_distinct_from_system_answer():
    runtime = object.__new__(FlightRuntime)
    runtime._composed = SimpleNamespace(values={"native_episode": {"perception_only": True}})
    assert runtime._result_kind() == "node"


@pytest.mark.parametrize("case", ["valid", "invalid", "finite"])
def test_perception_node_is_checkpointed_with_real_contact(tmp_path, monkeypatch, case):
    harness = _ActualEpisodeHarness(tmp_path, sample_count=1, max_revisions=0,
                                    max_contacts=4, planner_actions=("ACCEPT",))
    harness.config = replace(harness.config, perception_only=True)
    text = json.dumps({"candidates": [{"intent_id": "intent-a"}]}) if case == "valid" else "bad"
    harness._output = lambda key: text
    if case == "finite":
        original = fixture._wire_response
        def wire(value):
            doc = json.loads(original(value))
            doc["choices"][0]["finish_reason"] = "length"
            return json.dumps(doc).encode()
        monkeypatch.setattr(fixture, "_wire_response", wire)
    renderer = object.__new__(FlightRuntime)
    renderer._composed = SimpleNamespace(values={"native_episode": {"perception_only": True}})
    renderer._plan = harness.plan
    renderer._arm = lambda: harness.client.arm
    writer = RowCheckpointWriter(tmp_path / "checkpoint", harness.plan.run_id, "attempt", identity={"config_hash": "candidate-a"})
    renderer._row_checkpoint_writer = writer
    harness.trace.row_checkpoint = lambda **kwargs: renderer._checkpoint_row(
        **kwargs, client=harness.client, program=harness.program, policy=harness.program.program_id)
    try:
        summary = harness.run()
        cost = harness.close()["cost"]
    finally:
        writer.close()
    result, = summary.results
    assert result.result_kind == "node" and result.final_response_text == ""
    assert result.node_outcome["status"] == {"valid": "ok", "invalid": "invalid_output", "finite": "finite_completion"}[case]
    assert [key[1] for key in harness.dispatched] == ["perception"]
    assert cost["calls_used"] == 1
    document = read_checkpoint(writer.path)
    row, = document.rows
    assert row.keepable, row.reason
    record, = row.output_rows
    assert record["result_kind"] == "node" and record["text"] == text
    assert record["node_outcome"] == result.node_outcome
    trace = _verify_raw_trace(('\n'.join(row.trace_lines) + '\n').encode(), None)
    _verify_outputs_trace_binding((record,), trace)
    _verify_node_results("node", (record,), trace, {}, frozenset())
    broken = dict(record, node_outcome={**record['node_outcome'], 'status': 'invented'})
    with pytest.raises(BundleError):
        _verify_node_results("node", (broken,), trace, {}, frozenset())
    args = dict(run_id=harness.plan.run_id, attempt_id="attempt", attempt_dir=writer.path.parent,
                expected_checkpoint_sha256=document.sha256, materialize_dir=tmp_path / "recovered")
    recovered = read_checkpoint_predecessor(**args, node_config_hash="candidate-a")
    replay = recovered.complete[record["sample_id"]]
    assert replay.node_outcome == result.node_outcome and replay.final_response_text == ""
    with pytest.raises(ResumeRecoveryError, match="config identity"):
        read_checkpoint_predecessor(**args, node_config_hash="candidate-b")


def test_full_graph_and_node_share_perception_request_and_candidate_changes_identity(tmp_path):
    response = json.dumps({"candidates": [{"intent_id": "intent-a"}]})
    runs = []
    for mode in ("full", "node", "candidate"):
        harness = _ActualEpisodeHarness(tmp_path / mode, sample_count=1, max_revisions=0,
                                       max_contacts=4, planner_actions=("ACCEPT",))
        if mode != "full":
            harness.config = replace(harness.config, perception_only=True)
        original_output = harness._output
        harness._output = lambda key: response if key[1] == "perception" else original_output(key)
        if mode == "candidate":
            original_material = harness.material_for_phase
            def candidate_material(phase, context, sealed):
                material = original_material(phase, context, sealed)
                return replace(material, instruction=material.instruction + (" candidate" if phase == "perception" else ""))
            harness.material_for_phase = candidate_material
        try:
            harness.run()
            timings = tuple(harness.client.timings)
            harness.close()
        finally:
            pass
        key = next(k for k in harness.prepared if k[1] == "perception")
        runs.append((harness.prepared[key]["rendered"], timings[0].request_sha256))
    assert runs[0] == runs[1]
    assert runs[2][0] != runs[1][0] and runs[2][1] != runs[1][1]


def test_node_identity_ignores_only_operational_attempt_controls():
    values = {'native_episode': {'perception_only': True, 'template': 'candidate-a'},
              'native_contact_fit': {'run_id': 'fit-a', 'attempt_id': 'one', 'contact_cap': 4},
              'runtime_config_identity': {'path': '/a', 'sha256': 'a'}, 'dispatch': {'workers': 1}}
    changed = {**values, 'native_contact_fit': {**values['native_contact_fit'], 'attempt_id': 'two'},
               'runtime_config_identity': {'path': '/b', 'sha256': 'b'}, 'dispatch': {'workers': 2}}
    assert _node_scientific_identity(values) == _node_scientific_identity(changed)
    changed['native_episode'] = {**values['native_episode'], 'template': 'candidate-b'}
    assert _node_scientific_identity(values) != _node_scientific_identity(changed)


def test_node_identity_pins_template_audio_and_supply(tmp_path):
    from copy import deepcopy
    root = tmp_path / 'configs' / 'probes'
    root.mkdir(parents=True)
    template = root / 'p.txt'
    template.write_text('parent')
    values = {'native_episode': {'perception_only': True},
              'phase_material': {'perception': {'instruction_template': 'probes/p.txt'}},
              'prepared_phase_supply': {'sha256': 'a' * 64},
              'request_pipeline': {'shape': 'a'}}
    runtime = object.__new__(FlightRuntime)
    runtime._repo_root = tmp_path
    runtime._composed = SimpleNamespace(values=values)
    audio = SimpleNamespace(path='a.wav', sha256='b' * 64)
    runtime._samples = (SimpleNamespace(sample_id='s1', audio_ref=audio),)
    pin = runtime._node_checkpoint_identity()
    template.write_text('changed')
    assert runtime._node_checkpoint_identity() != pin
    template.write_text('parent')
    audio.sha256 = 'c' * 64
    assert runtime._node_checkpoint_identity() != pin
    audio.sha256 = 'b' * 64
    changed = deepcopy(values)
    changed['prepared_phase_supply']['sha256'] = 'd' * 64
    runtime._composed = SimpleNamespace(values=changed)
    assert runtime._node_checkpoint_identity() != pin


def test_raw_audio_group_v2_node_score_and_trace(tmp_path):
    from test_intent_slot_runner import recovery_harness
    from speech_aware_evidence_acquisition.analysis.perception_node_score import score

    h, policy = recovery_harness(tmp_path)
    h.config = replace(h.config, perception_only=True)
    renderer = object.__new__(FlightRuntime)
    renderer._composed = SimpleNamespace(values={"native_episode": {"perception_only": True}})
    renderer._plan = h.plan
    renderer._arm = lambda: h.client.arm
    writer = RowCheckpointWriter(tmp_path / "checkpoint", h.plan.run_id, "attempt", identity={"config_hash": "a"})
    renderer._row_checkpoint_writer = writer
    h.trace.row_checkpoint = lambda **kw: renderer._checkpoint_row(**kw, client=h.client, program=h.program, policy=h.program.program_id)
    try:
        summary = h.run()
        cost = h.close()['cost']
    finally:
        writer.close()
    row, = read_checkpoint(writer.path).rows
    record, = row.output_rows
    trace = _verify_raw_trace(('\n'.join(row.trace_lines) + '\n').encode(), None)
    _verify_outputs_trace_binding((record,), trace)
    _verify_node_results('node', (record,), trace, {}, frozenset())
    node = record['node_outcome']
    assert node['pipeline_config']['duplicate_candidate_policy'] == 'group_v2'
    assert node['response_format'] and node['evidence_inventory']['sample_id'] == record['sample_id']
    scored = score([{'sample_id': record['sample_id'], 'gold_intent': 'play_music'}], [record],
                   maximum=3, legal_ids={'play_music', 'play_radio'})
    assert scored['valid_output_count'] == 1 and scored['gold_recall_count'] == 1
    assert [key[1] for key in h.dispatched] == ['perception'] and cost['calls_used'] == 1


def test_runtime_dispatches_raw_audio_node_through_checkpoint(tmp_path, monkeypatch):
    from test_intent_slot_runner import recovery_harness
    from test_flight_row_checkpoint_resume import _ckpt_skeleton, _fly_all
    from speech_aware_evidence_acquisition.core.flight.resume import ResumeDeclaration
    from speech_aware_evidence_acquisition.core import preparation_fit
    original_fit_builder = preparation_fit.build_native_contact_fit_gate

    h, policy = recovery_harness(tmp_path)
    h.config = replace(h.config, perception_only=True)
    runtime = _ckpt_skeleton(h, monkeypatch, tmp_path / 'attempt', (h.samples[0].sample_id,))
    runtime._composed.values['native_episode']['perception_only'] = True
    runtime._composed.values['native_episode']['intent_slot_pipeline'] = policy
    runtime._composed.values['native_episode']['finite_completion_policy'] = 'deliver-latest-or-abstain'
    runtime._composed.values['native_episode']['review_loop'] = 'off'
    runtime._composed.values['carrier_lock_key'] = 'slurp'
    runtime._phase_program_material['query_source'] = 'speech_audio'
    runtime._composed.values['native_contact_fit'] = h.fit_gate.runtime._composed.values['native_contact_fit']
    def recorded_gate(harness):
        harness.fit_gate.runtime._checkpoint_fit_lineage = getattr(runtime if harness is h else recovery, '_checkpoint_fit_lineage', {})
        inner = harness.fit_gate.factory()
        def fit(request, **kwargs):
            _ActualEpisodeHarness.request_gate(harness, request)
            return inner(request, **kwargs)
        fit.receipt_for = inner.receipt_for
        return fit
    runtime._build_native_contact_fit_gate = lambda *_args, **_kwargs: recorded_gate(h)
    fit_runtime = h.fit_gate.runtime
    runtime.request_fit_inputs = fit_runtime.request_fit_inputs
    runtime.node_checkpoint_fit_inputs = lambda payloads, original, document: fit_runtime.request_fit_inputs(payloads)
    runtime.session_required_identity = fit_runtime.session_required_identity
    runtime.assert_generation_lease = fit_runtime.assert_generation_lease
    _fly_all(runtime, h)
    rows = runtime._canonical_output_rows(h.client)
    assert rows[0]['result_kind'] == 'node'
    assert runtime._seal_row_checkpoint(rows)['complete_rows'] == 1
    assert [key[1] for key in h.dispatched] == ['perception']
    doc = read_checkpoint(runtime._row_checkpoint_writer.path)
    runtime._run_id = h.plan.run_id
    runtime._resume_declarations = (ResumeDeclaration(attempt_id=h.plan.attempt_id, checkpoint_sha256=doc.sha256),)
    runtime._predecessor_attempt_dir = lambda _: runtime._row_checkpoint_writer.path.parent
    runtime._run_dir = lambda: tmp_path / 'replay'
    runtime._repo_root = tmp_path
    # FIT is produced by the real native gate and its endpoint journal.
    fit_runtime = h.fit_gate.runtime
    runtime.request_fit_inputs = fit_runtime.request_fit_inputs
    runtime.session_required_identity = fit_runtime.session_required_identity
    runtime._composed.values['native_contact_fit'] = fit_runtime._composed.values['native_contact_fit']
    info = runtime._load_resume_predecessor(runtime._composed.values, (h.samples[0].sample_id, 's2'))
    assert info['counts']['kept'] == 1 and runtime._resume_flown_sample_ids == ('s2',)
    pin, = doc.rows[0].fit_pins
    custody_path = __import__('pathlib').Path(pin['path'])
    original_custody = custody_path.read_bytes()
    custody_path.write_bytes(original_custody + b' ')
    with pytest.raises(FlightRuntimeError, match='FIT custody'):
        runtime._load_resume_predecessor(runtime._composed.values, (h.samples[0].sample_id, 's2'))
    custody_path.write_bytes(original_custody)
    seal_info = runtime._load_resume_predecessor(runtime._composed.values, (h.samples[0].sample_id,))
    assert seal_info['seal_only'] and runtime._resume_flown_sample_ids == ()
    runtime._output_rows = []
    sealed, _, _ = runtime._merged_canonical_rows()
    assert len(sealed) == 1 and len(h.dispatched) == 1
    runtime._output_rows = rows
    info = runtime._load_resume_predecessor(runtime._composed.values, (h.samples[0].sample_id, 's2'))
    lineage = info['predecessors'][0]['native_fit_lineage']
    assert lineage['endpoint_contacts'] == 1
    saved = json.loads((tmp_path / 'replay' / 'checkpoint-fit-lineage.json').read_bytes())
    assert saved['predecessors'][h.plan.attempt_id] == lineage
    assert lineage['fits'][0]['artifact_sha256'] == runtime._active_node_fit_gate.receipts[0]['sha256']
    assert runtime._resume.cost_totals()['calls_used'] == 1
    # Run a second actual native dispatch, with the same whole roster and the
    # loaded predecessor selecting the only sample still eligible to fly.
    from test_rails_runner import _plan
    second_plan = _plan(run_id=h.plan.run_id, attempt_id='attempt-two', planned_model_calls=6)
    monkeypatch.setattr(preparation_fit, 'build_native_contact_fit_gate', original_fit_builder)
    second, _ = recovery_harness(tmp_path / 'second', plan=second_plan,
                                 data=h.world.data_root, sample_ids=('s2',))
    second.config = replace(second.config, perception_only=True)
    recovery = _ckpt_skeleton(second, monkeypatch, tmp_path / 'attempt-two', (h.samples[0].sample_id, 's2'))
    recovery._composed = runtime._composed
    recovery._checkpoint_fit_lineage = runtime._checkpoint_fit_lineage
    recovery._resume = runtime._resume
    recovery._resume_flown_sample_ids = runtime._resume_flown_sample_ids
    recovery._phase_program_material['query_source'] = 'speech_audio'
    recovery._build_native_contact_fit_gate = lambda *_args, **_kwargs: recorded_gate(second)
    recovery.request_fit_inputs = second.fit_gate.runtime.request_fit_inputs
    recovery.node_checkpoint_fit_inputs = lambda payloads, original, document: second.fit_gate.runtime.request_fit_inputs(payloads)
    recovery.session_required_identity = second.fit_gate.runtime.session_required_identity
    recovery.assert_generation_lease = second.fit_gate.runtime.assert_generation_lease
    _fly_all(recovery, second)
    assert [(key[0], key[1]) for key in second.dispatched] == [('s2', 'perception')]
    merged, _, _ = recovery._merged_canonical_rows()
    assert {row['sample_id'] for row in merged} == {h.samples[0].sample_id, 's2'}
    assert recovery._resume.cost_totals()['calls_used'] == 1
    assert second.client.timings and len(second.client.timings) == 1
    new_fit = recovery._active_node_fit_gate.receipts[0]
    new_output = json.loads(__import__('pathlib').Path(new_fit['path']).read_bytes())['output']
    assert new_output['cumulative_contacts'] == 2
    # A FIT endpoint can have been charged before a generation intent was
    # journaled. Row-only recovery must not silently drop that spend.
    fit = runtime._composed.values['native_contact_fit']
    fit_root = runtime._data_root / 'prepared' / fit['run_id'] / fit['attempt_id']
    fit_root.mkdir(parents=True, exist_ok=True)
    (fit_root / 'orphan.request-fit.json.contacts.jsonl').write_text('{"contact":1}\n')
    with pytest.raises(FlightRuntimeError, match='unknown predecessor FIT'):
        runtime._load_resume_predecessor(runtime._composed.values, (h.samples[0].sample_id, 's2'))
    (fit_root / 'orphan.request-fit.json.contacts.jsonl').unlink()
    changed = dict(runtime._composed.values, native_episode={**runtime._composed.values['native_episode'], 'episode_id': 'other-candidate'})
    original_composed = runtime._composed
    runtime._composed = SimpleNamespace(values=changed, config_hash='different')
    with pytest.raises(FlightRuntimeError, match='config identity'):
        runtime._load_resume_predecessor(changed, (h.samples[0].sample_id, 's2'))
    runtime._composed = original_composed
    from test_contact_journal import payload
    h.client._adapter._contact_journal.intent(payload('unknown-p'))
    with pytest.raises(FlightRuntimeError, match='unknown contact outcome'):
        runtime._load_resume_predecessor(runtime._composed.values, (h.samples[0].sample_id, 's2'))


def test_checkpoint_known_fit_without_generation_is_reused(tmp_path):
    from pathlib import Path
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    from speech_aware_evidence_acquisition.core.native_fit_recovery import RecoverableNativeFitGate
    from test_native_fit_recovery import source

    fit_runtime, plan, replay, request, old = source(tmp_path)
    # FIT completed, but no generation intent was issued for this sample.
    for path in replay.source.root.glob('*.intent.json'):
        path.unlink()
    for path in replay.source.root.glob('*.response.json'):
        path.unlink()
    runtime = object.__new__(FlightRuntime)
    runtime._data_root = tmp_path
    runtime._run_id = plan.run_id
    runtime._plan = plan
    runtime._arm = fit_runtime._arm
    runtime._composed = fit_runtime._composed
    runtime.request_fit_inputs = fit_runtime.request_fit_inputs
    runtime.session_required_identity = fit_runtime.session_required_identity
    runtime.node_checkpoint_fit_inputs = lambda payloads, original, document: fit_runtime.request_fit_inputs(payloads)
    predecessor = SimpleNamespace(complete_sample_ids=(), canonical={}, fit_pins={})
    lineage = runtime._audit_node_checkpoint_custody(replay.source_plan.attempt_id, predecessor)
    assert lineage['endpoint_contacts'] == 1 and not lineage['fits'][0]['generation_complete']
    gate = RecoverableNativeFitGate(fit_runtime, plan, lambda: pytest.fail('tokenizer repeated'),
                                    checkpoint_fits=lineage['fits'])
    assert gate(request) == 7
    assert gate.receipts[0]['native_fit_reuse']['new_tokenizer_contacts'] == 0
    pin = Path(lineage['fits'][0]['custody_path'])
    pin.write_bytes(pin.read_bytes() + b' ')
    gate = RecoverableNativeFitGate(fit_runtime, plan, lambda: pytest.fail('tokenizer repeated'),
                                    checkpoint_fits=lineage['fits'])
    with pytest.raises(ValueError, match='custody changed'):
        gate(request)
