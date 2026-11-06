"""Raw-audio native Flight path, including semantic refusals and real wire replay."""
import json
from dataclasses import replace
import pytest
from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import canonical
from speech_aware_evidence_acquisition.rails.slurp_solver_protocol import response_format
from speech_aware_evidence_acquisition.rails.phase_material import PhaseMaterial
from test_llama_server_transport import _spawn_live_matching_process
from test_native_episode_runner import _ActualEpisodeHarness
from test_native_episode_config import config as raw_config


def harness(tmp_path, mode='batched', *, plan=None, data=None, replay=None, failure=None, samples=1, sample_ids=None):
    cap = 3 if mode == 'batched' else 5
    h = _ActualEpisodeHarness(tmp_path, sample_count=samples, max_revisions=0, max_contacts=cap,
                              planner_actions=(), arm='raw-speech-agent-native-tools', query_source='speech_audio',
                              contact_journal=True, execution_plan=plan, data_root=data, contact_replay=replay,
                              sample_ids=sample_ids)
    values = raw_config()
    native = values['native_episode']
    native.update(review_loop='off', finite_completion_policy='deliver-latest-or-abstain',
                  intent_slot_pipeline={'schema':'saea-intent-slot-pipeline-v1','slot_mode':mode,
                                        'max_candidates':3,'candidate_outside_recovery':False})
    for kind in ('reviewer','planner'): native['roles'][kind]['max_iterations']=0
    native['roles']['slot']={'phase_id':'slot','max_iterations':1}
    values['carrier_lock_key']='slurp'
    h.program=default_phase_program('phase-native-intent-slot-'+mode.replace('_','-')+'-v1')
    h.config=load_native_episode_config(values,h.program)
    candidate_rows=[{'intent_id':f'play_{action}','distinction':action,'evidence_refs':[],'unresolved_fields':[]}
                    for action in ('music','radio')]
    slot_rows=[{'candidate_id':r['intent_id'],'entities':[{'type':'person','filler':name}],
                'evidence_refs':[],'unresolved_fields':[]} for r,name in zip(candidate_rows,('Alice','Bob'))]
    pformat={'type':'json_schema','json_schema':{'name':'candidate_test','strict':True,'schema':{
        'type':'object','properties':{'candidates':{'type':'array','items':{'type':'object','properties':{
            'intent_id':{'enum':['play_music','play_radio']}}}}}}}}
    sformat={'type':'json_schema','json_schema':{'name':'slots_test','strict':True,'schema':{
        'type':'object','properties':{'candidates':{'type':'array','items':{'type':'object','properties':{
            'entities':{'type':'array','items':{'type':'object','properties':{'type':{'enum':['person']}}}}}}}}}}}
    formats={'perception':pformat,'slot':sformat,'solver':response_format({'play':['music','radio']},['person'])}
    original=h.material_for_phase
    def material(phase,context,sealed):
        return replace(original(phase,context,sealed),response_format=formats.get(phase))
    h.material_for_phase=material
    def output(key):
        phase,visit=key[1:3]
        if failure == phase: return '{}'
        if phase=='perception': return canonical({'candidates':candidate_rows})
        if phase=='slot': return canonical({'candidates':slot_rows if mode=='batched' else [slot_rows[visit-1]]})
        return canonical({'scenario':'play','action':'radio','entities':slot_rows[1]['entities']})
    h._output=output
    # Real native admission tokenizer path and its durable reuse certificate;
    # the tokenizer is the same model-free stub used by A0 recovery tests.
    from types import SimpleNamespace
    from test_native_contact_fit import _raw_runtime, _RawTokenizer
    from speech_aware_evidence_acquisition.core.native_fit_recovery import RecoverableNativeFitGate
    from speech_aware_evidence_acquisition.core.preparation_fit import build_native_contact_fit_gate
    runtime=_raw_runtime(h.world.data_root)
    runtime._samples=h.samples
    runtime._transport=h.transport
    runtime._composed.values['native_contact_fit']['contact_cap']=cap * samples
    runtime._adapter=SimpleNamespace(serving_geometry=lambda:{'slot_tokens':24576})
    h.fit_calls=[]
    class Tokenizer(_RawTokenizer):
        def request_json(self,*args):
            h.fit_calls.append(args)
            return super().request_json(*args)
    h.fit_gate=RecoverableNativeFitGate(runtime,h.plan,lambda:build_native_contact_fit_gate(
        runtime,h.program,h.plan,client_factory=Tokenizer),replay=replay)
    original_gate=h.request_gate
    def fit(request):
        original_gate(request)
        return h.fit_gate(request)
    h.request_gate=fit
    return h


def recovery_harness(tmp_path, *, plan=None, data=None, replay=None):
    from test_rails_runner import _plan
    plan = plan or _plan(run_id='SAEA-E-native-role-runner-proof', attempt_id='attempt-native-role-runner-proof', planned_model_calls=6)
    h = harness(tmp_path, plan=plan, data=data, replay=replay)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2',
        node_rails='sdk-handoff-v1', answer_only_fallback='off',
        coverage_recovery={'strategy': 'complete-original-candidate-set-v1', 'max_extra_contacts': 3})
    h.program = default_phase_program('phase-native-intent-slot-batched-recovery-v1')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy), max_contacts=6)
    from types import SimpleNamespace
    from test_native_contact_fit import _raw_runtime, _RawTokenizer
    from speech_aware_evidence_acquisition.core.native_fit_recovery import RecoverableNativeFitGate
    from speech_aware_evidence_acquisition.core.preparation_fit import build_native_contact_fit_gate
    runtime = _raw_runtime(h.world.data_root)
    runtime._samples, runtime._transport = h.samples, h.transport
    runtime._composed.values['native_contact_fit']['contact_cap'] = 6
    runtime._adapter = SimpleNamespace(serving_geometry=lambda: {'slot_tokens': 24576})
    h.fit_gate = RecoverableNativeFitGate(runtime, h.plan, lambda: build_native_contact_fit_gate(
        runtime, h.program, h.plan, client_factory=_RawTokenizer), replay=replay)
    original = h._output
    def output(key):
        value = json.loads(original(key))
        if key[1] == 'slot':
            value['candidates'] = [value['candidates'][0 if key[2] == 1 else 1]]
        elif key[1] == 'solver':
            value = {'candidate_id': 'play_radio'}
        return canonical(value)
    h._output = output
    return h, policy


def test_sdk_full_candidate_recovery_and_seal(tmp_path):
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    h, policy = recovery_harness(tmp_path)
    try:
        summary = h.run()
        raw = _raw_native(h, solver_role={'max_iterations': 1})
        raw['intent_slot_pipeline'] = policy
        outputs = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    finally:
        cost = h.close()['cost']
    assert cost['calls_used'] == 4
    assert len(h.fit_gate.receipts) == 4
    assert len({visit['execution_receipt']['final_contact']['agent_contact_id']
                for visit in summary.results[0].native_episode_receipt['role_visits']}) == 4
    assert json.loads(summary.results[0].final_response_text)['entities'][0]['filler'] == 'Bob'
    records = {rid: p for _, rid, p in h.trace.records}
    _verify_outputs_trace_binding(tuple(outputs), records)
    from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import validate_receipt
    receipt = summary.results[0].native_episode_receipt
    assert json.loads(validate_receipt(receipt, trace_records=records))['entities'][0]['filler'] == 'Bob'
    import copy
    block = receipt['intent_slot_pipeline']
    assert block['candidate_ids'] == ['play_music', 'play_radio']
    assert [source['candidate_ids'] for source in block['slot_sources']] == [
        ['play_music', 'play_radio'], ['play_radio']]
    assert [source['valid_ids'] for source in block['slot_sources']] == [
        ['play_music'], ['play_radio']]
    requests = [v['rendered'] for key, v in h.prepared.items() if key[1] == 'slot']
    assert len(requests) == 2
    assert 'play_music' not in json.dumps(requests[1]['response_format'])
    for index in (0, 1):
        changed = copy.deepcopy(records)
        response_id = block['slot_sources'][index]['source']['request_id'] + '/response'
        changed[response_id]['text'] += ' '
        with pytest.raises(ValueError):
            validate_receipt(receipt, trace_records=changed)
    bad = copy.deepcopy(receipt)
    bad['intent_slot_pipeline']['slot_sources'][1]['valid_ids'] = ['play_music']
    with pytest.raises(ValueError):
        validate_receipt(bad, trace_records=records)


@pytest.mark.parametrize('unknown', [False, True])
def test_sdk_recovery_resume_does_not_resend_known_or_unknown(tmp_path, unknown):
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    old, _ = recovery_harness(tmp_path / 'old')
    if unknown:
        output = old._output
        def fail_repair(key):
            if key[1] == 'slot' and key[2] == 2:
                raise TimeoutError('unknown repair response')
            return output(key)
        old._output = fail_repair
    else:
        sink = old.client._adapter._sink
        emit = sink.emit
        count = 0
        def stop(channel, rid, payload):
            nonlocal count
            result = emit(channel, rid, payload)
            if channel == 'MODEL_RESPONSE':
                count += 1
                if count == 3:
                    raise RuntimeError('stop after durable repair response')
            return result
        sink.emit = stop
    try:
        with pytest.raises(Exception):
            old.run()
    finally:
        old.close()
    old.client._adapter._attempt_handle.mark_aborted('recovery test source terminal')
    plan = replace(old.plan, attempt_id='coverage-successor')
    replay = NativeContactReplay(old.world.data_root, old.plan, plan, old.client.arm)
    new, _ = recovery_harness(tmp_path / 'new', plan=plan, data=old.world.data_root, replay=replay)
    gate = new.request_gate
    def track(request):
        gate(request)
        if replay.match(request.as_payload()) is not None:
            new.replayed_fixture_keys.add(next(reversed(new.prepared)))
    new.request_gate = track
    try:
        if unknown:
            with pytest.raises(Exception):
                new.run()
        else:
            result = new.run()
            assert json.loads(result.results[0].final_response_text)['entities'][0]['filler'] == 'Bob'
    finally:
        cost = new.close()['cost']
    assert [key[1] for key in new.dispatched] == ([] if unknown else ['solver'])
    assert cost['calls_used'] == (0 if unknown else 1)


def test_sdk_invalid_repair_retains_pending_row(tmp_path):
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    h, policy = recovery_harness(tmp_path)
    original = h._output
    def invalid(key):
        if key[1] == 'slot' and key[2] == 2:
            return '{}'
        return original(key)
    h._output = invalid
    try:
        summary = h.run()
        raw = _raw_native(h, solver_role={'max_iterations': 1})
        raw['intent_slot_pipeline'] = policy
        outputs = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    finally:
        cost = h.close()['cost']
    receipt = summary.results[0].native_episode_receipt
    assert cost['calls_used'] == 3
    assert receipt['terminal_disposition']['reason'] == 'seal_refusal'
    pending = receipt['slot_recovery_pending']
    assert list(pending['valid_results']) == ['play_music']
    assert pending['remaining_candidate_ids'] == ['play_radio']
    _verify_outputs_trace_binding(tuple(outputs), {rid: p for _, rid, p in h.trace.records})


@pytest.mark.parametrize('budget', [1, 3])
def test_sdk_known_invalid_repair_has_successor(tmp_path, budget):
    old, _ = recovery_harness(tmp_path / 'old')
    policy = json.loads(old.config.intent_slot_pipeline_json)
    policy['coverage_recovery']['max_extra_contacts'] = budget
    old.config = replace(old.config, intent_slot_pipeline_json=canonical(policy), max_contacts=3 + budget)
    output = old._output
    old._output = lambda key: '{}' if key[1] == 'slot' and key[2] == 2 else output(key)
    try:
        first = old.run()
        assert first.results[0].native_episode_receipt['slot_recovery_pending']['remaining_candidate_ids'] == ['play_radio']
        from test_template_variants_delivery import _runtime, _raw_native
        from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter, serialize_trace_records, sha256_file
        raw = _raw_native(old, solver_role={'max_iterations': 1})
        raw['intent_slot_pipeline'] = json.loads(old.config.intent_slot_pipeline_json)
        rows = _runtime(old, first, raw)._canonical_output_rows(old.client)
        writer = RowCheckpointWriter(old.world.data_root / 'runs' / old.plan.run_id / old.plan.attempt_id,
            old.plan.run_id, old.plan.attempt_id, identity={'protocol_hash': old.plan.protocol_hash,
            'split_identity_hash': old.plan.split_identity_hash})
        writer.append_row(sample_id=rows[0]['sample_id'], complete=True, output_rows=rows,
            trace_lines=serialize_trace_records(old.trace.records))
        writer.close()
    finally:
        old.close()
    old.client._adapter._attempt_handle.mark_aborted('known invalid successor source')
    plan = replace(old.plan, attempt_id='known-invalid-successor')
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournalError
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignCell
    from speech_aware_evidence_acquisition.core.flight.native_recovery import load_native_recovery, select_recovery_cell
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'analysis'))
    from native_recovery_receipt import prepare
    receipt_path = old.world.data_root / 'slot-successor-recovery.json'
    declaration = prepare(old.world.data_root, old.plan, plan.attempt_id, old.client.arm, receipt_path,
                          slot_checkpoint_sha256=sha256_file(writer.path))['declaration']
    cell = CampaignCell.from_document({'cell_id': 'coverage', 'model_fragment': 'm',
        'dataset_fragment': 'd', 'baseline_fragment': 'b', 'run_id': old.plan.run_id,
        'attempt_id': old.plan.attempt_id, 'readiness': {'state': 'ready'},
        'allowed_execution_stages': ['worker', 'check']})
    selected = select_recovery_cell(cell, declaration, data_root=old.world.data_root)
    assert selected.attempt_id == plan.attempt_id
    assert selected.document == {**cell.document, 'attempt_id': plan.attempt_id}
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    values = raw_config()
    values.update(transport='vllm', arm=old.client.arm, carrier_lock_key='slurp',
                  policy='phase-native-intent-slot-batched-recovery-v1')
    native = values['native_episode']
    native.update(review_loop='off', finite_completion_policy='deliver-latest-or-abstain',
                  intent_slot_pipeline=policy)
    for kind in ('reviewer', 'planner'):
        native['roles'][kind]['max_iterations'] = 0
    native['roles']['slot'] = {'phase_id': 'slot', 'max_iterations': 1}
    assert native_contact_custody_enabled(values)
    replay = load_native_recovery(declaration,
                                  data_root=old.world.data_root, plan=plan, arm=old.client.arm)
    original_checkpoint = writer.path.read_bytes()
    writer.path.write_bytes(original_checkpoint + b'changed')
    with pytest.raises(ContactJournalError):
        replay.slot_pending('s1')
    writer.path.write_bytes(original_checkpoint)
    new, _ = recovery_harness(tmp_path / 'new', plan=plan, data=old.world.data_root, replay=replay)
    new.config = replace(new.config, intent_slot_pipeline_json=canonical(policy), max_contacts=3 + budget)
    gate = new.request_gate
    def track(request):
        gate(request)
        if replay.match(request.as_payload()) is not None:
            new.replayed_fixture_keys.add(next(reversed(new.prepared)))
    new.request_gate = track
    try:
        result = new.run()
        if budget > 1:
            raw = _raw_native(new, solver_role={'max_iterations': 1})
            raw['intent_slot_pipeline'] = json.loads(new.config.intent_slot_pipeline_json)
            _runtime(new, result, raw)._canonical_output_rows(new.client)
            from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import validate_receipt
            validate_receipt(result.results[0].native_episode_receipt,
                             trace_records={rid: payload for _, rid, payload in new.trace.records})
    finally:
        cost = new.close()['cost']
    if budget == 1:
        assert result.results[0].final_response_text == ''
        assert not new.dispatched
        assert cost['calls_used'] == 0
        return
    assert json.loads(result.results[0].final_response_text)['entities'][0]['filler'] == 'Bob'
    assert [key[1] for key in new.dispatched] == ['slot', 'solver']
    assert cost['calls_used'] == 2
    assert cost['replayed_predecessor_cost']['calls'] == 3
    assert result.results[0].native_episode_receipt['intent_slot_pipeline']['slot_successor']['extra_contacts'] == 1
    from speech_aware_evidence_acquisition.core.bundle.native_replay import verify_native_replay
    from speech_aware_evidence_acquisition.core.bundle import BundleError
    import copy
    records = {rid: payload for _, rid, payload in new.trace.records}
    manifest = {'arm': new.client.arm, 'cost': cost, 'attempt': {'attempt_id': plan.attempt_id,
                'state_path': str(new.client._adapter._attempt_handle.path)}}
    assert len(verify_native_replay(manifest, records, plan)) == 3
    bad = copy.deepcopy(manifest)
    bad['cost']['replayed_predecessor_cost']['calls'] += 1
    with pytest.raises(BundleError):
        verify_native_replay(bad, records, plan)


def test_sdk_recovers_two_missing_candidates(tmp_path):
    h, _ = recovery_harness(tmp_path)
    original = h._output
    def output(key):
        value = json.loads(original(key))
        if key[1] == 'slot' and key[2] == 1:
            value['candidates'] = []
        elif key[1] == 'slot' and key[2] == 2:
            value['candidates'][0]['candidate_id'] = 'play_music'
        return canonical(value)
    h._output = output
    try:
        summary = h.run()
    finally:
        cost = h.close()['cost']
    assert cost['calls_used'] == 5
    assert json.loads(summary.results[0].final_response_text)['entities'][0]['filler'] == 'Bob'
    from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import validate_receipt
    validate_receipt(summary.results[0].native_episode_receipt,
                     trace_records={rid: p for _, rid, p in h.trace.records})


def test_sdk_recovery_budget_exhaustion_is_pending(tmp_path):
    h, policy = recovery_harness(tmp_path)
    policy['coverage_recovery']['max_extra_contacts'] = 1
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy), max_contacts=4)
    original = h._output
    def output(key):
        value = json.loads(original(key))
        if key[1] == 'slot':
            value['candidates'] = []
        return canonical(value)
    h._output = output
    try:
        result = h.run()
    finally:
        cost = h.close()['cost']
    receipt = result.results[0].native_episode_receipt
    assert cost['calls_used'] == 2
    assert receipt['terminal_disposition']['reason'] == 'seal_refusal'
    assert receipt['slot_recovery_pending']['original_candidate_ids'] == ['play_music', 'play_radio']
    assert 'budget exhausted' in receipt['terminal_disposition']['seal_refusal']['message']


@pytest.mark.parametrize('mode,contacts',[('batched',3),('per_candidate',4)])
def test_raw_audio_pipeline_and_bound_sources(tmp_path,mode,contacts):
    h=harness(tmp_path,mode)
    try: summary=h.run()
    finally: cost=h.close()['cost']
    assert cost['calls_used']==contacts
    row=summary.results[0]
    assert json.loads(row.final_response_text)['entities'][0]['filler']=='Bob'
    receipt=row.native_episode_receipt
    assert receipt['intent_slot_pipeline']['candidate_ids']==['play_music','play_radio']
    requests=[v['rendered'] for v in h.prepared.values()]
    assert len(requests)==contacts
    for request in requests:
        assert 'input_audio' in json.dumps(request)
    slot_requests=[v['rendered'] for k,v in h.prepared.items() if k[1]=='slot']
    assert all('perception_source' in json.dumps(r) for r in slot_requests)


@pytest.mark.parametrize('phase,contacts',[('perception',1),('slot',2),('solver',3)])
def test_semantic_failure_retains_paid_row(tmp_path,phase,contacts):
    h=harness(tmp_path,failure=phase)
    try:
        summary=h.run()
        from test_template_variants_delivery import _runtime, _raw_native
        from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
        raw=_raw_native(h,solver_role={'max_iterations':1})
        raw['intent_slot_pipeline']=json.loads(h.config.intent_slot_pipeline_json)
        outputs=_runtime(h,summary,raw)._canonical_output_rows(h.client)
    finally: cost=h.close()['cost']
    _verify_outputs_trace_binding(tuple(outputs),{rid:p for channel,rid,p in h.trace.records})
    assert len(summary.results)==1
    assert cost['calls_used']==contacts
    assert summary.results[0].native_episode_receipt['terminal_disposition']['reason']=='seal_refusal'


@pytest.mark.parametrize('stop_after,expected_new', [(2,2),(4,0)])
def test_partial_slot_and_final_decision_replay(tmp_path,stop_after,expected_new):
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from test_template_variants_delivery import _runtime, _raw_native
    old=harness(tmp_path/'source','per_candidate')
    sink=old.client._adapter._sink
    emit=sink.emit
    count=0
    def stop(channel,rid,payload):
        nonlocal count
        result=emit(channel,rid,payload)
        if channel=='MODEL_RESPONSE':
            count+=1
            if count==stop_after: raise RuntimeError('stop after durable response')
        return result
    sink.emit=stop
    try:
        with pytest.raises(Exception): old.run()
    finally: old.close()
    old.client._adapter._attempt_handle.mark_aborted('known terminal source in pipeline test')
    plan=replace(old.plan,attempt_id='intent-slot-recovery')
    replay=NativeContactReplay(old.world.data_root,old.plan,plan,old.client.arm)
    new=harness(tmp_path/'target','per_candidate',plan=plan,data=old.world.data_root,replay=replay)
    original=new.request_gate
    def gate(request):
        original(request)
        if replay.match(request.as_payload()) is not None:
            new.replayed_fixture_keys.add(next(reversed(new.prepared)))
    new.request_gate=gate
    try:
        summary=new.run()
        raw=_raw_native(new,solver_role={'max_iterations':1})
        raw['intent_slot_pipeline']=json.loads(new.config.intent_slot_pipeline_json)
        outputs=_runtime(new,summary,raw)._canonical_output_rows(new.client)
    finally: cost=new.close()['cost']
    assert len(new.dispatched)==expected_new
    assert cost['calls_used']==expected_new
    assert cost['replayed_predecessor_cost']['calls']==stop_after
    assert len(new.fit_calls)==expected_new
    assert len(new.fit_gate.receipts)==4
    assert all(r['native_fit_reuse']['new_tokenizer_contacts']==0 for r in new.fit_gate.receipts[:stop_after])
    records={rid:p for channel,rid,p in new.trace.records}
    _verify_outputs_trace_binding(tuple(outputs),records)
    if expected_new:
        assert [k[1] for k in new.dispatched]==['slot','solver']


def test_slot_branch_does_not_consume_sibling_completion(tmp_path):
    h=harness(tmp_path,'per_candidate')
    try: h.run()
    finally: h.close()
    requests=[v['rendered'] for k,v in h.prepared.items() if k[1]=='slot']
    assert 'Alice' not in json.dumps(requests[1])
    assert 'play_radio' in json.dumps(requests[1])


def test_changed_template_cannot_replay_old_response(tmp_path):
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    old=harness(tmp_path/'source')
    try: old.run()
    finally: old.close()
    old.client._adapter._attempt_handle.mark_aborted('test terminal release')
    plan=replace(old.plan,attempt_id='changed-template')
    replay=NativeContactReplay(old.world.data_root,old.plan,plan,old.client.arm)
    new=harness(tmp_path/'target',plan=plan,data=old.world.data_root,replay=replay)
    original=new.material_for_phase
    def changed(phase,context,sealed):
        m=original(phase,context,sealed)
        return replace(m,instruction=m.instruction+' changed')
    new.material_for_phase=changed
    try:
        with pytest.raises(Exception): new.run()
    finally: cost=new.close()['cost']
    assert cost['calls_used']==0


def test_unknown_slot_is_not_reissued(tmp_path):
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    old=harness(tmp_path/'source','per_candidate')
    original=old._output
    def unknown(key):
        if key[1]=='slot' and key[2]==2: raise TimeoutError('unknown second Slot response')
        return original(key)
    old._output=unknown
    try:
        with pytest.raises(Exception): old.run()
    finally: old.close()
    old.client._adapter._attempt_handle.mark_aborted('unknown-response recovery test')
    plan=replace(old.plan,attempt_id='unknown-slot')
    replay=NativeContactReplay(old.world.data_root,old.plan,plan,old.client.arm)
    new=harness(tmp_path/'target','per_candidate',plan=plan,data=old.world.data_root,replay=replay)
    original_gate=new.request_gate
    def gate(request):
        original_gate(request)
        if replay.match(request.as_payload()) is not None:
            new.replayed_fixture_keys.add(next(reversed(new.prepared)))
    new.request_gate=gate
    try:
        with pytest.raises(Exception): new.run()
    finally: cost=new.close()['cost']
    assert cost['calls_used']==0
    assert not new.dispatched
    assert not new.fit_calls


@pytest.mark.parametrize('valid', [True, False])
def test_slot_refs_bound_to_actual_current_audio_on_real_graph(tmp_path, valid):
    h=harness(tmp_path)
    original=h._output
    def output(key):
        value=json.loads(original(key))
        if key[1]=='slot':
            ref=h.speech_bindings[key[0]]['packet_id'] if valid else 'audio-of-another-row'
            for row in value['candidates']: row['evidence_refs']=[ref]
        return canonical(value)
    h._output=output
    try: summary=h.run()
    finally: cost=h.close()['cost']
    assert cost['calls_used']==(3 if valid else 2)
    receipt=summary.results[0].native_episode_receipt
    if valid:
        assert receipt['intent_slot_pipeline']['selected_candidate_id']=='play_radio'
        slot=next(v['rendered'] for k,v in h.prepared.items() if k[1]=='slot')
        assert 'evidence_inventory' in json.dumps(slot)
        assert h.speech_bindings[h.samples[0].sample_id]['packet_id'] in json.dumps(slot)
    else:
        assert receipt['terminal_disposition']['reason']=='seal_refusal'

@pytest.mark.parametrize('repeated', [False, True])
def test_runner_real_projection_keeps_sealed_short_id_and_duplicate_policy(tmp_path, repeated):
    from test_slot_binding_view import _config as projection_config, bind_slot_view_context
    from speech_aware_evidence_acquisition.rails.slurp_solver_protocol import response_format
    from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import validate_evidence_refs
    h=harness(tmp_path/'run')
    cfg=projection_config(h.world.data_root)
    cfg['native_episode']['intent_slot_pipeline']={'schema':'saea-intent-slot-pipeline-v1','duplicate_candidate_policy':'identical_v1'}
    cfg['native_episode']['roles']['slot']={'phase_id':'slot','max_iterations':1}
    cfg['knowledge_stores']['slot-contracts']['candidate_supply']['slot']=cfg['knowledge_stores']['slot-contracts']['candidate_supply']['solver'].copy()
    cfg['native_evidence_projection_params']['consumer_roles']=['slot','solver']
    context=bind_slot_view_context(cfg,h.world.data_root)
    h.transport._native_evidence_projection='native-candidate-slot-view-v1'
    h.transport._native_evidence_projection_context=context
    policy=json.loads(h.config.intent_slot_pipeline_json)
    policy.update(duplicate_candidate_policy='identical_v1',evidence_policy='unique_source_alias_v1')
    h.config=replace(h.config,intent_slot_pipeline_json=canonical(policy))
    original_material=h.material_for_phase
    def material(phase,ctx,sealed):
        result=original_material(phase,ctx,sealed)
        fmt=result.response_format
        if phase=='perception':
            fmt=json.loads(json.dumps(fmt));fmt['json_schema']['schema']['properties']['candidates']['items']['properties']['intent_id']['enum']=['alpha_one']
        elif phase=='solver':fmt=response_format({'alpha':['one']},['person'])
        return replace(result,response_format=fmt)
    h.material_for_phase=material
    candidate={'intent_id':'alpha_one','distinction':'same','evidence_refs':['audio-ref'],'unresolved_fields':[]}
    def output(key):
        phase=key[1]
        if phase=='perception':return canonical({'candidates':[candidate,candidate] if repeated else [candidate]})
        if phase=='slot':return canonical({'candidates':[{'candidate_id':'alpha_one','entities':[],'evidence_refs':[],'unresolved_fields':[]}]})
        return canonical({'scenario':'alpha','action':'one','entities':[]})
    # The actual audio packet is a permitted evidence ID, so its raw spelling
    # remains source-bound while the runner builds its native Slot context.
    candidate['evidence_refs']=[next(iter(h.speech_bindings.values()))['packet_id']]
    h._output=output
    try:summary=h.run()
    finally:h.close()
    receipt=summary.results[0].native_episode_receipt
    assert [v['stage'] for v in receipt['role_visits']]==['perception','slot','solver']
    assert receipt['intent_slot_pipeline']['candidate_ids']==['alpha_one']
    assert receipt['intent_slot_pipeline']['selected_candidate_id']=='alpha_one'
    assert json.loads(summary.results[0].final_response_text)['entities']==[]

@pytest.mark.parametrize('mode', ['batched', 'per_candidate'])
def test_v2_real_runner_three_nodes_and_raw_trace_replay(tmp_path, mode):
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from speech_aware_evidence_acquisition.core.bundle import BundleError
    from test_template_variants_delivery import _runtime, _raw_native
    h = harness(tmp_path, mode)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2')
    h.program = default_phase_program('phase-native-intent-slot-' + mode.replace('_', '-') + '-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    original = h._output
    def output(key):
        value = json.loads(original(key))
        if key[1] == 'perception':
            extra = dict(value['candidates'][0], distinction='second independent cue')
            value['candidates'].insert(1, extra)
        if key[1] == 'solver':
            value = {'candidate_id': 'play_radio'}
        return canonical(value)
    h._output = output
    try:
        summary = h.run()
        raw = _raw_native(h, solver_role={'max_iterations': 1})
        raw['intent_slot_pipeline'] = policy
        outputs = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    finally:
        cost = h.close()['cost']
    assert cost['calls_used'] == (3 if mode == 'batched' else 4)
    assert json.loads(summary.results[0].final_response_text) == {
        'scenario': 'play', 'action': 'radio', 'entities': [{'type': 'person', 'filler': 'Bob'}]}
    records = {rid: p for channel, rid, p in h.trace.records}
    _verify_outputs_trace_binding(tuple(outputs), records)
    assert len(summary.results[0].native_episode_receipt['intent_slot_pipeline']['grouping']['groups'][0]['sources']) == 2
    tampered = [dict(outputs[0], text='{}')]
    with pytest.raises(BundleError):
        _verify_outputs_trace_binding(tuple(tampered), records)

@pytest.mark.parametrize('mode', ['batched', 'per_candidate'])
@pytest.mark.parametrize('rails_enabled', [False, True])
def test_v2_v4g_normaliser_source_chain_and_trace_tamper(tmp_path, monkeypatch, mode, rails_enabled):
    """The real v4g implementation transforms the derived Slot answer before seal."""
    import copy
    import hashlib
    from speech_aware_evidence_acquisition.core.bundle import BundleError
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from speech_aware_evidence_acquisition.rails import delivery_normalisers as dn
    from speech_aware_evidence_acquisition.rails import slurp_projection_v3 as v3
    from speech_aware_evidence_acquisition.reproduction.slurp_scoring import legal_intent_inventory
    from test_template_variants_delivery import _runtime, _raw_native

    # Deterministic, model-free TRAIN-statistics stand-ins; use the registered
    # v4g and v3 implementations themselves in both runner and bundle replay.
    stats = dn.SlurpV4Statistics(article_strip=[], oclock_strip=False,
        intent_n={'play_radio': 10}, intent_with_entities={'play_radio': 10},
        intent_type={'play_radio': {'person': 10}})
    table = dn.ScopeRuleTable({}, sha256='0' * 64, floor=5)
    monkeypatch.setattr(dn, 'default_v4_statistics', lambda: stats)
    monkeypatch.setattr(dn, 'default_scope_rule_table', lambda: table)
    monkeypatch.setattr(v3, '_default_legal_intent_inventory',
                        lambda: legal_intent_inventory({'play': ('music', 'radio')}))

    h = harness(tmp_path, mode)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2')
    if rails_enabled: policy.update(node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1')
    h.program = default_phase_program('phase-native-intent-slot-' + mode.replace('_', '-') + '-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy),
                       deliver_normaliser_id=dn.DELIVER_NORMALISE_V4G)
    original = h._output
    def output(key):
        value = json.loads(original(key))
        if key[1] == 'slot':
            for row in value['candidates']:
                if row['candidate_id'] == 'play_radio':
                    row['entities'][0]['filler'] = '  BOB   Smith  '
        elif key[1] == 'solver':
            value = {'candidate_id': 'play_radio'}
        return canonical(value)
    h._output = output
    try:
        summary = h.run()
        raw = _raw_native(h, solver_role={'max_iterations': 1})
        raw['intent_slot_pipeline'] = policy
        raw['deliver_normaliser'] = {'normaliser': dn.DELIVER_NORMALISE_V4G}
        runtime = _runtime(h, summary, raw)
        # The generic mock harness uses fake-carrier; the registered v4g and
        # its Flight output contract are SLURP-only.
        runtime._plan = replace(h.plan, carrier_lock_key='slurp')
        outputs = runtime._canonical_output_rows(h.client)
    finally:
        cost = h.close()['cost']
    assert cost['calls_used'] == (3 if mode == 'batched' else 4)
    row = outputs[0]
    receipt = row['native_episode_receipt']
    block = receipt['intent_slot_pipeline']
    norm = receipt['deliver_normaliser']
    records = {rid: payload for _, rid, payload in h.trace.records}
    raw_decision = records[block['decision_source']['request_id'] + '/response']
    raw_text = raw_decision['text']
    sha = lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()
    assert json.loads(raw_text) == {'candidate_id': 'play_radio'}
    assert sha(raw_text) == raw_decision['response_text_sha256'] == block['decision_source']['response_text_sha256']
    assert raw_decision['response_sha256'] == block['decision_source']['response_sha256'] == row['response_sha256']
    derived = canonical({'scenario': 'play', 'action': 'radio',
                         'entities': [{'type': 'person', 'filler': '  BOB   Smith  '}]})
    assert block['derived_answer_sha256'] == norm['input_sha256'] == sha(derived)
    assert norm['normaliser_id'] == dn.DELIVER_NORMALISE_V4G and norm['changed'] is True
    assert json.loads(row['text'])['entities'] == [{'type': 'person', 'filler': 'bob smith'}]
    assert row['text'] != derived
    assert norm['output_sha256'] == row['response_text_sha256'] == sha(row['text'])
    _verify_outputs_trace_binding(tuple(outputs), records)

    for target in ('decision', 'slot', 'derived_hash', 'normaliser_input', 'normaliser_output', 'answer'):
        changed_row, changed_records = copy.deepcopy(row), copy.deepcopy(records)
        changed_block = changed_row['native_episode_receipt']['intent_slot_pipeline']
        changed_norm = changed_row['native_episode_receipt']['deliver_normaliser']
        if target == 'decision':
            changed_records[block['decision_source']['request_id'] + '/response']['text'] = canonical({'candidate_id': 'play_music'})
        elif target == 'slot':
            changed_records[block['slot_sources'][-1]['source']['request_id'] + '/response']['text'] += ' '
        elif target == 'derived_hash':
            changed_block['derived_answer_sha256'] = '0' * 64
        elif target == 'normaliser_input':
            changed_norm['input_sha256'] = '0' * 64
        elif target == 'normaliser_output':
            changed_norm['output_sha256'] = '0' * 64
        else:
            changed_row['text'] = '{}'
            changed_row['response_text_sha256'] = sha('{}')
        with pytest.raises(BundleError):
            _verify_outputs_trace_binding((changed_row,), changed_records)

@pytest.mark.parametrize('mode', ['batched', 'per_candidate'])
def test_sdk_candidate_hooks_actual_runner(mode, tmp_path, monkeypatch):
    import hashlib
    from speech_aware_evidence_acquisition.reproduction.ojw import native_roles
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from test_template_variants_delivery import _runtime, _raw_native
    events = []
    before, after = native_roles._RoleAuthorityRail.before_model_call, native_roles._RoleAuthorityRail.after_model_call
    async def record_before(self, ctx):
        events.append(('before', self._role_id))
        return await before(self, ctx)
    async def record_after(self, ctx):
        events.append(('after', self._role_id))
        return await after(self, ctx)
    monkeypatch.setattr(native_roles._RoleAuthorityRail, 'before_model_call', record_before)
    monkeypatch.setattr(native_roles._RoleAuthorityRail, 'after_model_call', record_after)
    h = harness(tmp_path, mode)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2', node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1')
    h.program = default_phase_program('phase-native-intent-slot-' + mode.replace('_', '-') + '-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    original = h._output
    def output(key):
        value = json.loads(original(key))
        if key[1] == 'perception':
            value['candidates'][0]['unresolved_fields'] = ['uncertain sample cue']
            value['candidates'].insert(1, dict(value['candidates'][0], distinction='different words', unresolved_fields=['another uncertainty']))
        elif key[1] == 'slot':
            for row in value['candidates']:
                row['entities'].append(dict(row['entities'][0]))
                row['unresolved_fields'] = ['uncertain entity']
        elif key[1] == 'solver':
            value = {'candidate_id': 'play_radio'}
        return canonical(value)
    h._output = output
    try:
        summary = h.run()
        raw = _raw_native(h, solver_role={'max_iterations': 1}); raw['intent_slot_pipeline'] = policy
        outputs = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    finally:
        cost = h.close()['cost']
    expected = ['perception'] + ['slot'] * (1 if mode == 'batched' else 2) + ['solver']
    assert events == [(side, role) for role in expected for side in ('before', 'after')]
    assert cost['calls_used'] == len(expected)
    row = outputs[0]; block = row['native_episode_receipt']['intent_slot_pipeline']
    assert len(json.loads(row['text'])['entities']) == 2
    records = {rid: payload for _, rid, payload in h.trace.records}
    _verify_outputs_trace_binding(tuple(outputs), records)
    assert len(block['node_handoffs']) == len(expected)
    assert hashlib.sha256(records[block['perception_source']['request_id'] + '/response']['text'].encode()).hexdigest() == block['perception_source']['response_text_sha256']
    assert 'uncertain sample cue' in str([v['rendered'] for k, v in h.prepared.items() if k[1] == 'slot'])

@pytest.mark.parametrize('target', ['digest', 'source', 'snapshot', 'knowledge', 'branch', 'normalized', 'request', 'inventory'])
def test_sdk_handoff_trace_tamper_rejected(tmp_path, target):
    import copy
    from speech_aware_evidence_acquisition.core.bundle import BundleError
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from test_template_variants_delivery import _runtime, _raw_native
    h = harness(tmp_path)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2', node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1')
    h.program = default_phase_program('phase-native-intent-slot-batched-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    original = h._output
    h._output = lambda key: canonical({'candidate_id': 'play_radio'}) if key[1] == 'solver' else original(key)
    try:
        summary = h.run()
        raw = _raw_native(h, solver_role={'max_iterations': 1}); raw['intent_slot_pipeline'] = policy
        rows = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    finally: h.close()
    records = {rid: payload for _, rid, payload in h.trace.records}
    _verify_outputs_trace_binding(tuple(rows), records)
    row = copy.deepcopy(rows[0]); records = copy.deepcopy(records)
    block = row['native_episode_receipt']['intent_slot_pipeline']
    handoff = block['node_handoffs'][1]
    if target == 'inventory':
        block['evidence_bindings'][0]['inventory']['sources'][0]['source_id'] = 'other-audio'
    else:
        keys = {'digest':'handoff_sha256', 'source':'source_response_sha256',
                'snapshot':'snapshot_sha256', 'branch':'branch_index',
                'normalized':'normalized_sha256', 'request':'target_request_sha256'}
        if target == 'knowledge': handoff['knowledge'].append({'id':'invented'})
        else: handoff[keys[target]] = 99 if target == 'branch' else '0'*64
    with pytest.raises(BundleError): _verify_outputs_trace_binding((row,), records)

@pytest.mark.parametrize('mode', ['batched', 'per_candidate'])
def test_sdk_snapshot_provider_supplies_next_node_and_preserves_raw(tmp_path, mode):
    from dataclasses import replace as dc_replace
    from test_rails_runner import _knowledge_snapshot
    from test_knowledge_stores import _supply
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    snapshot = _knowledge_snapshot()
    calls = []
    def supply(phase, context):
        calls.append((phase, context.sample_id))
        return _supply(snapshot, context)
    h = harness(tmp_path, mode)
    h.supply_for_phase = supply
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2', node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1')
    h.program = default_phase_program('phase-native-intent-slot-' + mode.replace('_', '-') + '-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    original = h._output
    def output(key):
        value = json.loads(original(key))
        if key[1] == 'perception':
            value['candidates'][0]['distinction'] = 'invented model knowledge MUST_NOT_BECOME_RULE'
        if key[1] == 'solver': value = {'candidate_id':'play_radio'}
        return canonical(value)
    h._output = output
    try:
        summary = h.run(snapshot=snapshot)
        raw = _raw_native(h, solver_role={'max_iterations': 1}); raw['intent_slot_pipeline'] = policy
        rows = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    finally: h.close()
    assert [phase for phase, _ in calls] == ['slot','solver']
    records = {rid: payload for _, rid, payload in h.trace.records}
    _verify_outputs_trace_binding(tuple(rows), records)
    receipt = rows[0]['native_episode_receipt']['intent_slot_pipeline']
    for handoff in receipt['node_handoffs'][1:]:
        payload = records[handoff['target_request_id'] + '/request']['payload']
        evidence = payload['supplied_evidence']
        assert any(e['evidence_id'] == snapshot.values[0].value_id and e['content'] == snapshot.values[0].content for e in evidence)
        assert 'MUST_NOT_BECOME_RULE' not in json.dumps(evidence)
        assert handoff['snapshot_sha256'] == snapshot.content_hash()
    assert any('MUST_NOT_BECOME_RULE' in payload.get('text','') for key,payload in records.items() if key.endswith('/response'))

@pytest.mark.parametrize('bad_phase', ['perception', 'slot'])
def test_sdk_hook_rejects_invalid_raw_refs_before_dedup(tmp_path, bad_phase):
    h = harness(tmp_path)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2', node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1')
    h.program = default_phase_program('phase-native-intent-slot-batched-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    original = h._output
    def output(key):
        value = json.loads(original(key))
        if key[1] == bad_phase:
            if bad_phase == 'perception':
                value['candidates'].insert(1, dict(value['candidates'][0], evidence_refs=['invented']))
            else:
                value['candidates'][0]['evidence_refs'] = ['invented', 'invented']
        elif key[1] == 'solver': value = {'candidate_id': 'play_radio'}
        return canonical(value)
    h._output = output
    try: summary = h.run()
    finally: cost = h.close()['cost']
    assert cost['calls_used'] == (1 if bad_phase == 'perception' else 2)
    assert summary.results[0].native_episode_receipt['terminal_disposition']['reason'] == 'seal_refusal'
    assert summary.results[0].final_response_text == ''


def test_node_rails_normalization_preserves_uncertainty_and_multiplicity():
    from speech_aware_evidence_acquisition.rails.candidate_handoff import normalized_response
    p = {'candidates': [{'intent_id':'play_music', 'distinction':'one', 'evidence_refs':[], 'unresolved_fields':['maybe x']},
                        {'intent_id':'play_music', 'distinction':'other', 'evidence_refs':[], 'unresolved_fields':['maybe y']}]}
    result = json.loads(normalized_response('perception', canonical(p), {}, {'max_candidates':3}))
    assert result['candidates'][0]['unresolved_fields'] == ['maybe x','maybe y']
    item = {'type':'person','filler':'Bob'}
    slot = {'candidates':[{'candidate_id':'play_music','entities':[item,item], 'evidence_refs':[], 'unresolved_fields':['uncertain']} ]}
    result = json.loads(normalized_response('slot', canonical(slot), {'candidates':[{'candidate_id':'play_music'}]}, {}))
    assert result['candidates'][0]['entities'] == [item,item]
    assert result['candidates'][0]['unresolved_fields'] == ['uncertain']


@pytest.mark.parametrize('mode', ['batched', 'per_candidate'])
def test_sdk_hooks_with_real_slot_view_projection(tmp_path, mode):
    from test_slot_binding_view import _config as projection_config, bind_slot_view_context
    from speech_aware_evidence_acquisition.rails.slurp_solver_protocol import response_format
    h = harness(tmp_path / 'run', mode)
    cfg = projection_config(h.world.data_root)
    cfg['native_episode']['intent_slot_pipeline'] = {'schema':'saea-intent-slot-pipeline-v2', 'duplicate_candidate_policy':'group_v2'}
    cfg['native_episode']['roles']['slot'] = {'phase_id':'slot','max_iterations':1}
    cfg['knowledge_stores']['slot-contracts']['candidate_supply']['slot'] = cfg['knowledge_stores']['slot-contracts']['candidate_supply']['solver'].copy()
    cfg['native_evidence_projection_params']['consumer_roles'] = ['slot','solver']
    context = bind_slot_view_context(cfg, h.world.data_root)
    from speech_aware_evidence_acquisition.rails.knowledge_stores import build_knowledge_store_supply
    router = build_knowledge_store_supply(context.knowledge_binding,
        program=default_phase_program('phase-native-intent-slot-' + mode.replace('_', '-') + '-v1'),
        data_root=h.world.data_root, legal_candidate_ids=('alpha_one', 'beta_two', 'gamma_three'))
    calls = []
    def supply(phase, current):
        calls.append(phase)
        return router.automatic_for_phase(phase, current)
    h.supply_for_phase = supply
    h.knowledge_store_binding = context.knowledge_binding
    h.transport._native_evidence_projection = 'native-candidate-slot-view-v1'
    h.transport._native_evidence_projection_context = context
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2', node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1')
    h.program = default_phase_program('phase-native-intent-slot-' + mode.replace('_', '-') + '-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    original_material = h.material_for_phase
    def material(phase, ctx, sealed):
        result = original_material(phase, ctx, sealed); fmt = result.response_format
        if phase == 'perception':
            fmt = json.loads(json.dumps(fmt)); fmt['json_schema']['schema']['properties']['candidates']['items']['properties']['intent_id']['enum'] = ['alpha_one', 'beta_two']
        elif phase == 'solver': fmt = response_format({'alpha':['one'], 'beta':['two']}, ['person'])
        return replace(result, response_format=fmt)
    h.material_for_phase = material
    candidate = {'intent_id':'alpha_one', 'distinction':'first', 'evidence_refs':[], 'unresolved_fields':['uncertain']}
    def output(key):
        if key[1] == 'perception': return canonical({'candidates':[candidate, dict(candidate, distinction='other'), dict(candidate, intent_id='beta_two')]})
        if key[1] == 'slot':
            ids = ['alpha_one', 'beta_two'] if mode == 'batched' else [['alpha_one'], ['beta_two']][key[2]-1]
            return canonical({'candidates':[{'candidate_id':cid,'entities':[], 'evidence_refs':[], 'unresolved_fields':[]} for cid in ids]})
        return canonical({'candidate_id':'alpha_one'})
    h._output = output
    try: summary = h.run(snapshot=context.knowledge_binding.composite_snapshot)
    finally: h.close()
    assert calls == ['slot', 'solver']
    rendered_slots = [json.dumps(v['rendered']) for k, v in h.prepared.items() if k[1] == 'slot']
    assert all('Alpha role.' in item or 'Beta role.' in item for item in rendered_slots)
    if mode == 'per_candidate':
        assert 'Alpha role.' in rendered_slots[0] and 'Beta role.' not in rendered_slots[0]
        assert 'Beta role.' in rendered_slots[1] and 'Alpha role.' not in rendered_slots[1]
    assert json.loads(summary.results[0].final_response_text)['action'] == 'one'
    records = {rid: payload for _,rid,payload in h.trace.records}
    projections = [v['org_native_evidence_projection'] for v in records.values() if 'org_native_evidence_projection' in v]
    scopes = [p.get('scope') for p in projections if isinstance(p.get('scope'), dict) and p['scope'].get('role_kind') in ('slot','solver')]
    assert scopes and all(s['fallback_reason'] is None for s in scopes), [(s['role_kind'],s['fallback_reason'],s['candidate_ids']) for s in scopes]
    assert [s['candidate_ids'] for s in scopes if s['role_kind'] == 'slot'] == ([['alpha_one', 'beta_two']] if mode == 'batched' else [['alpha_one'], ['beta_two']])
    assert [s['candidate_ids'] for s in scopes if s['role_kind'] == 'solver'] == [['alpha_one', 'beta_two']]
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    raw = _raw_native(h, solver_role={'max_iterations':1}); raw['intent_slot_pipeline'] = policy
    rows = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    _verify_outputs_trace_binding(tuple(rows), records)


def test_sdk_three_pinned_stores_remain_namespaced(tmp_path):
    from test_knowledge_stores import _config, _supply
    from speech_aware_evidence_acquisition.rails.knowledge_stores import load_knowledge_store_binding
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    store_config, _ = _config(tmp_path / 'stores', ('intent-boundaries','slot-contracts','train-exemplars'))
    store_binding = load_knowledge_store_binding(store_config, data_root=tmp_path / 'stores')
    snapshot = store_binding.composite_snapshot
    h = harness(tmp_path / 'run')
    h.knowledge_store_binding = store_binding
    h.supply_for_phase = lambda phase, context: _supply(snapshot, context)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2', node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1')
    h.program = default_phase_program('phase-native-intent-slot-batched-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    original = h._output
    h._output = lambda key: canonical({'candidate_id':'play_radio'}) if key[1] == 'solver' else original(key)
    try:
        summary = h.run(snapshot=snapshot)
        raw = _raw_native(h, solver_role={'max_iterations': 1}); raw['intent_slot_pipeline'] = policy
        rows = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    finally: h.close()
    records = {rid: payload for _,rid,payload in h.trace.records}
    _verify_outputs_trace_binding(tuple(rows), records)
    handoffs = rows[0]['native_episode_receipt']['intent_slot_pipeline']['node_handoffs']
    for handoff in handoffs:
        assert set(handoff['store_pins']) == {'intent-boundaries','slot-contracts','train-exemplars'}
        assert all(pin['composite_snapshot_sha256'] == snapshot.content_hash() for pin in handoff['store_pins'].values())
    assert any('::' in item['id'] for handoff in handoffs for item in handoff['knowledge'])


def test_sdk_hooks_two_samples_isolated_under_parallel_dispatch(tmp_path):
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    h = harness(tmp_path, samples=2)
    h.slot_count = 2
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2', node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1')
    h.program = default_phase_program('phase-native-intent-slot-batched-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    original = h._output
    def output(key):
        value = json.loads(original(key))
        if key[1] == 'slot':
            for row in value['candidates']: row['entities'][0]['filler'] = key[0]
        elif key[1] == 'solver': value = {'candidate_id':'play_radio'}
        return canonical(value)
    h._output = output
    try:
        summary = h.run()
        raw = _raw_native(h, solver_role={'max_iterations': 1}); raw['intent_slot_pipeline'] = policy
        rows = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    finally: cost = h.close()['cost']
    assert cost['calls_used'] == 6
    records = {rid: payload for _,rid,payload in h.trace.records}
    _verify_outputs_trace_binding(tuple(rows), records)
    for row in rows:
        assert json.loads(row['text'])['entities'][0]['filler'] == row['sample_id']
        assert all(item['sample_id'] == row['sample_id'] for item in row['native_episode_receipt']['intent_slot_pipeline']['node_handoffs'])


def test_sdk_before_invalid_handoff_stops_before_any_model_contact(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.reproduction.ojw import native_roles
    original = native_roles._RoleAuthorityRail.before_model_call
    async def corrupt(self, ctx):
        original_before = self._candidate_before
        if original_before is not None:
            def invalid(role, context):
                raise ValueError('invalid handoff identity')
            self._candidate_before = invalid
            try:
                return await original(self, ctx)
            finally:
                self._candidate_before = original_before
        return await original(self, ctx)
    monkeypatch.setattr(native_roles._RoleAuthorityRail, 'before_model_call', corrupt)
    h = harness(tmp_path)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2', node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1')
    h.program = default_phase_program('phase-native-intent-slot-batched-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    try:
        with pytest.raises(Exception): h.run()
    finally:
        cost = h.close()['cost']
    assert cost['calls_used'] == 0
    assert h.dispatched == {}


def test_sdk_shared_snapshot_absent_and_tampered_fail_replay(tmp_path):
    from pathlib import Path
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from test_template_variants_delivery import _runtime, _raw_native
    h = harness(tmp_path)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2', node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1')
    h.program = default_phase_program('phase-native-intent-slot-batched-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    output = h._output
    h._output = lambda key: canonical({'candidate_id':'play_radio'}) if key[1] == 'solver' else output(key)
    try:
        summary = h.run()
        raw = _raw_native(h, solver_role={'max_iterations':1}); raw['intent_slot_pipeline'] = policy
        rows = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    finally: h.close()
    records = {rid: payload for _, rid, payload in h.trace.records}
    node = next(v for v in records.values() if 'org_candidate_node_handoffs' in v)
    assert 'org_knowledge_snapshot' not in node
    path = Path(node['org_knowledge_snapshot_ref']['path'])
    _verify_outputs_trace_binding(tuple(rows), records)
    original = path.read_text()
    path.write_text('{}')
    with pytest.raises(Exception): _verify_outputs_trace_binding(tuple(rows), records)
    path.unlink()
    with pytest.raises(Exception): _verify_outputs_trace_binding(tuple(rows), records)
    path.write_text(original)
    _verify_outputs_trace_binding(tuple(rows), records)


@pytest.mark.parametrize('fallback,expected_calls', [('budgeted-v1', 4), ('off', 1)])
def test_sdk_budgeted_fallback_length_and_empty_preserves_cost(tmp_path, fallback, expected_calls):
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from test_native_episode_runner import _plan, _wire_response
    plan = _plan(run_id='SAEA-E-native-role-runner-proof', attempt_id='attempt-native-role-runner-proof', planned_model_calls=6)
    h = harness(tmp_path, plan=plan)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2', node_rails='sdk-handoff-v1', answer_only_fallback=fallback)
    h.program = default_phase_program('phase-native-intent-slot-batched-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    requests = []
    def post(_url, body):
        request = json.loads(body); requests.append(request)
        if len(requests) == 1:
            return json.dumps({'choices':[{'message':{'role':'assistant','content':''},'finish_reason':'length'}],
                               'usage':{'prompt_tokens':7,'completion_tokens':64}}).encode()
        phase = ('perception' if len(requests) == 2 else 'slot' if len(requests) == 3 else 'solver')
        text = h._output(('s1', phase, 1, 1)) if phase != 'solver' else canonical({'candidate_id':'play_radio'})
        return _wire_response(text)
    h.post_override = post
    h.transport._reasoning_content_guard = 'refuse-think-marker-in-content'
    try:
        summary = h.run(decoding_params={'temperature':0, 'max_tokens':64, 'thinking_budget_tokens':32,
                                          'chat_template_kwargs':{'enable_thinking':True}})
        raw = _raw_native(h, solver_role={'max_iterations':1}); raw['intent_slot_pipeline'] = policy
        rows = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    finally: cost = h.close()['cost']
    assert cost['calls_used'] == expected_calls == len(requests)
    records = {rid: payload for _, rid, payload in h.trace.records}
    _verify_outputs_trace_binding(tuple(rows), records)
    fallbacks = [p for channel, rid, p in h.trace.records if channel == 'TOOL' and rid.endswith('/reasoning-fallback')]
    if fallback == 'budgeted-v1':
        assert len(fallbacks) == 1 and fallbacks[0]['first_finish_reason'] == 'length'
        assert len(h.fit_calls) == 3
        assert json.loads(rows[0]['text'])['action'] == 'radio'
    else:
        assert not fallbacks and rows[0]['text'] == ''

@pytest.mark.parametrize('extra,expected_calls', [(3, 5), (1, 3)])
def test_sdk_coverage_answer_only_then_slot_shared_budget(tmp_path, extra, expected_calls):
    from test_native_episode_runner import _wire_response
    from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import validate_receipt
    h, policy = recovery_harness(tmp_path)
    policy['answer_only_fallback'] = 'budgeted-v1'
    policy['coverage_recovery']['max_extra_contacts'] = extra
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy), max_contacts=3 + extra)
    requests = []
    def post(_url, body):
        request = json.loads(body); requests.append(request)
        if len(requests) == 1:
            return json.dumps({'choices':[{'message':{'role':'assistant','content':''},'finish_reason':'length'}],
                               'usage':{'prompt_tokens':7,'completion_tokens':64}}).encode()
        phase, visit = {2: ('perception', 1), 3: ('slot', 1), 4: ('slot', 2), 5: ('solver', 1)}[len(requests)]
        return _wire_response(h._output(('s1', phase, visit, 1)))
    h.post_override = post
    h.transport._reasoning_content_guard = 'refuse-think-marker-in-content'
    try:
        result = h.run(decoding_params={'temperature':0, 'max_tokens':64, 'thinking_budget_tokens':32,
                                          'chat_template_kwargs':{'enable_thinking':True}})
    finally:
        cost = h.close()['cost']
    assert cost['calls_used'] == expected_calls == len(requests)
    assert len(h.fit_gate.receipts) == expected_calls
    receipt = result.results[0].native_episode_receipt
    if extra == 1:
        assert receipt['terminal_disposition']['reason'] == 'seal_refusal'
        return
    assert json.loads(result.results[0].final_response_text)['entities'][0]['filler'] == 'Bob'
    assert receipt['intent_slot_pipeline']['candidate_ids'] == ['play_music', 'play_radio']
    assert len(receipt['role_visits'][0]['execution_receipt']['contacts']) == 2
    records = {rid:p for _, rid, p in h.trace.records}
    validate_receipt(receipt, trace_records=records)
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    raw = _raw_native(h, solver_role={'max_iterations': 1}); raw['intent_slot_pipeline'] = policy
    rows = _runtime(h, result, raw)._canonical_output_rows(h.client)
    _verify_outputs_trace_binding(tuple(rows), records)
    import copy
    contacts = receipt['role_visits'][0]['execution_receipt']['contacts']
    for contact in contacts:
        tampered = copy.deepcopy(records)
        tampered[contact['request_id'] + '/response']['text'] += ' '
        with pytest.raises(ValueError):
            validate_receipt(receipt, trace_records=tampered)
    for source in receipt['intent_slot_pipeline']['slot_sources']:
        tampered = copy.deepcopy(records)
        tampered[source['source']['request_id'] + '/response']['text'] += ' '
        with pytest.raises(ValueError):
            validate_receipt(receipt, trace_records=tampered)


def test_sdk_answer_only_unknown_response_not_reissued(tmp_path):
    from test_native_episode_runner import _wire_response
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    old, policy = recovery_harness(tmp_path / 'old')
    policy['answer_only_fallback'] = 'budgeted-v1'
    old.config = replace(old.config, intent_slot_pipeline_json=canonical(policy))
    count = 0
    def post(_url, body):
        nonlocal count
        count += 1
        if count == 1:
            return json.dumps({'choices':[{'message':{'role':'assistant','content':''},'finish_reason':'length'}],
                               'usage':{'prompt_tokens':7,'completion_tokens':64}}).encode()
        raise TimeoutError('unknown answer-only response')
    old.post_override = post
    old.transport._reasoning_content_guard = 'refuse-think-marker-in-content'
    decoding = {'temperature':0, 'max_tokens':64, 'thinking_budget_tokens':32,
                'chat_template_kwargs':{'enable_thinking':True}}
    try:
        with pytest.raises(Exception): old.run(decoding_params=decoding)
    finally:
        old_cost = old.close()['cost']
    assert old_cost['calls_used'] == 2
    old.client._adapter._attempt_handle.mark_aborted('isolated unknown-response test')
    plan = replace(old.plan, attempt_id='answer-only-unknown-successor')
    replay = NativeContactReplay(old.world.data_root, old.plan, plan, old.client.arm)
    new, _ = recovery_harness(tmp_path / 'new', plan=plan, data=old.world.data_root, replay=replay)
    new.config = replace(new.config, intent_slot_pipeline_json=canonical(policy))
    new.transport._reasoning_content_guard = 'refuse-think-marker-in-content'
    new.post_override = lambda *_: pytest.fail('unknown request must never be reissued')
    try:
        with pytest.raises(Exception): new.run(decoding_params=decoding)
    finally:
        cost = new.close()['cost']
    assert cost['calls_used'] == 0


def test_sdk_invalid_answer_only_keeps_both_paid_contacts(tmp_path):
    from test_native_episode_runner import _wire_response
    from test_template_variants_delivery import _runtime, _raw_native
    h, policy = recovery_harness(tmp_path)
    policy['answer_only_fallback'] = 'budgeted-v1'
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    calls = 0
    def post(_url, _body):
        nonlocal calls
        calls += 1
        if calls == 1:
            return json.dumps({'choices':[{'message':{'role':'assistant','content':''},'finish_reason':'length'}],
                               'usage':{'prompt_tokens':7,'completion_tokens':64}}).encode()
        return _wire_response('{}')
    h.post_override = post
    h.transport._reasoning_content_guard = 'refuse-think-marker-in-content'
    try:
        result = h.run(decoding_params={'temperature':0, 'max_tokens':64, 'thinking_budget_tokens':32,
                                         'chat_template_kwargs':{'enable_thinking':True}})
        raw = _raw_native(h, solver_role={'max_iterations':1}); raw['intent_slot_pipeline'] = policy
        rows = _runtime(h, result, raw)._canonical_output_rows(h.client)
    finally:
        cost = h.close()['cost']
    assert calls == cost['calls_used'] == 2
    assert len(h.fit_gate.receipts) == 2
    assert rows[0]['native_episode_receipt']['terminal_disposition']['reason'] == 'seal_refusal'
    assert len(rows[0]['native_episode_receipt']['interrupted_role']['execution_receipt']['contacts']) == 2


def test_node_rails_preserves_wire_system_with_trailing_newline(tmp_path):
    h = harness(tmp_path)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2',
                  duplicate_candidate_policy='group_v2', node_rails='sdk-handoff-v1',
                  answer_only_fallback='budgeted-v1')
    h.program = default_phase_program('phase-native-intent-slot-batched-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    original = h.adapter.assemble_prompt
    def assemble(sample, supply):
        system, user = original(sample, supply)
        return ({**system, 'content': f"fake task\n\nTask contract:\ntask-contract-for-{sample.sample_id}\n"}, user)
    h.adapter.assemble_prompt = assemble
    try:
        result = h.run()
    finally:
        cost = h.close()['cost']
    assert len(result.results) == 1
    assert cost['calls_used'] == 3
    assert all(value['rendered']['messages'][0]['content'].endswith('\n')
               for value in h.prepared.values())



@pytest.mark.parametrize('bad_phase', ['perception', 'slot'])
@pytest.mark.parametrize('coverage', [False, True])
def test_sdk_quarantines_unsupplied_refs_with_source_bound_answer(tmp_path, bad_phase, coverage):
    """Scripted SDK transport; original response, FIT stub and bundle trace stay visible."""
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    h = recovery_harness(tmp_path)[0] if coverage else harness(tmp_path)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2',
                  node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1',
                  reference_normalizer='quarantine-unsupplied-v1')
    if coverage:
        policy['coverage_recovery'] = {'strategy':'complete-original-candidate-set-v1', 'max_extra_contacts':3}
    if not coverage:
        h.program = default_phase_program('phase-native-intent-slot-batched-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy), max_contacts=6 if coverage else 3)
    original = h._output
    def output(key):
        value = json.loads(original(key))
        if key[1] == bad_phase:
            value['candidates'][-1]['evidence_refs'] = ['slot-contracts::unsupplied']
        if key[1] == 'solver': value = {'candidate_id':'play_radio'}
        return canonical(value)
    h._output = output
    try:
        summary = h.run()
        raw = _raw_native(h, solver_role={'max_iterations':1}); raw['intent_slot_pipeline'] = policy
        rows = _runtime(h, summary, raw)._canonical_output_rows(h.client)
    finally:
        cost = h.close()['cost']
    records = {rid: payload for _,rid,payload in h.trace.records}
    _verify_outputs_trace_binding(tuple(rows), records)
    assert cost['calls_used'] == (4 if coverage else 3)
    assert json.loads(rows[0]['text']) == {'scenario':'play','action':'radio','entities':[{'type':'person','filler':'Bob'}]}
    block = rows[0]['native_episode_receipt']['intent_slot_pipeline']
    binding = next(b for b in block['evidence_bindings'] if b['stage'] == bad_phase)
    assert any(x['raw'] == 'slot-contracts::unsupplied' and x['status'] == 'quarantined'
               for x in binding['reference_normalization']['mappings'])
    import copy
    damaged = copy.deepcopy(records)
    trace = next(v['use_native_episode'] for v in damaged.values() if isinstance(v,dict) and v.get('use_native_episode',{}).get('intent_slot_pipeline'))
    trace['intent_slot_pipeline']['evidence_bindings'][0]['reference_normalization']['normalized_sha256'] = '0'*64
    with pytest.raises(Exception): _verify_outputs_trace_binding(tuple(rows), damaged)


def test_reference_normalizer_rejects_wrong_scope_and_quarantines_ambiguous_alias():
    from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import normalize_evidence_references
    inv={'sample_id':'s','candidate_ids':['play_radio'],'sources':[{'source_id':'a::x','kind':'knowledge'}, {'source_id':'b::x','kind':'knowledge'}]}
    text=canonical({'candidates':[{'candidate_id':'play_radio','entities':[{'type':'person','filler':'Bob'}], 'evidence_refs':['x','a::x','missing'], 'unresolved_fields':[]}]})
    clean, audit=normalize_evidence_references(text,inv,['play_radio'],policy='unique_source_alias_v1',normalizer='quarantine-unsupplied-v1')
    assert json.loads(clean)['candidates'][0]['evidence_refs']==['a::x']
    assert [r['status'] for r in audit]==['quarantined','supplied','quarantined']
    assert json.loads(clean)['candidates'][0]['entities']==[{'type':'person','filler':'Bob'}]
    with pytest.raises(ValueError): normalize_evidence_references(text,inv,['other'],policy='strict',normalizer='quarantine-unsupplied-v1')
    malformed, _ = normalize_evidence_references('{"candidates":[{"entities":[]}]}',inv,['play_radio'],policy='strict',normalizer='quarantine-unsupplied-v1')
    assert json.loads(malformed)["candidates"] == [{"entities":[]}]


def test_sdk_normalizer_answer_only_fallback_preserves_original_and_cost(tmp_path):
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from test_native_episode_runner import _plan, _wire_response
    plan = _plan(run_id='SAEA-E-native-role-runner-proof',attempt_id='attempt-native-role-runner-proof',planned_model_calls=6)
    h=harness(tmp_path,plan=plan)
    policy=json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2',duplicate_candidate_policy='group_v2',node_rails='sdk-handoff-v1',answer_only_fallback='budgeted-v1',reference_normalizer='quarantine-unsupplied-v1')
    h.program=default_phase_program('phase-native-intent-slot-batched-v2');h.config=replace(h.config,intent_slot_pipeline_json=canonical(policy))
    count=0
    def post(_url,_body):
        nonlocal count
        count+=1
        if count==1:
            return json.dumps({'choices':[{'message':{'role':'assistant','content':''},'finish_reason':'length'}], 'usage':{'prompt_tokens':7,'completion_tokens':64}}).encode()
        phase='perception' if count==2 else 'slot' if count==3 else 'solver'
        value=json.loads(h._output(('s1',phase,1,1))) if phase!='solver' else {'candidate_id':'play_radio'}
        if phase=='perception':value['candidates'][1]['evidence_refs']=['missing-source']
        return _wire_response(canonical(value))
    h.post_override=post;h.transport._reasoning_content_guard='refuse-think-marker-in-content'
    try:
        summary=h.run(decoding_params={'temperature':0,'max_tokens':64,'thinking_budget_tokens':32,'chat_template_kwargs':{'enable_thinking':True}})
        raw=_raw_native(h,solver_role={'max_iterations':1});raw['intent_slot_pipeline']=policy
        rows=_runtime(h,summary,raw)._canonical_output_rows(h.client)
    finally:cost=h.close()['cost']
    records={rid:payload for _,rid,payload in h.trace.records}
    _verify_outputs_trace_binding(tuple(rows),records)
    assert cost['calls_used']==count==4
    assert json.loads(rows[0]['text'])['action']=='radio'
    assert any(r.get('status')=='quarantined' for r in rows[0]['native_episode_receipt']['intent_slot_pipeline']['evidence_bindings'][0]['reference_normalization']['mappings'])

@pytest.mark.parametrize('repair_succeeds', [True, False])
def test_sdk_normalizer_does_not_repair_business_fields(tmp_path, repair_succeeds):
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    h,policy=recovery_harness(tmp_path)
    policy.update(answer_only_fallback='budgeted-v1',reference_normalizer='quarantine-unsupplied-v1',
                  coverage_recovery={'strategy':'complete-original-candidate-set-v1','max_extra_contacts':3})
    h.config=replace(h.config,intent_slot_pipeline_json=canonical(policy))
    original=h._output
    def output(key):
        value=json.loads(original(key))
        if key[1]=='slot':
            if key[2] > 1:
                cid = 'play_music' if key[2] == 2 else 'play_radio'
                value['candidates'][0] = {'candidate_id':cid, 'entities':[{'type':'person','filler':'Alice' if cid == 'play_music' else 'Bob'}], 'evidence_refs':[], 'unresolved_fields':[]}
            value['candidates'][0]['evidence_refs']=['not-supplied']
            if key[2]==1 or not repair_succeeds: value['candidates'][0].pop('entities')
        return canonical(value)
    h._output=output
    try:
        summary=h.run()
        raw=_raw_native(h,solver_role={'max_iterations':1});raw['intent_slot_pipeline']=policy
        rows=_runtime(h,summary,raw)._canonical_output_rows(h.client)
    finally: cost=h.close()['cost']
    records={rid:payload for _,rid,payload in h.trace.records}
    _verify_outputs_trace_binding(tuple(rows),records)
    if repair_succeeds:
        assert cost['calls_used']==5
        assert json.loads(rows[0]['text'])['entities']==[{'type':'person','filler':'Bob'}]
    else:
        assert cost['calls_used']==3
        assert rows[0]['text']==''
        assert rows[0]['native_episode_receipt']['terminal_disposition']['reason']=='seal_refusal'


def test_sdk_rejects_normalized_source_changed_before_settlement(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.rails import candidate_handoff
    original = candidate_handoff.normalized_response
    count = 0
    def corrupt_second(kind, text, context, config):
        nonlocal count
        result = original(kind, text, context, config)
        if kind == 'perception':
            count += 1
            if count == 2:
                return result + ' '
        return result
    monkeypatch.setattr(candidate_handoff, 'normalized_response', corrupt_second)
    h = harness(tmp_path)
    policy = json.loads(h.config.intent_slot_pipeline_json)
    policy.update(schema='saea-intent-slot-pipeline-v2', duplicate_candidate_policy='group_v2',
                  node_rails='sdk-handoff-v1', answer_only_fallback='budgeted-v1',
                  reference_normalizer='quarantine-unsupplied-v1')
    h.program = default_phase_program('phase-native-intent-slot-batched-v2')
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    try:
        with pytest.raises(Exception, match='candidate normalized source changed'):
            h.run()
    finally:
        cost = h.close()['cost']
    assert count == 2 and cost['calls_used'] == 1


def test_sdk_nonempty_perception_length_repair_reaches_decision(tmp_path):
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from test_native_episode_runner import _wire_response
    h, policy = recovery_harness(tmp_path)
    h.repo_root = __import__('pathlib').Path(__file__).resolve().parents[2]
    policy.update(answer_only_fallback='off', perception_length_recovery='compact-reask-v1',
                  reference_normalizer='quarantine-unsupplied-v1',
                  coverage_recovery={'strategy':'complete-original-candidate-set-v1','max_extra_contacts':3})
    h.config=replace(h.config,intent_slot_pipeline_json=canonical(policy))
    calls=[]
    def post(_url,body):
        calls.append(json.loads(body))
        if len(calls)==1:
            response=json.loads(_wire_response('{"candidates":[{"intent_id":"play_radio","distinction":"short"'+(' '*2000)))
            response['choices'][0]['finish_reason']='length'
            return json.dumps(response).encode()
        phase='perception' if len(calls)==2 else 'slot' if len(calls) in (3,4) else 'solver'
        if phase=='solver': text=canonical({'candidate_id':'play_radio'})
        else: text=h._output(('s1',phase,1 if len(calls)==3 else 2,1))
        return _wire_response(text)
    h.post_override=post
    try:
        summary=h.run(decoding_params={'temperature':0,'max_tokens':64})
        raw=_raw_native(h,solver_role={'max_iterations':1});raw['intent_slot_pipeline']=policy
        rows=_runtime(h,summary,raw)._canonical_output_rows(h.client)
    finally: cost=h.close()['cost']
    records={rid:payload for _,rid,payload in h.trace.records}
    _verify_outputs_trace_binding(tuple(rows),records)
    assert cost['calls_used']==5
    assert len(h.fit_gate.receipts)==5
    assert json.loads(rows[0]['text'])['action']=='radio'
    visit=rows[0]['native_episode_receipt']['role_visits'][0]
    assert len(visit['execution_receipt']['contacts'])==2
    assert 'previous Perception response stopped' in json.dumps(calls[1])
    import copy
    damaged=copy.deepcopy(records)
    parent=visit['execution_receipt']['contacts'][0]
    damaged[parent['request_id']+'/wire-response']['finish_reason']='stop'
    with pytest.raises(Exception, match='Perception repair parent'):
        _verify_outputs_trace_binding(tuple(rows),damaged)

@pytest.mark.parametrize('failure', ['child_length', 'child_invalid', 'unknown'])
def test_sdk_perception_length_repair_never_fabricates_or_reissues(tmp_path, failure):
    from test_native_episode_runner import _wire_response
    h,policy=recovery_harness(tmp_path)
    h.repo_root = __import__('pathlib').Path(__file__).resolve().parents[2]
    policy.update(answer_only_fallback='off',perception_length_recovery='compact-reask-v1',
                  coverage_recovery={'strategy':'complete-original-candidate-set-v1','max_extra_contacts':1})
    h.config=replace(h.config,intent_slot_pipeline_json=canonical(policy),max_contacts=4)
    # Harness uses a six-contact fixture plan and FIT ceiling; keep its graph
    # geometry and reduce only the shared extra-call budget in the config.
    h.config=replace(h.config,max_contacts=6)
    calls=0
    def post(_url,_body):
        nonlocal calls
        calls+=1
        if calls==2 and failure=='unknown': raise RuntimeError('unknown response transport')
        result=json.loads(_wire_response('{"candidates":'+(' '*100)))
        result['choices'][0]['finish_reason']='stop' if calls==2 and failure=='child_invalid' else 'length'
        return json.dumps(result).encode()
    h.post_override=post
    try:
        if failure=='unknown':
            with pytest.raises(Exception):h.run()
            journal=h.client._adapter._contact_journal
            import json as json_module
            intents=[json_module.loads(path.read_bytes()) for path in journal.root.glob('*.intent.json')]
            assert len(intents)==2
            states=[journal.inspect(item['request_id']) for item in intents]
            assert sorted(item['state'] for item in states)==['observed','unknown']
            unknown=next(item for item in states if item['state']=='unknown')
            assert unknown['intent']['agent_contact_id'].endswith(':perception-length-repair')
            assert 'previous Perception response stopped' in unknown['intent']['payload']['task_instruction']
        else:
            summary=h.run()
            assert summary.results[0].final_response_text==''
            receipt=summary.results[0].native_episode_receipt
            assert receipt['terminal_disposition']['reason']==('seal_refusal' if failure=='child_invalid' else 'finite_completion')
            from test_template_variants_delivery import _runtime, _raw_native
            from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
            raw=_raw_native(h,solver_role={'max_iterations':1});raw['intent_slot_pipeline']=policy
            rows=_runtime(h,summary,raw)._canonical_output_rows(h.client)
            records={rid:payload for _,rid,payload in h.trace.records}
            _verify_outputs_trace_binding(tuple(rows),records)
            from speech_aware_evidence_acquisition.analysis.native_episode_diagnostics import analyze_verified_native_episode_statistics
            analyze_verified_native_episode_statistics(sample_ids=('s1',),controller_policy_id=h.program.program_id,
                phase_results={'s1':summary.results[0].as_json()}, output_rows={'s1':rows[0]}, trace_records=records)
            contacts=receipt['interrupted_role']['execution_receipt']['contacts']
            assert len(contacts)==2
            for item in contacts:
                assert item['request_id']+'/request' in records
                assert item['request_id']+'/response' in records
                assert item['request_id']+'/cost' in records
    finally: cost=h.close()['cost']
    assert calls==2
    assert cost['calls_used'] >= 1


def test_perception_repair_template_pin_rejects_mutation_and_escape(tmp_path):
    import hashlib
    from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import load_repair_template
    path=tmp_path/'configs/probes/repair.txt';path.parent.mkdir(parents=True);path.write_text('Fixed diagnostic.\n')
    pin={'relpath':'configs/probes/repair.txt','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert load_repair_template(tmp_path,{'perception_repair_template':pin})=='Fixed diagnostic.\n'
    path.write_text('modified')
    with pytest.raises(ValueError,match='pin differs'):load_repair_template(tmp_path,{'perception_repair_template':pin})
    pin['relpath']='configs/probes/../../secret'
    with pytest.raises(ValueError):load_repair_template(tmp_path,{'perception_repair_template':pin})
