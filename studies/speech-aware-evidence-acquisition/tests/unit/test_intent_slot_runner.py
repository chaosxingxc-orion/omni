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


def harness(tmp_path, mode='batched', *, plan=None, data=None, replay=None, failure=None):
    cap = 3 if mode == 'batched' else 5
    h = _ActualEpisodeHarness(tmp_path, sample_count=1, max_revisions=0, max_contacts=cap,
                              planner_actions=(), arm='raw-speech-agent-native-tools', query_source='speech_audio',
                              contact_journal=True, execution_plan=plan, data_root=data, contact_replay=replay)
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
    runtime._composed.values['native_contact_fit']['contact_cap']=cap
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
