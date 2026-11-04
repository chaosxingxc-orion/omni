"""Distinct P graph/contract, protocol isolation, teacher routing and custody."""
import copy
import json
from pathlib import Path

import pytest
from speech_aware_evidence_acquisition.rails import perception_prompt_candidate as p
from speech_aware_evidence_acquisition.rails import solver_prompt_candidate as s
from speech_aware_evidence_acquisition.rails import gepa_prompt_material as material
from speech_aware_evidence_acquisition.rails.external_exploration import validate_proposals
from test_gepa_prompt_material import pair_inputs
from test_llama_server_transport import _spawn_live_matching_process


def inputs():
    root = Path(__file__).parents[2]
    original = (root/'configs/probes/slurp_native_role_perception_v2.txt').read_text()
    first, rest = original.split('\n\n', 1)
    count = lambda text: len(text.split())  # fixture only
    tokenizer = {'rendering': {'mode':'text','add_generation_prompt':False,
        'add_special_tokens':False,'truncation':False}, 'test_counter':'words'}
    frozen = {key:p.sha(key) for key in p.FROZEN_FIELDS}
    frozen['tokenizer_sha256'] = p.sha(tokenizer)
    contract = p.build_contract(base_template=p.BLOCK_START+first+p.BLOCK_END+'\n\n'+rest,
        frozen=frozen, feedback_query_ids=['slurp:train:1'], excluded_query_ids=['slurp:train:2'])
    args = dict(contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
    parent = p.seed_candidate(**args)
    mutation = {'schema':p.MUTATION_SCHEMA,'module':p.MODULE,
        'parent_sha256':parent['candidate_sha256'],'evidence_query_ids':['slurp:train:1'],
        'diagnosis':'A supplied boundary was not reflected in the candidate set.',
        'perception_instruction':'Compare requested operation and object against supplied boundaries. Retain up to three distinct speech-supported legal alternatives before narrowing.'}
    text = json.dumps(mutation)
    child = p.derive_candidate(text,parent=parent,expected_parent_sha256=parent['candidate_sha256'],**args)
    common = dict(args, tokenizer_identity=tokenizer, expected_tokenizer_sha256=p.sha(tokenizer))
    return original, mutation, dict(common,candidate=parent), dict(common,candidate=child,
        parent=parent,expected_parent_sha256=parent['candidate_sha256'],mutation_text=text)


def test_node_contract_preserves_protocol_and_frozen_solver():
    original, mutation, parent, child = inputs()
    assert 'solver_sha256' in parent['contract']['frozen']
    assert 'perception_sha256' not in parent['contract']['frozen']
    args = {k:v for k,v in parent.items() if not k.startswith('tokenizer') and k!='expected_tokenizer_sha256'}
    assert p.executable_template(**args) == original
    args = {k:v for k,v in child.items() if not k.startswith('tokenizer') and k!='expected_tokenizer_sha256'}
    output = p.executable_template(**args)
    assert output == mutation['perception_instruction']+'\n\n'+original.split('\n\n',1)[1]
    assert 'maxItems' not in mutation['perception_instruction']
    with pytest.raises(ValueError):
        s.validate_contract(parent['contract'],expected_contract_sha256=parent['expected_contract_sha256'])
    bad = copy.deepcopy(child['candidate']);bad['full_template'] += 'Change JSON schema'
    with pytest.raises(ValueError): p.executable_template(**{**args,'candidate':bad})


def test_teacher_validates_explicit_perception_contract_and_feedback_isolation():
    _, mutation, parent, _ = inputs()
    second={**mutation,'perception_instruction':'Keep legal alternatives supported by distinct supplied boundaries.'}
    profile={'algorithm':'gepa','min_candidates':2,'max_candidates':6}
    context={'contract':parent['contract'],'parent':parent['candidate']}
    doc={'algorithm':'gepa','candidates':[mutation,second]}
    assert validate_proposals(json.dumps(doc),profile,context,['slurp:train:1'])==doc
    second['evidence_query_ids']=['slurp:train:2']
    with pytest.raises(ValueError):validate_proposals(json.dumps(doc),profile,context,['slurp:train:1'])


def test_native_perception_pair_four_independent_p_s_arms(tmp_path):
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    base, setup = pair_inputs(tmp_path)
    original, _, parent, child = inputs()
    setup.update(parent_arguments=parent,child_arguments=child,
        delivering_template_tokens=parent['count_tokens'](original))
    result=material.compile_template_pair(tmp_path,base,**setup)
    built=result['baseline'];built.update(transport='vllm',carrier_lock_key='slurp')
    assert result['program_ceiling']==8
    assert built['native_episode']['roles']['solver']==base['native_episode']['roles']['solver']
    assert built['phase_material']==base['phase_material']
    native=load_native_episode_config(built,default_phase_program(built['policy']))
    assert native.perception_branch_group.branch_ids==('parent','child','aa','lock')
    assert native.perception_branch_group.level==2
    assert native.max_contacts==8
    pin = native.perception_branch_group.branch('child').instruction_template
    assert (tmp_path/'configs'/pin.relpath).read_text().startswith(child['candidate']['perception_instruction'])
    assert native_contact_custody_enabled(built)
    for fault in ('solver','knowledge','protocol','arm','ceiling'):
        bad=copy.deepcopy(built)
        if fault=='solver':bad['native_episode']['roles']['solver']['max_iterations']=2
        elif fault=='knowledge':bad['knowledge_stores']={}
        elif fault=='protocol':bad['phase_material']['perception']['response_format_id']='other'
        elif fault=='arm':bad['native_episode']['roles']['perception']['template_variants']['arms'][1]['instruction_template']['sha256']='0'*64
        else:bad['policy']='phase-native-module-v1-c4'
        assert not native_contact_custody_enabled(bad),fault


def test_actual_request_audit_rejects_shared_perception_and_changed_solver():
    from speech_aware_evidence_acquisition.rails.gepa_perception_request_audit import verify_inputs,ARMS
    templates={a:'P '+a for a in ARMS};requests={};responses={};sid='slurp:train:1'
    def request(rid,node,instruction,history):
        control={'contact_id':rid,'sample_id':sid,'role_id':node,
            'messages':[{'role':'system','content':'task'},{'role':'user','content':instruction+'\n\nNative episode limits: '+json.dumps({'remaining_model_contacts':17 if node=='perception' else 9},sort_keys=True)}]}
        text=json.dumps(control)
        return {'sample_id':sid,'request_id':rid,'task_instruction':'task','decoding_params':{'temperature':0},
            'raw_speech_query':{'sha256':'a'*64},'supplied_evidence':[],'evidence_provenance':[],
            'history':history+[{'role':'control','contact_id':rid,'content':text,'content_sha256':p.text_sha(text)}]}
    for arm in ARMS:
        text='note '+arm;digest=p.text_sha(text)
        responses[arm]={'text':text,'response_text_sha256':digest}
        state={'role':'agent_state','state_kind':'phase:perception','content':text,'content_sha256':digest,'source_response_sha256':digest}
        requests[arm]={'perception':request('p-'+arm,'perception',templates[arm],[]),
            'solver':request('s-'+arm,'solver','S',[state])}
    assert verify_inputs(sid,requests,responses,templates=templates,solver_template='S')['primary_contacts']==8
    for fault in ('shared_p','wrong_note','decoding','audio','template','knowledge'):
        bad=copy.deepcopy(requests)
        if fault=='shared_p':bad['child']['perception']=bad['parent']['perception']
        elif fault=='wrong_note':bad['child']['solver']['history'][0]=bad['parent']['solver']['history'][0]
        elif fault=='decoding':bad['child']['solver']['decoding_params']['temperature']=1
        elif fault=='audio':bad['child']['perception']['raw_speech_query']={'sha256':'b'*64}
        elif fault=='template':bad['child']['perception']['history'][-1]['content']+='changed'
        else:bad['child']['perception']['supplied_evidence']=[{'content':'injected'}]
        with pytest.raises(ValueError):verify_inputs(sid,bad,responses,templates=templates,solver_template='S')


def test_perception_workload_width_and_budget_are_explicit():
    from gepa_perception_workload import PerceptionWorkload,SCHEMA,KIND
    from test_gepa_pair_workload import config
    from registered_workload import assert_execution_binding
    cfg={**config(),'schema':SCHEMA}
    job=PerceptionWorkload(cfg)
    assert job.CONTACTS==8 and job.ARMS==['parent','child','aa','lock'] and job.KIND==KIND
    spec={'workload_kind':KIND,'job_config':cfg,'execution':cfg['execution'],'profile':'profile'}
    assert_execution_binding(spec,{'SAEA_ADMITTED_LEVEL':'16','SAEA_DEPLOYMENT_PROFILE':'profile'})
    with pytest.raises(ValueError):assert_execution_binding(spec,{'SAEA_ADMITTED_LEVEL':'96','SAEA_DEPLOYMENT_PROFILE':'profile'})
    with pytest.raises(ValueError):PerceptionWorkload(config())


def test_perception_release_rejects_material_type_and_claim_tampering(tmp_path,monkeypatch):
    import gepa_perception_material as m
    documents={name:{'name':name} for name in ('contract','parent','child','compiled_baseline','declaration')}
    for name,doc in documents.items():(tmp_path/(name+'.json')).write_text(json.dumps(doc))
    material={'schema':m.SCHEMA,'build':{},'documents':{n:m.pin(tmp_path/(n+'.json')) for n in documents},
        'normal_calls_per_row':8,'fallback_cap_per_row':16,'new_model_calls':0,'candidate_accepted':False,
        'external_request_sha256':'a'*64}
    result={**documents,'mutation':{'request_sha256':'a'*64}}
    monkeypatch.setattr(m,'derive',lambda *args:result)
    assert m.verify(tmp_path,tmp_path,material)==result
    for change in ({'normal_calls_per_row':4},{'candidate_accepted':True},{'external_request_sha256':'b'*64},{'schema':'saea-gepa-pair-material-v1'}):
        with pytest.raises(ValueError):m.verify(tmp_path,tmp_path,{**material,**change})
    (tmp_path/'child.json').write_text('{}')
    with pytest.raises(ValueError):m.verify(tmp_path,tmp_path,material)


@pytest.mark.parametrize("repeat_scope", ["shared-lock-perception", "independent-perception"])
def test_legacy_perception_pair_executes_all_eight_contacts(tmp_path, monkeypatch, repeat_scope):
    """The real compiled legacy config must execute, not merely load/custody-check."""
    from types import MappingProxyType
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from test_perception_template_variants import _build, _builder, _run, _requests_by_row, _control
    base, setup = pair_inputs(tmp_path / 'materials')
    original, _, parent, child = inputs()
    setup.update(parent_arguments=parent, child_arguments=child,
                 delivering_template_tokens=parent['count_tokens'](original))
    built = material.compile_template_pair(tmp_path / 'materials', base,
        perception_repeat_scope=repeat_scope, **setup)['baseline']
    built.update(transport='vllm', carrier_lock_key='slurp')
    config = load_native_episode_config(built, default_phase_program(built['policy']))
    assert config.prompt_contact_limits == 'legacy-c17-lock-v1'
    _, harness = _build(tmp_path / 'episode', monkeypatch)
    # Real runs can produce different greedy repeat outputs; equal fixture notes hid the bug.
    import test_perception_template_variants as variants
    monkeypatch.setitem(variants.NOTE, 3, 'repeat-own-note')
    # The fake transport has no live FIT service; keep all legacy graph constraints.
    from dataclasses import replace
    harness.config = replace(config, context_fit_policy="raise",
                             deliver_normaliser_id=harness.config.deliver_normaliser_id)
    harness.template_variant_materials = MappingProxyType({
        'parent': _builder(harness, 'p1'), 'child': _builder(harness, 'p2')})
    _run(harness)
    rows = _requests_by_row(harness)
    assert len(rows) == 4
    for sample_id, requests in rows.items():
        assert len(requests) == 8
        controls = [_control(request) for request in requests]
        ids = [control['contact_id'] for control in controls]
        assert len(set(ids)) == 8
        for role, offset in [('perception', 7), ('solver', 8)]:
            role_ids = [control['contact_id'] for control in controls if control['role_id'] == role]
            assert len(role_ids) == 4
            for branch, contact in zip(('parent', 'child', 'aa', 'lock'), role_ids):
                expected = 'phase-native-module-v1-c17'
                if branch != 'lock': expected += ':perception-arm:' + branch
                assert f'{expected}:{role}:{offset}:{sample_id}' in contact

        solver_requests = [r for r in requests if _control(r)['role_id'] == 'solver']
        aa_state, = [h for h in solver_requests[2]['history'] if h.get('role') == 'agent_state' and h.get('state_kind') == 'phase:perception']
        lock_state, = [h for h in solver_requests[3]['history'] if h.get('role') == 'agent_state' and h.get('state_kind') == 'phase:perception']
        assert lock_state['content'] == f'alpha-note:{sample_id}'
        expected_note = f'repeat-own-note:{sample_id}' if repeat_scope == 'independent-perception' else lock_state['content']
        assert aa_state['content'] == expected_note
        assert aa_state['source_response_sha256'] == p.text_sha(expected_note)
        if repeat_scope == 'independent-perception':
            assert solver_requests[2]['supplied_evidence'] != solver_requests[3]['supplied_evidence']


def test_independent_repeat_mode_is_explicit_and_unknown_scope_refused(tmp_path):
    base, setup = pair_inputs(tmp_path)
    original, _, parent, child = inputs()
    setup.update(parent_arguments=parent, child_arguments=child,
                 delivering_template_tokens=parent['count_tokens'](original))
    with pytest.raises(ValueError, match='unknown perception repeat scope'):
        material.compile_template_pair(tmp_path, base, perception_repeat_scope='guess', **setup)
    built = material.compile_template_pair(tmp_path, base, perception_repeat_scope='independent-perception', **setup)['baseline']
    assert material.independent_perception_repeat(built['native_episode']['consumption_module'])
    assert built['native_episode']['consumption_module']['gepa_pair']['schema'] == material.INDEPENDENT_PERCEPTION_PAIR_SCHEMA

    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    built.update(transport='vllm', carrier_lock_key='slurp')
    assert native_contact_custody_enabled(built)
    from gepa_perception_read import audit_inputs
    legacy = copy.deepcopy(built)
    legacy['native_episode']['consumption_module']['gepa_pair']['schema'] = material.PERCEPTION_PAIR_SCHEMA
    with pytest.raises(ValueError, match='explicit v2 repeat scope'):
        audit_inputs(tmp_path, legacy, None, [])
