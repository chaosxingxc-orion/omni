"""Native Flight mutation contracts; synthetic doubles are not NPU measurements."""
import copy
import json
from dataclasses import replace
import pytest
from speech_aware_evidence_acquisition.rails import gepa_mutation_runtime as m
from speech_aware_evidence_acquisition.rails import gepa_mutation_execution as execution
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha,canonical,UnknownLearningContact
from test_solver_prompt_candidate import fixture
from test_tfgrpo_dispatch import make_runtime


def setup(tmp_path,monkeypatch):
    rt,transport,captured=make_runtime(tmp_path,monkeypatch)
    contract,parent,mutation,count=fixture();sid=rt._samples[0].sample_id
    from speech_aware_evidence_acquisition.rails import solver_prompt_candidate as p
    contract=p.build_contract(base_template=contract['base_template'],frozen=contract['frozen'],
        feedback_query_ids=[sid],excluded_query_ids=[])
    parent=p.seed_candidate(contract=contract,expected_contract_sha256=contract['contract_sha256'],count_tokens=count)
    mutation.update(parent_sha256=parent['candidate_sha256'],evidence_query_ids=[sid],diagnosis='Test conditional evidence use.')
    source={'contract':contract,'parent':parent,'feedback_query_ids':[sid],'focus':'evidence-use',
        'historical_usage':{'calls':48,'prompt_tokens':1000,'completion_tokens':200},
        'observations':[{'query_id':sid,'raw_response':'old answer'}],
        'tokenizer_path':'tokenizer','tokenizer_identity':{'fixture':True}}
    rt._plan=replace(rt.plan,planned_model_calls=1)
    old=rt.frozen_core_client_kwargs;rt.frozen_core_client_kwargs=lambda:{**old(),'plan':rt.plan}
    reg,binding=m.build_registration(rt.plan,source=source,model_id='frozen-model',
        decoding_params={'max_tokens':100,'temperature':0},max_context_bytes=2000000,
        runtime_identity_sha256=sha(rt.session_required_identity()),roster_sha256='c'*64,
        resident_context=4096,reserve_tokens=64)
    spec={'schema':m.RUNTIME_SOURCE_SCHEMA,'source':{'fixture':True},'source_custody_sha256':sha(source)}
    (tmp_path/'mutation.json').write_bytes(canonical(spec))
    rt._composed.values.update(tfgrpo_learning_registration=reg,prepared_training_feedback={
        'schema':m.SOURCE_SCHEMA,'source':{'root':'data','relpath':'mutation.json','sha256':sha(spec)}})
    monkeypatch.setattr(m.evidence,'load_source',lambda *a,**k:copy.deepcopy(source))
    monkeypatch.setattr(m,'load_text_token_counter',lambda path:(count,{'fixture':True}))
    transport._responder=lambda payload:json.dumps(mutation)
    fits=[]
    def fit(prepared):
        fits.append(prepared.as_payload())
        return {'artifact_sha256':'d'*64,'every_request_fits':True,
            'payload_sha256':sha(prepared.as_payload()),'measured_prompt_tokens':12}
    from speech_aware_evidence_acquisition.core import tfgrpo_fit
    monkeypatch.setattr(tfgrpo_fit,'build_tfgrpo_fit_gate',lambda *a,**k:fit)
    return rt,transport,captured,fits,source,mutation


def test_real_dispatch_native_seal_and_prompt_kind(tmp_path,monkeypatch):
    rt,tr,captured,fits,source,_=setup(tmp_path,monkeypatch)
    result=rt.run_phases()
    assert result['contact_count']==len(tr.calls)==len(fits)==1
    candidate=json.loads((rt._run_dir()/'tfgrpo/learner/candidate.json').read_bytes())
    assert candidate['candidate_kind']=='solver_prompt' and candidate['historical_usage']==source['historical_usage']
    assert not candidate['deployment_accepted'] and candidate['learning_usage']['calls']==1
    assert result['learning']['schema']=='saea-gepa-mutation-flight-proposal-v1'
    sealed=rt.seal();assert sealed['cost']['calls_used']==1 and captured['finalized']


@pytest.mark.parametrize('fault',['stage','test','cap','source','plan','messages'])
def test_registration_drift_blocks_before_fit(tmp_path,monkeypatch,fault):
    rt,tr,_,fits,_,_=setup(tmp_path,monkeypatch);reg=rt._composed.values['tfgrpo_learning_registration']
    if fault=='stage':reg['contacts'][0]['stage']='other'
    elif fault=='test':reg['contacts'][0]['sample_id']='slurp:test:1'
    elif fault=='cap':reg['completion_cap']+=1
    elif fault=='source':(tmp_path/'mutation.json').write_text('{}')
    elif fault=='plan':reg['plan']['planned_model_calls']=2
    else:reg['contacts'][0]['messages_sha256']='f'*64
    with pytest.raises(ValueError):rt.run_phases()
    assert not tr.calls and not fits


def test_invalid_known_mutation_keeps_cost_and_is_never_resampled(tmp_path,monkeypatch):
    rt,tr,_,fits,_,_=setup(tmp_path,monkeypatch);tr._responder=lambda p:json.dumps({'invalid':True})
    for _ in range(2):
        with pytest.raises(ValueError):rt.run_phases()
    assert len(tr.calls)==len(fits)==1
    assert list((rt._run_dir()/'tfgrpo/learner/contacts').glob('*.response.json'))
    assert not (rt._run_dir()/'tfgrpo/learner/candidate.json').exists()
    rt._client.close()


def test_unknown_mutation_does_not_resend(tmp_path,monkeypatch):
    rt,tr,_,fits,_,_=setup(tmp_path,monkeypatch)
    def fail(payload):raise OSError('unknown send')
    tr._responder=fail
    with pytest.raises(OSError):rt.run_phases()
    with pytest.raises(UnknownLearningContact):rt.run_phases()
    assert len(tr.calls)==len(fits)==1
    rt._client.close()


def test_native_fit_context_is_checked_before_intent(tmp_path,monkeypatch):
    rt,tr,_,fits,_,_=setup(tmp_path,monkeypatch)
    reg=rt._composed.values['tfgrpo_learning_registration'];reg['resident_context']=170;reg['reserve_tokens']=64
    with pytest.raises(ValueError,match='wrapped mutation context'):rt.run_phases()
    assert len(fits)==1 and not tr.calls
    assert not list((rt._run_dir()/'tfgrpo/learner/contacts').glob('*.intent.json'))
    rt._client.close()


def test_request_validator_rejects_unregistered_prompt():
    binding={'model_id':'m','learning_config_sha256':'a','identity':'b','messages':[{'role':'system','content':'source'}]}
    expected=copy.deepcopy(execution.request(binding));m.request_validator(binding)(expected,None)
    expected['messages'][0]['content']='changed'
    with pytest.raises(ValueError):m.request_validator(binding)(expected,None)


def test_reflection_context_deduplicates_losslessly_without_dropping_evidence(tmp_path,monkeypatch):
    _,_,_,_,source,_=setup(tmp_path,monkeypatch)
    payload={'task_instruction':'same task','history':[
        {'role':'agent_state','content':'same P'},
        {'role':'control','content':json.dumps({'messages':[{'role':'system','content':'same task'},
            {'role':'user','content':'role instruction'}]},ensure_ascii=False,sort_keys=True,separators=(',',':'))}],
        'supplied_evidence':[{'evidence_id':'k','content':'knowledge'}],'extra':{'unchanged':1}}
    import hashlib
    payload['history'][-1]['content_sha256']=hashlib.sha256(payload['history'][-1]['content'].encode()).hexdigest()
    source['observations']=[{'query_id':'q1','request':payload},{'query_id':'q2','request':copy.deepcopy(payload)}]
    rendered=json.loads(m.evidence.messages(source)[1]['content']);shared=rendered['shared_content_by_sha256']
    def expand(value):
        if isinstance(value,list):return [expand(v) for v in value]
        if not isinstance(value,dict):return value
        if set(value)=={'content_sha256'}:return copy.deepcopy(shared[value['content_sha256']])
        result={k:expand(v) for k,v in value.items()}
        if 'content_canonical_json' in result:
            result['content']=json.dumps(result.pop('content_canonical_json'),ensure_ascii=False,sort_keys=True,separators=(',',':'))
        return result
    assert all(expand(r['request'])==payload for r in rendered['observations'])
    assert len(shared)==4


def test_explicit_feedback_id_policy_disambiguates_without_changing_evidence(tmp_path,monkeypatch):
    _,_,_,_,source,_=setup(tmp_path,monkeypatch)
    old=m.evidence.messages(source)
    source['mutation_prompt_policy']='explicit-feedback-query-ids-v1'
    updated=m.evidence.messages(source)
    body=json.loads(updated[1]['content']);previous=json.loads(old[1]['content'])
    assert body.pop('allowed_feedback_query_ids')==source['feedback_query_ids']
    assert body.pop('evidence_query_ids_contract')=={'min_items':1,'max_items':1,'unique_items':True,'source':'allowed_feedback_query_ids'}
    assert body==previous
    assert updated[0]['content'].startswith(old[0]['content'])
    assert 'train-exemplars::value' in updated[0]['content']
    source.pop('mutation_prompt_policy')
    assert m.evidence.messages(source)==old


def test_unknown_mutation_prompt_policy_is_rejected(tmp_path,monkeypatch):
    _,_,_,_,source,_=setup(tmp_path,monkeypatch)
    source['mutation_prompt_policy']='silently-cut-output'
    with pytest.raises(ValueError,match='unsupported mutation prompt policy'):
        m.evidence.messages(source)


def test_node_grounding_keeps_original_feedback_and_explicit_ids(tmp_path,monkeypatch):
    _,_,_,_,source,_=setup(tmp_path,monkeypatch)
    source['mutation_prompt_policy']='explicit-feedback-query-ids-v1'
    previous=m.evidence.messages(source)
    source['mutation_prompt_policy']='node-grounded-feedback-v2'
    updated=m.evidence.messages(source)
    body=json.loads(updated[1]['content']); old=json.loads(previous[1]['content'])
    for row in body['observations']:
        assert row.pop('recorded_node_feedback')['audio_available'] is False
    assert body == old
    assert 'other samples, not the current utterance' in updated[0]['content']
    assert updated[0]['content'].startswith(previous[0]['content'])
