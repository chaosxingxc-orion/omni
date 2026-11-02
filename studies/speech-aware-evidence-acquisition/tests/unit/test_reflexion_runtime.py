"""Reflexion through the real shared Flight adapter; explicit source/transport doubles."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from speech_aware_evidence_acquisition.rails import reflexion_runtime as m, reflexion_offline as method, tfgrpo_runtime
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha, UnknownLearningContact
from test_reflexion_offline import fixture, reflection, compact_reflection
from test_tfgrpo_dispatch import make_runtime

def setup(tmp_path,monkeypatch,**limits):
    runtime,transport,captured = make_runtime(tmp_path,monkeypatch)
    source,old = fixture(groups=1,**limits)
    model = 'frozen-model'; sid = runtime._samples[0].sample_id
    source['model_id'] = source['batch']['model_id'] = model
    parent = source['batch']['parent']; parent['model_id'] = model
    parent['sha256'] = sha({k:v for k,v in parent.items() if k != 'sha256'})
    group = source['batch']['groups'][0]; group['query_id'] = sid
    group['query'] = json.dumps({'sample_id':sid})
    for row in group['rollouts']:
        row.update(model_id=model,experience_sha256=parent['sha256'])
    runtime._plan = replace(runtime.plan,planned_model_calls=1)
    original = runtime.frozen_core_client_kwargs
    runtime.frozen_core_client_kwargs = lambda:{**original(),'plan':runtime.plan}
    config = {'model_id':model,'decoding_params':{'max_tokens':100,'temperature':0},
        'max_context_bytes':2000000,'runtime_identity_sha256':sha(runtime.session_required_identity())}
    doc = method.build_input(source,parent,learning_config_sha256=sha(m.learning_configuration(config)),
        **{k:v for k,v in old['bounds'].items() if k != 'max_calls'})
    count = lambda messages:10
    reg = m.build_registration(runtime.plan,source=source,document=doc,
        decoding_params=config['decoding_params'],max_context_bytes=config['max_context_bytes'],
        runtime_identity_sha256=config['runtime_identity_sha256'],roster_sha256='c'*64)
    spec = {'schema':m.RUNTIME_SOURCE_SCHEMA,'strategy':method.STRATEGY,'source':{},
        'tokenizer_relpath':'tokenizer','tokenizer_identity':{'fixture':True},'document':doc,
        'initial_measurement':method.reflect_messages(doc,0,method.empty_memory(),count),
        'phase_contract':'unknown-prompts-await-observed-predecessor'}
    (tmp_path/'method.json').write_bytes(canonical(spec))
    runtime._composed.values.update(tfgrpo_learning_registration=reg,
        prepared_training_feedback={'schema':m.SOURCE_SCHEMA,'source':{'root':'data','relpath':'method.json','sha256':sha(spec)}})
    monkeypatch.setattr(m,'load_token_counter',lambda path:(count,{'fixture':True}))
    monkeypatch.setattr(m,'load_finalized_shards',lambda *a,**k:copy.deepcopy(source))
    def answer(payload):
        stage = json.loads(payload['history'][0]['content'])['stage']
        assert stage == method.REFLECT
        answer = compact_reflection() if limits.get('reflector_contract') else reflection()
        if limits.get('reflector_contract'): answer['schema'] = limits['reflector_contract']
        return json.dumps(answer)
    transport._responder = answer
    fits = []
    def fit(prepared):
        fits.append(prepared.as_payload())
        return {'artifact_sha256':'d'*64,'every_request_fits':True,
            'payload_sha256':sha(prepared.as_payload()),'measured_prompt_tokens':12}
    from speech_aware_evidence_acquisition.core import tfgrpo_fit
    monkeypatch.setattr(tfgrpo_fit,'build_tfgrpo_fit_gate',lambda *a,**k:fit)
    return runtime,transport,captured,source,spec,fits


@pytest.mark.parametrize('contract', [None, method.COMPACT_CONTRACT, method.ERROR_FEEDBACK_CONTRACT])
def test_real_flight_dispatch_single_stage_seal_and_source_route(tmp_path,monkeypatch,contract):
    runtime,transport,captured,source,spec,fits=setup(tmp_path,monkeypatch,reflector_contract=contract)
    from speech_aware_evidence_acquisition.rails.tfgrpo_source import load_dynamic_source
    loaded,receipt=load_dynamic_source(runtime._composed.values,plan=runtime.plan,
        repo_root=tmp_path,data_root=tmp_path,sample_ids=(runtime._samples[0].sample_id,),roster_sha256='c'*64)
    assert not receipt['native_fit_completed'] and not fits
    result=runtime.run_phases()
    assert result['contact_count']==len(transport.calls)==len(fits)==1
    candidate=json.loads((runtime._run_dir()/'tfgrpo/learner/candidate.json').read_bytes())
    assert candidate['cumulative_known_usage']['calls']==313 and candidate['strategy']==method.STRATEGY
    assert not candidate['deployment_accepted']
    sealed=runtime.seal()
    assert sealed['cost']['calls_used']==1 and captured['finalized']


@pytest.mark.parametrize('fault',['stage','dependency','test','budget','cap','tokenizer','source','replay'])
def test_registration_source_drift_before_fit_or_generation(tmp_path,monkeypatch,fault):
    runtime,transport,_,_,spec,fits=setup(tmp_path,monkeypatch)
    reg=runtime._composed.values['tfgrpo_learning_registration']
    if fault=='stage':reg['contacts'][0]['stage']='ace_reflect'
    elif fault=='dependency':reg['contacts'][0]['depends_on']='0'*64
    elif fault=='test':reg['contacts'][0]['sample_id']='slurp:test:1'
    elif fault=='budget':reg['bounds']['max_calls']+=1
    elif fault=='cap':reg['completion_cap']+=1
    elif fault=='replay':runtime._native_contact_replay=object()
    elif fault=='source':(tmp_path/'method.json').write_text('{}')
    else:
        spec['tokenizer_identity']={'drift':True}
        (tmp_path/'method.json').write_bytes(canonical(spec))
        runtime._composed.values['prepared_training_feedback']['source']['sha256']=sha(spec)
    with pytest.raises(ValueError):runtime.run_phases()
    assert not fits and not transport.calls


def test_native_wrapper_budget_stops_before_generation_intent(tmp_path,monkeypatch):
    runtime,transport,_,_,_,fits=setup(tmp_path,monkeypatch,max_prompt_tokens_total=11)
    with pytest.raises(ValueError,match='native total'):runtime.run_phases()
    assert len(fits)==1 and not transport.calls
    assert not list((runtime._run_dir()/'tfgrpo/learner/contacts').glob('*.intent.json'))
    runtime._client.close()


def test_registration_rejects_foreign_reflector_contract(tmp_path,monkeypatch):
    runtime,transport,_,_,_,fits=setup(tmp_path,monkeypatch)
    reg=runtime._composed.values['tfgrpo_learning_registration']
    reg['bounds']['reflector_contract']='ace-contract'
    with pytest.raises(ValueError,match='exact finite Reflexion bounds'):
        m.validate_registration(reg,plan=runtime.plan,model_id=reg['model_id'])
    assert not fits and not transport.calls


def test_unknown_never_retries(tmp_path,monkeypatch):
    runtime,transport,_,_,_,fits=setup(tmp_path,monkeypatch)
    def fail(payload):raise OSError('unknown send')
    transport._responder=fail
    with pytest.raises(OSError):runtime.run_phases()
    with pytest.raises(UnknownLearningContact):runtime.run_phases()
    assert len(fits)==len(transport.calls)==1
    runtime._client.close()


def test_known_bridge_handoff_reconciles_without_resend(tmp_path,monkeypatch):
    runtime,transport,_,_,_,fits=setup(tmp_path,monkeypatch)
    original=tfgrpo_runtime.MeteredExperienceCaller.__call__;once=[]
    def crash(self,request):
        response=original(self,request)
        if not once:once.append(True);raise OSError('handoff')
        return response
    monkeypatch.setattr(tfgrpo_runtime.MeteredExperienceCaller,'__call__',crash)
    with pytest.raises(OSError):runtime.run_phases()
    runtime.run_phases()
    assert len(fits)==len(transport.calls)==1
    runtime.seal()


def test_request_prefix_rebuild_uses_fifo_and_rejects_missing_predecessor(tmp_path):
    _,doc=fixture(groups=3,max_items=1)
    root=tmp_path/'bridge';root.mkdir();rows=m.contact_roster(doc)
    caller=SimpleNamespace(root=root,observed=lambda request:pytest.fail('unobserved'))
    validate=m.request_validator(doc,lambda messages:10)
    memory=method.empty_memory(); first=method.reflect_messages(doc,0,memory,lambda x:10)
    validate(m._request(doc,rows[0],first),caller)
    with pytest.raises(UnknownLearningContact):
        validate(m._request(doc,rows[1],first),caller)
    requests={};answers={}
    for index in range(2):
        measured=method.reflect_messages(doc,index,memory,lambda x:10)
        key=m.contact_key(rows[index]['stage'],rows[index]['identity'])
        requests[key]=m._request(doc,rows[index],measured)
        answers[key]=json.dumps(reflection(index))
        (root/(key+'.intent.json')).write_bytes(canonical(requests[key]))
        (root/(key+'.response.json')).write_text('{}')
        checked=method.validate_reflection(answers[key],doc['groups'][index],200)
        memory,_=method.advance_memory(memory,checked,doc['groups'][index],index,doc['bounds'])
    def observed(request):
        key=m.contact_key(request['stage'],request['identity']);assert request==requests[key]
        return {'text':answers[key]}
    caller.observed=observed
    measured=method.reflect_messages(doc,2,memory,lambda x:10)
    assert len(json.loads(measured['messages'][1]['content'])['memory'])==1
    request=m._request(doc,rows[2],measured);validate(request,caller)
    request['messages'][1]['content']+=' changed'
    with pytest.raises(ValueError,match='prompt differs'):validate(request,caller)


def test_prior_spend_cannot_open_fresh_client(tmp_path,monkeypatch):
    runtime,transport,_,_,_,_=setup(tmp_path,monkeypatch)
    runtime.run_phases();runtime._client.close();del runtime._tfgrpo_caller
    with pytest.raises(ValueError,match='prior Reflexion spend'):m.run_registered_reflexion(runtime)
    assert len(transport.calls)==1
