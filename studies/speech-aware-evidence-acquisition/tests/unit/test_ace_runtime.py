"""ACE dynamic dispatch on real shared adapter with explicit transport/source doubles."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from speech_aware_evidence_acquisition.rails import ace_runtime as m, ace_incremental as ace, tfgrpo_runtime
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha, UnknownLearningContact
from test_ace_incremental import fixture, reflection, delta
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
    group['rollouts'][0].update(model_id=model,experience_sha256=parent['sha256'])
    runtime._plan = replace(runtime.plan,planned_model_calls=2)
    original = runtime.frozen_core_client_kwargs
    runtime.frozen_core_client_kwargs = lambda:{**original(),'plan':runtime.plan}
    config = {'model_id':model,'decoding_params':{'max_tokens':100,'temperature':0},
        'max_context_bytes':2000000,'runtime_identity_sha256':sha(runtime.session_required_identity())}
    doc = ace.build_input(source,parent,learning_config_sha256=sha(m.learning_configuration(config)),
        **{k:v for k,v in old['bounds'].items() if k != 'max_calls'})
    count = lambda messages:10
    reg = m.build_registration(runtime.plan,source=source,document=doc,
        decoding_params=config['decoding_params'],max_context_bytes=config['max_context_bytes'],
        runtime_identity_sha256=config['runtime_identity_sha256'],roster_sha256='c'*64)
    spec = {'schema':m.RUNTIME_SOURCE_SCHEMA,'strategy':ace.STRATEGY,'source':{},
        'tokenizer_relpath':'tokenizer','tokenizer_identity':{'fixture':True},'document':doc,
        'initial_measurement':ace.reflect_messages(doc,0,parent['items'],count),
        'phase_contract':'unknown-prompts-await-observed-predecessor'}
    (tmp_path/'ace.json').write_bytes(canonical(spec))
    runtime._composed.values.update(tfgrpo_learning_registration=reg,
        prepared_training_feedback={'schema':m.SOURCE_SCHEMA,'source':{'root':'data','relpath':'ace.json','sha256':sha(spec)}})
    monkeypatch.setattr(m,'load_token_counter',lambda path:(count,{'fixture':True}))
    monkeypatch.setattr(m,'load_finalized_shards',lambda *a,**k:copy.deepcopy(source))
    def answer(payload):
        stage = json.loads(payload['history'][0]['content'])['stage']
        return json.dumps(reflection() if stage == ace.REFLECT else delta())
    transport._responder = answer
    fits = []
    def fit(prepared):
        fits.append(prepared.as_payload())
        return {'artifact_sha256':'d'*64,'every_request_fits':True,
            'payload_sha256':sha(prepared.as_payload()),'measured_prompt_tokens':12}
    from speech_aware_evidence_acquisition.core import tfgrpo_fit
    monkeypatch.setattr(tfgrpo_fit,'build_tfgrpo_fit_gate',lambda *a,**k:fit)
    return runtime,transport,captured,source,spec,fits


def test_dynamic_dispatch_real_adapter_seal_and_truthful_phase_boundaries(tmp_path,monkeypatch):
    runtime,transport,captured,source,spec,fits = setup(tmp_path,monkeypatch)
    reg = runtime._composed.values['tfgrpo_learning_registration']
    loaded,receipt = m.load_runtime_source(runtime._composed.values,plan=runtime.plan,
        repo_root=tmp_path,data_root=tmp_path,sample_ids=(runtime._samples[0].sample_id,),roster_sha256='c'*64)
    assert receipt['subsequent_phases'] == 'AWAITING_OBSERVED_PREDECESSOR'
    assert not receipt['native_fit_completed'] and not fits
    assert all('messages_sha256' not in row for row in reg['contacts'])
    result = runtime.run_phases()
    assert result['contact_count'] == len(transport.calls) == len(fits) == 2
    candidate = json.loads((runtime._run_dir()/'tfgrpo/learner/candidate.json').read_bytes())
    assert candidate['cumulative_known_usage']['calls'] == 314
    assert len(candidate['contact_response_sha256']) == 2 and not candidate['deployment_accepted']
    sealed = runtime.seal()
    assert sealed['cost']['calls_used'] == 2 and captured['finalized']


@pytest.mark.parametrize('fault',['order','dependency','test','budget','cap','tokenizer','source','replay'])
def test_registration_source_drift_before_contact(tmp_path,monkeypatch,fault):
    runtime,transport,_,_,spec,fits = setup(tmp_path,monkeypatch)
    reg = runtime._composed.values['tfgrpo_learning_registration']
    if fault == 'order': reg['contacts'].reverse()
    elif fault == 'dependency': reg['contacts'][1]['depends_on'] = None
    elif fault == 'test': reg['contacts'][0]['sample_id'] = 'slurp:test:1'
    elif fault == 'budget': reg['bounds']['max_calls'] += 1
    elif fault == 'cap': reg['completion_cap'] += 1
    elif fault == 'replay': runtime._native_contact_replay = object()
    elif fault == 'source': (tmp_path/'ace.json').write_text('{}')
    else:
        spec['tokenizer_identity'] = {'drift':True}
        (tmp_path/'ace.json').write_bytes(canonical(spec))
        runtime._composed.values['prepared_training_feedback']['source']['sha256'] = sha(spec)
    with pytest.raises(ValueError):runtime.run_phases()
    assert not fits and not transport.calls


def test_native_wrapped_tokens_stop_before_generation_not_just_inner_prompt(tmp_path,monkeypatch):
    runtime,transport,_,_,_,fits = setup(tmp_path,monkeypatch,max_prompt_tokens_total=11)
    with pytest.raises(ValueError,match='native total'):runtime.run_phases()
    assert len(fits) == 1 and not transport.calls
    assert not list((runtime._run_dir()/'tfgrpo/learner/contacts').glob('*.intent.json'))
    runtime._client.close()


def test_unknown_holds_next_stage_and_never_retries(tmp_path,monkeypatch):
    runtime,transport,_,_,_,fits = setup(tmp_path,monkeypatch)
    def fail(payload):raise OSError('unknown send')
    transport._responder = fail
    with pytest.raises(OSError):runtime.run_phases()
    with pytest.raises(UnknownLearningContact):runtime.run_phases()
    assert len(transport.calls) == len(fits) == 1
    runtime._client.close()


def test_known_bridge_response_reconciles_without_resend(tmp_path,monkeypatch):
    runtime,transport,_,_,_,fits = setup(tmp_path,monkeypatch)
    original = tfgrpo_runtime.MeteredExperienceCaller.__call__; once=[]
    def crash(self,request):
        response = original(self,request)
        if not once:once.append(True); raise OSError('handoff')
        return response
    monkeypatch.setattr(tfgrpo_runtime.MeteredExperienceCaller,'__call__',crash)
    with pytest.raises(OSError):runtime.run_phases()
    runtime.run_phases()
    assert len(transport.calls) == len(fits) == 2
    runtime.seal()


def test_request_derivation_rejects_invented_prompt_and_unobserved_predecessor(tmp_path):
    _,document = fixture(groups=1)
    validator = m.request_validator(document,lambda messages:10)
    root = tmp_path/'bridge';root.mkdir()
    caller = SimpleNamespace(root=root,observed=lambda request:pytest.fail('no predecessor response'))
    rows = m.contact_roster(document)
    first = ace.reflect_messages(document,0,document['parent']['items'],lambda messages:10)
    request = m._request(document,rows[0],first)
    validator(request,caller)
    request['messages'][1]['content'] += ' changed'
    with pytest.raises(ValueError,match='prompt differs'):validator(request,caller)
    future = m._request(document,rows[1],first)
    with pytest.raises(UnknownLearningContact,match='predecessor'):validator(future,caller)


def test_prior_spend_refuses_fresh_client(tmp_path,monkeypatch):
    runtime,transport,_,_,_,_ = setup(tmp_path,monkeypatch)
    runtime.run_phases();runtime._client.close();del runtime._tfgrpo_caller
    with pytest.raises(ValueError,match='prior ACE spend'):m.run_registered_ace(runtime)
    assert len(transport.calls) == 2


def test_next_group_prompt_uses_observed_delta_and_preserves_signal_content_identity(tmp_path):
    _,document = fixture(groups=2)
    root = tmp_path/'bridge';root.mkdir(); rows = m.contact_roster(document)
    measured_calls=[]
    def count(messages):measured_calls.append(messages);return 10
    answers = {m.contact_key(rows[0]['stage'],rows[0]['identity']):reflection(),
               m.contact_key(rows[1]['stage'],rows[1]['identity']):delta()}
    requests={}
    first = ace.reflect_messages(document,0,document['parent']['items'],lambda messages:10)
    requests[m.contact_key(rows[0]['stage'],rows[0]['identity'])] = m._request(document,rows[0],first)
    metadata={k:{'helpful':0,'harmful':0,'uncertain':0,'events':[]} for k in document['parent']['items']}
    signal=reflection()['signals'][0]
    metadata['old']['harmful']=1
    metadata['old']['events']=[{'query_id':document['groups'][0]['query_id'],
        'item_content_sha256':sha(document['parent']['items']['old']),
        'association_origin':'reflector-retrospective','generator_item_exposure_verified':False,**signal}]
    curated=ace.curate_messages(document,0,document['parent']['items'],reflection(),metadata,lambda messages:10)
    requests[m.contact_key(rows[1]['stage'],rows[1]['identity'])]=m._request(document,rows[1],curated)
    for key in answers:
        (root/(key+'.intent.json')).write_bytes(canonical(requests[key]))
        (root/(key+'.response.json')).write_text('{}')
    def observed(request):
        key=m.contact_key(request['stage'],request['identity'])
        assert request==requests[key]
        return {'text':json.dumps(answers[key])}
    caller=SimpleNamespace(root=root,observed=observed)
    items=copy.deepcopy(document['parent']['items']);items['old']=delta()['deltas'][0]['content']
    next_message=ace.reflect_messages(document,1,items,lambda messages:10)
    validate=m.request_validator(document,count)
    validate(m._request(document,rows[2],next_message),caller)
    validate(m._request(document,rows[2],next_message),caller)
    assert len(measured_calls)==3  # Exact historical prompts counted once, then reused.
    assert json.loads(next_message['messages'][1]['content'])['offline_stale_policy']
