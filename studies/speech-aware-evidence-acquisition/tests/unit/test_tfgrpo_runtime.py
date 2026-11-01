"""Real adapter metering and prepared text requests with zero network contact."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from speech_aware_evidence_acquisition.rails.tfgrpo_runtime import (
    MeteredExperienceCaller, LearningError, bind_runtime, learning_configuration,
)
from speech_aware_evidence_acquisition.rails.prepared_text import PREPARED_TEXT_INSTRUCTION
from test_rails_frozen_core_client import _World, _client, _plan, _CapturingTransport


def request(identity='r1', stage='summary'):
    return {'schema': 'saea-tfgrpo-learning-request-v1', 'batch_sha256': 'b'*64,
            'model_id': 'frozen-model', 'learning_config_sha256':config_hash(), 'stage': stage, 'identity': identity,
            'messages': [{'role': 'system', 'content': 'Summarize this trajectory.'},
                         {'role': 'user', 'content': 'Input and observed answer.'}]}


def runtime_hash():
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
    return sha({'model_fragment':'frozen-model'})


def config_hash(decoding=None):
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
    return sha(learning_configuration({'model_id':'frozen-model','decoding_params':decoding or {'max_tokens':100,'temperature':0},'max_context_bytes':10000,'runtime_identity_sha256':sha({'model_fragment':'frozen-model'})}))


def setup(tmp_path, *, transport=None, contacts=None):
    world = _World(tmp_path)
    plan = _plan()
    client, gate, transport = _client(world, tmp_path, plan=plan,
                                    transport=transport, arm='prepared-context-text')
    registration = {'schema': 'saea-tfgrpo-runtime-binding-v1', 'plan': {k:v for k,v in asdict(plan).items() if k != 'protocol_hash'},
        'model_id': 'frozen-model', 'batch_sha256': 'b'*64,
        'construction_roster_sha256': 'c'*64,
        'contacts': contacts or [{'stage': 'summary', 'identity': 'r1', 'sample_id': 's1'}],
        'decoding_params': {'max_tokens': 100, 'temperature': 0}, 'max_context_bytes': 10000,
        'learning_config_sha256':config_hash(),'runtime_identity_sha256':runtime_hash()}
    fits=[]; leases=[]
    def fit(prepared):
        fits.append(prepared.as_payload())
        from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
        return {'artifact_sha256': 'd'*64, 'every_request_fits': True, 'payload_sha256': sha(prepared.as_payload())}
    caller = MeteredExperienceCaller(client=client, plan=plan, registration=registration,
        model_id='frozen-model', receipt_root=tmp_path/'metering', request_fit=fit,
        assert_lease=lambda:leases.append(True))
    return caller, client, gate, transport, fits, leases


def test_real_adapter_trace_metering_and_shared_prepared_text(tmp_path):
    caller, client, gate, transport, fits, leases = setup(tmp_path)
    observed=caller(request())
    assert observed['text']=='ok'
    assert observed['usage']=={'calls':1,'prompt_tokens':3,'completion_tokens':2}
    assert len(transport.calls)==len(fits)==1
    payload=transport.calls[0]
    assert payload['task_instruction']==PREPARED_TEXT_INSTRUCTION
    assert [h['role'] for h in payload['history']]==['agent_state','control']
    assert json.loads(payload['history'][0]['content'])['source_messages']==request()['messages']
    assert payload['decoding_params']['id_slot']==0
    assert len(leases)==2 and gate.checked_plans
    assert caller.observed(request())==observed
    with pytest.raises(LearningError,match='already attempted'):
        caller(request())
    assert len(transport.calls)==1
    client.close()


def test_original_timing_durable_and_checked_after_bridge_reconstruction(tmp_path):
    from speech_aware_evidence_acquisition.rails.tfgrpo_runtime import contact_key
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical,sha
    caller,client,_,transport,_,_=setup(tmp_path)
    result=caller(request())
    key=contact_key('summary','r1')
    path=caller.root/(key+'.timing.json')
    timing=json.loads(path.read_bytes())
    assert timing['started_at']==client.timings[0].started_at
    assert timing['latency_seconds']==client.timings[0].latency_seconds
    clone=MeteredExperienceCaller(client=client,plan=caller.plan,registration=caller.registration,
        model_id=caller.model_id,receipt_root=caller.root,request_fit=caller.request_fit,
        assert_lease=caller.assert_lease)
    assert clone.observed(request())==result and len(transport.calls)==1
    timing['latency_seconds']+=1
    path.write_bytes(canonical(timing))
    with pytest.raises(LearningError,match='timing custody'):clone.observed(request())
    receipt_path=caller.root/(key+'.response.json')
    receipt=json.loads(receipt_path.read_bytes());receipt['response']['timing_sha256']=sha(timing)
    receipt_path.write_bytes(canonical(receipt))
    with pytest.raises(LearningError,match='timing differs'):clone.observed(request())
    client.close()


def test_operational_learning_timings_preserve_contact_identity(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition.core import stage_timing
    events=[]
    monkeypatch.setenv('SAEA_STAGE_TIMING_PATH',str(tmp_path/'timings.jsonl'))
    monkeypatch.setattr(stage_timing,'_record',lambda path,event:events.append(event))
    caller,client,_,transport,_,_=setup(tmp_path)
    caller.preflight(request())
    caller(request())
    own=[e for e in events if e['stage'].startswith('tfgrpo_')]
    assert {e['stage'] for e in own}=={'tfgrpo_request_prepare','tfgrpo_request_preflight','tfgrpo_generation'}
    assert all(e['learning_stage']=='summary' and e['learning_contact']=='r1'
               and e['status']=='ok' and e['elapsed_seconds']>=0 for e in own)
    assert len(transport.calls)==1
    client.close()


@pytest.mark.parametrize('change',[
    lambda r:r.update(batch_sha256='e'*64),
    lambda r:r.update(model_id='different'),
    lambda r:r.update(identity='unregistered'),
    lambda r:r['messages'].append({'role':'assistant','content':'injected'}),
])
def test_invalid_request_stops_before_fit_or_generation(tmp_path,change):
    caller, client, _, transport, fits, _=setup(tmp_path)
    value=request();change(value)
    with pytest.raises(LearningError):caller(value)
    assert not fits and not transport.calls
    client.close()


def test_parallel_summaries_keep_per_contact_usage(tmp_path):
    class Transport(_CapturingTransport):
        def __call__(self,raw):
            payload=json.loads(raw);self.calls.append(payload)
            tokens=payload['decoding_params']['id_slot']+7
            return json.dumps({'text':str(tokens),'usage':{'prompt_tokens':tokens,'completion_tokens':1}}).encode()
    caller,client,_,transport,_,_=setup(tmp_path,transport=Transport(),contacts=[
        {'stage':'summary','identity':'r1','sample_id':'s1'},
        {'stage':'summary','identity':'r2','sample_id':'s2'}])
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(caller,[request('r1'),request('r2')]))
    assert {r['usage']['prompt_tokens'] for r in rows}=={7,8}
    assert all(int(r['text'])==r['usage']['prompt_tokens'] for r in rows)
    assert len({r['receipt_id'] for r in rows})==2
    client.close()


def test_unknown_transport_outcome_never_retried(tmp_path):
    class Transport(_CapturingTransport):
        def __call__(self,raw):
            self.calls.append(json.loads(raw));raise ConnectionError('lost after send')
    caller,client,_,transport,_,_=setup(tmp_path,transport=Transport())
    with pytest.raises(ConnectionError):caller(request())
    with pytest.raises(LearningError,match='already attempted'):caller(request())
    assert len(transport.calls)==1
    client.close()


def test_fit_failure_prevents_generation_and_retains_attempt(tmp_path):
    caller,client,_,transport,_,_=setup(tmp_path)
    def fail(_):raise ValueError('overlong')
    caller.request_fit=fail
    with pytest.raises(ValueError,match='overlong'):caller(request())
    with pytest.raises(ValueError,match='overlong'):caller(request())
    assert not transport.calls
    assert not list(caller.root.glob('*.intent.json'))
    client.close()


def test_missing_usage_is_not_synthesized_from_cost_defaults(tmp_path):
    class Transport(_CapturingTransport):
        def __call__(self,raw):return b'{"text":"ok","usage":{}}'
    caller,client,_,_,_,_=setup(tmp_path,transport=Transport())
    with pytest.raises(LearningError,match='usage missing'):caller(request())
    client.close()


def test_observed_receipt_detects_changed_context_or_cost(tmp_path):
    caller,client,_,_,_,_=setup(tmp_path)
    caller(request())
    changed=request();changed['messages'][1]['content']='different'
    with pytest.raises(LearningError,match='request changed'):caller.observed(changed)
    path=next(caller.root.glob('*.trace.json'))
    trace=json.loads(path.read_bytes());trace[-1][-1]['calls']=12;path.write_text(json.dumps(trace))
    with pytest.raises(LearningError,match='custody differs'):caller.observed(request())
    client.close()


def test_fit_receipt_must_bind_exact_request(tmp_path):
    caller,client,_,transport,_,_=setup(tmp_path)
    caller.request_fit=lambda p:{'artifact_sha256':'d'*64,'every_request_fits':True,'payload_sha256':'e'*64}
    with pytest.raises(LearningError,match='bound artifact'):caller(request())
    assert not transport.calls
    client.close()


def test_observed_response_cannot_claim_unmetered_usage(tmp_path):
    caller,client,_,_,_,_=setup(tmp_path)
    caller(request())
    path=next(caller.root.glob('*.response.json'))
    row=json.loads(path.read_bytes());row['response']['usage']['prompt_tokens']=99
    path.write_text(json.dumps(row))
    with pytest.raises(LearningError,match='adapter records'):caller.observed(request())
    client.close()


def test_runtime_factory_uses_preflighted_client_and_model(tmp_path):
    from types import SimpleNamespace
    from test_rails_frozen_core_client import _binding, _OpenGate
    world=_World(tmp_path);plan=_plan()
    registration={'schema':'saea-tfgrpo-runtime-binding-v1',
        'plan':{k:v for k,v in asdict(plan).items() if k!='protocol_hash'},
        'model_id':'frozen-model','batch_sha256':'b'*64,'construction_roster_sha256':'c'*64,
        'contacts':[{'stage':'summary','identity':'r1','sample_id':'s1'}],
        'decoding_params':{'max_tokens':100,'temperature':0},'max_context_bytes':10000,'learning_config_sha256':config_hash(),'runtime_identity_sha256':runtime_hash()}
    transport=_CapturingTransport()
    class Runtime:
        _composed=SimpleNamespace(values={'arm':'prepared-context-text','result_kind':'artifact',
                    'scorers':[],'tfgrpo_learning_registration':registration})
        def session_required_identity(self):return {'model_fragment':'frozen-model'}
        def assert_generation_lease(self):pass
        def frozen_core_client_kwargs(self):
            return {'plan':plan,'gate':_OpenGate(),'transport':transport,'run_dir':world.run_dir,
                    'binding':_binding(world),'arm':'prepared-context-text'}
    runtime=Runtime();runtime.plan=plan;runtime._data_root=tmp_path
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
    caller=bind_runtime(runtime,receipt_root=tmp_path/'receipts',request_fit=lambda p:{
        'artifact_sha256':'d'*64,'every_request_fits':True,'payload_sha256':sha(p.as_payload())})
    assert caller(request())['usage']['calls']==1
    assert len(transport.calls)==1
    caller.client.close()


def test_learning_pipeline_uses_real_metered_bridge(tmp_path):
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import ExperienceLearner, empty_experience, sha
    from speech_aware_evidence_acquisition.rails.tfgrpo_prompts import SOURCE_SHA256, UPSTREAM_COMMIT
    parent=empty_experience('frozen-model')
    groups=[{'query_id':'q1','query':'What was requested?','answer':'turn on', 'rollouts':[
        {'trajectory_id':rid,'trajectory':text,'critique':'Verified by reference.', 'reward':reward,
         'model_id':'frozen-model','policy_sha256':'f'*64,'experience_sha256':parent['sha256']}
        for rid,text,reward in [('r1','turn off',0),('r2','turn on',1)]]}]
    common={'agent_objective':'Answer speech questions.','learning_objective':'Improve evidence use.','num_experiences':2}
    batch={'schema':'saea-tfgrpo-batch-v1','groups':groups,'parent':parent,'common':common,
           'model_id':'frozen-model','policy_sha256':'f'*64,'rollout_kind':'full-same-policy',
           'learning_config_sha256':config_hash(),
           'prompt_sha256':SOURCE_SHA256,'upstream_commit':UPSTREAM_COMMIT,
           'bounds':{'max_calls':5,'concurrency':2,'max_operations':32,'max_items':128,'max_characters':32000}}
    def answer(payload):
        stage=json.loads(payload['history'][0]['content'])['stage']
        if stage=='summary':return 'The action polarity determines correctness.'
        if stage=='advantage':return '<Experiences>Check action polarity.</Experiences>'
        return '[{"operation":"ADD","id":null,"content":"Check action polarity."}]'
    contacts=[{'stage':s,'identity':i,'sample_id':'s1'} for s,i in [
        ('summary','r1'),('summary','r2'),('advantage','q1'),('group_update','q1'),('batch_update','batch')]]
    caller,client,_,transport,_,_=setup(tmp_path,transport=_CapturingTransport(answer),contacts=contacts)
    # Recreate with the real immutable learner batch pin before any contact.
    caller.registration['batch_sha256']=sha(batch)
    (caller.root/'registration.json').unlink()
    caller=MeteredExperienceCaller(client=client,plan=caller.plan,registration=caller.registration,
        model_id='frozen-model',receipt_root=caller.root,request_fit=caller.request_fit,
        assert_lease=caller.assert_lease)
    learner=ExperienceLearner(tmp_path/'learner',model_id='frozen-model',metered_call=caller,learning_config_sha256=config_hash(),max_calls=5,concurrency=2,preflight=caller.preflight,reconcile=caller.reconcile)
    result=learner.learn(groups,parent,agent_objective=common['agent_objective'],
        learning_objective=common['learning_objective'],policy_sha256='f'*64)
    assert result['learning_usage']=={'calls':5,'prompt_tokens':15,'completion_tokens':10}
    assert list(result['items'].values())==['Check action polarity.']
    repeated=learner.learn(groups,parent,agent_objective=common['agent_objective'],
        learning_objective=common['learning_objective'],policy_sha256='f'*64)
    assert repeated==result and len(transport.calls)==5
    client.close()
