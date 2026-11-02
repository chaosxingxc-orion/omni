import copy
import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).parents[2]
sys.path[:0] = [str(ROOT/'scripts/analysis'), str(Path(__file__).parent)]
import grounded_synthesis_execution as m
from grounded_synthesis_plan import plan_batches
from test_grounded_synthesis_plan import source, count


def fixture(tmp_path, *, caller=None, preflight=None, concurrency=2, groups=4):
    batch, ops = source()
    batch['model_id'] = 'model'
    if groups == 6:
        for i in range(4, 6):
            group = copy.deepcopy(batch['groups'][0]); group['query_id'] = f'slurp:train:{i}'
            batch['groups'].append(group)
            for j in range(2):
                row = copy.deepcopy(ops[j]); row['group_contact'] = m.sha({'stage':'group_update','identity':group['query_id']})
                row['response_file_sha256'] = m.sha({'source_response':group['query_id']})
                ops.append(row)
    bb, ob = m.canonical(batch), m.canonical(ops)
    tokenizer = {'fixture': 'a'*64}
    plan = plan_batches(bb, ob, source_batch_sha256=m.hashlib.sha256(bb).hexdigest(),
        source_operations_sha256=m.hashlib.sha256(ob).hexdigest(), count_tokens=count,
        tokenizer_identity=tokenizer, expected_groups=groups, expected_operations=groups*2,
        resident_context=500, completion_cap=100, safety_margin=100)
    custody = {'source_native_custody_verified':True, 'model_id':'model', 'unknown_contacts':0,
        'source_batch_sha256':plan['source_batch_sha256'],
        'source_operations_sha256':plan['source_operations_sha256'],
        'native_source_usage':[{'run_id':'failed', 'attempt_id':'a', 'usage':{'calls':308,'prompt_tokens':1000,'completion_tokens':500}},
                               {'run_id':'successful', 'attempt_id':'b', 'usage':{'calls':1,'prompt_tokens':10,'completion_tokens':5}}],
        'original_total_usage':{'calls':309,'prompt_tokens':1010,'completion_tokens':505}}
    calls, fits = [], []
    def default_call(request):
        calls.append(request)
        return {'model_id':'model','receipt_id':request['identity'],
                'text':'[{"operation":"ADD","id":null,"content":"conditional rule"}]',
                'usage':{'calls':1,'prompt_tokens':30,'completion_tokens':20}}
    learner = m.ExperienceLearner(tmp_path, model_id='model', metered_call=caller or default_call,
        learning_config_sha256='a'*64, max_calls=len(plan['shards']), concurrency=concurrency,
        preflight=preflight or (lambda request: fits.append(request)), max_operations=128)
    reg = m.registration_binding(plan, model_id='model', learning_config_sha256='a'*64,
                                 custody_sha256=m.sha(custody), max_operations=128,max_calls=len(plan['shards']))
    args = dict(registration=reg,count_tokens=count,tokenizer_identity=tokenizer,
                verify_source_custody=lambda:copy.deepcopy(custody))
    return learner, plan, bb, ob, args, custody, calls, fits


def run(f):
    learner, plan, bb, ob, args, *_ = f
    return m.execute_shards(learner, plan, bb, ob, **args)


def test_real_source_reader_contract_matches_executor(tmp_path, monkeypatch):
    from test_grounded_synthesis_source import fixture as source_fixture
    from grounded_synthesis_source import load_source
    spec, *_ = source_fixture(tmp_path, monkeypatch)
    custody = load_source(spec, data_root=tmp_path)
    source_pins = {key: custody[key] for key in
                   ('source_batch_sha256', 'source_operations_sha256')}
    assert m.validate_custody(custody, source_pins, 'm') == {
        'calls':309, 'prompt_tokens':1010, 'completion_tokens':202}


def test_success_resume_keeps_all_history_and_no_repeat(tmp_path):
    f = fixture(tmp_path)
    result = run(f)
    assert result['status'] == 'SHARDS_COMPLETE_NOT_CONSOLIDATED'
    assert result['cumulative_known_usage'] == {'calls':311,'prompt_tokens':1070,'completion_tokens':545}
    assert result['candidate_publishable'] is False and not (tmp_path/'candidate.json').exists()
    assert len(f[-2]) == len(f[-1]) == 2
    assert run(f) == result
    assert len(f[-2]) == len(f[-1]) == 2
    assert f[0].last_invocation == {'new_calls':0,'reused_calls':2}


def test_unknown_stops_recovery_before_any_unsent_shard(tmp_path):
    calls=[]
    def fail(request):
        calls.append(request); raise OSError('connection lost')
    f=fixture(tmp_path,caller=fail,concurrency=1)
    with pytest.raises(OSError): run(f)
    assert len(calls)==1
    f[0].caller=lambda request: pytest.fail('unknown contact must never be resent')
    with pytest.raises(m.UnknownLearningContact): run(f)
    assert len(list((tmp_path/'contacts').glob('*.intent.json')))==1
    progress=json.loads(sorted((tmp_path/'progress').glob('*.json'))[-1].read_bytes())
    assert progress['current_accounting']['unknown_contacts']==1
    assert progress['historical_usage']['calls']==309


def test_native_reconciliation_recovers_without_replay(tmp_path):
    f=fixture(tmp_path,concurrency=1)
    original=f[0].caller
    def unknown(request):
        raise OSError('lost response')
    f[0].caller=unknown
    with pytest.raises(OSError):run(f)
    def reconcile(request):
        response={'model_id':'model','receipt_id':'reconciled-original','text':'[]',
                  'usage':{'calls':1,'prompt_tokens':5,'completion_tokens':3}}
        return {'request_sha256':m.sha(request),'response':response,'response_sha256':m.sha(response)}
    f[0].reconcile=reconcile;f[0].caller=original
    result=run(f)
    assert len(f[-2])==1
    assert result['cumulative_known_usage']['calls']==311


@pytest.mark.parametrize('mutation',['history_total','duplicate_source','unknown','config','plan','no_preflight','source_model'])
def test_invalid_binding_rejected_before_model(tmp_path,mutation):
    f=fixture(tmp_path); learner,plan,bb,ob,args,custody,*_=f
    if mutation=='history_total':custody['original_total_usage']['calls']-=1
    elif mutation=='duplicate_source':custody['native_source_usage'].append(custody['native_source_usage'][0])
    elif mutation=='unknown':custody['unknown_contacts']=1
    elif mutation=='config':args['registration']['learning_config_sha256']='f'*64
    elif mutation=='plan':plan['shards'][0]['messages'][0]['content']+='changed'
    elif mutation=='no_preflight':learner.preflight=None
    else:learner.model_id='other'
    with pytest.raises(m.LearningError):run(f)
    assert not f[-2]


def test_changed_old_intent_rejected_before_new_work(tmp_path):
    f=fixture(tmp_path);run(f)
    path=next((tmp_path/'contacts').glob('*.intent.json'))
    d=json.loads(path.read_bytes());d['messages']=[];path.write_bytes(m.canonical(d))
    with pytest.raises(m.LearningError,match='foreign or changed'):run(f)
    assert len(f[-2])==2


def test_observed_invalid_response_is_preserved_not_resampled(tmp_path):
    f=fixture(tmp_path,concurrency=1)
    calls=[]
    def invalid(request):
        calls.append(request)
        return {'model_id':'model','receipt_id':request['identity'],'text':'invalid json',
                'usage':{'calls':1,'prompt_tokens':10,'completion_tokens':4}}
    f[0].caller=invalid
    with pytest.raises(m.LearningError):run(f)
    with pytest.raises(m.LearningError):run(f)
    assert len(calls)==1
    assert f[0].accounting()['known_usage']['calls']==1


def test_failure_drains_peer_and_does_not_start_third_shard(tmp_path):
    both=threading.Barrier(2); released=threading.Event();calls=[]
    f=fixture(tmp_path,groups=6)
    def caller(request):
        calls.append(request['identity']);both.wait(5)
        if request['identity'].endswith('shard-0'):
            released.set();raise OSError('first failure')
        assert released.wait(5)
        return {'model_id':'model','receipt_id':request['identity'],'text':'[]',
                'usage':{'calls':1,'prompt_tokens':10,'completion_tokens':4}}
    f[0].caller=caller
    with pytest.raises(OSError):run(f)
    assert len(calls)==2
    assert f[0].accounting()['observed_contacts']==1
    assert f[0].accounting()['unknown_contacts']==1
    assert (tmp_path/'shards/0001.json').exists()


def test_preflight_failure_spends_nothing_and_can_resume(tmp_path):
    f=fixture(tmp_path,concurrency=1)
    def reject(request):
        raise ValueError('native FIT rejected')
    f[0].preflight=reject
    with pytest.raises(ValueError,match='native FIT rejected'):run(f)
    assert f[0].accounting()['attempted_contacts']==0 and not f[-2]
    f[0].preflight=lambda request:None
    assert run(f)['current_accounting']['known_usage']['calls']==2


def test_custody_is_reverified_on_every_resume(tmp_path):
    f=fixture(tmp_path);run(f)
    def reject():
        raise ValueError('native source changed')
    f[4]['verify_source_custody']=reject
    with pytest.raises(ValueError,match='native source changed'):run(f)
    assert len(f[-2])==2


def test_observed_shard_output_tamper_is_not_replaced(tmp_path):
    f=fixture(tmp_path);run(f)
    path=tmp_path/'shards/0000.json';path.write_text('{}')
    with pytest.raises(m.LearningError,match='persisted shard execution evidence differs'):run(f)
    assert len(f[-2])==2 and path.read_text()=='{}'


def test_exact_registered_budget_required(tmp_path):
    f=fixture(tmp_path);f[0].max_calls+=1
    with pytest.raises(m.LearningError,match='exact one-call-per-shard budget'):run(f)
    assert not f[-2]
