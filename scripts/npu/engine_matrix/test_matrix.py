import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import threading
import pytest
P=Path(__file__).parent
spec=importlib.util.spec_from_file_location('matrix',P/'matrix.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_result_is_reused_without_second_call(tmp_path):
    calls=[]
    a,new=m.execute_request(tmp_path,'request',{'x':1},lambda:calls.append(1) or {'status':'ok'})
    b,again=m.execute_request(tmp_path,'request',{'x':1},lambda:pytest.fail('replayed'))
    assert new and not again and a==b and calls==[1]


def test_uncertain_request_refuses_retry(tmp_path):
    def crash():raise OSError('transport interrupted')
    with pytest.raises(OSError):m.execute_request(tmp_path,'request',{'x':1},crash)
    with pytest.raises(ValueError,match='unresolved'):m.execute_request(tmp_path,'request',{'x':1},lambda:pytest.fail('replayed'))


def test_body_change_refuses_completed_request(tmp_path):
    m.execute_request(tmp_path,'request',{'x':1},lambda:{'status':'ok'})
    with pytest.raises(ValueError,match='changed'):m.execute_request(tmp_path,'request',{'x':2},lambda:pytest.fail('replayed'))


def test_orphan_result_refused(tmp_path):
    m.durable(tmp_path/'x.result.json',{'request_id':'x','body_sha256':m.digest({})})
    with pytest.raises(ValueError,match='lacks intent'):m.resolve_request(tmp_path,'x',m.digest({}))


def candidate():
    return {'id':'c','container':'saea-omni-i0','url':'http://127.0.0.1:8200','model':m.MODEL,'image':m.IMAGE,
            'client_concurrency':2,'max_num_seqs':32,'max_num_batched_tokens':None,
            'enable_prefix_caching':True,'enable_chunked_prefill':True}


def doc():
    return {'Id':'id','Image':m.IMAGE,'State':{'Running':True,'Pid':100},'Args':['serve','model',
            '--served-model-name',m.MODEL,'--port','8200','--tensor-parallel-size','2',
            '--max-model-len','24576','--max-num-seqs','32']}


@pytest.mark.parametrize('field,value', [('Image','wrong')])
def test_live_image_mismatch(field,value):
    d=doc();d[field]=value
    with pytest.raises(ValueError):m.inspect_binding(candidate(),d,'enable_prefix_caching=True enable_chunked_prefill=True')


def test_live_flag_evidence_required():
    with pytest.raises(ValueError,match='startup'):m.inspect_binding(candidate(),doc(),'')
    assert m.inspect_binding(candidate(),doc(),'enable_prefix_caching=True enable_chunked_prefill=True')['image']==m.IMAGE


def test_explicit_batch_default_mismatch():
    d=doc();d['Args']+=['--max-num-batched-tokens','4096']
    with pytest.raises(ValueError,match='default'):m.inspect_binding(candidate(),d,'enable_prefix_caching=True enable_chunked_prefill=True')


def test_rounds_have_matched_lengths_and_fresh_content():
    tool=m.load_tool()
    a=m.plan_requests(8,3,'seed',tool,content_seed='round0');b=m.plan_requests(8,3,'seed',tool,content_seed='round1')
    assert [t.targets for t in a]==[t.targets for t in b]
    assert [t.audio_seconds for t in a]==[t.audio_seconds for t in b]
    assert a[0].seed!=b[0].seed


def test_tokenize_recovery_reuses_each_contact(tmp_path):
    tool=m.load_tool();t=m.plan_requests(1,3,'seed',tool)[0];config=tool.SweepConfig(base_urls=['http://none'],model=m.MODEL,levels=[1])
    calls=[]
    tool.tokenize_count=lambda *args:calls.append(1) or 10
    tool.fit_contact=lambda count,target:tool.FittedContact(0,0,target,count(0,0),1)
    first=m.fit_one(tool,config,t,tmp_path,threading.Event())
    assert len(calls)==3
    (tmp_path/'fit/0.json').unlink()  # interruption before aggregate checkpoint
    second=m.fit_one(tool,config,t,tmp_path,threading.Event())
    assert first==second and len(calls)==3


def test_stop_before_fit_sends_nothing(tmp_path):
    tool=m.load_tool();t=m.plan_requests(1,3,'seed',tool)[0];config=tool.SweepConfig(base_urls=['http://none'],model=m.MODEL,levels=[1])
    tool.tokenize_count=lambda *args:pytest.fail('sent after stop')
    stop=threading.Event();stop.set()
    with pytest.raises(InterruptedError):m.fit_one(tool,config,t,tmp_path,stop)
    assert not list(tmp_path.rglob('*.intent.json'))


def test_steady_counter_window_excludes_drain():
    samples=[{'time':i,'metrics':{'generation_tokens':i*10}} for i in range(6)]
    r=m.steady_summary(samples,1,3)
    assert r['samples']==3 and r['qualified'] and r['generation_tokens_s']==10
    assert not m.steady_summary(samples,4,3)['qualified']


def test_complete_trial_is_noop(tmp_path,monkeypatch):
    tool=m.load_tool();c=candidate();trajectories=m.plan_requests(1,3,'seed',tool)
    live=m.inspect_binding(c,doc(),'enable_prefix_caching=True enable_chunked_prefill=True')
    monkeypatch.setattr(m,'validate_live',lambda c:live)
    monkeypatch.setattr(tool,'preflight',lambda c:pytest.fail('completed trial contacted endpoint'))
    binding={'candidate':c,'live':live,'workload_sha256':m.digest([dataclasses.asdict(t) for t in trajectories]),
             'tool_sha256':m.hashlib.sha256(m.TOOL.read_bytes()).hexdigest(),'runner_sha256':m.hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
    m.durable(tmp_path/'binding.json',binding)
    for i in range(3):m.execute_request(tmp_path/'requests',str(i),{},lambda:{'status':'ok'})
    m.durable(tmp_path/'complete.json',{'completed':3,'new_requests':3,'errors':0,'request_result_hashes':{p.name:m.hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'requests').glob('*.result.json')}})
    result=m.run_trial(tool,c,tmp_path,trajectories,threading.Event())
    assert result['already_complete'] and result['new_requests']==0


def test_plan_validates_sufficient_load_and_unique_endpoints():
    p=m.read(P/'plan.json');m.validate_plan(p,0)
    p['trajectories']=1
    with pytest.raises(ValueError,match='three'):m.validate_plan(p,0)


def test_generation_error_stops_trajectory(tmp_path,monkeypatch):
    tool=m.load_tool();c=candidate();c['client_concurrency']=1;t=m.plan_requests(1,3,'seed',tool)
    monkeypatch.setattr(m,'validate_live',lambda c:m.inspect_binding(c,doc(),'enable_prefix_caching=True enable_chunked_prefill=True'))
    monkeypatch.setattr(tool,'preflight',lambda c:None)
    monkeypatch.setattr(tool,'read_metrics',lambda url:{'num_requests_running':0,'num_requests_waiting':0})
    monkeypatch.setattr(m,'fit_one',lambda *a:[tool.FittedContact(0,0,100,100,1)]*3)
    sent=[]
    def send(url,body,r,**kw):
        sent.append(r.request_id);r.status='error';r.completion_tokens=0;r.error='injected'
    monkeypatch.setattr(tool,'send_chat',send)
    stop=threading.Event();r=m.run_trial(tool,c,tmp_path,t,stop)
    assert stop.is_set() and r['errors']==1 and len(sent)==1 and not (tmp_path/'complete.json').exists()


def test_failed_candidate_does_not_cancel_healthy_peer(tmp_path, monkeypatch):
    failed = threading.Event()
    def trial(tool, candidate, root, trajectories, stop, barrier):
        if candidate['id'] == 'bad':
            stop.set(); failed.set()
            return {'errors': 1, 'interrupted': True}
        assert failed.wait(3)
        assert not stop.is_set()
        (root / 'healthy-complete').write_text('done')
        return {'errors': 0, 'interrupted': False}
    monkeypatch.setattr(m, 'run_trial', trial)
    with pytest.raises(RuntimeError, match='bad'):
        m.run_candidates(None, [{'id': 'bad'}, {'id': 'good'}], [tmp_path, tmp_path], [], threading.Event())
    assert (tmp_path / 'healthy-complete').read_text() == 'done'


def test_global_stop_reaches_all_candidates(tmp_path, monkeypatch):
    global_stop = threading.Event(); global_stop.set()
    seen = []
    def trial(tool, candidate, root, trajectories, stop, barrier):
        assert stop.wait(3); seen.append(candidate['id'])
        return {'interrupted': True}
    monkeypatch.setattr(m, 'run_trial', trial)
    with pytest.raises(RuntimeError):
        m.run_candidates(None, [{'id': 'a'}, {'id': 'b'}], [tmp_path, tmp_path], [], global_stop)
    assert sorted(seen) == ['a', 'b']


@pytest.mark.parametrize('value', [0, True, 3601, '1800'])
def test_invalid_request_timeout(value):
    plan=m.read(P/'plan.json');plan['rounds'][0][0]['request_timeout_s']=value
    with pytest.raises(ValueError,match='timeout'):m.validate_plan(plan,0)
