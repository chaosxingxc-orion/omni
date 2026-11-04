import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.error
import pytest

spec = importlib.util.spec_from_file_location('external', Path(__file__).parents[2]/'scripts/environment/configured_external_learner.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

@pytest.fixture
def setup(tmp_path, monkeypatch):
    messages = tmp_path/'messages.json'; messages.write_text('[{"role":"user","content":"test"}]')
    cfg = {'schema':'saea-external-learner-v1','endpoint':'https://example.invalid/v1/chat/completions',
        'model':'teacher','api_key_env':'TEST_TEACHER_KEY','messages':{'path':str(messages),
        'sha256':hashlib.sha256(messages.read_bytes()).hexdigest()},'max_tokens':100,
        'timeout_seconds':10,'output_root':str(tmp_path/'output')}
    config = tmp_path/'config.json'; config.write_text(json.dumps(cfg))
    monkeypatch.setenv('TEST_TEACHER_KEY','private-test-key')
    return config, cfg


def test_timeout_durable_no_repeat(setup, monkeypatch):
    config,cfg=setup
    class Opener:
        def open(self,*a,**kw): raise TimeoutError('private-test-key must never appear')
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    first=m.execute(config)
    assert first['state']=='unknown' and first['usage'] is None
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:pytest.fail('duplicate spend'))
    assert m.execute(config)==first
    assert all('private-test-key' not in p.read_text() for p in Path(cfg['output_root']).glob('*.json'))


def test_success_reuses_receipt(setup,monkeypatch):
    config,cfg=setup
    class Response:
        status=200;headers={'x-request-id':'id'}
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self):return b'{"model":"teacher","usage":{"total_tokens":4},"choices":[{"finish_reason":"stop"}]}'
    class Opener:
        def open(self,*a,**kw):return Response()
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    first=m.execute(config);assert first['usage']['total_tokens']==4
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:pytest.fail('duplicate spend'))
    assert m.execute(config)==first


def test_crash_after_intent_does_not_resend(setup):
    config,cfg=setup;root=Path(cfg['output_root']);root.mkdir(mode=0o700)
    m.persist(root/'intent.json',{'config':cfg,'messages_sha256':cfg['messages']['sha256']})
    with pytest.raises(ValueError,match='reconcile'):m.execute(config)


def test_messages_tamper_rejected(setup):
    config,cfg=setup;Path(cfg['messages']['path']).write_text('[]')
    with pytest.raises(ValueError,match='messages changed'):m.execute(config)


def test_redirect_refused():
    assert m.NoRedirect().redirect_request(None,None,302,'',{},'https://other.invalid') is None


def test_http_error_never_retries(setup,monkeypatch):
    config,cfg=setup
    class Opener:
        def open(self,*a,**kw):raise urllib.error.HTTPError('url',429,'rate',{},None)
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    assert m.execute(config)['state']=='http_error'
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:pytest.fail('duplicate spend'))
    assert m.execute(config)['http_status']==429

@pytest.mark.parametrize('complete',[True,False])
def test_stream_journal_and_terminal_required(setup,monkeypatch,complete):
    config,cfg=setup;cfg['stream']=True;config.write_text(json.dumps(cfg))
    events=[{'id':'r1','model':'teacher','choices':[{'index':0,'delta':{'reasoning_content':'reason'},'finish_reason':None}]},
        {'id':'r1','model':'teacher','choices':[{'index':0,'delta':{'content':'OK'},'finish_reason':'stop'}]},
        {'id':'r1','model':'teacher','choices':[],'usage':{'prompt_tokens':10,'completion_tokens':5}}]
    lines=[]
    for event in events:lines.extend([('data: '+json.dumps(event)+'\n').encode(),b'\n'])
    if complete:lines.extend([b'data: [DONE]\n',b'\n'])
    class Response:
        status=200;headers={'x-request-id':'rid','content-type':'text/event-stream'}
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def __iter__(self):return iter(lines)
    class Opener:
        def open(self,req,**kw):
            body=json.loads(req.data);assert body['stream_options']=={'include_usage':True}
            return Response()
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    result=m.execute(config);root=Path(cfg['output_root'])
    assert (root/'response.sse').read_bytes()==b''.join(lines)
    assert json.loads((root/'headers.json').read_text())['request_id']=='rid'
    if complete:
        assert result['state']=='response_received'
        assert result['usage']['prompt_tokens']==10
        response=json.loads((root/'response.json').read_text())
        assert response['choices'][0]['message']=={'role':'assistant','content':'OK','reasoning_content':'reason'}
    else:
        assert result['state']=='unknown' and not (root/'response.json').exists()
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:pytest.fail('no redispatch'))
    assert m.execute(config)==result


def test_total_deadline_interrupts_blocking_read_without_resend(setup,monkeypatch):
    import time
    config,cfg=setup;cfg['total_timeout_seconds']=1;config.write_text(json.dumps(cfg))
    class Opener:
        def open(self,*a,**kw):time.sleep(5);pytest.fail('deadline did not interrupt')
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    result=m.execute(config)
    assert result['state']=='unknown' and result['error_type']=='TotalDeadlineExceeded'
    assert result['elapsed_seconds']<4 and m.signal.getitimer(m.signal.ITIMER_REAL)==(0.0,0.0)
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:pytest.fail('no deadline retry'))
    assert m.execute(config)==result


def test_explicit_thinking_mode_forwarded(setup,monkeypatch):
    config,cfg=setup;cfg['thinking']={'type':'disabled'};config.write_text(json.dumps(cfg))
    class Opener:
        def open(self,request,**kw):
            assert json.loads(request.data)['thinking']=={'type':'disabled'}
            raise TimeoutError()
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    assert m.execute(config)['state']=='unknown'


@pytest.mark.parametrize('value',[0,True,-2,'60'])
def test_invalid_total_deadline_rejected_before_dispatch(setup,value):
    config,cfg=setup;cfg['total_timeout_seconds']=value;config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='total wall'):m.execute(config)
    assert not Path(cfg['output_root']).exists()


def research_config(config, cfg):
    policy = config.parent/'research.md'
    policy.write_text('Analyze mechanisms and counterexamples, not case-by-case repetition.')
    cfg['research_prompt'] = {'path': policy.name,
        'sha256': hashlib.sha256(policy.read_bytes()).hexdigest()}
    cfg['max_tokens'] = 131072
    config.write_text(json.dumps(cfg))
    return policy


def test_research_policy_in_actual_request_preserves_full_evidence(setup, monkeypatch):
    config, cfg = setup
    evidence = [{'role': 'system', 'content': 'Task-specific role'},
        {'role': 'user', 'content': 'Full evidence ' * 20000}]
    raw = json.dumps(evidence).encode()
    Path(cfg['messages']['path']).write_bytes(raw)
    cfg['messages']['sha256'] = hashlib.sha256(raw).hexdigest()
    policy = research_config(config, cfg)
    class Opener:
        def open(self, request, **kw):
            body = json.loads(request.data)
            assert body['max_tokens'] == 131072
            assert body['messages'][0] == {'role': 'system', 'content': policy.read_text()}
            assert body['messages'][1:] == evidence
            raise TimeoutError()
    monkeypatch.setattr(m.urllib.request, 'build_opener', lambda *a: Opener())
    assert m.execute(config)['state'] == 'unknown'
    sent = json.loads((Path(cfg['output_root'])/'request.json').read_bytes())
    assert sent['messages'][1:] == evidence
    monkeypatch.setattr(m.urllib.request, 'build_opener', lambda *a: pytest.fail('duplicate spend'))
    assert m.execute(config)['state'] == 'unknown'


@pytest.mark.parametrize('change', ['tamper', 'budget', 'empty'])
def test_research_policy_invalid_before_dispatch(setup, change):
    config, cfg = setup
    policy = research_config(config, cfg)
    if change == 'tamper':
        policy.write_text('changed')
    elif change == 'budget':
        cfg['max_tokens'] = 102400
    else:
        policy.write_text(' ')
        cfg['research_prompt']['sha256'] = hashlib.sha256(policy.read_bytes()).hexdigest()
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='research'):
        m.execute(config)
    assert not Path(cfg['output_root']).exists()
