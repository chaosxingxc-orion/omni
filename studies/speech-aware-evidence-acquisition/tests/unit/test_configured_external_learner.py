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
    diagnostic=json.loads((Path(cfg['output_root'])/'diagnostics.json').read_bytes())
    assert diagnostic['usage_received'] and diagnostic['finish_received']
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


def test_analysis_budget_is_validated_and_injected_without_provider_fields(setup, monkeypatch):
    config, cfg = setup
    research_config(config, cfg)
    cfg['max_tokens'] = 131072
    cfg['analysis_budget'] = {
        'input_context_target_tokens': 262144,
        'max_reasoning_tokens_target': 49152,
        'max_mechanism_dimensions': 6,
        'max_hypotheses': 6,
        'representative_evidence_per_hypothesis': 3,
        'counterexamples_per_hypothesis': 1,
        'representative_case_target': 12,
        'stop_when_evidence_sufficient': True,
    }
    config.write_text(json.dumps(cfg))
    class Opener:
        def open(self, request, **kw):
            body = json.loads(request.data)
            assert 'analysis_budget' not in body
            assert '49152' in body['messages'][0]['content']
            assert '262144' in body['messages'][0]['content']
            assert body['messages'][1:] == [{'role': 'user', 'content': 'test'}]
            raise TimeoutError()
    monkeypatch.setattr(m.urllib.request, 'build_opener', lambda *a: Opener())
    assert m.execute(config)['state'] == 'unknown'
    intent = json.loads((Path(cfg['output_root'])/'intent.json').read_bytes())
    assert intent['config']['analysis_budget']['representative_case_target'] == 12


def test_analysis_budget_cannot_exceed_output_budget(setup):
    config, cfg = setup
    research_config(config, cfg)
    cfg['analysis_budget'] = {
        'input_context_target_tokens': 262144,
        'max_reasoning_tokens_target': 131073,
        'max_mechanism_dimensions': 6,
        'max_hypotheses': 6,
        'representative_evidence_per_hypothesis': 3,
        'counterexamples_per_hypothesis': 1,
        'representative_case_target': 12,
        'stop_when_evidence_sufficient': True,
    }
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='exceeds output budget'):
        m.execute(config)


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


def test_batched_lines_durable_before_yield_and_flush_error_tail(tmp_path, monkeypatch):
    import http.client
    diagnostic = m.ResponseDiagnostics(m.time.monotonic())
    synced = []
    monkeypatch.setattr(m.os, 'fsync', lambda fd: synced.append(True))
    def source():
        yield b'first\n'
        yield b'second\n'
        raise http.client.IncompleteRead(b'partial', 30)
    path = tmp_path/'journal'
    with path.open('wb') as handle:
        iterator = m.durable_stream_lines(source(), handle, diagnostic, 65536, 10)
        assert next(iterator) == b'first\n'
        assert path.read_bytes() == b'first\nsecond\n' and len(synced) == 1
        assert next(iterator) == b'second\n'
        with pytest.raises(http.client.IncompleteRead): next(iterator)
    assert diagnostic.data['durable_body_bytes'] == 13


def test_batched_stream_done_does_not_wait_for_disconnect(tmp_path, monkeypatch):
    diagnostic = m.ResponseDiagnostics(m.time.monotonic())
    def source():
        for _ in range(100):
            yield b'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}\n'
            yield b'\n'
        yield b'data: [DONE]\n'; yield b'\n'
        pytest.fail('read after terminal event')
    result, _ = m.read_stream(source(), tmp_path, diagnostic)
    assert result['choices'][0]['message']['content'] == 'x'*100
    assert diagnostic.data['body_fsync_calls'] == 1


@pytest.mark.parametrize('key,value',[('journal_batch_bytes',0),('journal_batch_bytes',True),
    ('journal_batch_bytes',1048577),('journal_batch_seconds',11)])
def test_invalid_batch_rejected_before_dispatch(setup,key,value):
    config,cfg=setup;cfg[key]=value;config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='bounded journal'):m.execute(config)
    assert not Path(cfg['output_root']).exists()


@pytest.mark.parametrize('stream', [True, False])
def test_incomplete_read_preserves_partial_and_diagnostics_without_retry(setup, monkeypatch, stream):
    import http.client
    config, cfg = setup
    cfg['stream'] = stream
    config.write_text(json.dumps(cfg))
    prefix = b'data: {"id":"r","choices":[{"index":0,"delta":{"reasoning_content":"analysis"}}]}\n\n'
    partial = b'data: {"choices": [ private-test-key'
    class Response:
        status = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __iter__(self):
            for line in prefix.splitlines(keepends=True):
                yield line
            raise http.client.IncompleteRead(partial, 100)
        def read(self): raise http.client.IncompleteRead(partial, 100)
    class Opener:
        def open(self, *a, **kw): return Response()
    monkeypatch.setattr(m.urllib.request, 'build_opener', lambda *a: Opener())
    result = m.execute(config)
    root = Path(cfg['output_root'])
    assert result['state'] == 'unknown' and result['error_type'] == 'IncompleteRead'
    assert result['usage'] is None and (root/'response.partial.bin').read_bytes() == partial
    assert (root/'response.partial.bin').stat().st_mode & 0o777 == 0o600
    diagnostic = json.loads((root/'diagnostics.json').read_bytes())
    expected = len(partial) + (len(prefix) if stream else 0)
    assert diagnostic['received_body_bytes'] == diagnostic['durable_body_bytes'] == expected
    assert diagnostic['partial_exception_bytes'] == len(partial)
    assert diagnostic['failure_phase'] == 'body-read'
    assert diagnostic['done_received'] is False and diagnostic['finish_received'] is False
    assert diagnostic['first_body_data_elapsed_seconds'] is not None
    assert diagnostic['last_body_data_elapsed_seconds'] >= diagnostic['first_body_data_elapsed_seconds']
    assert diagnostic['body_fsync_calls'] >= 1
    if stream:
        assert (root/'response.sse').read_bytes() == prefix
        assert diagnostic['sse_events'] == 1 and diagnostic['reasoning_characters'] == 8
    assert 'private-test-key' not in json.dumps(result) + json.dumps(diagnostic)
    monkeypatch.setattr(m.urllib.request, 'build_opener', lambda *a: pytest.fail('duplicate spend'))
    assert m.execute(config) == result


def test_http_error_body_is_private_and_not_in_summary(setup, monkeypatch):
    import io
    config, cfg = setup
    body = b'{"error":"private-test-key sensitive task content"}'
    class Opener:
        def open(self, *a, **kw):
            raise urllib.error.HTTPError('url', 400, 'bad', {}, io.BytesIO(body))
    monkeypatch.setattr(m.urllib.request, 'build_opener', lambda *a: Opener())
    result = m.execute(config)
    root = Path(cfg['output_root'])
    assert result['state'] == 'http_error' and result['http_status'] == 400
    assert (root/'http-error-body.bin').read_bytes() == body
    assert (root/'http-error-body.bin').stat().st_mode & 0o777 == 0o600
    diagnostic = json.loads((root/'diagnostics.json').read_bytes())
    assert diagnostic['received_body_bytes'] == diagnostic['durable_body_bytes'] == len(body)
    assert 'sensitive task content' not in json.dumps(result) + json.dumps(diagnostic)
    monkeypatch.setattr(m.urllib.request, 'build_opener', lambda *a: pytest.fail('duplicate spend'))
    assert m.execute(config) == result


def test_body_wait_and_disk_time_have_distinct_measurements(tmp_path, monkeypatch):
    # A deterministic fake clock proves local fsync time is not counted as read wait.
    clock = [10.0]
    monkeypatch.setattr(m.time, 'monotonic', lambda: clock[0])
    diagnostic = m.ResponseDiagnostics(clock[0])
    def read():
        clock[0] += 3.0
        return b'abc'
    class Handle:
        def write(self, raw): clock[0] += 1.0
        def flush(self): clock[0] += 2.0
        def fileno(self): return 123
    monkeypatch.setattr(m.os, 'fsync', lambda fd: clock.__setitem__(0, clock[0] + 4.0))
    raw = diagnostic.read(read)
    diagnostic.received(raw)
    diagnostic.write(Handle(), raw)
    assert diagnostic.data['body_read_wait_seconds'] == 3.0
    assert diagnostic.data['body_write_flush_seconds'] == 3.0
    assert diagnostic.data['body_fsync_seconds'] == 4.0
    assert diagnostic.data['first_body_data_elapsed_seconds'] == 3.0
    assert diagnostic.data['durable_body_bytes'] == 3


@pytest.mark.parametrize('fault', ['null', 'bool', 'missing', 'equal', 'no_policy', 'stop'])
def test_analysis_budget_invalid_before_dispatch(setup, fault):
    config, cfg = setup
    research_config(config, cfg)
    example = Path(__file__).parents[2]/'configs/learners/glm-flash-external.example.json'
    cfg['analysis_budget'] = json.loads(example.read_bytes())['analysis_budget']
    if fault == 'null': cfg['analysis_budget'] = None
    elif fault == 'bool': cfg['analysis_budget']['max_hypotheses'] = True
    elif fault == 'missing': cfg['analysis_budget'].pop('representative_case_target')
    elif fault == 'equal': cfg['analysis_budget']['max_reasoning_tokens_target'] = cfg['max_tokens']
    elif fault == 'no_policy': cfg.pop('research_prompt')
    else: cfg['analysis_budget']['stop_when_evidence_sufficient'] = False
    config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): m.execute(config)
    assert not Path(cfg['output_root']).exists()


def test_total_deadline_from_worker_thread_uses_process_main(tmp_path, monkeypatch):
    messages = tmp_path/'messages.json'; messages.write_text('[{"role":"user","content":"thread deadline"}]')
    cfg = {'schema':'saea-external-learner-v1','endpoint':'https://127.0.0.1:1/v1/chat/completions',
        'model':'teacher','api_key_env':'TEST_TEACHER_KEY','messages':{'path':str(messages),
        'sha256':hashlib.sha256(messages.read_bytes()).hexdigest()},'max_tokens':131072,
        'timeout_seconds':1,'total_timeout_seconds':1,'output_root':str(tmp_path/'output')}
    config = tmp_path/'config.json'; config.write_text(json.dumps(cfg)); monkeypatch.setenv('TEST_TEACHER_KEY','private-test-key')
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(m.execute, config)
        result = future.result(timeout=10)
        assert result['state'] == 'unknown'
    result = json.loads((tmp_path/'output/result.json').read_bytes())
    assert result['state'] == 'unknown' and result['automatic_retry'] is False
    assert result['error_type'] in ('URLError', 'ConnectionRefusedError', 'TimeoutError')


def test_four_threaded_production_deadlines_success_timeout_and_no_repeat(tmp_path, monkeypatch):
    """Child interpreters retain real signal handling; only HTTP is controlled."""
    import concurrent.futures
    import os
    import signal
    import time
    support = tmp_path/'support'; support.mkdir()
    (support/'sitecustomize.py').write_text('''
import json, os, time, urllib.request
from pathlib import Path
class Response:
    status=200
    headers={'content-type':'text/event-stream'}
    def __init__(self, slow): self.slow=slow
    def __enter__(self): return self
    def __exit__(self,*a): pass
    def __iter__(self):
        if self.slow: time.sleep(4)
        events=[{'id':'r1','model':'teacher','choices':[{'index':0,'delta':{'content':'complete'},'finish_reason':'stop'}]},
                {'id':'r1','model':'teacher','choices':[],'usage':{'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}}]
        for event in events:
            yield ('data: '+json.dumps(event)+'\\n').encode(); yield b'\\n'
        yield b'data: [DONE]\\n'; yield b'\\n'
class Opener:
    def open(self, request, timeout):
        assert timeout==600
        body=json.loads(request.data)
        label=body['messages'][0]['content']
        with open(os.environ['TEST_OPEN_LOG'],'a') as f: f.write(label+'\\n')
        return Response(label=='slow')
urllib.request.build_opener=lambda *a:Opener()
''')
    monkeypatch.setenv('PYTHONPATH', str(support)+os.pathsep+os.environ.get('PYTHONPATH',''))
    monkeypatch.setenv('TEST_TEACHER_KEY','private-test-key')
    log=tmp_path/'opens.log';monkeypatch.setenv('TEST_OPEN_LOG',str(log))
    configs=[]
    for label in ['fast-a','slow','fast-b','fast-c']:
        root=tmp_path/label;root.mkdir()
        messages=root/'messages.json';messages.write_text(json.dumps([{'role':'user','content':label}]))
        cfg={'schema':'saea-external-learner-v1','endpoint':'https://example.invalid/v1/chat/completions',
             'model':'teacher','api_key_env':'TEST_TEACHER_KEY','messages':{'path':str(messages),'sha256':hashlib.sha256(messages.read_bytes()).hexdigest()},
             'max_tokens':131072,'timeout_seconds':600,'total_timeout_seconds':1 if label=='slow' else 2400,
             'stream':True,'output_root':str(root/'response')}
        path=root/'config.json';path.write_text(json.dumps(cfg));configs.append(path)
    before=signal.getitimer(signal.ITIMER_REAL);started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(m.execute,configs))
    assert time.monotonic()-started < 8
    assert [r['state'] for r in results]==['response_received','unknown','response_received','response_received']
    assert results[1]['error_type']=='TotalDeadlineExceeded'
    assert results[1]['elapsed_seconds'] < 3
    assert signal.getitimer(signal.ITIMER_REAL)==before
    assert sorted(log.read_text().splitlines())==['fast-a','fast-b','fast-c','slow']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(m.execute,configs))==results
    assert len(log.read_text().splitlines())==4
    for path,result in zip(configs,results):
        root=path.parent/'response'
        assert (root/'intent.json').exists()
        assert json.loads((root/'result.json').read_bytes())==result
        if result['state']=='response_received':
            assert hashlib.sha256((root/'response.sse').read_bytes()).hexdigest()==result['response_sha256']


def test_threaded_deadline_unknown_intent_never_redispatches(setup):
    import concurrent.futures
    config,cfg=setup;cfg['total_timeout_seconds']=2400;config.write_text(json.dumps(cfg))
    root=Path(cfg['output_root']);root.mkdir(mode=0o700)
    m.persist(root/'intent.json',{'config':cfg,'messages_sha256':cfg['messages']['sha256']})
    before=(root/'intent.json').read_bytes()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for _ in range(2):
            with pytest.raises(RuntimeError,match='terminal receipt'):
                pool.submit(m.execute,config).result(timeout=10)
    assert (root/'intent.json').read_bytes()==before
    assert not (root/'request.json').exists() and not (root/'result.json').exists()


def test_urlerror_reason_is_structured_but_stays_unknown(setup, monkeypatch):
    import socket
    config,cfg=setup
    class Opener:
        def open(self,*a,**kw):
            raise urllib.error.URLError(socket.gaierror(-3,'private-test-key'))
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    r=m.execute(config)
    assert r['state']=='unknown' and r['automatic_retry'] is False
    assert r['reason_type']=='gaierror' and r['reason_errno']==-3
    assert r['failure_phase']=='request-open'
    assert 'private-test-key' not in json.dumps(r)
    diagnostic=json.loads((Path(cfg['output_root'])/'diagnostics.json').read_text())
    assert diagnostic['reason_errno']==-3
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:pytest.fail('must not retry'))
    assert m.execute(config)==r


@pytest.mark.parametrize('mode', [None, 'inherit-environment', 'direct'])
def test_executor_proxy_mode_matches_probe(setup, monkeypatch, mode):
    config,cfg=setup
    if mode is not None: cfg['proxy_mode']=mode
    config.write_text(json.dumps(cfg))
    seen=[]
    class Opener:
        def open(self,*a,**kw): raise TimeoutError('private proxy details')
    def build(*handlers):
        seen.extend(handlers)
        return Opener()
    monkeypatch.setattr(m.urllib.request, 'build_opener', build)
    result=m.execute(config)
    explicit=[h for h in seen if isinstance(h,m.urllib.request.ProxyHandler)]
    assert len(explicit)==(1 if mode=='direct' else 0)
    if explicit: assert explicit[0].proxies=={}
    assert result['state']=='unknown' and result['automatic_retry'] is False


def test_executor_invalid_proxy_mode_before_intent(setup):
    config,cfg=setup;cfg['proxy_mode']='guess';config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='proxy mode'):m.execute(config)
    assert not Path(cfg['output_root']).exists()
