import copy
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/analysis'))
from server_observability import parse_metrics,difference


@pytest.fixture
def collector_clock(monkeypatch):
    import server_observability as m
    class Clock:
        now=0
        on_sleep=None

        def sleep(self, seconds):
            assert 0 < seconds <= 1
            self.now+=seconds
            if self.on_sleep:self.on_sleep(self.now)
    clock=Clock()
    monkeypatch.setattr(m.time,'monotonic',lambda:clock.now)
    monkeypatch.setattr(m.time,'sleep',clock.sleep)
    return clock

def snap(count,total,labels='model_name="private-model"'):
    text=f'vllm:request_queue_time_seconds_count{{{labels}}} {count}\nvllm:request_queue_time_seconds_sum{{{labels}}} {total}\nvllm:request_queue_time_seconds_bucket{{{labels},le="+Inf"}} {count}\nvllm:num_requests_running{{{labels}}} 2\n'
    return {'window_id':'run','mode':'debug','engine_identity':{'boot':'a','process_start':1},'instances':{'i0':{'metrics':parse_metrics(text)}}}

def test_real_histogram_window_delta_is_aggregate_and_missing_stays_unknown():
    a,b=snap(10,100),snap(14,112);r=difference(a,b);v=r['instances']['i0']
    assert v['histograms']['request_queue_time_seconds']['mean_seconds']==3
    assert v['histograms']['request_queue_time_seconds']['cumulative_buckets']=={'+Inf':4}
    assert v['histograms']['request_prefill_time_seconds']['mean_seconds'] is None
    assert not r['isolated_window_verified']
    assert 'private-model' not in str(a)

@pytest.mark.parametrize('fault',['decrease','labels','identity','missing'])
def test_never_report_reset_or_mixed_identity_as_latency(fault):
    a,b=snap(10,100),snap(14,112)
    if fault=='decrease':b=snap(1,2)
    if fault=='labels':b=snap(14,112,'model_name="new"')
    if fault=='missing':b['instances']['i0']={'available':False}
    if fault=='identity':
        b['engine_identity']['process_start']=2
        with pytest.raises(ValueError):difference(a,b)
    else:assert difference(a,b)['instances']['i0']['available'] is False

def test_empty_window_has_no_fabricated_zero_latency():
    r=difference(snap(10,100),snap(10,100))
    assert r['instances']['i0']['histograms']['request_queue_time_seconds']['mean_seconds'] is None

def test_duplicate_series_refused():
    with pytest.raises(ValueError):parse_metrics('vllm:prompt_tokens_total 1\nvllm:prompt_tokens_total 2\n')


def test_background_collector_uses_profile_and_terminal_event(tmp_path,monkeypatch):
    import json
    import server_observability as m
    profile=tmp_path/'debug.json';profile.write_text(json.dumps({'mode':'debug','client_detail':True,'server_metrics_interval_seconds':600}))
    stop=tmp_path/'terminal.json';stop.write_text('{}');calls=[]
    def capture(*a,**kw):calls.append(kw);return snap(10,100)
    monkeypatch.setattr(m,'snapshot',capture)
    m.collect_windows({'i0':'http://localhost/metrics'},window_id='run',mode='debug',config=profile,out=tmp_path/'out',stop_file=stop,max_snapshots=10)
    assert len(calls)==1
    assert json.loads((tmp_path/'out/complete.json').read_bytes())['terminal_observed'] is True
    assert json.loads((tmp_path/'out/config.json').read_bytes())['effective']['server_metrics_interval_seconds']==600


def test_server_profile_hash_matches_effective_settings(tmp_path,monkeypatch):
    import hashlib,json
    import server_observability as m
    effective={'mode':'normal','client_detail':False,'server_metrics_interval_seconds':600}
    profile=tmp_path/'profile.json';profile.write_text(json.dumps(effective))
    stop=tmp_path/'done';stop.touch()
    monkeypatch.setattr(m,'snapshot',lambda *a,**kw:snap(1,2))
    m.collect_windows({},window_id='run',mode='normal',config=profile,out=tmp_path/'out',stop_file=stop,max_snapshots=2)
    cfg=json.loads((tmp_path/'out/config.json').read_bytes())
    assert cfg['configuration_sha256']==hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()


def test_terminal_status_requires_matching_full_id_and_final_snapshot(tmp_path,monkeypatch):
    import json
    import server_observability as m
    cid='a'*64
    config=tmp_path/'profile.json';config.write_text(json.dumps({'mode':'normal','client_detail':False,'server_metrics_interval_seconds':600}))
    status=tmp_path/'status.json';status.write_text(json.dumps({'state':'exited','container_id':cid,'running':False,'exit_code':17}))
    monkeypatch.setattr(m,'snapshot',lambda *a,**kw:snap(1,2))
    m.collect_windows({},window_id='run',mode='normal',config=config,out=tmp_path/'out',terminal_status=status,expected_container_id=cid,max_snapshots=2)
    complete=json.loads((tmp_path/'out/complete.json').read_bytes())
    assert complete['terminal_observed'] is True
    assert complete['final_snapshot_after_terminal'] is True
    assert complete['terminal_status']['exit_code']==17


@pytest.mark.parametrize('payload', [
    {'state':'exited','container_id':'b'*64,'running':False},
    {'state':'exited','container_id':'a'*63,'running':False},
    {'state':'exited','container_id':'a'*64,'running':True},
    {'state':'banana','container_id':'a'*64,'running':False},
    {'state':'exited','container_id':'a'*64,'running':0},
    {'state':'exited','container_id':'a'*64,'running':False,'exit_code':'0'},
    {'state':'not-launched'},
    [],
])
def test_terminal_status_rejects_corrupt_or_changed_identity(tmp_path,payload):
    import json
    import server_observability as m
    path=tmp_path/'status.json';path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):m._read_terminal_status(path,'a'*64)


@pytest.mark.parametrize('state',['running','created','paused','restarting',None])
def test_terminal_running_and_missing_continue_without_final_claim(tmp_path,monkeypatch,collector_clock,state):
    import json
    import server_observability as m
    cid='c'*64; config=tmp_path/'profile.json'
    config.write_text(json.dumps({'mode':'normal','client_detail':False,'server_metrics_interval_seconds':600}))
    status=tmp_path/'status.json'
    if state is not None:status.write_text(json.dumps({'state':state,'container_id':cid,'running':state!='created'}))
    captured=[]
    def capture(*a,**kw):
        captured.append(collector_clock.now)
        return snap(1,2)
    monkeypatch.setattr(m,'snapshot',capture)
    m.collect_windows({},window_id='run',mode='normal',config=config,out=tmp_path/'out',terminal_status=status,expected_container_id=cid,max_snapshots=2)
    complete=json.loads((tmp_path/'out/complete.json').read_bytes())
    assert complete['terminal_observed'] is False
    assert complete['termination_reason']=='snapshot-limit'
    assert captured==[0,600]
    assert collector_clock.now==600


def test_terminal_and_stop_file_are_mutually_exclusive(tmp_path):
    import json
    import server_observability as m
    config=tmp_path/'profile.json';config.write_text(json.dumps({'mode':'normal','client_detail':False,'server_metrics_interval_seconds':600}))
    with pytest.raises(ValueError):m.collect_windows({},window_id='run',mode='normal',config=config,out=tmp_path/'out',stop_file=tmp_path/'stop',terminal_status=tmp_path/'status',expected_container_id='a'*64,max_snapshots=2)


@pytest.mark.parametrize('terminal_state',['exited','dead'])
def test_event_after_periodic_snapshot_is_collected_exactly_once(tmp_path,monkeypatch,collector_clock,terminal_state):
    import json
    import server_observability as m
    cid='d'*64;status=tmp_path/'status.json';config=tmp_path/'profile.json'
    config.write_text(json.dumps({'mode':'normal','client_detail':False,'server_metrics_interval_seconds':600}))
    # Start missing, then receive a live launcher update, then a failed terminal.
    def update(now):
        if now==5:status.write_text(json.dumps({'container_id':cid,'state':'running','running':True}))
        if now==607:status.write_text(json.dumps({'container_id':cid,'state':terminal_state,'running':False,'exit_code':9}))
    collector_clock.on_sleep=update
    captured=[]
    def capture(*a,**kw):
        captured.append(collector_clock.now)
        return snap(len(captured),len(captured)*2)
    monkeypatch.setattr(m,'snapshot',capture)
    m.collect_windows({},window_id='run',mode='normal',config=config,out=tmp_path/'out',terminal_status=status,expected_container_id=cid,max_snapshots=10)
    assert captured==[0,600,607]
    complete=json.loads((tmp_path/'out/complete.json').read_bytes())
    assert complete['terminal_observed'] and complete['snapshots']==3
    assert complete['terminal_status']['exit_code']==9
    assert complete['scientific_success']=='verify shared runtime artifacts separately'
    assert len(list((tmp_path/'out').glob('*.delta.json')))==2


@pytest.mark.parametrize('during_collection',[False,True])
def test_changed_status_during_wait_or_collection_never_writes_completion(tmp_path,monkeypatch,collector_clock,during_collection):
    import json
    import server_observability as m
    cid='d'*64;status=tmp_path/'status.json';config=tmp_path/'profile.json'
    config.write_text(json.dumps({'mode':'normal','client_detail':False,'server_metrics_interval_seconds':600}))
    status.write_text(json.dumps({'container_id':cid,'state':'running','running':True}))
    def replace():status.write_text(json.dumps({'container_id':'e'*64,'state':'exited','running':False}))
    def capture(*a,**kw):
        if during_collection:replace()
        return snap(1,2)
    if not during_collection:collector_clock.on_sleep=lambda now:replace()
    monkeypatch.setattr(m,'snapshot',capture)
    with pytest.raises(ValueError,match='identity changed'):
        m.collect_windows({},window_id='run',mode='normal',config=config,out=tmp_path/'out',terminal_status=status,expected_container_id=cid,max_snapshots=2)
    assert not (tmp_path/'out/complete.json').exists()


@pytest.mark.parametrize('raw',['{', 'null', '\ufffd'])
def test_corrupt_status_fails_before_metrics(tmp_path,monkeypatch,raw):
    import json
    import server_observability as m
    config=tmp_path/'profile.json';config.write_text(json.dumps({'mode':'normal','client_detail':False,'server_metrics_interval_seconds':600}))
    status=tmp_path/'status.json';status.write_text(raw)
    def unexpected(*a,**kw):pytest.fail('metrics must not be contacted')
    monkeypatch.setattr(m,'snapshot',unexpected)
    with pytest.raises(ValueError):
        m.collect_windows({},window_id='run',mode='normal',config=config,out=tmp_path/'out',terminal_status=status,expected_container_id='a'*64,max_snapshots=2)
    assert not (tmp_path/'out').exists()


def test_stop_event_during_snapshot_gets_a_later_final_snapshot(tmp_path,monkeypatch,collector_clock):
    import json
    import server_observability as m
    config=tmp_path/'profile.json';config.write_text(json.dumps({'mode':'normal','client_detail':False,'server_metrics_interval_seconds':600}))
    stop=tmp_path/'done';captures=[]
    def capture(*a,**kw):
        captures.append(stop.exists())
        stop.touch()
        return snap(1,2)
    monkeypatch.setattr(m,'snapshot',capture)
    m.collect_windows({},window_id='run',mode='normal',config=config,out=tmp_path/'out',stop_file=stop,max_snapshots=2)
    assert captures==[False,True]
    assert collector_clock.now==0


@pytest.mark.parametrize('options',[
    [], ['--terminal-status','/status'], ['--expected-container-id','a'*64],
    ['--terminal-status','/status','--expected-container-id','a'*12],
    ['--terminal-status','/status','--expected-container-id','G'*64],
    ['--stop-file','/stop','--expected-container-id','a'*64],
    ['--stop-file','/stop','--terminal-status','/status','--expected-container-id','a'*64],
])
def test_cli_rejects_unpaired_or_ambiguous_stop_source(monkeypatch,options):
    import server_observability as m
    monkeypatch.setattr(sys,'argv',['observer','watch','--endpoint','i=http://localhost/metrics','--window-id','run','--config','/config','--out','/out',*options])
    with pytest.raises(SystemExit) as error:m.main()
    assert error.value.code==2
