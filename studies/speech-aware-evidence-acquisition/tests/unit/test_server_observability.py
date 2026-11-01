import copy
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/analysis'))
from server_observability import parse_metrics,difference

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
