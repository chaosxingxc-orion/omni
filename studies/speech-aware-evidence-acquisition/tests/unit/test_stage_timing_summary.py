import importlib.util
import json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('summary',Path(__file__).resolve().parents[2]/'scripts/analysis/summarize_stage_timing.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_nested_not_double_counted_and_partial_tail(tmp_path):
    p=tmp_path/'events';events=[{'schema':'saea-stage-timing-v1','pid':1,'thread':1,'request_id':'r','sample_id':'s','status':'ok','stage':stage,'elapsed_seconds':seconds} for stage,seconds in [('request_total',10),('transport_total',8),('http_request',6)]]
    p.write_text(''.join(json.dumps(e)+'\n' for e in events)+'{"partial"')
    r=m.summarize(p)
    assert r['incomplete_tail'] and r['observed_sample_count']==1
    assert r['request_validation_accounting_and_response_handling_s']['mean_s']==2
    assert r['http_request_s']['mean_s']==6
    p.write_text(''.join(json.dumps(e)+'\n' for e in events+[events[0]]))
    with pytest.raises(ValueError,match='duplicate'):m.summarize(p)


def test_complete_sample_filter_does_not_mix_partial_costs(tmp_path):
    p=tmp_path/'events'
    events=[{'schema':'saea-stage-timing-v1','pid':1,'thread':1,'request_id':sample,'sample_id':sample,'status':'ok','stage':'request_total','elapsed_seconds':seconds} for sample,seconds in [('done',10),('inflight',100)]]
    p.write_text(''.join(json.dumps(e)+'\n' for e in events))
    result=m.summarize(p,{'done','missing'})
    assert result['events']==1 and result['total_journal_events']==2
    assert result['stages']['request_total']['invocations']['mean_s']==10
    assert result['completed_rows_without_timing']==['missing']
    assert result['sample_scope']=='verified-complete-checkpoint-rows'


def _debug(stage, *, sample=None, status='ok', values=None):
    event={'schema':'saea-stage-timing-v1','pid':1,'thread':1,'stage':stage,
        'status':status,'elapsed_seconds':2,'detail':'debug','observability_mode':'debug',
        'run_id':'run1','attempt_id':'attempt1','measurements':values or {}}
    if sample:event['sample_id']=sample
    return event


def _write(path,events):
    path.write_text(''.join(json.dumps(event)+'\n' for event in events))


def test_debug_measurements_keep_failed_and_missing_distinct(tmp_path):
    p=tmp_path/'timing'
    _write(p,[_debug('runtime_identity_full_hash_process_floor',values={'verified_bytes':4096,'file_count':1}),
        _debug('runtime_identity_full_hash_process_floor',status='error'),
        _debug('runtime_identity_cache_hit_memory',values={'verified_bytes':4096,'cache_hits':1})])
    r=m.summarize(p)
    hashed=r['stages']['runtime_identity_full_hash_process_floor']
    assert hashed['errors']==1
    assert hashed['measurements']['verified_bytes']=={'observed_count':1,'missing_count':1,'sum':4096,'mean':4096,'min':4096,'max':4096}
    assert r['stages']['runtime_identity_cache_hit_memory']['measurements']['cache_hits']['sum']==1
    assert r['observability']['configuration_verified'] is False
    assert r['observability']['run_id']=='run1'


def test_sample_filter_preserves_separate_unscoped_startup(tmp_path):
    p=tmp_path/'timing'
    _write(p,[_debug('startup',values={'verified_bytes':99}),_debug('work',sample='done'),_debug('work',sample='inflight')])
    result=m.summarize(p,{'done'})
    assert result['events']==1 and result['total_journal_events']==3
    assert set(result['stages'])=={'work'}
    assert result['process_scope']['events']==1
    assert result['process_scope']['stages']['startup']['measurements']['verified_bytes']['sum']==99


def test_configuration_hash_and_identity_are_verified(tmp_path):
    import hashlib
    p=tmp_path/'timing';_write(p,[_debug('work')])
    effective={'mode':'debug','client_detail':True,'server_metrics_interval_seconds':600,'run_id':'run1','attempt_id':'attempt1'}
    digest=hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()
    snapshot={'schema':'saea-observability-config-v1','effective':effective,'configuration_sha256':digest}
    sidecar=p.with_name(p.name+'.config.json');sidecar.write_text(json.dumps(snapshot))
    assert m.summarize(p)['observability']['configuration_sha256']==digest
    snapshot['configuration_sha256']='0'*64;sidecar.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError,match='hash mismatch'):m.summarize(p)
    snapshot['configuration_sha256']=digest;sidecar.write_text(json.dumps(snapshot))
    event=_debug('work');event['run_id']='other';_write(p,[event])
    with pytest.raises(ValueError,match='identity differs'):m.summarize(p)


def test_legacy_has_unknown_configuration_and_no_debug_values(tmp_path):
    p=tmp_path/'timing'
    event=_debug('work');event.pop('observability_mode');event.pop('run_id');event.pop('attempt_id');event.pop('measurements')
    _write(p,[event]);r=m.summarize(p)
    assert r['observability']['effective'] is None
    assert r['observability']['observability_mode'] is None
    assert r['stages']['work']['measurements']=={}


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,True,'bad'])
def test_invalid_numeric_measurements_are_rejected(tmp_path,value):
    p=tmp_path/'timing';_write(p,[_debug('work',values={'verified_bytes':value})])
    with pytest.raises(ValueError,match='invalid debug measurement'):m.summarize(p)


def test_mixed_attempts_rejected_and_raw_fields_not_copied(tmp_path):
    p=tmp_path/'timing';first=_debug('work',values={'prompt':'secret','request_bytes':3})
    _write(p,[first]);r=m.summarize(p)
    assert 'secret' not in json.dumps(r)
    second=_debug('work');second['attempt_id']='other';_write(p,[first,second])
    with pytest.raises(ValueError,match='mixed observability'):m.summarize(p)
