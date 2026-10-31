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
