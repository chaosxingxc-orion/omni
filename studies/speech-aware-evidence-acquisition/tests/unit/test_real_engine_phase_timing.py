import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
spec=importlib.util.spec_from_file_location('phase',Path(__file__).resolve().parents[2]/'scripts/analysis/real_engine_phase_timing.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def inputs():
    output={'native_episode_receipt':{'perception_branches':{'branches':[{'branch_id':'lock','contact_id':'pc','solver_contact_id':'sc'}]},'role_visits':[{'execution_receipt':{'contacts':[{'agent_contact_id':'pc','request_id':'z'},{'agent_contact_id':'sc','request_id':'a'}]}}]}}
    row=SimpleNamespace(sample_id='s',output_rows=[output],keep_defect=lambda:None,trace_lines=[json.dumps({'record_id':rid+'/cost','payload':{'latency_seconds':duration}}) for rid,duration in [('z',10),('a',20)]])
    events=[{'stage':'request_total','sample_id':'s','request_id':rid,'started_monotonic':start,'elapsed_seconds':duration,'pid':1} for rid,start,duration in [('z',100,10),('a',115,20)]]
    return [row],events


def test_roles_from_receipt_not_request_sort():
    rows,events=inputs();r=m.join(rows,events)
    assert r['metrics']['perception/scientific_request']['mean_s']==10
    assert r['metrics']['solver/scientific_request']['mean_s']==20
    assert r['metrics']['perception_end_to_solver_request_start']['mean_s']==5
    assert r['metrics']['perception_start_to_solver_end']['mean_s']==35
    assert r['missing_timing_entries']==2


def test_duplicate_boundary_refused():
    rows,events=inputs()
    with pytest.raises(ValueError,match='ambiguous'):m.join(rows,events+[events[0]])


def test_c2_roles_without_diagnostic_branches():
    output={'native_episode_receipt':{'role_visits':[{'role_id':role,'execution_receipt':{'contacts':[{'request_id':rid}]}} for role,rid in [('perception','p'),('solver','s')]]}}
    assert m.roles(output)=={'perception':'p','solver':'s'}
