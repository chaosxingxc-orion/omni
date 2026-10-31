import importlib.util
import json
from pathlib import Path
import pytest
from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter,serialize_trace_records
spec=importlib.util.spec_from_file_location('progress',Path(__file__).resolve().parents[2]/'scripts/analysis/real_engine_progress.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(tmp):
    roster=tmp/'roster.json';roster.write_text(json.dumps({'count':2,'items':[{'sample_id':'a'},{'sample_id':'b'}],'split_identity_hash':'b'*64}))
    w=RowCheckpointWriter(tmp/'attempt','RUN','attempt',identity={'protocol_hash':'a'*64,'split_identity_hash':'b'*64})
    trace=serialize_trace_records([('MODEL_REQUEST','r/request',{'payload':{'sample_id':'a'}}),('COST','r/cost',{'prompt_tokens':20,'completion_tokens':5,'latency_seconds':8,'audio_seconds':2}),('USE','a/timing',{'use_sample_execution_timing':{'batch_started_ns':0,'admitted_ns':1000000000,'finished_ns':10000000000}})])
    w.append_row(sample_id='a',complete=True,output_rows=[{'sample_id':'a','text':'answer'}],trace_lines=trace)
    w.close();return w.path,roster


def test_real_progress_not_contact_count_or_partial_winner(tmp_path):
    ck,r=fixture(tmp_path)
    d=m.report(ck,r,state='running',elapsed_seconds=20,cards=8,instances=2)
    assert d['completed_rows']==1 and d['total_rows']==2
    assert d['wall_seconds_per_complete_row']==20 and d['estimated_remaining_seconds']==20
    assert d['checkpoint_recorded_cost_lower_bound']['prompt_tokens']==20
    assert d['complete_sample_timings']['admitted_to_finished']['mean_s']==9
    assert not d['full_roster_comparison_eligible']
    assert m.report(ck,r,state='failed',elapsed_seconds=20,cards=8,instances=2)['estimated_remaining_seconds'] is None
    with pytest.raises(ValueError,match='full roster'):m.report(ck,r,state='complete',elapsed_seconds=20,cards=8,instances=2)


def test_foreign_roster_and_torn_tail_are_not_completed(tmp_path):
    ck,r=fixture(tmp_path)
    with ck.open('ab') as f:f.write(b'{"partial"')
    d=m.report(ck,r,state='stopped',elapsed_seconds=20,cards=8,instances=2)
    assert d['completed_rows']==1 and d['torn_tail_bytes']>0 and d['estimated_remaining_seconds'] is None
    doc=json.loads(r.read_bytes());doc['split_identity_hash']='c'*64;r.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='roster identity'):m.report(ck,r,state='running',elapsed_seconds=20,cards=8,instances=2)
