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


def test_sample_residual_wait_is_joined_per_row():
    rows, events = inputs()
    rows[0].trace_lines.append(json.dumps({'record_id': 'sample-timing', 'payload': {'use_sample_execution_timing': {'admitted_ns': 90_000_000_000, 'finished_ns': 137_000_000_000}}}))
    result = m.join(rows, events)['metrics']
    assert result['admission_to_perception_request_start']['mean_s'] == 10
    assert result['solver_request_end_to_sample_finished']['mean_s'] == 2
    assert result['admission_to_sample_finished']['mean_s'] == 47


def add_sample_boundary(row, record_id='sample-timing'):
    row.trace_lines.append(json.dumps({'record_id': record_id, 'payload': {
        'use_sample_execution_timing': {
            'admitted_ns': 90_000_000_000, 'finished_ns': 137_000_000_000}}}))


def complete_events(events):
    return events + [{**event, 'stage': 'http_request'} for event in events]


def test_missing_sample_boundary_is_visible_despite_complete_request_events():
    rows, events = inputs()
    result = m.join(rows, complete_events(events))
    assert result['missing_timing_entries'] == result['missing_sample_count'] == 0
    assert result['sample_timing_coverage'] == {
        'expected_rows': 1, 'boundary_present_rows': 0, 'boundary_missing_rows': 1,
        'joined_phase_rows': 0, 'unjoined_phase_rows': 1}
    assert not result['all_timing_complete']


def test_complete_sample_boundary_and_request_events_report_complete():
    rows, events = inputs()
    add_sample_boundary(rows[0])
    result = m.join(rows, complete_events(events))
    assert result['sample_timing_coverage'] == {
        'expected_rows': 1, 'boundary_present_rows': 1, 'boundary_missing_rows': 0,
        'joined_phase_rows': 1, 'unjoined_phase_rows': 0}
    assert result['all_timing_complete']


def test_present_boundary_without_role_event_is_not_a_completed_join():
    rows, events = inputs()
    add_sample_boundary(rows[0])
    result = m.join(rows, events[:1])
    coverage = result['sample_timing_coverage']
    assert coverage['boundary_present_rows'] == 1
    assert coverage['boundary_missing_rows'] == 0
    assert coverage['joined_phase_rows'] == 0
    assert coverage['unjoined_phase_rows'] == 1
    assert not result['all_timing_complete']


@pytest.mark.parametrize('same_id', [True, False])
def test_duplicate_sample_boundaries_refused_even_without_role_events(same_id):
    rows, events = inputs()
    add_sample_boundary(rows[0])
    add_sample_boundary(rows[0], 'sample-timing' if same_id else 'other-timing')
    with pytest.raises(ValueError, match='ambiguous sample execution timing'):
        m.join(rows, [])


def test_empty_rows_do_not_claim_complete_timing():
    result = m.join([], [])
    assert result['sample_timing_coverage']['expected_rows'] == 0
    assert not result['all_timing_complete']
