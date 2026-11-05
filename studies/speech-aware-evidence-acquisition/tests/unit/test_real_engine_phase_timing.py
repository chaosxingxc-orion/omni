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
    assert m.join(rows,events+[events[0]])['row_classification']['unsupported_row'] == 1


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
    assert m.join(rows, [])['row_classification']['unsupported_row'] == 1


def test_empty_rows_do_not_claim_complete_timing():
    result = m.join([], [])
    assert result['sample_timing_coverage']['expected_rows'] == 0
    assert not result['all_timing_complete']


def test_slot_partition_clips_and_unions_only_pre_p_waits():
    def event(start, seconds, sample='s'):
        return {'stage': 'dispatch_trajectory_slot_wait', 'sample_id': sample,
            'pid': 1, 'status': 'ok', 'started_monotonic': start, 'elapsed_seconds': seconds}
    events = [event(85, 10), event(94, 4), event(110, 10), event(90, 10, 'other')]
    assert m.slot_wait_partition(events, sample_id='s', pid=1,
        admitted=90, request_started=100) == (8, 2)
    assert m.slot_wait_partition([], sample_id='s', pid=1,
        admitted=90, request_started=100) is None


def test_slot_wait_join_retains_unknown_and_exposes_residual():
    rows, events = inputs(); add_sample_boundary(rows[0])
    events.append({'stage': 'dispatch_trajectory_slot_wait', 'sample_id': 's',
        'pid': 1, 'status': 'ok', 'started_monotonic': 92, 'elapsed_seconds': 6})
    result = m.join(rows, complete_events(events[:2]) + events[2:])
    assert result['metrics']['pre_perception_observed_slot_wait']['mean_s'] == 6
    assert result['metrics']['pre_perception_other_elapsed']['mean_s'] == 4
    assert result['pre_perception_slot_coverage'] == {'joined_rows': 1, 'rows_without_observed_wait': 0}
    missing = m.join(rows, events[:2])
    assert missing['pre_perception_slot_coverage']['rows_without_observed_wait'] == 1
    assert 'pre_perception_observed_slot_wait' not in missing['metrics']


@pytest.mark.parametrize('change', [{'pid': 2}, {'elapsed_seconds': -1},
    {'elapsed_seconds': float('nan')}, {'status': 'error'}])
def test_invalid_slot_intervals_cannot_be_attributed(change):
    event = {'stage': 'dispatch_trajectory_slot_wait', 'sample_id': 's',
        'pid': 1, 'status': 'ok', 'started_monotonic': 92, 'elapsed_seconds': 6, **change}
    with pytest.raises(ValueError):
        m.slot_wait_partition([event], sample_id='s', pid=1, admitted=90, request_started=100)


def test_three_nodes_and_refusal_are_distinct():
    rows, events = inputs()
    output = {'native_episode_receipt': {'accepted_solver_receipt': {}, 'role_visits': []}}
    output['native_episode_receipt']['accepted_solver_receipt'] = {'ok': True}
    for role, rid in [('perception', 'z'), ('slot', 'x'), ('solver', 'a')]:
        output['native_episode_receipt']['role_visits'].append({'role_id': role, 'execution_receipt': {'contacts': [{'request_id': rid}]}})
    rows[0].output_rows = [output]
    rows[0].trace_lines.append(json.dumps({'record_id': 'x/cost', 'payload': {'latency_seconds': 3}}))
    events[1]['started_monotonic'] = 120
    events.append({**events[0], 'request_id':'x', 'started_monotonic':112, 'elapsed_seconds':3})
    result = m.join(rows, events)
    assert result['metrics']['perception_end_to_slot_request_start']['mean_s'] == 2
    assert result['metrics']['slot_end_to_solver_request_start']['mean_s'] == 5
    assert 'perception_end_to_solver_request_start' not in result['metrics']
    output['native_episode_receipt']['terminal_disposition'] = {'reason': 'seal_refusal'}
    result = m.join(rows, events)
    assert result['row_classification']['refusal'] == 1
    assert not result['metrics'] and result['terminal_metrics']


def test_empty_refusal_finite_and_unsupported_counted():
    rows, events = inputs()
    rows[0].output_rows = [{'native_episode_receipt': {'role_visits': [], 'terminal_disposition': {'reason': 'seal_refusal'}}}]
    result = m.join(rows, events)
    assert result['row_classification']['refusal'] == result['row_classification']['empty_chain'] == 1
    rows[0].output_rows[0]['native_episode_receipt']['terminal_disposition']['reason'] = 'finite_completion'
    assert m.join(rows, events)['row_classification']['finite_or_other_terminal'] == 1
    rows, events = inputs()
    rows[0].output_rows = [{'native_episode_receipt': {'role_visits': [{'role_id':'x', 'execution_receipt': {'contacts':[{'request_id':'a'}, {'request_id':'b'}]}}]}}]
    assert m.join(rows, events)['row_classification']['unsupported_chain'] == 1


@pytest.mark.parametrize('change', [{'pid':2}, {'elapsed_seconds':-1}, {'started_monotonic':float('nan')}, {'started_monotonic':105}])
def test_invalid_request_boundaries_rejected(change):
    rows, events = inputs(); events[1].update(change)
    assert m.join(rows, events)['row_classification']['unsupported_row'] == 1


def test_repeated_request_id_in_chain_is_unsupported():
    rows, events = inputs()
    rows[0].output_rows = [{'native_episode_receipt': {'role_visits': [{'role_id':r, 'execution_receipt': {'contacts':[{'request_id':'z'}]}} for r in ('perception','solver')]}}]
    assert m.join(rows, events)['row_classification']['unsupported_chain'] == 1


@pytest.mark.parametrize('change,reason', [({'pid': 2}, 'request_http_pid_mismatch'),
    ({'started_monotonic': 99}, 'http_outside_request'), ({'status': 'error'}, 'request_status_not_ok')])
def test_http_mismatch_isolated_per_row(change, reason):
    rows, events = inputs(); events = complete_events(events)
    events[2].update(change)
    result = m.join(rows, events, include_rows=True)
    assert result['sample_rows'][0]['unsupported_reason'] == reason
    assert not result['all_timing_complete']


def test_invalid_receipt_and_unsupported_do_not_claim_all_complete():
    rows, events = inputs(); rows[0].output_rows = [{'native_episode_receipt': None}]
    result = m.join(rows, events, include_rows=True)
    assert result['row_classification']['invalid_receipt'] == 1
    assert result['processed_rows'] == 1 and not result['all_timing_complete']


def test_bad_row_does_not_discard_other_row():
    rows, events = inputs(); bad, bad_events = inputs()
    bad[0].sample_id = 'other'; bad_events = [{**e, 'sample_id':'other'} for e in bad_events]
    bad_events[1]['elapsed_seconds'] = -1
    result = m.join(rows + bad, events + bad_events, include_rows=True)
    assert result['processed_rows'] == 2 and result['rows'] == 1
    assert result['row_classification']['unsupported_row'] == 1
    assert result['metrics']['perception/scientific_request']['n'] == 1
