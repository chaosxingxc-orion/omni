import base64
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import h1_thinking_comparison as m
from test_tfrl_h1_observation_read import inputs


def test_identical_arms_have_zero_paired_deltas_and_no_repairs():
    result,items=inputs();out=m.paired_scores(dict.fromkeys(m.ARMS,result),items,draws=20)
    for arm in ('off','short'):
        assert out['comparisons'][arm]['task_repairs']==0
        assert out['comparisons'][arm]['task_breaks']==0
        for ci in out['comparisons'][arm]['contrasts'].values():
            assert ci['point_estimate']==ci['ci_low']==ci['ci_high']==0
    assert not out['automatic_adoption']


@pytest.mark.parametrize('fault',['missing_arm','roster','scorer','slots'])
def test_refuses_noncomparable_results(fault):
    result,items=inputs();results={a:copy.deepcopy(result) for a in m.ARMS}
    if fault=='missing_arm':results.pop('short')
    elif fault=='roster':results['off']['per_sample'].pop()
    elif fault=='scorer':results['off']['aggregate']['evaluator_commit']='different'
    else:results['off']['per_sample'][0]['slot_f1_exact_fn']=-1
    with pytest.raises(ValueError):m.paired_scores(results,items,draws=2)


def test_only_declared_policy_and_run_identity_are_normalized():
    base={'thinking_policy':{},'prepared_phase_supply':{},'what_this_is':'x',
        'native_contact_fit':{'run_id':'old','attempt_id':'fit','contact_cap':626},'prompt':'same'}
    other=copy.deepcopy(base);other['thinking_policy']={'perception':False};other['native_contact_fit']['run_id']='new'
    assert m.invariant(base)==m.invariant(other)
    other['prompt']='changed'
    assert m.invariant(base)!=m.invariant(other)


def test_costs_keep_unattributed_fallback_and_wire_reasoning():
    rows=[{'native_episode_receipt':{'role_visits':[{'role_id':'perception',
        'execution_receipt':{'contacts':[{'request_id':'p'}]}}]}}]
    cost={'calls':1,'prompt_tokens':10,'completion_tokens':3,'latency_seconds':2.0}
    wire={'choices':[{'message':{'content':'{}','reasoning_content':'thinking'},'finish_reason':'stop'}]}
    trace={'p/cost':cost,'fallback/cost':cost,'p/wire-response':{
        'response_b64':base64.b64encode(json.dumps(wire).encode()).decode()}}
    out=m.native_costs(rows,trace)
    assert out['perception']['contacts_with_reasoning']==1
    assert out['unattributed']['calls']==1 and out['unattributed']['wire_records']==0
    del trace['p/cost']
    with pytest.raises(ValueError,match='lack native cost'):m.native_costs(rows,trace)


def test_pooled_slu_and_binary_task_repairs_are_distinct():
    result,items=inputs();results={a:copy.deepcopy(result) for a in m.ARMS}
    # Repair one formerly wrong row with large word/char weight: pooled delta
    # must use counts, while task repairs still count this as one sample.
    row=results['off']['per_sample'][200]
    row.update(scenario_action_exact=1,slot_f1_exact_fp=0,slot_f1_exact_fn=0)
    for kind in ('word','char'):
        row.update({f'slu_f1_{kind}_tp':20,f'slu_f1_{kind}_fp':0,f'slu_f1_{kind}_fn':0})
    ids=[i['sample_id'] for i in items]
    table={r['sample_id']:r for r in results['off']['per_sample']}
    value=m.pooled_slu_f1(table,ids);results['off']['aggregate']['official']['slu_f1']=value
    out=m.paired_scores(results,items,draws=20)['comparisons']['off']
    assert out['task_repairs']==1 and out['task_breaks']==0
    assert out['contrasts']['slu_f1_pp']['point_estimate']==pytest.approx(100*(value-200/313))
    assert out['contrasts']['intent_pp']['point_estimate']==pytest.approx(100/313)


def test_timing_refuses_checkpoint_from_another_arm_roster(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core import row_checkpoint
    monkeypatch.setattr(row_checkpoint, 'read_checkpoint', lambda _, **kwargs: SimpleNamespace(
        rows=[SimpleNamespace(sample_id='other', keep_defect=lambda: None)]))
    with pytest.raises(ValueError, match='checkpoint differs'):
        m.phase_timing(tmp_path, 'run', tmp_path/'timing', ['expected'], 'a'*64)


def test_timing_keeps_incomplete_coverage_visible(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core import row_checkpoint
    import real_engine_phase_timing
    monkeypatch.setattr(row_checkpoint, 'read_checkpoint', lambda _, **kwargs: SimpleNamespace(
        rows=[SimpleNamespace(sample_id='expected', keep_defect=lambda: None)]))
    timing = tmp_path/'timing'; timing.write_text('{}\n')
    monkeypatch.setattr(real_engine_phase_timing, 'analyze', lambda *_: {
        'rows': 1, 'missing_timing_entries': 1, 'timing_partial_tail': False})
    result = m.phase_timing(tmp_path, 'run', timing, ['expected'], 'a'*64)
    assert not result['complete_request_timing']
    assert result['timing_sha256'] == m.sha(timing)


def test_timing_refuses_a_changing_source(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core import row_checkpoint
    import real_engine_phase_timing
    monkeypatch.setattr(row_checkpoint, 'read_checkpoint', lambda _, **kwargs: SimpleNamespace(
        rows=[SimpleNamespace(sample_id='expected', keep_defect=lambda: None)]))
    timing = tmp_path/'timing'; timing.write_text('{}\n')
    def analyze(*_):
        timing.write_text('{}\n{}\n')
        return {'rows': 1, 'missing_timing_entries': 0, 'timing_partial_tail': False}
    monkeypatch.setattr(real_engine_phase_timing, 'analyze', analyze)
    with pytest.raises(ValueError, match='changed'):
        m.phase_timing(tmp_path, 'run', timing, ['expected'], 'a'*64)
