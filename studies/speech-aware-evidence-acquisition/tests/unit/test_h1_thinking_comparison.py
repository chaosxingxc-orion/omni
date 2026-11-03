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


def h2_inputs():
    result, items = inputs()
    result['per_sample'].pop()
    items.pop()
    result['aggregate']['samples'] = 312
    result['aggregate']['official']['slu_f1'] = 200/312
    return result, items


def test_h2_two_arm_paired_screening_uses_312_rows():
    result, items = h2_inputs()
    out = m.paired_scores(dict.fromkeys(('off', 'baseline'), result), items,
                          half='h2', draws=20)
    assert out['rows'] == 312
    assert set(out['comparisons']) == {'off'}
    assert out['comparisons']['off']['task_ties'] == 312
    for ci in out['comparisons']['off']['contrasts'].values():
        assert ci['point_estimate'] == ci['ci_low'] == ci['ci_high'] == 0
    assert not out['automatic_adoption']
    assert m.identity(1, m.config('off', 'h2'))[1] == 'SAEA-SLURP-H2-PERCEPTION-THINKING-OFF-V1'


@pytest.mark.parametrize('fault', ['h1_roster', 'short_arm', 'missing_reference', 'implicit_h1'])
def test_h2_refuses_population_or_arm_substitution(fault):
    result, items = h2_inputs()
    results = dict.fromkeys(('off', 'baseline'), result)
    half = 'h2'
    if fault == 'h1_roster':
        result, items = inputs()
        results = dict.fromkeys(('off', 'baseline'), result)
    elif fault == 'short_arm':
        results['short'] = result
    elif fault == 'missing_reference':
        results.pop('baseline')
    else:
        half = 'h1'
    with pytest.raises(ValueError):
        m.paired_scores(results, items, half=half, draws=2)


def test_read_scope_requires_explicit_h2_and_freeze():
    spec = {'schema': 'saea-h2-thinking-comparison-read-v1', 'data': '/data',
            'arms': {'off': {}, 'baseline': {}}, 'candidate_freeze': {}}
    assert m.read_scope(spec) == ('h2', ('off', 'baseline'), 312)
    for bad in ({k: v for k, v in spec.items() if k != 'candidate_freeze'},
                {**spec, 'schema': 'unknown'},
                {**spec, 'arms': {**spec['arms'], 'short': {}}}):
        with pytest.raises(ValueError):
            m.read_scope(bad)
    legacy = {'schema': 'saea-h1-thinking-comparison-read-v1', 'data': '/data',
              'arms': dict.fromkeys(m.ARMS, {})}
    assert m.read_scope(legacy) == ('h1', m.ARMS, 313)


def test_h2_reader_matches_actual_and_registered_freeze(tmp_path):
    relative = Path('configs/evaluation/agent-thinking-candidate-freeze-v1.json')
    path = tmp_path/relative
    path.parent.mkdir(parents=True)
    path.write_bytes((m.ROOT/relative).read_bytes())
    comparison = m.config('off', 'h2')
    expected = m.frozen_candidate(tmp_path, comparison)
    candidate = {'candidate_freeze': expected}
    assert m.assert_candidate_freeze(tmp_path, comparison, candidate, expected) == expected
    with pytest.raises(ValueError, match='preregistered candidate freeze'):
        m.assert_candidate_freeze(tmp_path, comparison, {}, expected)
    with pytest.raises(ValueError, match='preregistered candidate freeze'):
        m.assert_candidate_freeze(tmp_path, comparison, candidate,
                                  {**expected, 'sha256': '0'*64})
    # A semantically allowed edit still changes the pre-H2 provenance hash.
    value = json.loads(path.read_bytes())
    value['selection_basis'] = 'changed after freeze'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='preregistered candidate freeze'):
        m.assert_candidate_freeze(tmp_path, comparison, candidate, expected)


def test_h2_run_reads_each_own_arm_and_keeps_freeze_provenance(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core import bundle
    result, items = h2_inputs()
    data = tmp_path/'data'
    arms = {}
    calls = []
    relative = Path('configs/evaluation/agent-thinking-candidate-freeze-v1.json')
    for arm in ('off', 'baseline'):
        repo = tmp_path/arm
        freeze = repo/relative
        freeze.parent.mkdir(parents=True)
        freeze.write_bytes((m.ROOT/relative).read_bytes())
        comparison = m.config(arm, 'h2')
        name, rid = m.identity(1, comparison)
        pin = m.frozen_candidate(repo, comparison)
        candidate = {'candidate_freeze': pin, 'roster': {'sha256': 'a'*64},
                     'dataset': 'same', 'runtime_binding': 'same', 'gain_lock': 'same'}
        files = {
            repo/'docs/readiness'/name/'candidate.json': candidate,
            repo/f'configs/baseline/{name}.json': {'native_contact_fit': {'run_id': rid},
                                                'thinking_policy': {'arm': arm}},
            data/'runs'/rid/'generation-v1'/(rid+'.scores.json'): {'slurp-slu-official-train': result}}
        for path, value in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        arms[arm] = {'repo': str(repo), 'attempt_id': 'generation-v1'}
    spec = {'schema': 'saea-h2-thinking-comparison-read-v1', 'data': str(data),
            'arms': arms, 'candidate_freeze': pin}
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(m, 'pinned', lambda *_: {'items': items})
    monkeypatch.setattr(bundle, 'verify_bundle', lambda *_: SimpleNamespace(outputs_rows=[], trace_records={}))
    def observation(repo, data, out, *, comparison, attempt_id):
        calls.append((comparison, attempt_id))
        out.mkdir(parents=True)
        return {'rows': 312}
    monkeypatch.setattr(m.observation, 'run', observation)
    paired = m.paired_scores
    monkeypatch.setattr(m, 'paired_scores', lambda results, items, **kw: paired(results, items, draws=2, **kw))
    out = m.run(path, data/'work/codex/result')
    assert out['schema'] == 'saea-h2-thinking-comparison-result-v1'
    assert out['candidate_freeze'] == pin
    assert out['paired']['rows'] == 312
    assert calls == [(m.config(arm, 'h2'), 'generation-v1') for arm in ('off', 'baseline')]
    assert any('prior development-selection exposure' in v for v in out['paired']['limitations'])
    assert out['new_model_calls'] == out['new_scoring_jobs'] == 0
