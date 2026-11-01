import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import tfgrpo_h2_read as m
from test_tfrl_current_half_result import result


def inputs():
    items=[{'sample_id':f'row{i}','utterance_id':f'u{i//2}','stratum':'alarm' if i<156 else 'calendar'} for i in range(312)]
    results={}
    for arm in ('learned','lock'):
        rows=[]
        for i,item in enumerate(items):
            ok=i<200 if arm=='lock' else i<210
            row=result(ok)['per_sample'][0];row['sample_id']=item['sample_id'];rows.append(row)
        aggregate=result()['aggregate'];aggregate.update(samples=312)
        aggregate['official']['slu_f1']=(200 if arm=='lock' else 210)/312
        results[arm]={'per_sample':rows,'aggregate':aggregate}
    return results,items


def test_paired_counts_and_gate_never_claim_promotion():
    results,items=inputs();out=m.statistics(results,items,draws=20)
    assert out['metrics']['slu_f1']['point_pp']==pytest.approx(1000/312)
    assert out['numeric_gate_passed'] and not out['candidate_accepted'] and not out['claims_stable_gain']
    assert out['scenario_table']['calendar']['delta_pp']['intent']>0
    same=m.statistics({'learned':results['lock'],'lock':results['lock']},items,draws=20)
    assert same['metrics']['slu_f1']['ci95_lo_pp']==same['metrics']['slu_f1']['ci95_hi_pp']==0
    assert not same['numeric_gate_passed']


@pytest.mark.parametrize('fault',['missing','duplicate','scorer','micro','cluster','intent_regression'])
def test_refuse_partial_or_misbound_comparison(fault):
    results,items=inputs()
    if fault=='missing':results['learned']['per_sample'].pop()
    elif fault=='duplicate':results['learned']['per_sample'][-1]=results['learned']['per_sample'][0]
    elif fault=='scorer':results['learned']['aggregate']['evaluator_commit']='other'
    elif fault=='micro':results['learned']['aggregate']['official']['slu_f1']=.1
    elif fault=='cluster':items[200]['utterance_id']=items[0]['utterance_id']
    else:
        for row in results['learned']['per_sample']:row['scenario_action_exact']=0
        assert not m.statistics(results,items,draws=20)['numeric_gate_passed'];return
    with pytest.raises(ValueError):m.statistics(results,items,draws=20)


def test_label_slices_use_observed_relations_and_expose_regressions():
    results, items = inputs()
    labels = {row['sample_id']: {'intent': 'alarm_set' if i < 156 else 'calendar_set',
              'slots': ['time', 'date'] if i < 156 else []} for i, row in enumerate(items)}
    out = m.label_diagnostics(results, items, labels)
    assert len(out['observed_intent_slot']) == 2
    assert {row['intent'] for row in out['observed_intent_slot']} == {'alarm_set'}
    assert sum(row['rows'] for row in out['intent']) == 312
    assert [row['rows'] for row in out['slot_presence']] == [156, 156]
    assert out['no_gold_slots']['rows'] == 156
    calendar = next(row for row in out['intent'] if row['label'] == 'calendar_set')
    assert calendar['intent_repaired_rows'] == 10
    assert calendar['intent_regressed_rows'] == 0
    assert calendar['delta_pp']['intent'] == pytest.approx(1000/156)
    swapped = m.label_diagnostics({'learned':results['lock'], 'lock':results['learned']}, items, labels)
    assert swapped['no_gold_slots']['intent_regressed_rows'] == 10
    assert 'sample_id' not in str(out) and 'row0' not in str(out)


@pytest.mark.parametrize('fault', ['missing', 'duplicate_slot', 'empty_intent', 'partial_scores'])
def test_label_diagnostics_refuse_incomplete_or_ambiguous_membership(fault):
    results, items = inputs()
    labels = {row['sample_id']: {'intent': 'alarm_set', 'slots': ['time']} for row in items}
    if fault == 'missing': labels.pop('row0')
    elif fault == 'duplicate_slot': labels['row0']['slots'] = ['time', 'time']
    elif fault == 'empty_intent': labels['row0']['intent'] = ''
    else: results['learned']['per_sample'].pop()
    with pytest.raises(ValueError): m.label_diagnostics(results, items, labels)


def test_original_labels_bind_split_utterance_and_scenario():
    from types import SimpleNamespace as NS
    item = {'sample_id':'slurp:train:1', 'utterance_id':'u1', 'stratum':'alarm'}
    record = NS(sample_id=item['sample_id'], utterance_id='u1', split='train', scenario='alarm',
                intent='alarm_set', entities=[NS(type='time'), NS(type='time')])
    assert m.label_membership([item], [record]) == {item['sample_id']:{'intent':'alarm_set','slots':['time']}}
    for field, value in [('utterance_id', 'wrong'), ('stratum', 'wrong'), ('sample_id', 'slurp:test:1')]:
        with pytest.raises(ValueError):m.label_membership([{**item, field:value}], [record])
    with pytest.raises(ValueError):m.label_membership([item], [record, record])


def test_original_numeric_utterance_matches_real_namespaced_roster():
    from types import SimpleNamespace as NS
    item={'sample_id':'slurp:train:000049:audio-real','utterance_id':'slurp:train:000049','stratum':'alarm'}
    record=NS(sample_id=item['sample_id'],utterance_id='49',split='train',scenario='alarm',intent='alarm_set',entities=[])
    assert m.label_membership([item],[record])[item['sample_id']]['intent']=='alarm_set'
    record.utterance_id='50'
    with pytest.raises(ValueError,match='identity'):m.label_membership([item],[record])
    assert not m.same_train_utterance('slurp:test:000049','49')
