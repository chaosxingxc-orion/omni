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
