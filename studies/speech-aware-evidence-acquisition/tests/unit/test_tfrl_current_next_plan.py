from pathlib import Path
import copy,sys
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/analysis'))
import tfrl_current_next_plan as m
from test_tfrl_v2_consumption import make_bank,SCOPE,H1,H2


def inputs(tmp_path):
    rosters={h:{'roster_id':m.sampler.cm.HALF_ROSTER_ID[half],'split_identity_hash':SCOPE[half],
        'items':[{'sample_id':f'{h}-{i}','audio_seconds':1.0} for i in range(n)]}
        for h,half,n in [('h1',H1,313),('h2',H2,312)]}
    receipts={}
    for h,half in [('h1',H1),('h2',H2)]:
        _,r=make_bank(tmp_path/h,kind='half',sources=(half,),source_rows=(h+'-0',))
        receipts[h]=r
    return rosters,receipts


def test_crossfit_held_plans_have_current_seeds_without_synthetic_capacity(tmp_path):
    rosters,receipts=inputs(tmp_path);before=copy.deepcopy(receipts)
    plans=m.derive(rosters,receipts,iteration=2)
    assert receipts==before
    for h,p in plans.items():
        assert p['iteration']==2 and p['banks']['supply_bound_items'] is None
        assert p['banks']['bank_source_halves']==['R01C-'+('H2' if h=='h1' else 'H1')]
        assert len(m.sampler.execution_columns(p))==4
        assert all(not b['cost_forecast_available'] for b in p['budget'].values())


@pytest.mark.parametrize('fault',['self','future','row_overlap','source_identity','too_many_iterations'])
def test_refuses_invalid_crossfit_handoff(tmp_path,fault):
    r,b=inputs(tmp_path);iteration=2
    if fault=='self':b['h2']=b['h1']
    elif fault=='future':b['h2']['iteration']=2
    elif fault=='row_overlap':r['h2']['items'][0]['sample_id']=r['h1']['items'][0]['sample_id']
    elif fault=='source_identity':b['h2']['source_split_identities']={H2:'f'*64}
    else:iteration=4
    with pytest.raises((ValueError,m.sampler.SamplerError)):m.derive(r,b,iteration=iteration)


def test_explicit_actual_bounds_replace_legacy_forecast(tmp_path):
    rosters,receipts=inputs(tmp_path)
    plans=m.derive(rosters,receipts,iteration=2,control_bounds={'h1':55,'h2':55})
    for plan in plans.values():
        assert plan['banks']['supply_bound_items']==47
        assert plan['banks']['supply_allocation']['control_max_items']==55


@pytest.mark.parametrize('gain,flag',[(.9,False),(float('nan'),True),(1.1,False)])
def test_first_iteration_kill_gate_cannot_be_bypassed(gain,flag):
    with pytest.raises(ValueError):m.require_first_construction_gate({'rows':625,'iteration':1,
        'vote_minus_lock_pp':gain,'construction_gate_passed':flag,'threshold_pp':1.0})


def test_construction_pass_is_not_system_acceptance():
    result=m.require_first_construction_gate({'rows':625,'iteration':1,'vote_minus_lock_pp':1.,
        'construction_gate_passed':True,'threshold_pp':1.0})
    assert result['preparation_gate_passed'] and not result['system_accepted']
