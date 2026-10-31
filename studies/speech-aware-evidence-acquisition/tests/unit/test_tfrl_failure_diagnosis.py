import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import tfrl_failure_diagnosis as m


def result(intent,tp,fp,fn):
    row={'sample_id':'r','scenario_action_exact':intent}
    for kind in ('word','char'):
        for k,v in [('tp',tp),('fp',fp),('fn',fn)]:row[f'slu_f1_{kind}_{k}']=v
    for k,v in [('tp',tp),('fp',fp),('fn',fn)]:row['slot_f1_exact_'+k]=v
    return {'per_sample':[row],'aggregate':{'samples':1,'identity_check':{'passed':True},
        'official':{'slu_f1':2*tp/(2*tp+fp+fn) if tp+fp+fn else 0}}}


def test_vote_can_improve_intent_while_losing_slots_and_candidate_headroom():
    r={a:result(1,2,0,0) for a in m.ARMS}
    r['lock']=result(0,1,0,1);r['vote']=result(1,0,0,2)
    before=copy.deepcopy(r)
    x=m.diagnose({'row_plan':[{'row_id':'r'}]},r)
    assert r==before
    assert x['counts']['intent_lock0_vote1']==1
    assert x['counts']['vote_row_f1_worse_lock']==1
    assert x['counts']['oracle_improvement_missed_by_vote']==1
    assert x['counts']['vote_loses_exact_slot_tp_vs_lock']==1
    assert x['vote_minus_lock_slu_counts']['word']=={'tp':-1,'fp':0,'fn':1}


def test_correct_minority_candidate_does_not_imply_correct_vote():
    r={a:result(0,0,1,1) for a in (*m.ARMS,'vote')};r['s2']=result(1,1,0,0)
    x=m.diagnose({'row_plan':[{'row_id':'r'}]},r)
    assert x['counts']['correct_candidate_intent_lost_by_vote']==1
    assert x['counts']['all_candidates_wrong_intent']==0


@pytest.mark.parametrize('fault',['missing','duplicate','slot_nan','slot_negative','micro'])
def test_incomplete_or_invalid_counts_refused(fault):
    r={a:result(1,1,0,0) for a in (*m.ARMS,'vote')};plan={'row_plan':[{'row_id':'r'}]}
    if fault=='missing':del r['s3']
    elif fault=='duplicate':plan['row_plan']*=2
    elif fault=='micro':r['vote']['aggregate']['official']['slu_f1']=.5
    else:r['vote']['per_sample'][0]['slot_f1_exact_tp']=float('nan') if fault=='slot_nan' else -1
    with pytest.raises(ValueError):m.diagnose(plan,r)
