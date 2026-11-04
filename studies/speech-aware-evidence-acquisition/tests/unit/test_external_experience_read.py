import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/analysis'))
import external_experience_read as m
from test_gepa_pair_read import fixture

def test_actual_two_arm_official_summary_and_no_promotion():
 r,items=fixture();r={'learned':r['child'],'lock':r['lock']};r['lock']['per_sample'][0]['scenario_action_exact']=0
 v=m.summarize(r,items)
 assert v['comparisons']['learned_vs_lock']['strict_repairs']==1
 assert not v['candidate_accepted'] and v['rows']==4

@pytest.mark.parametrize('fault',['missing','duplicate','scorer','nan','negative','bool','intent'])
def test_invalid_evidence_rejected(fault):
 r,items=fixture();r={'learned':r['child'],'lock':r['lock']}
 if fault=='missing':r['learned']['per_sample'].pop()
 elif fault=='duplicate':items[1]=items[0]
 elif fault=='scorer':r['learned']['aggregate']['scorer_version']='other'
 elif fault=='intent':r['learned']['per_sample'][0]['scenario_action_exact']=2
 else:r['learned']['per_sample'][0]['slot_f1_exact_tp']={'nan':float('nan'),'negative':-1,'bool':True}[fault]
 with pytest.raises(ValueError):m.summarize(r,items)
