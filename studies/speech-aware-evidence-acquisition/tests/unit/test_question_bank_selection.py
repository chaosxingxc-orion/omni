import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import question_bank_selection as q

C={'min_per_label':8,'scale':2,'cap_per_label':64,'boundary_fraction':.2,'margin_weight':0.,'crosshub_weight':0.,'redundancy_weight':0.}

def test_midrank_ties_and_single():
    assert np.allclose(q.midrank([3,1,1,5]),[2/3,1/6,1/6,1])
    assert q.midrank([7]).tolist()==[.5]

def test_dedup_same_class_but_not_cross_class():
    m=[{'intent':'a','sentence_sha256':'x'},{'intent':'a','sentence_sha256':'x'},{'intent':'b','sentence_sha256':'x'}]
    g=q.group_members(m,['a','b','c']);assert g['a'][1].tolist()==[0] and g['b'][1].tolist()==[0] and len(g['c'][0])==0

def test_small_class_coverage_boundary_fallback_and_stable_ties():
    matrix=np.eye(4); cfg={**C,'boundary_fraction':.2}
    chosen,audit,cover,closest=q.greedy(matrix,np.arange(4),[0,1,2,3],[0,0,0,0],[0,1,1,1],cfg)
    assert chosen==[0,1,2,3] and not any(x['boundary_slot'] for x in audit)
    assert np.allclose(cover,1) and closest.tolist()==[0,1,2,3]

def test_budget_and_boundary_protects_supported_low_margin():
    sim=np.eye(25); chosen,audit,_,_=q.greedy(sim,np.arange(25),np.arange(25),np.zeros(25),np.ones(25),C)
    assert len(chosen)==10 and sum(x['boundary_slot'] for x in audit)==2 and chosen[:2]==[0,1]

def test_crosshub_weight_and_cache_identity(tmp_path):
    assert np.allclose(q.crosshub(np.array([[1,2],[0,2],[0,1]]),['a','a','b']),[1,1,1])
    p=tmp_path/'key';p.write_text('a');one=q.cache_identity([p]);p.write_text('b');assert one!=q.cache_identity([p])

def test_config_exact_no_hidden_source_or_nan():
    with pytest.raises(ValueError):q.validate_config({**C,'source_ids':[1]})
    with pytest.raises(ValueError):q.validate_config({**C,'margin_weight':float('nan')})

def test_runtime_rank_uses_mask_only_and_frozen_count():
    from question_bank_selection_cli import rank
    members=[{'intent':'a','sentence_sha256':'x'},{'intent':'a','sentence_sha256':'y'},{'intent':'b','sentence_sha256':'x'}]
    matrix=np.array([[.9,.7,.8]])
    result=rank(matrix,members,['a','b'],np.array([True,False,True]))
    assert result['mean5']==[[0,1]] and result['coverage']==[[0,1]] and result['coverage_frozen']==[[0,1]]
