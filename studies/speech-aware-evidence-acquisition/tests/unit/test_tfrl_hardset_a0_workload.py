import copy
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
import tfrl_hardset_a0_workload as m
from test_tfrl_current_banked_cells import documents

def test_derives_real_two_contact_graph_without_changing_inputs(tmp_path):
    source=documents(tmp_path)[0];saved=copy.deepcopy(source)
    base=m.derive_a0(ROOT,source)
    assert source==saved
    assert base['policy']=='phase-native-module-v1-c2'
    assert base['knowledge_stores']==source['knowledge_stores']
    assert base['phase_material']==source['phase_material']
    assert base['native_episode']['roles']['perception']==source['native_episode']['roles']['perception']
    assert 'decoding_variants' not in base['native_episode']['roles']['solver']
    assert 'prepared_phase_supply' not in base
    assert base['native_episode']['prompt_contact_limits']=='legacy-c17-lock-v1'

@pytest.mark.parametrize('fault',['columns','bank','policy'])
def test_rejects_unreviewed_source(tmp_path,fault):
    source=documents(tmp_path)[0]
    if fault=='columns':source['native_episode']['consumption_module']['columns'].pop()
    elif fault=='bank':source['native_episode']['consumption_module']['tfrl_v2_bank']={}
    else:source['policy']='changed'
    with pytest.raises((ValueError,m.cm.ConsumptionModuleConfigError)):m.derive_a0(ROOT,source)

def test_subset_preserves_queue_order_media_durations_and_split_identity():
    parent={'items':[{'sample_id':'a','audio_seconds':1.5},{'sample_id':'b','audio_seconds':2.0}], 'seed':4}
    queue=[{'sample_id':'b','job_kind':'independent_a0_difficulty_observation','model_calls_authorized':False}]
    result=m.population(parent,queue)
    assert result['items']==[parent['items'][1]]
    assert result['split_identity_hash']==m.split_identity_hash(m.prefixed('slurp',['b']))
    assert result['seed']==4
    with pytest.raises(ValueError):m.population(parent,queue*2)
    queue[0]['model_calls_authorized']=True
    with pytest.raises(ValueError):m.population(parent,queue)
