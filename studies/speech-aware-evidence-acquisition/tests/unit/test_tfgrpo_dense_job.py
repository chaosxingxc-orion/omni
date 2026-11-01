import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import tfgrpo_dense_job as job
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience,sha,DENSE_REWARD


def half():
    parent=empty_experience('model');policy={'fixture':True};ph=sha(policy);groups=[];custody=[];items=[]
    for i in range(51):
        sid=f'slurp:train:{i:06d}';rolls=[];items.append({'sample_id':sid,'stratum':'home'})
        for k in range(4):
            rid=f'{sid}-{k}';score={'scenario_action_exact':k%2,'slot_f1_exact_tp':0,'slot_f1_exact_fp':1,'slot_f1_exact_fn':1,'strict_task_success':0,'reward_mode':DENSE_REWARD,'scored_answer_sha256':job.sha_bytes(b'normalized')}
            rolls.append({'trajectory_id':rid,'trajectory':json.dumps({'scored_answer_after_fixed_normalization':'normalized'}),'critique':json.dumps(score),'reward':.5*(k%2),
                'model_id':'model','policy_sha256':ph,'experience_sha256':parent['sha256']});custody.append({'trajectory_id':rid,'scored_answer_sha256':job.sha_bytes(b'normalized')})
        groups.append({'query_id':sid,'query':'task','answer':'TRAIN reference','rollouts':rolls})
    return {'schema':'saea-tfgrpo-dense-h1-groups-v1','reward_mode':DENSE_REWARD,'source_half':'h1','evaluation_half':'h2',
        'rollout_kind':'shared-perception-adaptation','parent':parent,'model_id':'model','policy':policy,'policy_sha256':ph,
        'groups':groups,'custody':custody,'roster_items':items}


def test_full51_registration_counts_and_immutable_batch():
    h=half();r,s,g,p=job.documents(h,{'model_fragment':'model'},{'decoding_params':{'max_tokens':4096}},[])
    assert len(g['contacts'])==307 and p.planned_model_calls==307 and r['count']==51
    assert sha(s['batch'])==g['batch_sha256']
    assert s['batch']['common']['reward_mode']==DENSE_REWARD
    assert g['construction_roster_sha256']==job.sha_bytes(job.base.raw(r))


@pytest.mark.parametrize('kind',['reward','parent','identity','duplicate','full_success'])
def test_source_drift_rejected(kind):
    h=half()
    if kind=='reward':h['groups'][0]['rollouts'][0]['reward']=.3
    elif kind=='parent':h['parent']['items']={'x':'candidate'}
    elif kind=='identity':h['groups'][0]['rollouts'][0]['experience_sha256']='f'*64
    elif kind=='duplicate':h['custody'][1]=h['custody'][0]
    else:h['groups'][0]['rollouts'][0]['reward']=1
    with pytest.raises(ValueError):job.documents(h,{'model_fragment':'model'},{'decoding_params':{'max_tokens':4096}},[])


def test_final_budget_checked_before_generation_registration():
    with pytest.raises(ValueError,match='16384'):
        job.require_generation_budget({'decoding_params':{'max_tokens':4096,'thinking_budget_tokens':3072}},
            {'reasoning_budget_tokens':3072,'answer_reserve_tokens':1024})
    job.require_generation_budget({'decoding_params':{'max_tokens':16384,'thinking_budget_tokens':3072}},
        {'reasoning_budget_tokens':3072,'answer_reserve_tokens':13312})
