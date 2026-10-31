from pathlib import Path
import sys,copy,json
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'scripts/analysis'),str(Path(__file__).resolve().parents[2]/'scripts/flights/nx')]
import pytest
import tfrl_h1_remaining as m
import tfrl_v2_sampler as sampler

def test_repair_derivation_keeps_seed_method_and_exact_complement():
    roster={'half':'R01C-H1','roster_id':sampler.cm.HALF_ROSTER_ID['R01C-H1'],'split_identity_hash':'a'*64,
            'items':[{'sample_id':f'fixture-{i}','audio_seconds':1.0} for i in range(310)]}
    plan=sampler.build_plan(roster,'R01C-H1',1,execution_run_id=m.RUN,execution_attempt_id=m.ATTEMPT)
    base={'native_episode':{'consumption_module':{'columns':sampler.execution_columns(plan)}},
          'policy':'unchanged-policy','native_contact_fit':{},'prepared_phase_supply':{'sha256':'b'*64}}
    exp={'cells':[{}],'population':{},'populations':[{}],'execution_plan':{},'stage3_measurement':{}}
    original=copy.deepcopy(base);before=copy.deepcopy(exp)
    b,e=m.derive_documents(base,exp,roster,plan)
    assert base==original and exp==before
    assert b['native_episode']==base['native_episode']
    assert b['policy']==base['policy']
    assert b['native_contact_fit']['contact_cap']==2170
    assert e['execution_plan']['planned_model_calls']==4340
    assert e['populations'][0]['ids']==[r['sample_id'] for r in roster['items']]
    assert e['cells'][0]['readiness']['state']=='held'
    roster['items'].pop()
    with pytest.raises(ValueError,match='310'):m.derive_documents(base,exp,roster,plan)
