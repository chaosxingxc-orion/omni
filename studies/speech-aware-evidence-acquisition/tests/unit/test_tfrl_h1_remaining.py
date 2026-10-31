from pathlib import Path
import sys,copy,json
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'scripts/analysis'),str(Path(__file__).resolve().parents[2]/'scripts/flights/nx')]
import pytest
import tfrl_h1_remaining as m
import tfrl_v2_sampler as sampler

def test_repair_derivation_keeps_seed_method_and_exact_complement():
    root=Path(__file__).resolve().parents[2]
    base=json.loads((root/f'configs/baseline/{m.PARENT}.json').read_bytes())
    exp=json.loads((root/f'configs/experiment/{m.PARENT}.json').read_bytes())
    roster=json.loads((root/'docs/receipts/rosters/slurp-stage3-construction-r01c-h1.json').read_bytes())
    roster['items']=roster['items'][3:];roster['split_identity_hash']='a'*64
    plan=sampler.build_plan(roster,'R01C-H1',1,execution_run_id=m.RUN,execution_attempt_id=m.ATTEMPT)
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
