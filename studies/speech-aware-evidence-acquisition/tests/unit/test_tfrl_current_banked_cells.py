import copy,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
import tfrl_current_banked_cells as m
from test_tfrl_current_next_plan import inputs
from tfrl_current_next_plan import derive as plans
from test_decoding_variants import _decoding_block
from test_tfrl_seed_execution import plan as seed_plan
from test_tfrl_v2_consumption import SCOPE


def documents(tmp_path):
    rosters,receipts=inputs(tmp_path)
    plan=plans(rosters,receipts,iteration=2,control_bounds={'h1':55,'h2':55})['h1']
    block=_decoding_block();block['columns']=m.sampler.execution_columns(seed_plan())
    block['prompt_contact_limits']='legacy-c17-lock-v1'
    base=m.cm.compile_module(block,m.carrier_module_hooks(repo_root=ROOT),repo_root=ROOT).values()
    base['carrier_lock_key']='slurp';base['native_episode']['consumption_module']=block
    base['native_contact_fit']={'run_id':'old-FIT','attempt_id':'fit-v1','contact_cap':2191}
    base['prepared_phase_supply']={'old':'must not reuse'}
    exp={'cells':[{'dataset_fragment':'actual625','model_fragment':'selected','cell_id':'old','baseline_fragment':'old',
        'run_id':'old','attempt_id':'old','readiness':{'state':'ready'}}],
        'execution_plan':{'planned_model_calls':1,'planned_speech_audio_seconds':1,'planned_gpu_hours':16},
        'stage3_measurement':{}}
    receipt=receipts['h2'];pin={'relpath':'banks/h2/bank-receipt.json','sha256':m.rail.receipt_sha256(receipt)}
    return base,exp,rosters['h1'],plan,receipt,pin


def test_banked_cell_compiles_keeps_controls_and_stays_held(tmp_path):
    args=documents(tmp_path);original=copy.deepcopy(args)
    name,base,exp=m.derive(ROOT,*args,SCOPE)
    assert args==original
    assert name=='slurp-tfrl-current-bank-i2-h1-v1'
    assert exp['cells'][0]['readiness']['state']=='held' and exp['cells'][0]['allowed_execution_stages']==['check']
    assert exp['execution_plan']['planned_model_calls']==14*313
    assert 'prepared_phase_supply' not in base
    assert base['knowledge_stores']==args[0]['knowledge_stores']
    module=base['native_episode']['consumption_module']
    assert module['tfrl_supply_allocation']['candidate_max_items']==47
    assert module['tfrl_v2_bank']['consumer']=='R01C-H1'
    assert base['native_episode']['roles']['solver']['decoding_variants']['arms'][-2:]==args[0]['native_episode']['roles']['solver']['decoding_variants']['arms'][-2:]


@pytest.mark.parametrize('fault',['bank','iteration','bound','roster','pin'])
def test_banked_cell_refuses_wrong_source_before_any_write(tmp_path,fault):
    base,exp,roster,plan,receipt,pin=documents(tmp_path)
    if fault=='bank':receipt['source_halves']=['R01C-H1']
    elif fault=='iteration':plan['iteration']=4
    elif fault=='bound':plan['banks']['supply_bound_items']=43
    elif fault=='roster':roster['items'].pop()
    else:pin['sha256']='f'*64
    with pytest.raises(ValueError):m.derive(ROOT,base,exp,roster,plan,receipt,pin,SCOPE)
