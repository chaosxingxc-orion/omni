import copy
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/flights/nx'),str(ROOT/'scripts/analysis')]
import r01c_current_comparator as m
from derive_lock_only_baseline import derive as lock_only


@pytest.fixture
def source():
    b=lock_only(m.read(ROOT/'configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json'),repo=ROOT)
    b['native_contact_fit']={'run_id':'original-fit','attempt_id':'original'}
    e={'cells':[{'model_fragment':'original-model'}],'service_session':'original-session',
        'stage3_measurement':{'rows':500,'read_roster':'old','claims_benefit':False}}
    pins={p:str(i)*64 for i,p in enumerate((m.inputs.SEGMENTS,m.inputs.CONTENT,m.inputs.RECEIPT,
         m.inputs.ALIAS_RECEIPT,m.inputs.PROVENANCE),1)}
    return b,e,pins


def test_derivation_preserves_c2_recall_and_uses_current625(source):
    b,e,pins=source;before=copy.deepcopy((b,e))
    derived,experiment=m.derive(ROOT,b,e,pins)
    assert (b,e)==before
    assert derived['policy']==b['policy']=='phase-native-module-v1-c2'
    for key in ('roles','prompt_contact_limits','deliver_normaliser','max_revisions'):
        assert derived['native_episode'][key]==b['native_episode'][key]
    solver=derived['knowledge_stores']['train-exemplars']['phase_supply']['solver']
    old=b['knowledge_stores']['train-exemplars']['phase_supply']['solver']
    assert {k:v for k,v in solver.items() if k!='query_source'}=={k:v for k,v in old.items() if k!='query_source'}
    assert solver['query_source']['tool_contact_ledger_row']==m.inputs.RUN
    assert derived['knowledge_stores']['slot-contracts']==b['knowledge_stores']['slot-contracts']
    assert experiment['cells'][0]['model_fragment']=='original-model'
    assert experiment['service_session']=='original-session'
    assert experiment['cells'][0]['allowed_execution_stages']==['check']
    assert len(experiment['populations'][0]['ids'])==625
    assert experiment['split_id']==m.read(ROOT/m.inputs.DATASET)['split_name']
    assert experiment['cells'][0]['readiness']['state']=='held'
    assert not any('FOLD-B' in x['relpath'] or 'fold-b' in x['relpath'] for x in derived['preparation_source_receipts'])


@pytest.mark.parametrize('fault',['missing','zero','nonhex','recall','loop'])
def test_refuses_placeholder_and_method_drift(source,fault):
    b,e,pins=source
    if fault=='missing':pins.pop(m.inputs.CONTENT)
    elif fault=='zero':pins[m.inputs.CONTENT]='0'*64
    elif fault=='nonhex':pins[m.inputs.CONTENT]='z'*64
    elif fault=='recall':b['knowledge_stores']['train-exemplars']['phase_supply']['solver']['k']=1
    else:b['native_episode']['review_loop']='on'
    with pytest.raises((ValueError,m.cm.ConsumptionModuleConfigError)):m.derive(ROOT,b,e,pins)
