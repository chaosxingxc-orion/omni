import copy,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/flights/nx'),str(ROOT/'scripts/analysis')]
import fold_a_c2_baseline as m
from test_fold_a_c2_comparator import source


def test_restored_agreement_preserves_actual_c2_semantics():
    b,e,p=source();before=copy.deepcopy((b,e))
    base,exp=m.derive(ROOT,b,e,p,'a'*64,'b'*64)
    assert (b,e)==before
    assert base['policy']=='phase-native-module-v1-c2'
    assert base['native_episode']['roles']==b['native_episode']['roles']
    block=base['native_episode']['consumption_module']
    assert block['lock']=='v2.2-npu-port' and block['columns']==[]
    phase=base['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    assert phase['k']==60 and phase['rank_weights']['w_agree']==1
    assert phase['agreement_source']['sha256']=='a'*64
    assert phase['agreement_source']['tool_contact_ledger_row']==m.agreement.ARTIFACT
    assert base['native_contact_fit']['contact_cap']==2160
    assert exp['cells'][0]['run_id']==m.RUN and ' ' not in m.RUN
    assert exp['cells'][0]['allowed_execution_stages']==['check']
    assert exp['cells'][0]['readiness']['state']=='held'
    assert exp['population']['count']==1080 and 'prepared_phase_supply' not in base
    assert base['knowledge_stores']['train-exemplars']['snapshot']==b['knowledge_stores']['train-exemplars']['snapshot']


@pytest.mark.parametrize('pin',['0'*64,'z'*64,'',None])
def test_invalid_agreement_pin_refused(pin):
    b,e,p=source()
    with pytest.raises(ValueError):m.derive(ROOT,b,e,p,pin,'b'*64)
