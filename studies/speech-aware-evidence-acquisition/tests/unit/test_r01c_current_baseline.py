import copy
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/flights/nx'), str(ROOT / 'scripts/analysis')]
import r01c_current_baseline as m
from test_r01c_current_comparator import source


def test_restores_real_agreement_without_role_or_prompt_drift(source):
    b, e, pins = source
    before = copy.deepcopy((b, e))
    base, exp = m.derive(ROOT, b, e, pins, 'a' * 64, 'b' * 64)
    assert (b, e) == before
    assert base['policy'] == 'phase-native-module-v1-c2'
    assert base['native_episode']['roles'] == b['native_episode']['roles']
    block = base['native_episode']['consumption_module']
    assert block['prompt_contact_limits'] == 'legacy-c17-lock-v1'
    assert block['columns'] == []
    assert block['lock'] == 'v2.2-npu-port-r01c'
    phase = base['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    assert phase['k'] == 60 and phase['rank_weights']['w_agree'] == 1
    assert phase['agreement_source']['sha256'] == 'a' * 64
    assert phase['agreement_source']['tool_contact_ledger_row'] == m.agreement.ARTIFACT
    assert base['native_contact_fit']['contact_cap'] == 1250
    assert exp['cells'][0]['allowed_execution_stages'] == ['check']
    assert exp['cells'][0]['readiness']['state'] == 'held'
    assert exp['population']['count'] == 625
    assert 'prepared_phase_supply' not in base


@pytest.mark.parametrize('pin', ['0' * 64, 'z' * 64, '', None])
def test_missing_real_agreement_rejected(source, pin):
    b, e, pins = source
    with pytest.raises(ValueError): m.derive(ROOT, b, e, pins, pin, 'b' * 64)
