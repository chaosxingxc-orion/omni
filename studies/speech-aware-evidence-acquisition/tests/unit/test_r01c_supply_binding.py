from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/flights/nx'))
import r01c_current_supply_binding as m


def fixture():
    return ({'policy': 'phase-native-module-v1-c2'}, {'cells': [{'cell_id': m.baseline.CELL,
        'run_id': m.baseline.RUN, 'readiness': {'state': 'held', 'requirements': []}, 'allowed_execution_stages': ['check']}]},
        {'schema': 'saea-r01c-current-supply-v1', 'rows': 625, 'model_calls': 0, 'readiness': 'held',
         'registered': False, 'supply': {'relpath': f'prepared/{m.baseline.RUN}/phase-supply-v1.json', 'sha256': 'a' * 64}})


def test_bind_only_pin_preserves_held():
    b, e, p = fixture(); nb, ne = m.bound_documents(b, e, p)
    assert nb == {**b, 'prepared_phase_supply': p['supply']}
    assert ne['cells'][0]['readiness']['state'] == 'held'
    assert ne['cells'][0]['allowed_execution_stages'] == ['check']
    assert 'prepared_phase_supply' not in b


@pytest.mark.parametrize('fault', ['partial', 'foreign', 'ready', 'bound', 'contacts'])
def test_refuse_incorrect_source(fault):
    b, e, p = fixture()
    if fault == 'partial': p['rows'] = 624
    elif fault == 'foreign': p['supply']['relpath'] = 'prepared/other.json'
    elif fault == 'ready': e['cells'][0]['readiness']['state'] = 'ready'
    elif fault == 'bound': b['prepared_phase_supply'] = p['supply']
    else: p['model_calls'] = 1
    with pytest.raises(ValueError): m.bound_documents(b, e, p)
