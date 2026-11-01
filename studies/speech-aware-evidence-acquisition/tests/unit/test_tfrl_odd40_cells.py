import copy
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
import tfrl_odd40_cells as m
from test_tfrl_current_banked_cells import documents


def inputs(tmp_path):
    base, exp, roster, *_ = documents(tmp_path)
    base['knowledge_stores']['train-exemplars']['phase_supply']['solver'].update(
        k=20, fuse_policy='saea-fuse-supplyfilter-v1', fuse_coverage=dict(m=20, c=20, top_l=20),
        rank_weights=dict(w_cos=1., w_prior=0., w_agree=0.))
    exp['population'] = {'split_identity_hash': roster['split_identity_hash']}
    exp['populations'] = [{'ids': [r['sample_id'] for r in roster['items']]}]
    exp['service_session'] = 'spent-source-session'
    exp['cells'][0]['readiness']['requirements'] = [{'kind': 'decision', 'relpath': 'old'}]
    return base, exp, roster


def test_fresh_identity_whole_method_budgets_and_original_controls(tmp_path):
    args = inputs(tmp_path)
    before = copy.deepcopy(args)
    name, base, exp, proof = m.derive(ROOT, *args, 'h1')
    assert args == before
    assert name == 'slurp-tfrl-odd40-i1-h1-v1'
    cell, = exp['cells']
    assert cell['readiness']['state'] == 'held'
    assert cell['readiness']['requirements'] == []
    assert cell['allowed_execution_stages'] == ['check']
    assert 'service_session' not in exp and 'prepared_phase_supply' not in base
    assert base['native_contact_fit']['run_id'] == cell['run_id'] + '-FIT'
    assert base['native_contact_fit']['contact_cap'] == 1565
    assert exp['execution_plan']['planned_model_calls'] == 3130
    assert exp['execution_plan']['planned_speech_audio_seconds'] == 3130
    assert exp['populations'] == args[1]['populations']
    assert base['knowledge_stores'] == args[0]['knowledge_stores']
    assert proof['method_program_ceiling'] == 5
    assert proof['generation_authorized'] is False


@pytest.mark.parametrize('fault', ['duplicate', 'different_ids', 'split', 'duration_nan', 'duration_negative', 'half'])
def test_refuses_population_drift_and_invalid_duration(tmp_path, fault):
    base, exp, roster = inputs(tmp_path)
    half = 'h1'
    if fault == 'duplicate': roster['items'][1]['sample_id'] = roster['items'][0]['sample_id']
    elif fault == 'different_ids': exp['populations'][0]['ids'][0] = 'different'
    elif fault == 'split': exp['population']['split_identity_hash'] = 'wrong'
    elif fault == 'duration_nan': roster['items'][0]['audio_seconds'] = float('nan')
    elif fault == 'duration_negative': roster['items'][0]['audio_seconds'] = -1
    else: half = 'h2'
    with pytest.raises(ValueError): m.derive(ROOT, base, exp, roster, half)


@pytest.mark.parametrize('kind', ['runs', 'attempts', 'prepared'])
@pytest.mark.parametrize('suffix', ['', '-FIT'])
def test_never_reuses_generation_or_fit_identity(tmp_path, kind, suffix):
    (tmp_path / kind / ('run' + suffix)).mkdir(parents=True)
    with pytest.raises(ValueError): m.require_unspent(tmp_path, 'run', preparation=True)


def test_supply_requires_complete_exact_real_roster():
    m.verify_output({'facets': {}, 'values': {'perception': {'a': {}, 'b': {}}}}, ['a', 'b'])
    for output in ({'facets': {}, 'values': {}},
                   {'facets': {}, 'values': {'perception': {'a': {}}}},
                   {'facets': {'solver': {'a': {}, 'b': {}}}, 'values': {'perception': {'a': {}, 'b': {}}}}):
        with pytest.raises(ValueError): m.verify_output(output, ['a', 'b'])


def test_paths_cannot_escape_declared_root(tmp_path):
    with pytest.raises(ValueError): m.confined(tmp_path, '../foreign')
