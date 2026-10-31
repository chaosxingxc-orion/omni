"""Full Agent admission preserves the roster and separates final from branch scores."""
import json
from pathlib import Path
import shutil
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/flights/nx'))
import agent_port_full_registration as writer


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    data.mkdir()
    shutil.copytree(REPO / 'configs', repo / 'configs')
    shutil.copytree(REPO / writer.PKG, repo / writer.PKG)
    # The checked-in phase-width package is already prepared for its real
    # admission.  Registration tests need the pre-registration fixture state.
    (repo / writer.PKG / 'root-admission-draft-v1.json').unlink(missing_ok=True)
    spec = writer.read(repo / writer.PKG / 'successor-draft-v1.json')
    exp_path = repo / 'configs/experiment' / (spec['cell'] + '.json')
    exp = writer.read(exp_path)
    exp['cells'][0]['readiness']['state'] = 'held'
    exp_path.write_text(json.dumps(exp))
    base_path = repo / 'configs/baseline' / (spec['baseline'] + '.json')
    base = writer.read(base_path)
    base.pop('prepared_phase_supply', None)
    base_path.write_text(json.dumps(base))
    shutil.copytree(REPO / writer.SOURCE, repo / writer.SOURCE)
    ledger = repo / 'docs/exposure-ledger.md'; ledger.write_text('# Prior ledger\n')
    spec = writer.read(repo / writer.PKG / 'successor-draft-v1.json')
    exp = writer.read(repo / 'configs/experiment' / (spec['cell'] + '.json'))
    binding = {'session_id': exp['service_session'],
               'model_fragment': {'relpath': 'configs/model/' + exp['cells'][0]['model_fragment'] + '.json'},
               'lock_rows_per_minute_per_instance': 6.0, 'tensor_parallel_size': 2}
    binding_path = tmp_path / 'binding.json'; binding_path.write_text(json.dumps(binding))
    monkeypatch.setattr(writer, 'validate_binding', lambda *args: binding)
    monkeypatch.setattr(writer, 'accepted_proof', lambda *args: ({'run_id': 'PROOF'}, {'proof': 'a'*64}))
    monkeypatch.setattr(writer.subprocess, 'check_output', lambda argv, **kw: b'' if 'status' in argv else 'b'*40+'\n')
    return repo, data, spec, binding_path


def test_complete_registration_and_one_shot_scope(setup):
    repo, data, spec, binding = setup
    result = writer.register(repo, data, binding, data / 'archive')
    assert result['held'] and result['card_hours'] != 76.6
    exp = writer.read(repo / 'configs/experiment' / (spec['cell'] + '.json'))
    assert exp['cells'][0]['readiness']['state'] == 'held'
    assert len(exp['populations'][0]['ids']) == 1080
    package = repo / writer.PKG
    recipe = writer.read(package / 'analysis-recipe-v1.json')
    assert recipe['score_surface'] == 'sealed-final-delivery' and recipe['new_scoring_jobs'] == 0
    gen = writer.read(package / 'generation-plan-registered-v1.json')
    fit = writer.read(package / 'fit-plan-registered-v1.json')
    assert gen['execution_plan']['planned_model_calls'] == 36720
    assert fit['execution_plan']['planned_model_calls'] == 18360
    assert gen['execution_plan']['planned_gpu_hours'] == result['card_hours']
    draft = writer.read(package / 'root-admission-draft-v1.json')
    assert draft['eligible_run_ids'] == [spec['run_id'], spec['fit_run_id']]
    assert not (data / spec['decision_relpath']).exists()
    assert (data / 'archive/before/docs/exposure-ledger.md').read_text() == '# Prior ledger\n'
    with pytest.raises(ValueError, match='already registered'):
        writer.register(repo, data, binding, data / 'another-archive')


def test_dry_registration_preserves_all_source_bytes(setup):
    repo, data, spec, binding = setup
    before = {str(p.relative_to(repo)): p.read_bytes() for p in repo.rglob('*') if p.is_file()}
    result = writer.register(repo, data, binding, data / 'archive', dry_run=True)
    assert result['dry_run'] and result['model_contacts'] == 0
    assert len(result['outputs']) == result['files']
    assert not (data / 'archive').exists()
    assert {str(p.relative_to(repo)): p.read_bytes() for p in repo.rglob('*') if p.is_file()} == before


def test_shared_registration_gate_resolves_new_recipe(setup):
    repo, data, spec, binding = setup
    writer.register(repo, data, binding, data / 'archive')
    # The actual structural gate reads the real composed module and roster receipts.
    shutil.copytree(REPO / 'docs/receipts', repo / 'docs/receipts')
    shutil.copytree(REPO / 'scripts/checks', repo / 'scripts/checks')
    sys.path.insert(0, str(REPO / 'scripts/checks'))
    import flight_registration_check as gate
    result = gate.evaluate(repo, data, Path('docs/exposure-ledger.md'), cells=[spec['cell']])
    cell, = result['cells']
    assert cell['read_run_id'] == spec['read_run_id']
    # This staging-only writer deliberately leaves mint and supply preparation pending.
    # All other structural findings (including exact carrier tokens) are defects.
    kinds = {item['kind'] for item in cell['findings']}
    assert kinds == {'not-ready', 'receipt-missing', 'registered-supply-binding'}, json.dumps(cell['findings'], indent=2)


@pytest.mark.parametrize('fault', ['roster', 'baseline', 'spent', 'no-proof', 'deployment'])
def test_refuses_before_mutation(setup, monkeypatch, fault):
    repo, data, spec, binding = setup
    ep = repo / 'configs/experiment' / (spec['cell'] + '.json')
    bp = repo / 'configs/baseline' / (spec['baseline'] + '.json')
    if fault == 'roster':
        exp = writer.read(ep); exp['populations'][0]['ids'].pop(); ep.write_text(json.dumps(exp))
    elif fault == 'baseline':
        base = writer.read(bp); base['policy'] = 'different'; bp.write_text(json.dumps(base))
    elif fault == 'spent':
        (data / 'attempts' / spec['run_id']).mkdir(parents=True)
    elif fault == 'deployment':
        exp = writer.read(ep); exp['service_session'] = 'different'; ep.write_text(json.dumps(exp))
    else:
        def refuse(*args): raise ValueError('proof not accepted')
        monkeypatch.setattr(writer, 'accepted_proof', refuse)
    before = ep.read_bytes()
    with pytest.raises(ValueError): writer.register(repo, data, binding, data / 'archive')
    assert ep.read_bytes() == before and not (data / 'archive').exists()
    assert not (repo / writer.PKG / 'root-admission-draft-v1.json').exists()
