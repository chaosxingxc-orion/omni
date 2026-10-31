"""The full Agent coordinator checks before contact and never replays a failure."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/flights/nx'))
import agent_port_full_campaign as coordinator
from test_agent_port_full_registration import setup


@pytest.fixture
def ready(setup):
    repo, data, spec, binding = setup
    coordinator.registration.register(repo, data, binding, data / 'archive')
    ep = repo / 'configs/experiment' / (spec['cell'] + '.json')
    exp = coordinator.registration.read(ep)
    exp['cells'][0]['readiness']['state'] = 'ready'
    ep.write_text(json.dumps(exp))
    old = data / spec['supply_reference']
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(json.dumps({'output': {'rows': 'frozen synthetic'}}))
    supply = data / 'prepared' / spec['run_id'] / 'phase-supply-v1.json'
    supply.parent.mkdir(parents=True, exist_ok=True)
    supply.write_bytes(old.read_bytes())
    bp = repo / 'configs/baseline' / (spec['baseline'] + '.json')
    base = coordinator.registration.read(bp)
    base['prepared_phase_supply'] = {'relpath': str(supply.relative_to(data)), 'sha256': coordinator.registration.sha(supply)}
    bp.write_text(json.dumps(base))
    return repo, data, spec


def test_complete_commands_and_validated_scope(ready):
    repo, data, spec = ready
    plan = coordinator.validate(repo, data)
    checks, campaign, diagnostics = coordinator.commands(repo, data, data / 'runtime', repo / 'lock', plan)
    assert [name for name, _ in checks] == ['registration', 'preflight']
    assert campaign.count('campaign') == 1
    assert campaign[campaign.index('--experiment') + 1] == spec['cell']
    assert [name for name, _ in diagnostics] == ['columns', 'read', 'repeat-custody']
    assert all('stage3_rollout_score.py' not in arg for arg in campaign)


@pytest.mark.parametrize('fault', ['held', 'wrong-experiment', 'wrong-run', 'roster', 'supply', 'spent', 'fit-spent', 'recipe'])
def test_validation_refuses_before_launch(ready, fault):
    repo, data, spec = ready
    ep = repo / 'configs/experiment' / (spec['cell'] + '.json')
    exp = coordinator.registration.read(ep)
    if fault == 'held': exp['cells'][0]['readiness']['state'] = 'held'
    elif fault == 'wrong-experiment': exp['experiment_id'] = 'old'
    elif fault == 'wrong-run': exp['cells'][0]['run_id'] = 'other'
    elif fault == 'roster': exp['populations'][0]['ids'].pop()
    elif fault == 'spent': (data / 'attempts' / spec['run_id']).mkdir(parents=True)
    elif fault == 'fit-spent':
        (data / 'prepared' / spec['fit_run_id'] / spec['fit_attempt_id']).mkdir(parents=True)
    elif fault == 'supply': (data / spec['supply_reference']).write_text('{"output": "changed"}')
    else:
        p = repo / coordinator.registration.PKG / 'analysis-recipe-v1.json'
        p.write_text(p.read_text() + '\n')
    ep.write_text(json.dumps(exp))
    with pytest.raises(ValueError): coordinator.validate(repo, data)


@pytest.mark.parametrize('failure', [None, 'preflight', 'campaign', 'fit', 'columns', 'repeat-custody'])
def test_execution_records_terminal_steps_without_replay(ready, monkeypatch, failure):
    repo, data, spec = ready
    calls = []
    def run(args, **kwargs):
        name = Path(kwargs['stdout'].name).stem
        calls.append(name)
        return SimpleNamespace(returncode=int(name == failure))
    monkeypatch.setattr(coordinator, 'subprocess', SimpleNamespace(
        check_output=lambda args, **kw: 'a'*40+'\n' if 'rev-parse' in args else b'', run=run, STDOUT=-2))
    def fit(*args):
        if failure == 'fit': raise ValueError('incomplete FIT')
        return {'rows': 1080, 'status': 'PASS'}
    monkeypatch.setattr(coordinator.fit, 'build', fit)
    monkeypatch.setattr(coordinator.delivered, 'build', lambda *args: {'rows': 1080, 'aggregate': {'synthetic': True}})
    out = data / 'coordinator'
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            coordinator.execute(repo, data, data / 'runtime', repo / 'lock', out)
        assert (out / 'failed.json').exists() and not (out / 'complete.json').exists()
    else:
        assert coordinator.execute(repo, data, data / 'runtime', repo / 'lock', out)['rows'] == 1080
        assert calls == ['registration', 'preflight', 'campaign', 'columns', 'read', 'repeat-custody']
    if failure == 'preflight': assert calls == ['registration', 'preflight']
    if failure in ('campaign', 'fit'): assert calls == ['registration', 'preflight', 'campaign']
    before = list(calls)
    with pytest.raises(ValueError, match='fresh external'):
        coordinator.execute(repo, data, data / 'runtime', repo / 'lock', out)
    assert calls == before
