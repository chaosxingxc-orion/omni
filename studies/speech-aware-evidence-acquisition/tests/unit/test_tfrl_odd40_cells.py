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


def binding_fixture(tmp_path, monkeypatch):
    import speech_aware_evidence_acquisition.contracts as contracts
    import speech_aware_evidence_acquisition.core.bundle.git_commit as git
    import speech_aware_evidence_acquisition.core.flight.factory as factory
    monkeypatch.setattr(contracts, 'assert_execution_environment', lambda: None)
    monkeypatch.setattr(git, 'require_clean_study_repo', lambda p: 'a' * 40)
    calls = []
    class Runtime:
        def verify_prepared_artifacts(self):
            return [{'identity_ok': True, 'seed_ok': True}]
    def build(*args, **kwargs):
        calls.append(kwargs)
        return Runtime()
    monkeypatch.setattr(factory, 'build_flight_runtime', build)
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(m.raw(value))
    pins, preps = {}, []
    lock = tmp_path / 'datasets.lock.json'
    write(lock, {'pinned': True})
    for half in ('h1', 'h2'):
        name, run, n = m.identity(half)
        base = {}
        exp = {'populations': [{'ids': [str(i) for i in range(n)]}], 'cells': [{
            'cell_id': name, 'baseline_fragment': name, 'run_id': run,
            'model_fragment': 'selected', 'dataset_fragment': 'real625', 'attempt_id': 'generation-v1',
            'allowed_execution_stages': ['check'], 'readiness': {'state': 'held', 'requirements': []}}]}
        for kind, doc in (('baseline', base), ('experiment', exp)):
            rel = f'configs/{kind}/{name}.json'
            write(repo / rel, doc)
            pins[rel] = m.sha(repo / rel)
        rel = f'prepared/{run}/phase-supply-v1.json'
        write(data / rel, {'output': {'facets': {}, 'values': {'perception': {str(i): {} for i in range(n)}}}})
        preps.append({'schema': 'saea-odd40-supply-v1', 'rows': n, 'cell': name, 'run_id': run,
            'source_hashes': pins, 'data_hashes': {}, 'dataset_lock_sha256': m.sha(lock),
            'model_calls': 0, 'tokenizer_calls': 0, 'readiness': 'held', 'registered': False,
            'supply': {'relpath': rel, 'sha256': m.sha(data / rel)}})
    derivation = tmp_path / 'derivation.json'
    write(derivation, {'schema': 'saea-odd40-held-cells-v1', 'rows': 625,
        'dependency_hashes': {}, 'output_hashes': pins, 'data_hashes': {}})
    paths = []
    for i, prep in enumerate(preps):
        prep['derivation_sha256'] = m.sha(derivation)
        path = tmp_path / f'prep{i}.json'
        write(path, prep)
        paths.append(path)
    return repo, data, derivation, lock, paths, data / 'work/codex/binding', calls


def test_binding_invokes_shared_verifier_and_keeps_halves_held(tmp_path, monkeypatch):
    *args, calls = binding_fixture(tmp_path, monkeypatch)
    result = m.bind(*args)
    assert result['rows'] == 625 and result['generation_authorized'] is False
    assert len(calls) == 2
    for half in ('h1', 'h2'):
        name, run, _ = m.identity(half)
        base = m.read(args[0] / f'configs/baseline/{name}.json')
        exp = m.read(args[0] / f'configs/experiment/{name}.json')
        assert base['prepared_phase_supply']['relpath'] == f'prepared/{run}/phase-supply-v1.json'
        assert exp['cells'][0]['readiness']['state'] == 'held'
        assert exp['cells'][0]['allowed_execution_stages'] == ['check']


@pytest.mark.parametrize('fault', ['foreign_supply', 'supply_bytes', 'source_bytes', 'spent_fit', 'lock'])
def test_binding_rejects_changed_evidence_before_mutation(tmp_path, monkeypatch, fault):
    *args, calls = binding_fixture(tmp_path, monkeypatch)
    repo, data, derivation, lock, preps, out = args
    prep = m.read(preps[0])
    if fault == 'foreign_supply':
        prep['supply']['relpath'] = 'prepared/foreign/phase-supply-v1.json'
        preps[0].write_bytes(m.raw(prep))
    elif fault == 'supply_bytes': (data / prep['supply']['relpath']).write_text('{}')
    elif fault == 'source_bytes':
        (repo / f"configs/baseline/{prep['cell']}.json").write_text('{} ')
    elif fault == 'spent_fit': (data / 'attempts' / (prep['run_id'] + '-FIT')).mkdir(parents=True)
    else: lock.write_text('{}')
    with pytest.raises(ValueError): m.bind(*args)
    assert not calls and not out.exists()
    assert 'prepared_phase_supply' not in m.read(repo / f"configs/baseline/{prep['cell']}.json")


def test_binding_never_claims_success_when_shared_seed_check_fails(tmp_path, monkeypatch):
    import speech_aware_evidence_acquisition.core.flight.factory as factory
    *args, calls = binding_fixture(tmp_path, monkeypatch)
    class BadRuntime:
        def verify_prepared_artifacts(self):
            return [{'identity_ok': True, 'seed_ok': False}]
    monkeypatch.setattr(factory, 'build_flight_runtime', lambda *a, **k: BadRuntime())
    with pytest.raises(ValueError, match='identity/seed'): m.bind(*args)
    assert (args[-1] / 'failed.json').exists()
    assert not (args[-1] / 'complete.json').exists()
