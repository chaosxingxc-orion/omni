import copy
import json
from pathlib import Path
import shutil
import sys
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/analysis'))
import tfrl_hardset_a0_registration as m


@pytest.fixture
def workload(tmp_path):
    repo, data = tmp_path/'repo', tmp_path/'data'
    name = m.NAME
    rels = [f'configs/{k}/{name}.json' for k in ('baseline', 'experiment')]
    rels += [f'docs/receipts/rosters/{name}.json', m.LOCK,
             'configs/dataset/slurp-speech-native-stage3-construction-r01c.json']
    for rel in rels:
        target = repo/rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
    base = m.read(repo/rels[0]); exp = m.read(repo/rels[1]); exp['service_session'] = m.SESSION
    ids = exp['populations'][0]['ids']
    # Synthetic contents verify custody/membership only; no performance claims.
    supply = data/'prepared/supply.json'; supply.parent.mkdir(parents=True)
    supply.write_bytes(m.raw({'output': {'facets': {}, 'values': {'perception': {i: [] for i in ids}}}}))
    base['prepared_phase_supply'] = m.pin(data, 'prepared/supply.json')
    (repo/rels[0]).write_bytes(m.raw(base))
    spec = {'cell': name, 'experiment': name, 'run_id': m.RUN, 'attempt_id': 'generation-v1',
            'fit_run_id': m.RUN+'-FIT', 'fit_attempt_id': 'fit-v1', 'rows': 439,
            'service_session': m.SESSION, 'profile': m.PROFILE, 'runtime_receipt': m.RECEIPT,
            'roster': m.pin(repo, rels[2]), 'gain_lock': m.pin(repo, m.LOCK),
            'dataset': m.pin(repo, rels[4]), 'split_identity_hash': exp['population']['split_identity_hash'],
            'scientific_baseline_sha256': m.scientific_hash(base)}
    return repo, data, spec, exp, m.read(repo/rels[4])


def test_accepts_fixed439_subset_of_unchanged625(workload):
    repo, data, spec, exp, view = workload
    assert m.validate(repo, data, m.NAME, spec, exp, view) == 439


@pytest.mark.parametrize('fault', ['roster', 'budget', 'fit', 'policy', 'dataset', 'supply'])
def test_rejects_custody_budget_or_program_change(workload, fault):
    repo, data, spec, exp, view = workload
    if fault == 'roster': exp['populations'][0]['ids'].reverse()
    elif fault == 'budget': exp['execution_plan']['planned_model_calls'] = 878
    elif fault == 'dataset': view['items'].pop()
    elif fault == 'supply': (data/'prepared/supply.json').write_text('{}')
    else:
        p = repo/f'configs/baseline/{m.NAME}.json'; base = m.read(p)
        if fault == 'fit': base['native_contact_fit']['contact_cap'] = 1
        else: base['policy'] = 'phase-direct-only-v1'
        p.write_bytes(m.raw(base))
    with pytest.raises(ValueError): m.validate(repo, data, m.NAME, spec, exp, view)


def test_registered_workload_routes_narrow_kind(workload, monkeypatch):
    import registered_workload
    repo, data, spec, exp, view = workload
    spec['workload_kind'] = 'tfrl-hardset-a0-observation'
    (repo/f'configs/experiment/{m.NAME}.json').write_bytes(m.raw(exp))
    assert registered_workload.validate_workload(repo, data, m.NAME, spec) == 439
    spec['workload_kind'] = 'unknown'
    with pytest.raises(ValueError): registered_workload.validate_workload(repo, data, m.NAME, spec)


def test_existing_spend_blocks_registration(tmp_path):
    repo, data = tmp_path/'repo', tmp_path/'data'
    (repo/'docs').mkdir(parents=True); (repo/'docs/exposure-ledger.md').write_text('')
    m.require_unspent(repo, data)
    (data/'attempts'/m.RUN).mkdir(parents=True)
    with pytest.raises(ValueError, match='existing registration or spend'): m.require_unspent(repo, data)


def test_stage_and_register_write_exact_two_budgets_without_contacts(workload, monkeypatch):
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    repo, data, spec, exp, view = workload
    monkeypatch.setattr(git_commit, 'require_clean_study_repo', lambda _: 'a'*40)
    bp = repo/f'configs/baseline/{m.NAME}.json'; base = m.read(bp)
    supply = base.pop('prepared_phase_supply'); bp.write_bytes(m.raw(base))
    exp.pop('service_session'); ep = repo/f'configs/experiment/{m.NAME}.json'; ep.write_bytes(m.raw(exp))
    # Build the fixture from the public protocol; resident receipt is execution-local.
    model = m.read(ROOT/'configs/model/bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget3072-max4096-closemarker-port-v2.json')
    model['runtime_receipt'] = m.RECEIPT
    target = repo/f'configs/model/{m.MODEL}.json'; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(m.raw(model))
    target = repo/m.AUTH; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/m.AUTH, target)
    (repo/'docs/exposure-ledger.md').write_text('')
    work = data/'work/codex/inputs'; work.mkdir(parents=True)
    derivation = work/'derived.json'; derivation.write_bytes(m.raw({'schema': 'saea-hardset-a0-held-workload-v1',
        'rows': 439, 'output_hashes': {str(p.relative_to(repo)): m.sha(p) for p in (bp, ep)}}))
    prepared = work/'prepared.json'; prepared.write_bytes(m.raw({'schema': 'saea-hardset-a0-first-p-supply-v1',
        'rows': 439, 'run_id': m.RUN, 'derivation_sha256': m.sha(derivation), 'supply': supply,
        'generation_authorized': False, 'new_model_calls': 0, 'new_tokenizer_calls': 0}))
    out = data/'work/codex/staged'
    assert m.stage(repo, data, prepared, derivation, out)['output_values_reused']
    result = m.register(repo, data, out/'complete.json')
    assert result['readiness'] == 'held'
    pkg = repo/'docs/readiness'/m.NAME
    generation = m.read(pkg/'generation-plan-registered-v1.json')
    fit = m.read(pkg/'fit-plan-registered-v1.json')
    assert generation['execution_plan']['planned_model_calls'] == 1756
    assert fit['execution_plan']['planned_model_calls'] == 878
    assert generation['execution_plan']['protocol_hash'] == fit['execution_plan']['protocol_hash']
    assert '| '+m.RUN+' |' in (repo/'docs/exposure-ledger.md').read_text()
    with pytest.raises(ValueError): m.register(repo, data, out/'complete.json')
