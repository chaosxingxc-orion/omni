"""Final correction preserves learning coverage while budgeting one real call."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/analysis'))
import tfgrpo_final_job as m
import tfgrpo_learning_job as original
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience


def fixture(monkeypatch):
    model = m.final_model_values(m.read(m.ROOT / 'configs/model/bf16-thinking-tfgrpo-tp4-ctx48k-v1.json'))
    parent = empty_experience('model')
    groups = []
    for i in range(3):
        qid = f'slurp:train:{i}:audio'
        rolls = [{'trajectory_id': f'{qid}-{j}', 'trajectory': 'sealed output',
            'critique': 'official counts', 'reward': int(j == 0), 'model_id': 'model',
            'policy_sha256': 'a' * 64, 'experience_sha256': parent['sha256']} for j in range(4)]
        groups.append({'query_id': qid, 'query': 'fixed P', 'answer': 'gold', 'rollouts': rolls})
    batch = {'schema': 'saea-tfgrpo-batch-v1', 'groups': groups, 'parent': parent,
        'common': {'agent_objective': 'task', 'learning_objective': 'learn', 'num_experiences': 2},
        'model_id': 'model', 'policy_sha256': 'a' * 64,
        'rollout_kind': 'shared-perception-adaptation', 'learning_config_sha256': 'b' * 64,
        'prompt_sha256': m.SOURCE_SHA256, 'upstream_commit': m.UPSTREAM_COMMIT,
        'bounds': {'max_calls': 19, 'concurrency': 128, 'max_operations': 128,
                   'max_items': 128, 'max_characters': 16000},
        'summary_import': {'relpath': 'old-summary.json', 'sha256': 'c' * 64}}
    source = {'schema': 'saea-tfgrpo-learning-source-v1', 'batch': batch, 'provenance': []}
    roster = {'schema': 'saea-roster-receipt-v1', 'roster_id': 'old',
        'roster_role': 'construction', 'carrier_lock_key': 'slurp', 'split_role': 'discovery',
        'count': 3, 'items': [{'sample_id': g['query_id'], 'stratum': 'alarm'} for g in groups]}
    custody = {'source_all_responses_observed': True, 'original_total_usage': {'calls': 199}}
    def verified(spec, **kwargs):
        assert kwargs['groups'] == groups and kwargs['parent'] == parent
        assert kwargs['common'] == batch['common'] and kwargs['model_id'] == 'model'
        assert spec == {'relpath': 'final-import.json', 'sha256': 'd' * 64}
        return [], custody
    monkeypatch.setitem(sys.modules, 'speech_aware_evidence_acquisition.rails.tfgrpo_final_import',
                        SimpleNamespace(load_import=verified))
    kwargs = {'final_import': {'relpath': 'final-import.json', 'sha256': 'd' * 64},
        'runtime_identity': {'model_fragment': 'model'}, 'model_values': model,
        'provenance': [], 'data_root': Path('/data')}
    return source, roster, kwargs


def test_one_contact_preserves_all_learning_groups_and_parent(monkeypatch):
    source, roster, kwargs = fixture(monkeypatch)
    before = copy.deepcopy((source, roster, kwargs))
    new_roster, new_source, reg, plan, custody = m.documents(source, roster, **kwargs)
    assert plan.planned_model_calls == 1 and reg['contacts'] == [
        {'stage': 'batch_update', 'identity': 'batch', 'sample_id': source['batch']['groups'][0]['query_id']}]
    assert len(new_roster['items']) == new_roster['count'] == 1
    assert new_source['batch']['groups'] == source['batch']['groups']
    assert len(new_source['batch']['groups']) == 3
    assert 'summary_import' not in new_source['batch']
    assert new_source['batch']['final_import'] == kwargs['final_import']
    assert new_source['batch']['bounds']['max_calls'] == 1
    assert new_source['batch']['bounds']['concurrency'] == 1
    assert reg['batch_sha256'] == m.sha(new_source['batch'])
    assert custody['original_total_usage']['calls'] == 199
    assert (source, roster, kwargs) == before


@pytest.mark.parametrize('fault', ['model', 'tokens', 'reasoning', 'reserve', 'roster', 'duplicate', 'tied', 'prompt'])
def test_invalid_successor_does_not_bypass_custody(monkeypatch, fault):
    source, roster, kwargs = fixture(monkeypatch)
    if fault == 'model': kwargs['runtime_identity']['model_fragment'] = 'other'
    elif fault == 'tokens': kwargs['model_values']['decoding_params']['max_tokens'] = 4096
    elif fault == 'reasoning': kwargs['model_values']['reasoning_budget_tokens'] = 2048
    elif fault == 'reserve': kwargs['model_values']['answer_reserve_tokens'] = 1024
    elif fault == 'roster': roster['items'].pop()
    elif fault == 'duplicate': source['batch']['groups'].append(copy.deepcopy(source['batch']['groups'][0]))
    elif fault == 'tied':
        for r in source['batch']['groups'][0]['rollouts']: r['reward'] = 0
    else: source['batch']['prompt_sha256'] = '0' * 64
    with pytest.raises(ValueError): m.documents(source, roster, **kwargs)


def test_final_fragment_is_explicit_copy_with_unchanged_engine():
    old = m.read(m.ROOT / 'configs/model/bf16-thinking-tfgrpo-tp4-ctx48k-v1.json')
    before = copy.deepcopy(old)
    result = m.final_model_values(old)
    assert old == before
    assert result['server_args'] == old['server_args']
    assert result['runtime_receipt'] == old['runtime_receipt']
    assert result['decoding_params']['max_tokens'] == 16384
    assert result['answer_reserve_tokens'] + result['reasoning_budget_tokens'] == 16384
    assert original.learning_decoding(result)['thinking_budget_tokens'] == 3072


def prepared(tmp_path, monkeypatch):
    source, roster, kwargs = fixture(monkeypatch)
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    old_job, job = data / 'work/codex/old', data / 'work/codex/new'
    old_job.mkdir(parents=True)
    model = repo / 'configs/model/model.json'
    model.parent.mkdir(parents=True)
    model.write_bytes(m.raw(kwargs['model_values']))
    for name, doc in [('source.json', source), ('construction-roster.json', roster)]:
        (old_job / name).write_bytes(m.raw(doc))
    (old_job / 'complete.json').write_bytes(m.raw({'artifacts': {
        name: m.file_sha(old_job / name) for name in ('source.json', 'construction-roster.json')}}))
    identity = data / 'identity.json'; identity.write_bytes(m.raw(kwargs['runtime_identity']))
    monkeypatch.setattr(m.base, 'deployment_identity', lambda *_: kwargs['runtime_identity'])
    monkeypatch.setattr('speech_aware_evidence_acquisition.contracts.assert_execution_environment', lambda: None)
    from speech_aware_evidence_acquisition.rails import tfgrpo_source
    monkeypatch.setattr(tfgrpo_source, 'load_dynamic_source', lambda *a, **k: None)
    m.prepare(repo, data, old_job, kwargs['final_import'], job, identity)
    return repo, data, old_job, job


def test_prepare_and_verify_exact_derivation(tmp_path, monkeypatch):
    repo, data, old_job, job = prepared(tmp_path, monkeypatch)
    complete, reg, roster = m.verify_job(repo, data, job)
    assert complete['groups'] == 3 and complete['runtime_rows'] == 1
    assert complete['generation_calls'] == complete['fit_calls'] == 1
    assert complete['new_model_calls'] == 0
    assert len(reg['contacts']) == len(roster['items']) == 1


@pytest.mark.parametrize('fault', ['groups', 'cost', 'source', 'roster', 'manifest'])
def test_reject_self_consistent_changed_job(tmp_path, monkeypatch, fault):
    repo, data, old_job, job = prepared(tmp_path, monkeypatch)
    complete = m.read(job / 'complete.json')
    if fault == 'groups': complete['groups'] = 1
    elif fault == 'cost': complete['historical_custody_sha256'] = '0' * 64
    elif fault == 'source':
        source = m.read(job / 'source.json'); source['batch']['bounds']['max_calls'] = 2
        (job / 'source.json').write_bytes(m.raw(source))
        complete['artifacts']['source.json'] = m.file_sha(job / 'source.json')
    elif fault == 'roster':
        roster = m.read(job / 'construction-roster.json'); roster['items'] = []
        (job / 'construction-roster.json').write_bytes(m.raw(roster))
        complete['artifacts']['construction-roster.json'] = m.file_sha(job / 'construction-roster.json')
    else: (old_job / 'complete.json').write_text('{}')
    (job / 'complete.json').write_bytes(m.raw(complete))
    with pytest.raises(ValueError): m.verify_job(repo, data, job)


def test_register_one_fit_generation_contact_and_all_learning_coverage(tmp_path, monkeypatch):
    import shutil
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    monkeypatch.setattr(git_commit, 'require_clean_study_repo', lambda _: 'a' * 40)
    repo, data, old_job, job = prepared(tmp_path, monkeypatch)
    complete, reg, roster = m.verify_job(repo, data, job)
    monkeypatch.setattr(m, 'verify_job', lambda *_: (complete, reg, roster))
    for rel in ('configs/baseline/system-group-update-text-v1.json',
                'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json',
                'configs/experiment/slurp-tfrl-hardset-a0-observation-r1-v1.json',
                'configs/dataset/slurp-speech-native-stage3-construction-r01c.json'):
        target = repo / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m.ROOT / rel, target)
    (repo / 'docs').mkdir(); (repo / 'docs/exposure-ledger.md').write_text('')
    result = m.register(repo, data, job)
    assert result['calls'] == result['fit_calls'] == result['rows'] == 1
    assert result['learned_groups'] == 3
    exp = m.read(repo / f'configs/experiment/{m.NAME}.json')
    assert exp['tfgrpo_contact_fit']['contact_cap'] == 1
    assert exp['populations'][0]['ids'] == [roster['items'][0]['sample_id']]
    for kind in ('generation', 'fit'):
        doc = m.read(repo / 'docs/readiness' / m.NAME / (kind + '-plan.json'))
        assert doc['execution_plan']['planned_model_calls'] == 1
    assert complete['groups'] == 3
    with pytest.raises(ValueError, match='existing'): m.register(repo, data, job)


def test_admission_uses_exact_final_job_and_shared_decision(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.core import config
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    from speech_aware_evidence_acquisition.core.session import serving
    from tfrl_current_generation import SESSION
    repo, data, old_job, job = prepared(tmp_path, monkeypatch)
    complete, reg, roster = m.verify_job(repo, data, job)
    monkeypatch.setattr(git_commit, 'require_clean_study_repo', lambda _: 'a' * 40)
    monkeypatch.setattr(m, 'verify_job', lambda *_: (complete, reg, roster))
    pkg = repo / 'docs/readiness' / m.NAME; pkg.mkdir(parents=True)
    exp = {'cells': [{'cell_id': m.NAME, 'dataset_fragment': 'dataset',
        'readiness': {'state': 'held', 'requirements': []}}], 'tfgrpo_learning_registration': reg}
    ep = repo / f'configs/experiment/{m.NAME}.json'; ep.parent.mkdir(parents=True)
    ep.write_bytes(m.raw(exp))
    for kind in ('generation', 'fit'):
        (pkg / (kind + '-plan.json')).write_bytes(m.raw({
            'execution_plan': {'protocol_hash': 'old'}, 'config_sha256': 'old'}))
    ledger = repo / 'docs/exposure-ledger.md'; ledger.write_text(m.file_sha(pkg / 'generation-plan.json'))
    authority = repo / m.base.AUTH; authority.parent.mkdir(parents=True)
    authority.write_text('Authorized finite learning.')
    receipt = repo / 'receipt.json'; receipt.write_bytes(m.raw({'deployment': {'instances': []}}))
    identity = m.read(job / 'runtime-identity.json')
    monkeypatch.setattr(serving, 'find_resident_session', lambda _: (data, {'session_id': SESSION, **identity}))
    monkeypatch.setattr(serving, 'assert_identity_matches', lambda *_: None)
    monkeypatch.setattr(config, 'read_fragment', lambda *_: {})
    monkeypatch.setattr(config, 'resolve_runtime_receipt_path', lambda *_: receipt)
    monkeypatch.setattr(config, 'compose', lambda *_: SimpleNamespace(config_hash='b' * 64, values={'arm': 'prepared-context-text'}))
    def mint(root, data, coordinator, dry, *, family, tool):
        draft = m.read(pkg / 'root-admission-draft-v1.json')
        assert 'one final consolidation' in draft['decision']
        draft['status'] = 'decided'
        target = Path(data) / family.DECISION_RELPATH
        target.parent.mkdir(parents=True); target.write_bytes(m.raw(draft))
        return {'out': str(target)}
    monkeypatch.setitem(sys.modules, 'thinking_system_mint_receipt', SimpleNamespace(mint=mint))
    result = m.admit(repo, data, job)
    assert result['new_model_calls'] == 0
    assert m.read(ep)['cells'][0]['readiness']['state'] == 'ready'
    plan = m.read(pkg / 'generation-plan.json')
    assert plan['config_sha256'] == 'b' * 64
    assert m.file_sha(pkg / 'generation-plan.json') in ledger.read_text()
