"""Full-test contracts use the real fixed roster; no synthetic cost estimates."""
import copy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
import final_report_workload as m

MODEL = 'bf16-thinking-tfgrpo-tp4-ctx48k-v1'
PROFILE = 'qwen3-omni-tfgrpo-2xtp4-s128-b2048-ctx48k-v1'


def put(root, rel, doc):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.raw(doc))
    return m.pin(root, rel)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    repo.mkdir(); data.mkdir()
    for rel in [m.ROSTER, f'configs/dataset/{m.DATASET}.json', m.AUTH, m.POFF_AUTH,
                m.DEVELOPMENT_LOCK, m.MODEL_SOURCE,
                'scripts/flights/nx/l338_port_pins.json',
                'docs/receipts/knowledge/slurp-kb-size-pool-annotated-exemplars-minus-r01.json',
                'configs/baseline/slurp-native-default-plan-lock-baseline-v1.json',
                *[value[0] for value in m.SOURCES.values()],
                f'configs/model/{MODEL}.json', f'configs/deployment/{PROFILE}.json']:
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    for directory in ('configs/probes', 'configs/modules', 'configs/consumption'):
        if (ROOT / directory).exists():
            shutil.copytree(ROOT / directory, repo / directory, dirs_exist_ok=True)
    # Armed compiler components require their actual admission records.
    def copy_records(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'record' and isinstance(item, str) and (ROOT / item).is_file():
                    destination = repo / item
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / item, destination)
                else:
                    copy_records(item)
        elif isinstance(value, list):
            for item in value:
                copy_records(item)
    for source, _ in m.SOURCES.values():
        copy_records(m.read(ROOT / source))
    (repo / 'docs/exposure-ledger.md').write_text('Historical exposures and costs remain.\n')
    model = m.read(repo / f'configs/model/{MODEL}.json')
    # The live alias is runtime-local; public source keeps its identical model
    # contract under the deployment fragment. Do not fabricate a new contract.
    put(repo, 'configs/model/bf16-thinking-real500-tp4-s128-v1.json', model)
    roster, view = m.population(repo)
    name, run = m.identity('direct')
    doc = {'schema': 'saea-owner-split-allowance-v1', 'study': 'speech-aware-evidence-acquisition',
        'status': 'authorised', 'dated_owner_decision': 'Owner 2026-10-30 full-test decision; independent frozen baseline authorized 2026-11-02.',
        'authorization_basis': {'relpath': m.AUTH, 'sha256': m.AUTH_SHA},
        'venue': 'Owner authorizes this NPU reproduction study, no paper publication.',
        'split': {'carrier_lock_key': 'slurp', 'split_identity_hash': roster['split_identity_hash'],
            'roster_id': roster['roster_id'], 'roster_receipt_relpath': m.ROSTER,
            'roster_receipt_sha256': m.ROSTER_SHA},
        'permitted_run_ids': [run, run + '-FIT', run + '-READ'],
        'read_run_ids': [run, run + '-READ'], 'preparation_run_ids': [run + '-FIT'],
        'evaluation_policy': copy.deepcopy(m.POLICY)}
    allowance_pin = put(repo, 'docs/receipts/owner/direct-fulltest-v1.json', doc)
    config = {'schema': 'saea-final-report-job-config-v1', 'method': 'direct',
        'source_baseline': {'relpath': m.SOURCES['direct'][0], 'sha256': m.SOURCES['direct'][1]},
        'runtime': {'model_fragment': MODEL, 'profile': PROFILE,
            'service_session': 'test-fixture-resident', 'runtime_receipt': model['runtime_receipt']},
        'budget': {'generation_calls': 26156, 'fit_calls': 13078, 'gpu_hours': 100.0},
        'execution': {'admitted_level': 96},
        'allowance': {'path': allowance_pin['relpath'], 'sha256': allowance_pin['sha256']}}
    monkeypatch.setattr(m, 'environment', lambda repo: 'a' * 40)
    return repo, data, config


def fake_direct_runtime(monkeypatch, error='program has no supplied first phase to prepare'):
    from speech_aware_evidence_acquisition.core.flight import factory
    from speech_aware_evidence_acquisition.core.preparation import PreparationError
    calls = []
    class Runtime:
        def phase_supply_artifact_identity(self):
            calls.append('identity')
            if error is not None:
                raise PreparationError(error)
            return {}
        def verify_prepared_artifacts(self):
            calls.append('verify')
            return []
    monkeypatch.setattr(factory, 'build_flight_runtime', lambda *a, **kw: Runtime())
    return calls


def stage_and_prepare(configured, monkeypatch):
    repo, data, config = configured
    stage = data / 'work/codex/stage'
    m.stage(repo, data, config, stage)
    fake_direct_runtime(monkeypatch)
    prep = data / 'work/codex/prepared'
    m.prepare(repo, data, stage / 'complete.json', prep)
    return repo, data, config, stage / 'complete.json', prep / 'complete.json'


def test_direct_derives_exact_full_roster_and_frozen_prompts(configured):
    repo, data, config = configured
    base, exp, roster, view = m.derive(repo, data, config)
    source = m.read(repo / m.SOURCES['direct'][0])
    assert len(exp['populations'][0]['ids']) == len(roster['items']) == 13078
    assert exp['populations'][0]['ids'] == [r['sample_id'] for r in view['items']]
    assert base['phase_material'] == source['phase_material']
    assert base['task_instruction_template'] == source['task_instruction_template']
    assert base['scorers'] == ['slurp-slu-official-test', 'cost']
    assert base['serving_slot_tokens'] == 49152
    assert base['live_phase_fit']['contact_cap'] == 13078
    assert base['live_phase_fit']['run_id'] == m.identity('direct')[1] + '-FIT'
    assert exp['execution_plan']['split_role'] == 'final-report'
    assert exp['execution_plan']['planned_model_calls'] == 26156
    assert exp['execution_plan']['planned_speech_audio_seconds'] == 73824
    assert exp['cells'][0]['readiness']['state'] == 'held'
    assert exp['stage3_algorithm']['operators_used'] == []
    assert base['serving_slots'] == source['serving_slots'] == 4
    assert m.execution_binding(repo, config) == {'admitted_level': 96,
        'instances': 2, 'total_client_concurrency': 192,
        'environment': {'SAEA_ADMITTED_LEVEL': '96'}}


@pytest.mark.parametrize('level', [None, True, 4, 95, 128, '96'])
def test_execution_requires_explicit_stable_c96(configured, level):
    repo, data, config = configured
    if level is None:
        config.pop('execution')
    else:
        config['execution']['admitted_level'] = level
    with pytest.raises(ValueError): m.derive(repo, data, config)


def test_c96_uses_actual_shared_runtime_width_without_geometry_change(configured):
    from speech_aware_evidence_acquisition.core.flight.runtime import FlightRuntime
    repo, _, config = configured
    profile = m.read(repo / f'configs/deployment/{PROFILE}.json')
    runtime = object.__new__(FlightRuntime)
    runtime._admitted_level = config['execution']['admitted_level']
    runtime._deployment_profile_object = lambda: SimpleNamespace(
        instances=profile['instances'], admitted_level=profile['admitted_level'],
        max_num_seqs=profile['max_num_seqs'])
    assert runtime._flight_width() == 192
    profile['admitted_level'] = 64
    put(repo, f'configs/deployment/{PROFILE}.json', profile)
    with pytest.raises(ValueError, match='cannot raise'): m.execution_binding(repo, config)


def test_explicit_model_alias_requires_real_source_and_profile_compatibility(configured):
    repo, data, config = configured
    alias = 'bf16-thinking-real500-tp4-s128-v1'
    original = m.read(repo / f'configs/model/{MODEL}.json')
    put(repo, f'configs/model/{alias}.json', original)
    config['runtime']['model_fragment'] = alias
    # The declared deployment's differently named fragment has identical task
    # and server arguments, so the production alias does not change the method.
    m.derive(repo, data, config)
    changed = copy.deepcopy(original); changed['decoding_params']['max_tokens'] = 16384
    put(repo, f'configs/model/{MODEL}.json', changed)
    with pytest.raises(ValueError, match='deployment differs'): m.derive(repo, data, config)
    put(repo, f'configs/model/{MODEL}.json', original)
    (repo / m.MODEL_SOURCE).unlink()
    with pytest.raises(ValueError, match='pinned input'): m.derive(repo, data, config)


def test_allowance_records_existing_authority_once(configured):
    repo, data, config = configured
    result = m.create_allowance(repo, data, 'direct')
    config['allowance'] = result['allowance']
    assert result['generation_authorized'] is False
    assert result['new_model_calls'] == 0 and result['readiness'] == 'held'
    m.derive(repo, data, config)
    with pytest.raises(FileExistsError):
        m.create_allowance(repo, data, 'direct')


def test_final_report_recovery_receipt_preserves_allowance(configured, monkeypatch):
    from dataclasses import replace
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.contracts import plan as contract
    from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay
    from speech_aware_evidence_acquisition.core.flight.native_recovery import load_native_recovery
    from native_recovery_receipt import prepare
    repo, data, config = configured
    monkeypatch.setattr(contract, '_STUDY_REPO_ROOT', repo)
    _, exp, _, view = m.derive(repo, data, config)
    _, run = m.identity('direct')
    plan = ExecutionPlan(run_id=run, attempt_id='generation-v1',
        carrier_lock_key='slurp', split_identity_hash=view['split_identity_hash'],
        protocol_hash='a' * 64, owner_allowance=config['allowance'],
        **exp['execution_plan']).validate()
    path = data / 'attempts' / run / 'generation-v1.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(m.raw({'state': 'ABORTED', 'run_id': run,
        'attempt_id': plan.attempt_id, 'carrier_lock_key': 'slurp',
        'actual': {'calls': 0, 'audio_seconds': 0.0}}))
    target = replace(plan, attempt_id='recovery-v1')
    replay = NativeContactReplay(data, plan, target, 'raw-speech-agent-native-tools')
    assert replay.source_binding()['plan']['owner_allowance'] == config['allowance']
    out = data / 'work/codex/recovery.json'
    result = prepare(data, plan, target.attempt_id, 'raw-speech-agent-native-tools', out)
    assert m.read(out)['target_plan']['owner_allowance'] == config['allowance']
    load_native_recovery(result['declaration'], data_root=data, plan=target, arm='raw-speech-agent-native-tools')
    with pytest.raises(ValueError):
        load_native_recovery(result['declaration'], data_root=data,
            plan=replace(target, owner_allowance=None), arm='raw-speech-agent-native-tools')


@pytest.mark.parametrize('fault', ['subset', 'duplicate', 'order', 'label', 'audio'])
def test_exact_dataset_pin_refuses_population_changes(configured, fault):
    repo, data, config = configured
    path = repo / f'configs/dataset/{m.DATASET}.json'
    view = m.read(path)
    if fault == 'subset': view['items'].pop()
    elif fault == 'duplicate': view['items'][-1] = view['items'][0]
    elif fault == 'order': view['items'].reverse()
    elif fault == 'label': view['items'][0]['sentence'] = 'gold must not enter runtime'
    else: view['items'][0]['media_relpath'] = 'synthetic.wav'
    path.write_bytes(m.raw(view))
    with pytest.raises(ValueError, match='pinned input'): m.derive(repo, data, config)


@pytest.mark.parametrize('fault', ['allowance', 'source', 'generation_cap', 'fit_cap', 'hours', 'model', 'profile'])
def test_frozen_configuration_refuses_drift(configured, fault):
    repo, data, config = configured
    if fault == 'allowance': config['allowance']['sha256'] = '0' * 64
    elif fault == 'source': config['source_baseline']['sha256'] = '0' * 64
    elif fault == 'generation_cap': config['budget']['generation_calls'] = 13078
    elif fault == 'fit_cap': config['budget']['fit_calls'] = 1
    elif fault == 'hours': config['budget']['gpu_hours'] = float('nan')
    elif fault == 'model':
        path = repo / f'configs/model/{MODEL}.json'; doc = m.read(path)
        doc['decoding_params']['max_tokens'] = 16384; path.write_bytes(m.raw(doc))
    else:
        path = repo / f'configs/deployment/{PROFILE}.json'; doc = m.read(path)
        doc['instances'][1]['cards'] = [0, 1, 2, 3]; path.write_bytes(m.raw(doc))
    with pytest.raises((ValueError, RuntimeError)): m.derive(repo, data, config)


def test_prepare_uses_shared_flight_and_does_not_fabricate_supply(configured, monkeypatch):
    repo, data, config = configured
    stage = data / 'work/codex/stage'; m.stage(repo, data, config, stage)
    calls = fake_direct_runtime(monkeypatch)
    out = data / 'work/codex/prepared'
    result = m.prepare(repo, data, stage / 'complete.json', out)
    assert calls == ['identity', 'verify']
    assert result['supply'] is None and result['readiness'] == 'held'
    assert result['new_model_calls'] == 0
    assert not (data / 'prepared').exists()
    assert (repo / 'docs/exposure-ledger.md').read_text().startswith('Historical exposures')


@pytest.mark.parametrize('failure', ['missing source snapshot', None])
def test_prepare_does_not_hide_dependency_errors(configured, monkeypatch, failure):
    repo, data, config = configured
    stage = data / 'work/codex/stage'; m.stage(repo, data, config, stage)
    fake_direct_runtime(monkeypatch, failure)
    out = data / 'work/codex/prepared'
    with pytest.raises(Exception): m.prepare(repo, data, stage / 'complete.json', out)
    assert not (out / 'complete.json').exists()


@pytest.fixture
def registered(configured, monkeypatch):
    repo, data, config, stage, prep = stage_and_prepare(configured, monkeypatch)
    # Real compose and real ExecutionPlan/allowance validation are intentional.
    result = m.register(repo, data, stage, prep)
    name = result['candidate']; pkg = repo / 'docs/readiness' / name
    spec = m.read(pkg / 'candidate.json')
    exp = m.read(repo / f'configs/experiment/{name}.json')
    view = m.read(repo / f'configs/dataset/{m.DATASET}.json')
    return repo, data, name, spec, exp, view, pkg


def test_registration_shared_validator_and_admission_are_complete(registered):
    import registered_workload
    import admit_real_engine_candidate as admit
    repo, data, name, spec, exp, view, pkg = registered
    assert registered_workload.validate_workload(repo, data, name, spec) == 13078
    assert admit.test_contact_authorized(repo, data, name, spec, exp) is True
    assert exp['cells'][0]['readiness']['state'] == 'held'
    assert m.read(pkg / 'generation-plan-registered-v1.json')['execution_plan']['planned_model_calls'] == 26156
    assert m.read(pkg / 'fit-plan-registered-v1.json')['execution_plan']['planned_model_calls'] == 13078
    assert m.read(pkg / 'fit-plan-registered-v1.json')['execution_plan']['planned_speech_audio_seconds'] == 36912
    assert m.read(pkg / 'analysis-recipe-v1.json')['complete_manifest_only'] is True
    assert (repo / 'docs/exposure-ledger.md').read_text().startswith('Historical exposures')
    assert spec['execution']['total_client_concurrency'] == 192
    assert m.read(pkg / 'protocol-registered-v1.json')['execution'] == spec['execution']
    with pytest.raises(ValueError, match='existing'): m.require_unspent(repo, data, 'direct')


@pytest.mark.parametrize('fault', ['subset', 'budget', 'learning', 'scorer', 'preparation', 'authority', 'execution'])
def test_registered_drift_cannot_gain_test_authority(registered, fault):
    import admit_real_engine_candidate as admit
    repo, data, name, spec, exp, view, pkg = registered
    if fault == 'subset': exp['populations'][0]['ids'].pop()
    elif fault == 'budget': exp['execution_plan']['planned_model_calls'] += 1
    elif fault == 'learning': exp['stage3_algorithm']['operators_used'] = ['context_application']
    elif fault == 'authority': spec['authority'] = 'unrelated.md'
    elif fault == 'execution': spec['execution']['admitted_level'] = 128
    elif fault == 'preparation': (data / spec['preparation']['relpath']).unlink()
    else:
        path = repo / f'configs/baseline/{name}.json'; base = m.read(path)
        base['scorers'] = ['slurp-slu-official-train', 'cost']; path.write_bytes(m.raw(base))
    with pytest.raises(ValueError): admit.test_contact_authorized(repo, data, name, spec, exp)


def test_train_workload_cannot_self_declare_test_authority(configured):
    import admit_real_engine_candidate as admit
    repo, data, config = configured
    assert admit.test_contact_authorized(repo, data, 'train',
        {'workload_kind': 'tfrl-h1-fresh-observation', 'test_contact_authorized': True}, {}) is False


def test_unknown_spend_and_staged_file_never_blindly_repeat(configured):
    repo, data, config = configured
    run = m.identity('direct')[1]
    (data / 'attempts' / run).mkdir(parents=True)
    with pytest.raises(ValueError, match='spend'): m.stage(repo, data, config, data / 'work/codex/stage')


def test_a0_missing_real_speech_inputs_remains_unprepared(configured):
    repo, data, config = configured
    source = m.read(repo / m.SOURCES['a0'][0])
    view = m.population(repo)[1]
    with pytest.raises(ValueError, match='real full-test'):
        m.a0_inputs(repo, data, {'method': 'a0'}, view, source)


def test_allowance_binds_original_user_authority_and_run_partition(configured):
    repo, data, config = configured
    path = repo / config['allowance']['path']; doc = m.read(path)
    doc['authorization_basis']['sha256'] = '0' * 64
    path.write_bytes(m.raw(doc)); config['allowance']['sha256'] = m.sha(path)
    with pytest.raises(ValueError, match='owner authority'): m.derive(repo, data, config)


def test_admission_run_resume_require_same_c96_environment(configured):
    from registered_workload import assert_execution_binding
    repo, _, config = configured
    spec = {'workload_kind': m.KIND, 'job_config': config, 'profile': PROFILE,
        'execution': m.execution_binding(repo, config)}
    env = {'SAEA_DEPLOYMENT_PROFILE': PROFILE, 'SAEA_ADMITTED_LEVEL': '96'}
    assert_execution_binding(spec, env)
    for value in (None, '4', '128', '096'):
        changed = dict(env)
        if value is None: changed.pop('SAEA_ADMITTED_LEVEL')
        else: changed['SAEA_ADMITTED_LEVEL'] = value
        with pytest.raises(ValueError, match='C96'): assert_execution_binding(spec, changed)
    with pytest.raises(ValueError):
        assert_execution_binding(spec, {**env, 'SAEA_DEPLOYMENT_PROFILE': 'another'})
    assert_execution_binding({'workload_kind': 'tfrl-h1-fresh-observation'}, {})


@pytest.fixture
def poff_configured(configured):
    repo, data, config = configured
    config['method'] = 'a0-poff'
    config['source_baseline'] = dict(zip(('relpath', 'sha256'), m.SOURCES['a0-poff']))
    config['development_lock'] = {'relpath': m.DEVELOPMENT_LOCK, 'sha256': m.DEVELOPMENT_LOCK_SHA}
    config['runtime']['model_fragment'] = m.read(repo / m.DEVELOPMENT_LOCK)['model_fragment']
    config['budget'].update(generation_calls=52312, fit_calls=26156)
    config['allowance'] = m.create_allowance(repo, data, 'a0-poff')['allowance']
    return repo, data, config


@pytest.fixture
def poff_derivable(poff_configured, monkeypatch):
    """Isolate derivation; production vector custody remains checked by a0_inputs."""
    repo, data, config = poff_configured
    agreement_receipt = put(data, 'fixtures/agreement-receipt.json', {'artifact_id': 'SAEA-QUERY-UNIT'})
    inputs = {key: {'relpath': 'fixtures/' + key + '.jsonl', 'sha256': 'a' * 64}
              for key in ('segment_query', 'content_query', 'agreement', 'content_provenance')}
    inputs.update(agreement_receipt=agreement_receipt, source_receipts=[], query_ledger_row='SAEA-QUERY-UNIT')
    monkeypatch.setattr(m, 'a0_inputs', lambda *args: inputs)
    return repo, data, config


def test_poff_uses_separate_identity_authority_and_cannot_resume_old_a0(poff_configured):
    repo, data, config = poff_configured
    old_run = m.identity('a0')[1]
    (data / 'attempts' / old_run).mkdir(parents=True)
    assert m.identity('a0-poff') == ('slurp-a0-poff-fulltest-v1', 'SAEA-SLURP-A0-POFF-FULLTEST-V1')
    m.require_unspent(repo, data, 'a0-poff')
    with pytest.raises(ValueError, match='spend'): m.require_unspent(repo, data, 'a0')
    allowance = m.allowance(repo, config, m.population(repo)[0])
    assert allowance['authorization_basis'] == m.authority('a0-poff')
    assert old_run not in allowance['permitted_run_ids']
    assert m.authority('a0') == m.authority('direct') == {'relpath': m.AUTH, 'sha256': m.AUTH_SHA}
    # Historical hold is retained; the successor authority enables this new method.
    assert m.read(repo / m.DEVELOPMENT_LOCK)['full_test_held'] is True


def test_poff_only_changes_thinking_from_shared_a0_derivation(poff_derivable):
    repo, data, config = poff_derivable
    new, exp, roster, view = m.derive(repo, data, config)
    old_config = copy.deepcopy(config)
    old_config['method'] = 'a0'
    old_config.pop('development_lock')
    old_config['allowance'] = m.create_allowance(repo, data, 'a0')['allowance']
    old, _, _, _ = m.derive(repo, data, old_config)
    assert new.pop('thinking_policy') == m.read(repo / m.DEVELOPMENT_LOCK)['thinking_policy']
    for key in ('native_contact_fit', 'what_this_is'):
        new.pop(key); old.pop(key)
    assert new == old
    assert len(roster['items']) == len(view['items']) == exp['population']['count'] == 13078
    assert exp['execution_plan']['planned_model_calls'] == 52312


@pytest.mark.parametrize('fault', ['missing_lock', 'lock_pin', 'lock_bytes', 'model_alias', 'model', 'source', 'authority'])
def test_poff_rejects_development_source_and_runtime_drift(poff_configured, fault):
    repo, data, config = poff_configured
    if fault == 'missing_lock': config.pop('development_lock')
    elif fault == 'lock_pin': config['development_lock']['sha256'] = '0' * 64
    elif fault == 'lock_bytes':
        lock = m.read(repo / m.DEVELOPMENT_LOCK)
        lock['thinking_policy']['solver']['reasoning_budget_tokens'] = 1024
        put(repo, m.DEVELOPMENT_LOCK, lock)
    elif fault == 'model_alias': config['runtime']['model_fragment'] = MODEL
    elif fault == 'model':
        path = f"configs/model/{config['runtime']['model_fragment']}.json"
        model = m.read(repo / path); model['decoding_params']['temperature'] = 0.9
        put(repo, path, model)
    elif fault == 'source':
        path = m.SOURCES['a0-poff'][0]; source = m.read(repo / path)
        source['knowledge_stores']['train-exemplars']['snapshot']['sha256'] = '0' * 64
        put(repo, path, source)
    else:
        path = config['allowance']['path']; doc = m.read(repo / path)
        doc['authorization_basis'] = m.authority('a0')
        put(repo, path, doc); config['allowance']['sha256'] = m.sha(repo / path)
    with pytest.raises(ValueError): m.check_config(repo, config, m.population(repo)[0])


def test_old_method_cannot_silently_adopt_new_policy(configured):
    repo, data, config = configured
    config['development_lock'] = {'relpath': m.DEVELOPMENT_LOCK, 'sha256': m.DEVELOPMENT_LOCK_SHA}
    with pytest.raises(ValueError, match='historical methods'): m.derive(repo, data, config)


def test_poff_still_requires_all_real_full_test_inputs(poff_configured):
    repo, data, config = poff_configured
    with pytest.raises(ValueError, match='real full-test'): m.derive(repo, data, config)


@pytest.mark.parametrize('fault', ['perception', 'solver', 'knowledge'])
def test_poff_staged_scientific_drift_is_rejected(poff_derivable, fault):
    repo, data, config = poff_derivable
    out = data / 'work/codex/stage'
    m.stage(repo, data, config, out)
    path = f"configs/baseline/{m.identity('a0-poff')[0]}.json"
    base = m.read(repo / path)
    if fault == 'perception': base['thinking_policy']['perception']['enable_thinking'] = True
    elif fault == 'solver': base['thinking_policy']['solver']['reasoning_budget_tokens'] = 1024
    else: base['knowledge_stores']['train-exemplars']['snapshot']['sha256'] = '0' * 64
    put(repo, path, base)
    with pytest.raises(ValueError, match='staged configuration changed'):
        m.staged_inputs(repo, data, out / 'complete.json')


def test_poff_registers_and_validates_shared_a0_preparation(poff_derivable, monkeypatch):
    import tfrl_odd40_cells
    from speech_aware_evidence_acquisition.core.flight import factory
    repo, data, config = poff_derivable
    verified = []
    monkeypatch.setattr(tfrl_odd40_cells, 'verify_output', lambda output, ids: verified.append((output, len(ids))))
    class Runtime:
        def prepare_phase_supply(self, target, max_workers):
            put(data, str(target.relative_to(data)), {'output': {'unit_fixture': True}})
    monkeypatch.setattr(factory, 'build_flight_runtime', lambda *args, **kw: Runtime())
    out = data / 'work/codex/stage'
    m.stage(repo, data, config, out)
    prepared = data / 'work/codex/prepared'
    m.prepare(repo, data, out / 'complete.json', prepared)
    result = m.register(repo, data, out / 'complete.json', prepared / 'complete.json')
    name = result['candidate']
    spec = m.read(repo / f'docs/readiness/{name}/candidate.json')
    exp = m.read(repo / f'configs/experiment/{name}.json')
    assert spec['authority'] == m.POFF_AUTH
    assert m.validate(repo, data, name, spec, exp, m.population(repo)[1]) == 13078
    assert verified and all(count == 13078 for _, count in verified)
    assert m.read(repo / f'configs/baseline/{name}.json')['prepared_phase_supply'] is not None


@pytest.fixture
def lane_configured(poff_derivable):
    repo, data, config = poff_derivable
    for rel in [m.LANE_AUTH, f'configs/deployment/{m.LANE_PROFILE}.json', f'configs/model/{m.LANE_MODEL}.json']:
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    config['method'] = 'a0-poff-lane'
    config['runtime'].update(model_fragment=m.LANE_MODEL, profile=m.LANE_PROFILE,
        runtime_receipt=m.read(repo / f'configs/model/{m.LANE_MODEL}.json')['runtime_receipt'],
        service_session='serving-fulltest-tp4-lane-v1')
    config['allowance'] = m.create_allowance(repo, data, config['method'])['allowance']
    return repo, data, config


def test_lane_keeps_full_population_and_scientific_policy(lane_configured):
    repo, data, config = lane_configured
    base, exp, roster, view = m.derive(repo, data, config)
    assert len(roster['items']) == len(view['items']) == 13078
    assert base['thinking_policy'] == m.read(repo / m.DEVELOPMENT_LOCK)['thinking_policy']
    assert m.execution_binding(repo, config)['total_client_concurrency'] == 96
    assert m.execution_binding(repo, config)['instances'] == 1
    assert m.identity(config['method'])[1] == 'SAEA-SLURP-A0-POFF-LANE-FULLTEST-V1'


@pytest.mark.parametrize('fault', ['cards', 'port', 'tp', 'instances'])
def test_lane_rejects_resource_escape(lane_configured, fault):
    repo, data, config = lane_configured
    rel = f'configs/deployment/{m.LANE_PROFILE}.json'; profile = m.read(repo / rel)
    if fault == 'cards': profile['instances'][0]['cards'] = [4, 5, 6, 7]
    elif fault == 'port': profile['instances'][0]['port'] = 8202
    elif fault == 'tp': profile['tensor_parallel_size'] = 2
    else: profile['instances'].append(copy.deepcopy(profile['instances'][0]))
    put(repo, rel, profile)
    with pytest.raises(ValueError): m.derive(repo, data, config)
