"""New producer identities reuse the historical full H2 executable contract."""
import copy
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
import tfgrpo_h2_job as job
import tfgrpo_h2_paired as paired
import registered_workload
from test_tfgrpo_h2_job import workload
from test_tfgrpo_consumption import release


def evaluation(suffix='one'):
    name = 'ace-shared-h2-' + suffix
    return {'schema': paired.EVALUATION_SCHEMA, 'name': name,
        'run_id': 'SAEA-ACE-SHARED-H2-' + suffix.upper(),
        'template': 'probes/' + name + '-experience.txt',
        'producer_strategy': 'ace_incremental_offline_v1',
        'runtime': {'model_fragment': job.MODEL, 'service_session': job.SESSION,
            'profile': job.PROFILE, 'runtime_receipt': job.RECEIPT},
        'candidate_release': {'relpath': 'release.json', 'sha256': 'e' * 64}}


def bind_producer(data, params, config):
    # Only method binding is under test; native custody has independent tests.
    candidate_path = data / params['candidate']['relpath']
    candidate = job.read(candidate_path)
    candidate['strategy'] = config['producer_strategy']
    candidate_path.write_bytes(job.raw(candidate))
    params['candidate'] = job.pin(data, params['candidate']['relpath'])
    release_path = data / 'release.json'
    release_path.write_bytes(job.raw({'strategy': config['producer_strategy'], 'candidate': params['candidate']}))
    config['candidate_release'] = job.pin(data, 'release.json')


@pytest.fixture
def shared_workload(workload, request):
    repo, data, spec, exp, view, derived = workload
    config = evaluation(); config['producer_strategy'] = getattr(request, 'param', 'ace_incremental_offline_v1')
    name, run = config['name'], config['run_id']
    base = job.read(repo / f'configs/baseline/{job.NAME}.json')
    component = next(c for c in base['native_episode']['consumption_module']['components']
        if c['component_id'] == job.consumption.COMPONENT)
    params = component['params']; bind_producer(data, params, config)
    params['candidate_release'] = config['candidate_release']
    params['template']['relpath'] = config['template']
    block = base['native_episode']['consumption_module']
    block['columns'][0]['template']['relpath'] = 'configs/' + config['template']
    base['native_episode']['roles']['solver']['template_variants']['arms'][0]['instruction_template']['relpath'] = 'configs/' + config['template']
    base['native_contact_fit']['run_id'] = run + '-FIT'
    exp['cells'][0].update(cell_id=name, run_id=run, baseline_fragment=name)
    exp['stage3_algorithm'].update(id=paired.evaluation_algorithm(config)[0], operators_used=['context_application'])
    spec.update(cell=name, experiment=name, run_id=run, fit_run_id=run + '-FIT',
        workload_kind='learning-full-h2-paired', evaluation_config=config,
        scientific_baseline_sha256=job.scientific_hash(base))
    for rel, doc in [(f'configs/model/{job.MODEL}.json', {'runtime_receipt': job.RECEIPT}),
            (f'configs/deployment/{job.PROFILE}.json', {'model_fragment': job.MODEL}),
            (f'configs/baseline/{name}.json', base), (f'configs/experiment/{name}.json', exp)]:
        path = repo / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(job.raw(doc))
    return repo, data, spec, exp, view, base


@pytest.mark.parametrize('shared_workload', list(paired.EVALUATION_METHODS), indirect=True)
def test_shared_registered_full_h2_dispatch(shared_workload):
    repo, data, spec, exp, view, base = shared_workload
    assert registered_workload.validate_workload(repo, data, spec['cell'], spec) == 312
    assert base['native_contact_fit']['contact_cap'] == 936
    assert exp['execution_plan']['planned_model_calls'] == 1872
    from speech_aware_evidence_acquisition.core import stage3_algorithms
    from speech_aware_evidence_acquisition.core.base import STAGE3_ALGORITHMS
    record = STAGE3_ALGORITHMS.resolve(exp['stage3_algorithm']['id'])
    assert set(exp['stage3_algorithm']['operators_used']) <= set(record.operators)


@pytest.mark.parametrize('fault', ['release', 'missing_config', 'subset', 'order'])
def test_shared_dispatch_rejects_drift(shared_workload, fault):
    repo, data, spec, exp, view, base = shared_workload
    if fault == 'release': spec['evaluation_config']['candidate_release']['sha256'] = 'f' * 64
    elif fault == 'missing_config': spec.pop('evaluation_config')
    elif fault == 'subset': exp['populations'][0]['ids'].pop()
    else: exp['populations'][0]['ids'].reverse()
    (repo / f'configs/experiment/{spec["cell"]}.json').write_bytes(job.raw(exp))
    with pytest.raises(ValueError): registered_workload.validate_workload(repo, data, spec['cell'], spec)


def test_two_configurations_derive_without_mutating_globals(release, monkeypatch):
    _, data, params = release
    params['consumer_ids'] = [r['sample_id'] for r in job.read(ROOT / job.ROSTER)['items']]
    params['base_template']['relpath'] = job.read(ROOT / 'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json')['phase_material']['solver']['instruction_template']
    monkeypatch.setattr(paired.consumption, 'verify', lambda *a, **kw: None)
    original = (paired.NAME, paired.RUN, job.NAME, job.RUN)
    results = []
    for suffix, strategy in zip(('one', 'two'), paired.EVALUATION_METHODS):
        config = evaluation(suffix); config['producer_strategy'] = strategy; p = copy.deepcopy(params)
        bind_producer(data, p, config)
        p['candidate_release'] = config['candidate_release']; p['template']['relpath'] = config['template']
        result = paired.derive(ROOT, data, p, fit_estimate={'slot_tokens':24576,
            'completion_ceiling':4096, 'base_prompt_tokens':2000, 'tokens_per_supplied_item':235},
            base_template_tokens=400, learned_template_tokens=500, evaluation=config)
        results.append(result['experiment']['cells'][0]['run_id'])
        assert result['experiment']['stage3_algorithm']['id'] == paired.EVALUATION_METHODS[strategy][0]
        assert result['evaluation']['rows'] == 312
        assert result['evaluation']['normal_generation_calls'] == 936
    assert results == ['SAEA-ACE-SHARED-H2-ONE', 'SAEA-ACE-SHARED-H2-TWO']
    assert (paired.NAME, paired.RUN, job.NAME, job.RUN) == original


@pytest.mark.parametrize('key,value', [('name', paired.NAME), ('run_id', paired.RUN),
    ('producer_strategy', 'unsupported-reflexion'), ('name', '../escape'), ('template', 'probes/wrong.txt')])
def test_explicit_identity_rejects_unsupported_or_historical_override(key, value):
    config = evaluation(); config[key] = value
    with pytest.raises(ValueError): paired.evaluation_identity(config)


def test_independent_runs_do_not_share_spend_identity(tmp_path):
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    (repo / 'docs').mkdir(parents=True); (repo / 'docs/exposure-ledger.md').write_text('')
    one, two = evaluation('one'), evaluation('two')
    (data / 'attempts' / one['run_id']).mkdir(parents=True)
    with pytest.raises(ValueError, match='existing H2'): job.require_unspent(repo, data, one)
    job.require_unspent(repo, data, two)


@pytest.mark.parametrize('fault', [None, 'unknown_key', 'malformed_pin'])
def test_shared_compiler_accepts_only_exact_release_contract(release, fault):
    from test_tfgrpo_h2_paired import paired_block
    from speech_aware_evidence_acquisition.rails import consumption_module as cm
    from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    params = copy.deepcopy(release[2]); params['candidate_release'] = evaluation()['candidate_release']
    if fault == 'unknown_key':params['semantic_approval_override'] = True
    elif fault == 'malformed_pin':params['candidate_release'] = {'path':'release.json', 'sha256':'e' * 64}
    base, block = paired_block(params)
    if fault:
        with pytest.raises((ValueError, cm.ConsumptionModuleConfigError)):cm.compile_module(block, carrier_module_hooks(repo_root=ROOT), repo_root=ROOT)
    else:
        compiled = cm.compile_module(block, carrier_module_hooks(repo_root=ROOT), repo_root=ROOT)
        output = compiled.apply_to(base); output['native_episode']['consumption_module'] = block
        cm.verify_declared_module(block, output, repo_root=ROOT)
        assert native_contact_custody_enabled({**output, 'transport':'vllm', 'carrier_lock_key':'slurp'})


@pytest.mark.parametrize('shared_workload', list(paired.EVALUATION_METHODS), indirect=True)
@pytest.mark.parametrize('fault', ['declared_method', 'candidate_method', 'release_method', 'algorithm_label'])
def test_h2_rejects_producer_mislabel_even_with_updated_pins(shared_workload, fault):
    repo, data, spec, exp, view, base = shared_workload
    config = spec['evaluation_config']
    other = next(s for s in paired.EVALUATION_METHODS if s != config['producer_strategy'])
    params = next(c['params'] for c in base['native_episode']['consumption_module']['components']
                  if c['component_id'] == job.consumption.COMPONENT)
    if fault == 'declared_method':
        config['producer_strategy'] = other
        exp['stage3_algorithm']['id'] = paired.EVALUATION_METHODS[other][0]
    elif fault == 'algorithm_label':
        exp['stage3_algorithm']['id'] = paired.EVALUATION_METHODS[other][0]
    else:
        rel = job.read(data / params['candidate_release']['relpath'])
        if fault == 'candidate_method':
            path = data / params['candidate']['relpath']; candidate = job.read(path)
            candidate['strategy'] = other; path.write_bytes(job.raw(candidate))
            params['candidate'] = job.pin(data, params['candidate']['relpath'])
            rel['candidate'] = params['candidate']
        else:
            rel['strategy'] = other
        (data / params['candidate_release']['relpath']).write_bytes(job.raw(rel))
        params['candidate_release'] = job.pin(data, params['candidate_release']['relpath'])
        config['candidate_release'] = params['candidate_release']
        spec['scientific_baseline_sha256'] = job.scientific_hash(base)
    (repo / f'configs/baseline/{spec["cell"]}.json').write_bytes(job.raw(base))
    (repo / f'configs/experiment/{spec["cell"]}.json').write_bytes(job.raw(exp))
    with pytest.raises(ValueError, match='producer|program'):
        registered_workload.validate_workload(repo, data, spec['cell'], spec)
