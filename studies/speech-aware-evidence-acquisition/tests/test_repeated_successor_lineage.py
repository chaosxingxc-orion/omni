"""CPU compiler -> real business validator; source authentication is a fixture boundary."""
import copy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
spec = importlib.util.spec_from_file_location('lineage_compiler', ROOT / 'scripts/environment/configured_checkpoint_successor.py')
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)
import gepa_pair_workload as business
from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import (
    ImageReleaseCompatibility, RELEASE_SCHEMA, MULTI_SCHEMA, STUDY_PREFIX)


@pytest.fixture
def lineage(tmp_path, monkeypatch):
    names = dict(model='m', dataset='d', baseline='e', experiment='e')
    budget = dict(execution_profile='bounded-discovery-probe', split_role='discovery',
                  planned_model_calls=8, planned_gpu_hours=1, planned_speech_audio_seconds=20)
    root = dict(cell='e', experiment='e', baseline='e', model_fragment='m',
                dataset={'relpath': 'configs/dataset/d.json'}, run_id='RUN', attempt_id='generation-v1',
                fit_run_id='ROOT-FIT', fit_attempt_id='fit-v1', protocol_hash='b'*64,
                composed_config_hash='c'*64, split_identity_hash='d'*64,
                execution={'admitted_level': 2}, job_config={'execution': {'admitted_level': 2}},
                workload_kind=business.KIND)
    base = dict(arm='fixture', native_contact_fit=dict(run_id='ROOT-FIT', attempt_id='fit-v1', contact_cap=4))
    root['scientific_baseline_sha256'] = business.scientific_hash(base)
    cell = dict(cell_id='e', model_fragment='m', dataset_fragment='d', baseline_fragment='e',
                run_id='RUN', attempt_id=root['attempt_id'], readiness={'state': 'held'})
    fragments = dict(model={}, dataset={}, baseline=base,
                     experiment={'cells': [cell], 'execution_plan': budget})
    plan = dict(cell='e', experiment='e', composed_config_hash=root['composed_config_hash'],
                execution_plan=asdict(ExecutionPlan(run_id='RUN', attempt_id=root['attempt_id'],
                    carrier_lock_key='slurp', split_identity_hash=root['split_identity_hash'],
                    protocol_hash=root['protocol_hash'], **budget)))
    assets, sources = {}, {}
    def save(path, doc):
        blob = compiler.encode(doc)
        assets[path] = blob
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        return {'relpath': path, 'sha256': compiler.digest(blob)}
    save('docs/readiness/e/generation-plan-registered-v1.json', plan)
    def bridge(target):
        evidence = {k: {'path': '/fixture/' + k, 'sha256': 'a'*64} for k in
                    ('commit_object','tree_listing','source_archive','release_manifest','source_review','reference_pins')}
        evidence['startup_data_root'] = '/fixture/data'
        return dict(schema=RELEASE_SCHEMA, predecessor_commit='a'*40, predecessor_fragments=names,
                    successor_attempt_id=target, change='image-release-operational-v1', evidence=evidence)
    def capture(prior, frags):
        sources[prior] = {STUDY_PREFIX + 'configs/' + kind + '/' + names[kind] + '.json': compiler.encode(doc)
                          for kind, doc in frags.items()}
    capture(root['attempt_id'], fragments)
    def old_source(self, *, run_id=None, predecessor_attempt_id=None):
        assert run_id == 'RUN'
        return sources[predecessor_attempt_id]
    monkeypatch.setattr(ImageReleaseCompatibility, 'old_source', old_source)
    root_pin = save('docs/root.json', root)
    first_bridge = bridge('middle')
    first_declaration = {'attempt_id': root['attempt_id'], 'checkpoint_sha256': 'e'*64}
    b1, e1, middle = compiler.build_successor_documents(original=root, plan=plan, fragments=fragments,
        target=dict(attempt_id='middle', fit_run_id='MID-FIT', fit_attempt_id='mid-fit'),
        dispatch=None, width=4, supply={}, declarations=first_declaration, bridge=first_bridge,
        registration_relpath=root_pin['relpath'], registration_sha256=root_pin['sha256'], data=tmp_path)
    # Legacy one-hop documents did not carry the immediate predecessor fields.
    middle.pop('immediate_predecessor_registration')
    middle_plan = copy.deepcopy(plan)
    middle_plan['execution_plan'].update(attempt_id='middle', protocol_hash=middle['protocol_hash'])
    middle_plan['composed_config_hash'] = middle['composed_config_hash']
    middle_plan_pin = save('docs/middle-plan.json', middle_plan)
    middle_pin = save('docs/middle.json', middle)
    capture('middle', dict(fragments, baseline=b1, experiment=e1))
    declarations = [first_declaration, {'attempt_id': 'middle', 'checkpoint_sha256': 'f'*64}]
    bridges = dict(schema=MULTI_SCHEMA, successor_attempt_id='last',
                   bridges={d['attempt_id']: bridge('last') for d in declarations})
    b2, e2, last = compiler.build_successor_documents(original=middle, plan=middle_plan,
        fragments=dict(fragments, baseline=b1, experiment=e1),
        target=dict(attempt_id='last', fit_run_id='LAST-FIT', fit_attempt_id='last-fit'),
        dispatch=None, width=8, supply={}, declarations=declarations, bridge=bridges,
        registration_relpath=middle_pin['relpath'], registration_sha256=middle_pin['sha256'],
        predecessor_plan=middle_plan_pin, data=tmp_path)
    return dict(repo=tmp_path, root=root, root_pin=root_pin, base=base, middle=middle,
                first=(b1, e1), last=last, final=(b2, e2), sources=sources, save=save)


def check(case, *, first=False):
    base, exp = case['first' if first else 'final']
    return business.validate_checkpoint_successor(case['repo'], case['middle' if first else 'last'],
                                                exp['cells'][0], base, case['base'])


def test_compiler_two_hops_real_business_validator_and_legacy_one_hop(lineage):
    assert check(lineage, first=True) == lineage['base']['native_contact_fit']
    assert check(lineage) == lineage['base']['native_contact_fit']
    assert lineage['last']['predecessor_registration'] == lineage['root_pin']
    assert lineage['last']['immediate_predecessor_registration']['relpath'] == 'docs/middle.json'
    assert lineage['middle']['attempt_id'] == 'middle'


@pytest.mark.parametrize('fault', ['root', 'cycle', 'missing', 'same-run', 'science', 'budget',
                                   'ancestor-budget', 'ancestor-science', 'ancestor-cycle', 'checkpoint-pin',
                                   'plan-budget', 'extra-ancestor', 'missing-immediate-plan', 'fit-run-reuse', 'fit-attempt-reuse'])
def test_repeated_successor_rejects_invalid_lineage(lineage, fault):
    last = lineage['last']; b2, e2 = lineage['final']
    if fault == 'missing-immediate-plan':
        del last['immediate_predecessor_plan']
        with pytest.raises(ValueError, match='immediate predecessor plan pin required'):
            check(lineage)
        return
    elif fault in ('fit-run-reuse', 'fit-attempt-reuse'):
        key = 'run_id' if fault == 'fit-run-reuse' else 'attempt_id'
        b2['native_contact_fit'][key] = lineage['middle']['fit_' + key]
        last['fit_' + key] = b2['native_contact_fit'][key]
        with pytest.raises(ValueError, match='fresh FIT identity repeats a predecessor'):
            check(lineage)
        return
    elif fault == 'root':
        last['predecessor_registration'] = last['immediate_predecessor_registration']
    elif fault == 'cycle':
        from speech_aware_evidence_acquisition.core.registration_lineage import registration_lineage
        pin = {'relpath': 'docs/loop.json', 'sha256': 'a'*64}
        doc = dict(last, predecessor_registration=pin, immediate_predecessor_registration=pin)
        with pytest.raises(ValueError, match='cyclic'):
            registration_lineage(doc, lambda _: doc)
        return
    elif fault == 'missing':
        (lineage['repo'] / 'docs/root.json').unlink()
    elif fault == 'same-run':
        middle = dict(lineage['middle'], run_id='OTHER')
        last['immediate_predecessor_registration'] = lineage['save']('docs/middle.json', middle)
    elif fault == 'science':
        last['scientific_baseline_sha256'] = '0'*64
    elif fault == 'budget':
        b2['native_contact_fit']['contact_cap'] += 1
    elif fault in ('ancestor-budget', 'ancestor-science', 'ancestor-cycle', 'checkpoint-pin'):
        files = lineage['sources']['middle']
        path = STUDY_PREFIX + ('configs/baseline/e.json' if fault == 'ancestor-science' else 'configs/experiment/e.json')
        doc = compiler.strict_json(files[path])
        if fault == 'ancestor-budget': doc['execution_plan']['planned_model_calls'] += 1
        elif fault == 'ancestor-science': doc['arm'] = 'changed'
        elif fault == 'ancestor-cycle': doc['cells'][0]['resume_from'] = {'attempt_id': 'middle', 'checkpoint_sha256': 'f'*64}
        else: doc['cells'][0]['resume_from']['checkpoint_sha256'] = '0'*64
        files[path] = compiler.encode(doc)
    elif fault == 'plan-budget':
        pin = last['immediate_predecessor_plan']
        doc = compiler.strict_json((lineage['repo'] / pin['relpath']).read_bytes())
        doc['execution_plan']['planned_model_calls'] += 1
        last['immediate_predecessor_plan'] = lineage['save'](pin['relpath'], doc)
    else:
        # Dropping the immediate ancestor cannot silently downgrade to one hop.
        e2['cells'][0]['resume_from'].pop()
        del e2['cells'][0]['checkpoint_compatibility']['bridges']['middle']
    with pytest.raises((ValueError, FileNotFoundError)):
        check(lineage)


def test_lineage_catalog_imports_only_required_closure(lineage, tmp_path):
    from speech_aware_evidence_acquisition.core.registration_lineage import registration_lineage
    assets = {}
    catalog = {}
    for name in ('docs/root.json', 'docs/middle.json', 'docs/middle-plan.json',
                 'docs/readiness/e/generation-plan-registered-v1.json'):
        path = lineage['repo'] / name
        catalog[name] = {'path': str(path), 'sha256': compiler.digest(path.read_bytes())}
    stale = tmp_path / 'stale.json'
    stale.write_bytes(b'{"stale": true}')
    catalog['docs/readiness/e/terminal-custody-result.json'] = {
        'path': str(stale), 'sha256': compiler.digest(stale.read_bytes())}
    manifest = tmp_path / 'catalog.json'
    manifest.write_bytes(compiler.encode({'schema': 'saea-experiment-assets-v1', 'files': catalog}))
    cfg = {'lineage_assets_manifest': {'path': str(manifest), 'sha256': compiler.digest(manifest.read_bytes())}}
    compiler.complete_lineage_inputs(cfg, assets, lineage['last'])
    assert set(assets) == set(catalog) - {'docs/readiness/e/terminal-custody-result.json'}
    root_pin, root, docs = registration_lineage(lineage['last'], lambda pin: compiler.strict_json(assets[pin['relpath']]))
    assert root_pin == lineage['root_pin'] and root == lineage['root']
    assert len(docs) == 2
    assets['docs/root.json'] = b'{}'
    with pytest.raises(ValueError, match='conflicting'):
        compiler.complete_lineage_inputs(cfg, assets, lineage['last'])


def test_lineage_missing_closure_fails_without_repository_fallback(lineage):
    with pytest.raises(ValueError, match='lineage asset missing'):
        compiler.complete_lineage_inputs({}, {}, lineage['last'])


def test_registration_lineage_is_bounded():
    from speech_aware_evidence_acquisition.core.registration_lineage import registration_lineage
    current = dict(run_id='RUN', predecessor_registration={'relpath': 'docs/root.json', 'sha256': 'a'*64})
    sequence = iter(range(1000))
    def load(_):
        index = next(sequence)
        return dict(current, immediate_predecessor_registration={
            'relpath': 'docs/' + str(index) + '.json', 'sha256': 'a'*64})
    with pytest.raises(ValueError, match='excessive'):
        registration_lineage(current, load)


@pytest.fixture
def package_chain(lineage, tmp_path, monkeypatch):
    """Actual package prepare/build/finalize and workload body; external gates are fixtures."""
    import configured_terminal_custody_package as package
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core.flight import campaign, factory, native_recovery
    import tfrl_odd40_cells
    data = tmp_path / 'data'; data.mkdir()
    repo = tmp_path / 'repo'; repo.mkdir()
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compiler.encode(value))
        return package.pin(path)
    binding = dict(model_fragment='m', profile='profile', service_session='session', runtime_receipt='receipts/runtime.json')
    job_config = dict(schema=business.SCHEMA, name='e', run_id='RUN', sample_ids=['slurp:train:1'],
                      runtime_binding=binding, execution={'admitted_level': 4},
                      material={'relpath': 'docs/material.json', 'sha256': 'a'*64})
    view = {}; roster = dict(items=[{'sample_id': 'slurp:train:1'}], split_identity_hash='d'*64)
    dataset = lineage['save']('configs/dataset/d.json', view)
    roster_pin = lineage['save']('docs/receipts/rosters/e.json', roster)
    source_pin = lineage['save'](business.observation.SOURCE_EXP, {})
    lock = lineage['save'](business.observation.LOCK, {'status': 'locked_full_train_fold_b'})
    monkeypatch.setattr(business.observation, 'LOCK_SHA', lock['sha256'])
    for doc in (lineage['root'], lineage['middle']):
        doc.update(rows=1, job_config=copy.deepcopy(job_config), material=job_config['material'],
                   dataset=dataset, roster=roster_pin, source_experiment=source_pin,
                   gain_lock=lock, runtime_binding=binding, **{k:binding[k] for k in ('profile','service_session','runtime_receipt')},
                   authority='docs/authority.md', decision_relpath='coordination/old.json')
        doc['job_config']['execution'] = copy.deepcopy(doc['execution'])
    lineage['middle']['predecessor_registration'] = lineage['save']('docs/root.json', lineage['root'])
    middle = lineage['middle']; lineage['save']('docs/middle.json', middle)
    candidate = lineage['save']('docs/readiness/e/candidate.json', middle)
    plan_pin = {'relpath': 'docs/middle-plan.json', 'sha256': compiler.digest((tmp_path/'docs/middle-plan.json').read_bytes())}
    supply = put(data/'supply.json', {'output': {'items': []}})
    supply = dict(relpath='supply.json', sha256=supply['sha256'])
    b1, e1 = copy.deepcopy(lineage['first']); b1['prepared_phase_supply'] = supply
    fragment_docs = dict(model={}, dataset=view, baseline=b1, experiment=e1)
    for kind, doc in fragment_docs.items(): lineage['save']('configs/'+kind+'/'+dict(model='m',dataset='d',baseline='e',experiment='e')[kind]+'.json', doc)
    lineage['save']('docs/authority.md', {})
    lineage['save']('docs/exposure-ledger.md', {'historical_calls': 99})
    paths = [candidate['relpath'], plan_pin['relpath'], roster_pin['relpath'], source_pin['relpath'],lock['relpath'],
             'docs/authority.md','docs/exposure-ledger.md', *['configs/'+kind+'/'+name+'.json' for kind,name in dict(model='m',dataset='d',baseline='e',experiment='e').items()]]
    files = {name: package.pin(tmp_path/name) for name in paths}
    manifest = put(tmp_path/'manifest.json', {'schema':'saea-experiment-assets-v1','files':files})
    catalog_files = {name: package.pin(tmp_path/name) for name in ('docs/root.json','docs/readiness/e/generation-plan-registered-v1.json')}
    catalog = put(tmp_path/'catalog.json', {'schema':'saea-experiment-assets-v1','files':catalog_files})
    moved = {}
    for name, pin in catalog_files.items():
        path = tmp_path/'relocated'/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(Path(pin['path']).read_bytes()); moved[name] = package.pin(path)
    bridge = copy.deepcopy(lineage['final'][1]['cells'][0]['checkpoint_compatibility'])
    evidence = put(data/'evidence.json', {'fixture': True})
    mappings = {}
    for aid, item in bridge['bridges'].items():
        mappings[aid] = {}
        for key, value in item['evidence'].items():
            if isinstance(value,dict): item['evidence'][key] = evidence; mappings[aid][key] = evidence
    cfg = dict(schema='saea-checkpoint-successor-compiler-v1', mode='finalize', repo=str(repo), data=str(data),
               output=str(tmp_path/'compiled'), assets_manifest=manifest, lineage_assets_manifest=catalog,
               candidate=candidate, plan=plan_pin, fragments=dict(model='m',dataset='d',baseline='e',experiment='e'),
               target=dict(attempt_id='last',fit_run_id='LAST-FIT',fit_attempt_id='last-fit'), dispatch=None,width=8,
               supply={'old':supply,'new':supply},terminal=None,
               admission=dict(workspace=str(tmp_path/'workspace'),decision_relpath='coordination/new.json',
                              predecessor_decision=dict(relpath='coordination/old.json',sha256=evidence['sha256'])))
    prepare = dict(schema='saea-terminal-package-prepare-v1', source_manifest=manifest,source_paths=files,
                   bridge=put(tmp_path/'bridge.json',bridge),bridge_paths=mappings,
                   compiler=put(tmp_path/'template.json',cfg), lineage_paths=moved,
                   admission=dict(cfg['admission'],registration_relpath='docs/direct.json'),
                   predecessor_decision_file=evidence,lock=evidence,output=str(tmp_path/'prepared'),package_output=str(tmp_path/'package'))
    terminal = dict(schema='saea-checkpoint-terminal-custody-result-v1',ancestors=[{'fixture_only':True}],activation_authorized=False)
    monkeypatch.setattr(package,'scan',lambda roots: [])
    monkeypatch.setattr(package,'validate_terminal_custody',lambda **kw: terminal)
    monkeypatch.setattr(compiler,'validate_terminal_custody',lambda **kw: terminal)
    monkeypatch.setattr(compiler,'validate_gate_inputs',lambda *a: None)
    monkeypatch.setattr(native_recovery,'native_contact_custody_enabled',lambda *a: True)
    monkeypatch.setattr(tfrl_odd40_cells,'verify_output',lambda *a: None)
    # Keep validate_workload, PairWorkload.validate and validate_checkpoint_successor real.
    monkeypatch.setattr(business.PairWorkload,'derive',lambda *a: (copy.deepcopy(lineage['base']),
        {'cells':[dict(e1['cells'][0],attempt_id='generation-v1')], 'execution_plan':e1['execution_plan']},roster,view,binding))
    monkeypatch.setattr(business.PairWorkload,'DATASET','d')
    calls=[]
    def runtime(*a, **kw):
        calls.append('runtime-after-business')
        return SimpleNamespace(_admitted_level=8,_deployment_profile='profile')
    monkeypatch.setattr(factory,'build_flight_runtime',runtime)
    def build():
        result=package.prepare(prepare)
        # Old catalog assets go offline after relocation; consumers must use new paths.
        for pin in catalog_files.values(): Path(pin['path']).unlink()
        held=compiler.strict_json(Path(result['held']['path']).read_bytes())
        pkg=held['config']
        pkg.update(resume_from=lineage['final'][1]['cells'][0]['resume_from'],evidence=[evidence],
                   ancestors=[dict(attempt_id='middle',plan=files[plan_pin['relpath']],generation_root=str(data),
                                   checkpoint=evidence,prepared_root=str(data))])
        return package.build(pkg)
    return dict(build=build,prepare=prepare,cfg=cfg,package=package,compiler=compiler,moved=moved,
                catalog=catalog,tmp=tmp_path,calls=calls,put=put)


def test_package_prepare_build_two_hop_finalize_real_business_body(package_chain):
    case=package_chain
    result=case['build']()
    cfg=compiler.strict_json(Path(result['compiler']['path']).read_bytes())
    assert cfg['lineage_assets_manifest']['path'].endswith('/prepared/lineage-assets.json')
    assert compiler.execute(result['compiler']['path'])['status']=='HELD'
    assert case['calls']==['runtime-after-business']
    out=case['tmp']/'compiled'
    candidate=compiler.strict_json((out/'docs/readiness/e/candidate.json').read_bytes())
    assert candidate['predecessor_registration']['relpath']=='docs/root.json'
    assert candidate['immediate_predecessor_registration']['relpath']=='docs/direct.json'
    assert (out/'docs/exposure-ledger.md').read_bytes()==(case['tmp']/'docs/exposure-ledger.md').read_bytes()
    assert compiler.strict_json((out/'pin-index.json').read_bytes())['source_lineage_manifest']==cfg['lineage_assets_manifest']
    provenance=compiler.strict_json((case['tmp']/'prepared/lineage-relocation.json').read_bytes())
    assert provenance['source_manifest']==case['catalog']
    assert provenance['paths']==case['moved']


@pytest.mark.parametrize('fault',['manifest-tamper','manifest-missing','asset-tamper','asset-missing','mapping-missing','repinned-asset','bridge-missing'])
def test_package_prepare_catalog_rejection(package_chain,fault):
    case=package_chain
    if fault.startswith('manifest'):
        path=Path(case['catalog']['path'])
    else:
        path=Path(next(iter(case['moved'].values()))['path'])
    if fault=='mapping-missing': case['prepare']['lineage_paths'].pop(next(iter(case['moved'])))
    elif fault=='bridge-missing': case['prepare']['bridge_paths'].pop('middle')
    elif fault=='repinned-asset':
        path.write_bytes(b'changed')
        case['prepare']['lineage_paths'][next(iter(case['moved']))] = case['package'].pin(path)
    elif fault.endswith('missing'): path.unlink()
    else: path.write_bytes(b'changed')
    with pytest.raises((ValueError,OSError)): case['package'].prepare(case['prepare'])
    assert not (case['tmp']/'prepared').exists()


@pytest.mark.parametrize('phase',['build','after-audit','compiler-prepare','compiler-finalize'])
def test_catalog_revalidated_at_consumption(package_chain,monkeypatch,phase):
    case=package_chain
    if phase=='after-audit':
        original = case['package'].validate_terminal_custody
        def changed_after_audit(**kwargs):
            result = original(**kwargs)
            Path(next(iter(case['moved'].values()))['path']).write_bytes(b'changed')
            return result
        monkeypatch.setattr(case['package'],'validate_terminal_custody',changed_after_audit)
        with pytest.raises(ValueError,match='hash'): case['build']()
        assert not (case['tmp']/'package').exists()
    elif phase=='build':
        real=case['package'].validate_compiler_template
        def changed(cfg):
            Path(cfg['lineage_assets_manifest']['path']).write_bytes(b'changed')
            return real(cfg)
        monkeypatch.setattr(case['package'],'validate_compiler_template',changed)
        with pytest.raises(ValueError,match='hash'): case['build']()
        assert not (case['tmp']/'package').exists()
    else:
        result=case['build'](); cfg=compiler.strict_json(Path(result['compiler']['path']).read_bytes())
        if phase=='compiler-prepare':
            cfg['mode']='prepare'; del cfg['terminal']; del cfg['admission']
        Path(next(iter(case['moved'].values()))['path']).unlink()
        path=case['tmp']/'consume.json'; path.write_bytes(compiler.encode(cfg))
        with pytest.raises(ValueError,match='missing'): compiler.execute(path)
        assert not (case['tmp']/'compiled').exists()


def test_package_optional_catalog_without_relocation(package_chain):
    case = package_chain
    del case['prepare']['lineage_paths']
    result = case['package'].prepare(case['prepare'])
    cfg = compiler.strict_json((Path(result['output'])/'compiler-template.json').read_bytes())
    case['package'].validate_compiler_template(cfg)
    assert compiler.strict_json(Path(cfg['lineage_assets_manifest']['path']).read_bytes()) == compiler.strict_json(Path(case['catalog']['path']).read_bytes())


def test_compiler_prepare_preserves_relocated_closure(package_chain):
    case = package_chain
    result = case['build']()
    cfg = compiler.strict_json(Path(result['compiler']['path']).read_bytes())
    cfg['mode'] = 'prepare'
    del cfg['terminal']; del cfg['admission']
    path = case['tmp']/'compiler-prepare.json'; path.write_bytes(compiler.encode(cfg))
    assert compiler.execute(path)['status'] == 'HELD_STATIC_ONLY'
    output = case['tmp']/'compiled'
    for name, pin in case['moved'].items():
        assert (output/name).read_bytes() == Path(pin['path']).read_bytes()
    index = compiler.strict_json((output/'pin-index.json').read_bytes())
    assert index['source_lineage_manifest'] == cfg['lineage_assets_manifest']
