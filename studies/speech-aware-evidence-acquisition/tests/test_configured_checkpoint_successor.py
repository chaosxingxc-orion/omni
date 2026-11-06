import importlib.util
import json
import hashlib
from pathlib import Path
import pytest
import sys
import types
import os

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/environment/configured_checkpoint_successor.py'
spec = importlib.util.spec_from_file_location('checkpoint_successor_compiler', SCRIPT)
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(blob)
    return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest()}


@pytest.fixture
def case(tmp_path):
    repo, data = tmp_path/'repo', tmp_path/'data'
    repo.mkdir(); data.mkdir()
    supply = write(data/'supply.json', {'output': {'items': [1, 2]}})
    supply_pin = {'relpath': 'supply.json', 'sha256': supply['sha256']}
    assets = {}
    def asset(name, value):
        assets[name] = write(tmp_path/'assets'/name, value)
        return {'relpath': name, 'sha256': assets[name]['sha256']}
    candidate = {'experiment':'exp', 'baseline':'base', 'model_fragment':'model',
                 'dataset':{'relpath':'configs/dataset/data.json'}, 'run_id':'RUN',
                 'attempt_id':'generation-v1', 'fit_run_id':'FIT', 'fit_attempt_id':'fit-v1',
                 'protocol_hash':'protocol', 'composed_config_hash':'config'}
    cp = asset('docs/readiness/exp/candidate.json', candidate)
    plan = asset('docs/readiness/generation-plan-registered-v1.json', {
        'execution_plan':{'run_id':'RUN', 'attempt_id':'generation-v1', 'protocol_hash':'protocol'},
        'composed_config_hash':'config'})
    for kind, name in [('model','model'),('dataset','data'),('experiment','exp')]:
        asset('configs/'+kind+'/'+name+'.json', {})
    asset('configs/baseline/base.json', {'prepared_phase_supply':supply_pin})
    manifest = write(tmp_path/'manifest.json', {'schema':'saea-experiment-assets-v1','files':assets})
    config = {'schema':'saea-checkpoint-successor-compiler-v1', 'mode':'prepare',
        'repo':str(repo), 'data':str(data), 'output':str(tmp_path/'output'),
        'assets_manifest':manifest, 'candidate':cp, 'plan':plan,
        'fragments':{'model':'model','dataset':'data','baseline':'base','experiment':'exp'},
        'target':{'attempt_id':'generation-v2','fit_run_id':'FIT2','fit_attempt_id':'fit-v2'},
        'width':32, 'dispatch':None, 'supply':{'old':supply_pin,'new':supply_pin}}
    config_path = tmp_path/'config.json'
    def run():
        config_path.write_text(json.dumps(config))
        return compiler.execute(config_path)
    return config, run, tmp_path


def test_prepare_atomic_held_and_preserves_original(case):
    cfg, run, tmp = case
    result = run()
    assert result['status'] == 'HELD_STATIC_ONLY'
    out = tmp/'output'
    assert json.loads((out/'held-template.json').read_bytes())['activation_authorized'] is False
    assert (out/cfg['plan']['relpath']).read_bytes() == (tmp/'assets'/cfg['plan']['relpath']).read_bytes()
    pins = json.loads((out/'pin-index.json').read_bytes())['assets']
    assert set(pins) == set(json.loads((tmp/'manifest.json').read_bytes())['files'])
    for name, pin in pins.items():
        assert Path(pin['path']) == out/name
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == pin['sha256']
    with pytest.raises(ValueError, match='fresh output'): run()


@pytest.mark.parametrize('fail', [False, True])
def test_compiler_hash_uses_declared_root_and_restores_environment(case, monkeypatch, fail):
    cfg, run, _ = case
    monkeypatch.setenv('SAEA_DATA_ROOT', '/unrelated/root')
    monkeypatch.delenv('SPEECHRL_DATA_DIR', raising=False)
    from speech_aware_evidence_acquisition.core.config import _compose_fragments
    fragments = {'model': {'model_path': cfg['data'] + '/models/frozen'},
                 'dataset': {}, 'baseline': {}, 'experiment': {}}
    names = {key: key for key in fragments}
    def execute(document):
        assert document == cfg
        assert os.environ['SAEA_DATA_ROOT'] == cfg['data']
        assert os.environ['SPEECHRL_DATA_DIR'] == cfg['data']
        actual = _compose_fragments(names, fragments, cfg['data']).config_hash
        canonical = dict(fragments, model={'model_path': '${SAEA_DATA_ROOT}/models/frozen'})
        assert actual == _compose_fragments(names, canonical, cfg['data']).config_hash
        if fail:
            raise ValueError('validation failed')
        return actual
    monkeypatch.setattr(compiler, '_execute', execute)
    if fail:
        with pytest.raises(ValueError, match='validation failed'):
            run()
    else:
        assert len(run()) == 64
    assert os.environ['SAEA_DATA_ROOT'] == '/unrelated/root'
    assert 'SPEECHRL_DATA_DIR' not in os.environ


@pytest.mark.parametrize('mutator', [
    lambda c: c.update(extra=True),
    lambda c: c['target'].update(attempt_id='generation-v1'),
    lambda c: c['candidate'].update(sha256='0'*64),
    lambda c: c.update(output=c['data']+'/unsafe'),
    lambda c: c.update(mode='finalize'),
])
def test_refuses_invalid_or_unsupported_without_output(case, mutator):
    cfg, run, tmp = case
    mutator(cfg)
    with pytest.raises(ValueError): run()
    assert not (tmp/'output').exists()


@pytest.mark.parametrize('blob', [b'{"mode":"prepare","mode":"finalize"}', b'{"x":NaN}'])
def test_strict_json_refuses_ambiguous_input(blob):
    with pytest.raises(ValueError):
        compiler.strict_json(blob)


def finalize_case(case):
    cfg, run, tmp = case
    evidence = write(tmp/'evidence.json', {'test': True})
    cfg['mode'] = 'finalize'
    manifest = json.loads(Path(cfg['assets_manifest']['path']).read_bytes())
    candidate_path = Path(manifest['files'][cfg['candidate']['relpath']]['path'])
    candidate = json.loads(candidate_path.read_bytes())
    candidate['authority'] = 'docs/decisions/authority.md'
    candidate_pin = write(candidate_path, candidate)
    manifest['files'][cfg['candidate']['relpath']] = candidate_pin
    cfg['candidate']['sha256'] = candidate_pin['sha256']
    authority = tmp/'assets/docs/decisions/authority.md'
    authority.parent.mkdir(parents=True, exist_ok=True)
    authority.write_text('Original authorization\n')
    manifest['files'][candidate['authority']] = {
        'path': str(authority), 'sha256': hashlib.sha256(authority.read_bytes()).hexdigest()}
    ledger = tmp/'assets/docs/exposure-ledger.md'
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('Frozen predecessor exposure rows\n')
    manifest['files']['docs/exposure-ledger.md'] = {
        'path': str(ledger), 'sha256': hashlib.sha256(ledger.read_bytes()).hexdigest()}
    from speech_aware_evidence_acquisition.contracts import FrozenCoreGate,E0_RECEIPT_SCHEMA,STUDY_SLUG
    deliverables={}
    for name in FrozenCoreGate.E0_DELIVERABLES:
        relative='docs/receipts/artifacts/'+name+'.json'
        pin=write(tmp/'assets'/relative,{'fixture':name});manifest['files'][relative]=pin
        deliverables[name]={'artifact':'artifacts/'+name+'.json','sha256':pin['sha256'],'status':'CLOSED'}
    manifest['files']['docs/receipts/e0-closure.json']=write(tmp/'assets/docs/receipts/e0-closure.json',
        {'schema':E0_RECEIPT_SCHEMA,'study':STUDY_SLUG,'deliverables':deliverables})
    manifest['files']['docs/receipts/runtime.json']=write(tmp/'assets/docs/receipts/runtime.json',{'fixture':True})
    cfg['assets_manifest'] = write(Path(cfg['assets_manifest']['path']), manifest)
    cfg['terminal'] = {'evidence': [evidence], 'lock': evidence, 'resume_from': {},
                       'bridge': {}, 'registration_relpath': 'docs/readiness/previous.json'}
    cfg['admission'] = {'workspace': str(tmp/'workspace'),
                        'decision_relpath': 'coordination/exp/new.json',
                        'predecessor_decision': {'relpath': 'coordination/exp/old.json', 'sha256': 'f'*64}}
    return cfg, run, tmp


def test_missing_exposure_ledger_fails_before_terminal_audit(case, monkeypatch):
    cfg, run, tmp = finalize_case(case)
    manifest = json.loads(Path(cfg['assets_manifest']['path']).read_bytes())
    del manifest['files']['docs/exposure-ledger.md']
    cfg['assets_manifest'] = write(Path(cfg['assets_manifest']['path']), manifest)
    def unexpected(**kwargs):
        pytest.fail('terminal audit must not run without complete admission inputs')
    monkeypatch.setattr(compiler, 'validate_terminal_custody', unexpected)
    with pytest.raises(ValueError, match='pinned predecessor exposure ledger'):
        run()
    assert not (tmp/'output').exists()


def test_preparation_sources_require_complete_pinned_asset_closure(tmp_path):
    blob = b'frozen source receipt'
    data = tmp_path/'data'
    data.mkdir()
    (data/'source.json').write_bytes(blob)
    pins = [{'root': root, 'relpath': rel, 'sha256': compiler.digest(blob)}
            for root, rel in [('repository', 'docs/source.json'), ('data', 'source.json')]]
    assets = {'configs/baseline/base.json': compiler.encode({'preparation_source_receipts': pins}),
              'docs/source.json': blob}
    assert compiler.validate_preparation_inputs(assets, {'baseline': 'base'}, data) == 2
    del assets['docs/source.json']
    with pytest.raises(ValueError, match='missing from experiment assets'):
        compiler.validate_preparation_inputs(assets, {'baseline': 'base'}, data)
    assets['docs/source.json'] = b'changed'
    with pytest.raises(ValueError, match='hash differs'):
        compiler.validate_preparation_inputs(assets, {'baseline': 'base'}, data)
    assets['docs/source.json'] = blob
    (data/'source.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        compiler.validate_preparation_inputs(assets, {'baseline': 'base'}, data)


def test_preparation_source_cannot_escape_data_root(tmp_path):
    data = tmp_path/'data'
    data.mkdir()
    external = tmp_path/'external.json'
    external.write_bytes(b'secret')
    (data/'source.json').symlink_to(external)
    pins = [{'root': 'data', 'relpath': 'source.json', 'sha256': compiler.digest(b'secret')}]
    assets = {'configs/baseline/base.json': compiler.encode({'preparation_source_receipts': pins})}
    with pytest.raises(Exception, match='escapes|outside|confined'):
        compiler.validate_preparation_inputs(assets, {'baseline': 'base'}, data)


def test_terminal_rejection_propagates(case, monkeypatch):
    cfg, run, tmp = finalize_case(case)
    def reject(**kwargs):
        raise ValueError('unresolved custody')
    monkeypatch.setattr(compiler, 'validate_terminal_custody', reject)
    with pytest.raises(ValueError, match='unresolved custody'):
        run()
    assert not (tmp/'output').exists()


def test_terminal_result_required(case, monkeypatch):
    cfg, run, tmp = finalize_case(case)
    monkeypatch.setattr(compiler, 'validate_terminal_custody', lambda **kw: None)
    with pytest.raises(ValueError, match='no durable result'):
        run()
    assert not (tmp/'output').exists()


def test_finalize_publication_and_failure_cleanup(case, monkeypatch):
    cfg, run, tmp = finalize_case(case)
    terminal = {'schema': 'saea-checkpoint-terminal-custody-result-v1',
                'ancestors': [{'attempt_id': 'generation-v1'}], 'activation_authorized': False}
    monkeypatch.setattr(compiler, 'validate_terminal_custody', lambda **kw: terminal)
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from dataclasses import asdict
    generation = ExecutionPlan(run_id='RUN', attempt_id='generation-v2',
        execution_profile='bounded-discovery-probe', carrier_lock_key='slurp', split_role='discovery',
        split_identity_hash='a'*64, planned_model_calls=6, planned_gpu_hours=1,
        planned_speech_audio_seconds=12, protocol_hash='b'*64)
    generation_doc = {'execution_plan': asdict(generation), 'protocol_document': {}}
    fit_doc = {'execution_plan': asdict(generation), 'protocol_document': {}}
    monkeypatch.setattr(compiler, '_successor_plan_documents', lambda **kw:
                        (generation_doc, fit_doc, {'schema': 'fixture'}))
    def derive(**kwargs):
        return ({'new': 'baseline'}, {'new': 'experiment'},
                {**kwargs['original'], 'attempt_id': 'generation-v2', 'profile': 'profile',
                 'cell': 'exp', 'decision_relpath': 'coordination/exp/old.json',
                 'split_identity_hash': 'a'*64, 'predecessor_registration': {'relpath': 'docs/readiness/previous.json', 'sha256': 'a'*64}})
    monkeypatch.setattr(compiler, 'build_successor_documents', derive)
    calls = []
    module = types.ModuleType('registered_workload')
    def validate(repo, data, name, spec):
        from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
        assert json.loads(resolve_asset(repo, cfg['plan']['relpath']).read_bytes())['execution_plan']['attempt_id'] == 'generation-v1'
        assert json.loads(resolve_asset(repo, cfg['candidate']['relpath']).read_bytes())['attempt_id'] == 'generation-v2'
        calls.append('validate')
        return 1
    module.validate_workload = validate
    module.assert_execution_binding = lambda *a, **kw: calls.append('binding')
    monkeypatch.setitem(sys.modules, 'registered_workload', module)
    from speech_aware_evidence_acquisition.core.flight import campaign, factory
    cell = types.SimpleNamespace(run_id='RUN', attempt_id='generation-v2', resume_from={}, checkpoint_compatibility={})
    monkeypatch.setattr(campaign, 'load_campaign_cells', lambda *a: [cell])
    def fail(*a, **kw):
        raise ValueError('runtime refused')
    monkeypatch.setattr(factory, 'build_flight_runtime', fail)
    with pytest.raises(ValueError, match='runtime refused'):
        run()
    assert not (tmp/'output').exists()
    assert not list(tmp.glob('.output.*'))
    def runtime(*a, **kw):
        assert os.environ['SAEA_ADMITTED_LEVEL'] == '32'
        assert os.environ['SAEA_DEPLOYMENT_PROFILE'] == 'profile'
        calls.append('runtime')
        return types.SimpleNamespace(_admitted_level=32, _deployment_profile='profile',
            _composed=types.SimpleNamespace(values={}), _build_execution_plan=lambda *a: generation)
    monkeypatch.setattr(factory, 'build_flight_runtime', runtime)
    assert run()['status'] == 'HELD'
    manifest = json.loads((tmp/'output/assets.json').read_bytes())
    assert all(Path(pin['path']).is_file() for pin in manifest['files'].values())
    assert (tmp/'output'/cfg['plan']['relpath']).read_bytes() == (tmp/'assets'/cfg['plan']['relpath']).read_bytes()
    assert calls == ['validate', 'binding', 'validate', 'binding', 'runtime']
    index = json.loads((tmp/'output/pin-index.json').read_bytes())
    assert json.loads(Path(index['terminal_custody']['path']).read_bytes()) == terminal
    admission = json.loads(Path(index['admission_config']['path']).read_bytes())
    assert all(set(pin) == {'relpath', 'sha256'} for pin in admission['plans'].values())


def test_derive_real_campaign_and_identity(tmp_path):
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import SCHEMA
    fragments = {'model': {'model': 'fixed'}, 'dataset': {'dataset': 'fixed'},
        'baseline': {'arm': 'test', 'native_contact_fit': {'run_id': 'fit-old', 'attempt_id': 'fit-v1'},
                     'prepared_phase_supply': {'relpath': 'supply', 'sha256': 'a'*64}},
        'experiment': {'cells': [{'cell_id': 'exp', 'model_fragment': 'model', 'dataset_fragment': 'data',
            'baseline_fragment': 'base', 'run_id': 'run', 'attempt_id': 'old', 'readiness': {'state':'ready'}}]}}
    original = {'run_id':'run', 'attempt_id':'old', 'fit_run_id':'fit-old', 'fit_attempt_id':'fit-v1',
        'protocol_hash':'old-protocol', 'composed_config_hash':'old-config', 'model_fragment':'model',
        'dataset':{'relpath':'configs/dataset/data.json'}, 'baseline':'base', 'experiment':'exp',
        'split_identity_hash':'b'*64, 'execution':{'admitted_level':32},
        'job_config':{'execution':{'admitted_level':32}}}
    plan = {'execution_plan':{'run_id':'run', 'attempt_id':'old', 'protocol_hash':'old-protocol'},
            'composed_config_hash':'old-config'}
    bridge = {'schema': SCHEMA, 'predecessor_commit':'c'*40,
        'predecessor_fragments':{'model':'model','dataset':'data','baseline':'base','experiment':'exp'},
        'successor_attempt_id':'new', 'change':'contact-executor-width-v1', 'reviewed_source_changes':{}}
    args = dict(original=original, plan=plan, fragments=fragments,
        target={'attempt_id':'new','fit_run_id':'fit-new','fit_attempt_id':'fit-v2'}, dispatch=None, width=32,
        supply={'relpath':'new-supply','sha256':'d'*64}, declarations={'attempt_id':'old','checkpoint_sha256':'e'*64},
        bridge=bridge, registration_relpath='docs/readiness/old.json', registration_sha256='f'*64, data=tmp_path)
    base, exp, successor = compiler.build_successor_documents(**args)
    assert exp['cells'][0]['readiness']['state'] == 'held'
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignCell
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import CheckpointCompatibility
    CampaignCell.from_document(exp['cells'][0])
    CheckpointCompatibility.from_document(bridge, attempt_id='new')
    assert base['native_contact_fit']['run_id'] == 'fit-new'
    assert successor['composed_config_hash'] != original['composed_config_hash']
    assert fragments['baseline']['native_contact_fit']['run_id'] == 'fit-old'
    args['target']['fit_run_id'] = 'fit-old'
    with pytest.raises(ValueError, match='identity reused'):
        compiler.build_successor_documents(**args)


@pytest.mark.parametrize('fault',['closure','artifact','runtime','tamper'])
def test_gate_input_closure_refuses_before_terminal_audit(case,monkeypatch,fault):
    cfg,run,tmp=finalize_case(case)
    manifest=json.loads(Path(cfg['assets_manifest']['path']).read_bytes())
    name={'closure':'docs/receipts/e0-closure.json','artifact':'docs/receipts/artifacts/d1_identity.json',
          'runtime':'docs/receipts/runtime.json','tamper':'docs/receipts/artifacts/d1_identity.json'}[fault]
    if fault=='tamper':manifest['files'][name]=write(tmp/'changed.json',{'wrong':True})
    else:del manifest['files'][name]
    cfg['assets_manifest']=write(Path(cfg['assets_manifest']['path']),manifest)
    monkeypatch.setattr(compiler,'validate_terminal_custody',lambda **kw:pytest.fail('late terminal audit reached'))
    with pytest.raises(ValueError,match='E0|runtime'):run()
    assert not (tmp/'output').exists()
