"""Successor plan gates run before admission publishes a receipt."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
from admit_real_engine_candidate import validate_successor_plans


def document(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return {'relpath': path.relative_to(path.parents[2]).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(repo):
    candidate = {'cell':'c', 'experiment':'c', 'run_id':'RUN', 'attempt_id':'next',
                 'fit_run_id':'RUN-FIT', 'fit_attempt_id':'fit-next',
                 'composed_config_hash':'a'*64, 'protocol_hash':'b'*64,
                 'split_identity_hash':'c'*64, 'service_session':'session'}
    execution = {'execution_profile':'bounded-discovery-probe', 'split_role':'discovery',
                 'planned_model_calls':8, 'planned_gpu_hours':1, 'planned_speech_audio_seconds':10}
    experiment = {'cells':[{'baseline_fragment':'c'}], 'execution_plan':execution}
    baseline = repo/'configs/baseline/c.json'
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({'native_contact_fit':{'run_id':'RUN-FIT','attempt_id':'fit-next','contact_cap':4}}))
    root = repo/'docs/readiness'
    protocol = document(root/'protocol-next.json', {'schema':'fixture'})
    plans = {}
    for kind, rid, aid, calls in [('generation','RUN','next',8),('fit','RUN-FIT','fit-next',4)]:
        plan = {'schema':f'saea-registered-{kind}-plan-v1','plan_kind':kind,'cell':'c','experiment':'c',
                'composed_config_hash':'a'*64,'expected_session_id':'session','protocol_document':protocol,
                'execution_plan':{**execution, 'planned_model_calls':calls,'run_id':rid,'attempt_id':aid,
                  'carrier_lock_key':'slurp','split_identity_hash':'c'*64,'protocol_hash':'b'*64}}
        plans[kind] = document(root/f'{kind}-next.json',plan)
    return candidate,experiment,{**plans,'protocol':protocol}


def test_pinned_successor_plans_and_tamper(tmp_path):
    spec, exp, pins = fixture(tmp_path)
    docs = validate_successor_plans(tmp_path, spec, exp, pins)
    assert docs['generation']['execution_plan']['attempt_id'] == 'next'
    path = tmp_path/pins['fit']['relpath']
    doc = json.loads(path.read_text());doc['execution_plan']['planned_model_calls'] = 5
    path.write_text(json.dumps(doc));pins['fit']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='calls differ'):
        validate_successor_plans(tmp_path, spec, exp, pins)


def test_fresh_original_plan_is_explicit_and_budget_checked(tmp_path):
    spec, exp, pins = fixture(tmp_path)
    for kind in ('generation', 'fit', 'protocol'):
        old=tmp_path/pins[kind]['relpath']
        new=tmp_path/'docs/readiness/c'/f"{kind}-{'plan-' if kind != 'protocol' else ''}registered-v1.json"
        new.parent.mkdir(parents=True, exist_ok=True);old.rename(new)
        pins[kind]['relpath']=new.relative_to(tmp_path).as_posix()
    for kind in ('generation', 'fit'):
        path=tmp_path/pins[kind]['relpath'];doc=json.loads(path.read_bytes());doc['protocol_document']=pins['protocol']
        path.write_text(json.dumps(doc));pins[kind]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    new=tmp_path/pins['generation']['relpath']
    with pytest.raises(ValueError,match='distinct new paths'):
        validate_successor_plans(tmp_path,spec,exp,pins)
    assert validate_successor_plans(tmp_path,spec,exp,pins,fresh=True)['generation']['execution_plan']['planned_model_calls']==8
    plan=json.loads(new.read_bytes());plan['execution_plan']['planned_model_calls']=9;new.write_text(json.dumps(plan))
    pins['generation']['sha256']=hashlib.sha256(new.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='calls differ'):
        validate_successor_plans(tmp_path,spec,exp,pins,fresh=True)


def test_fresh_ledger_rewrites_only_new_plan_pins():
    from admit_real_engine_candidate import fresh_ledger
    keys=['date','run/experiment id','execution profile','scope','model/tool','speech carrier + split',
          'split role','split identity hash','consumed','calls','speech audio seconds','gpu hours',
          'inherited exposure','artifacts','notes']
    def row(run):
        return '| '+' | '.join(['2026-01-01',run,'bounded-discovery-probe','fixture','model','slurp','discovery',
                                'c'*64,'no','8','10','1','historical','docs/p.json','Plan SHA256 '+'a'*64])+' |\n'
    history=row('HISTORY');text='| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+history+row('RUN')+row('FIT')
    original={k:{'relpath':'docs/p.json','sha256':'a'*64} for k in ('generation','fit')}
    result=fresh_ledger(text,'RUN','FIT',{'generation':'b'*64,'fit':'d'*64},original)
    assert history in result and 'Plan SHA256 '+'b'*64 in result and 'Plan SHA256 '+'d'*64 in result
    with pytest.raises(ValueError,match='unconsumed'):
        fresh_ledger(text.replace('| RUN |','| MISSING |'),'RUN','FIT',{'generation':'b'*64,'fit':'d'*64},original)
    with pytest.raises(ValueError,match='registration pin differs'):
        fresh_ledger(text.replace('docs/p.json','docs/other.json'),'RUN','FIT',{'generation':'b'*64,'fit':'d'*64},original)


def test_reject_original_generation_path_and_mutated_pin(tmp_path):
    spec, exp, pins = fixture(tmp_path)
    bad = copy.deepcopy(pins);bad['generation']['relpath']='docs/readiness/generation-plan-registered-v1.json'
    with pytest.raises(ValueError, match='distinct new paths'):
        validate_successor_plans(tmp_path, spec, exp, bad)
    bad = copy.deepcopy(pins);bad['protocol']['sha256']='0'*64
    with pytest.raises(ValueError, match='pin differs'):
        validate_successor_plans(tmp_path, spec, exp, bad)


def test_predecessor_receipt_identity_and_new_destination(tmp_path):
    from admit_real_engine_candidate import validate_predecessor_decision
    original = {'decision_relpath':'coordination/old.json','decision_id':'did','authority':'docs/authority.md',
                'run_id':'RUN','fit_run_id':'FIT','attempt_id':'generation-v1','fit_attempt_id':'fit-v1'}
    spec = copy.deepcopy(original)
    old = tmp_path/'coordination/old.json';old.parent.mkdir()
    old.write_text(json.dumps({'schema':'saea-campaign-decision-v1','status':'decided','decision_id':'did',
        'authorization_basis':'docs/authority.md','eligible_run_ids':['RUN','FIT'],
        'eligible_attempt_ids':['generation-v1','fit-v1'], 'budget':{'generation':{},'fit':{}}}))
    pin = {'relpath':'coordination/old.json','sha256':hashlib.sha256(old.read_bytes()).hexdigest()}
    assert validate_predecessor_decision(tmp_path,spec,original,pin,'coordination/next.json')['status'] == 'decided'
    with pytest.raises(ValueError):
        validate_predecessor_decision(tmp_path,spec,original,pin,'coordination/old.json')
    spec['decision_id']='changed'
    with pytest.raises(ValueError, match='does not authorize'):
        validate_predecessor_decision(tmp_path,spec,original,pin,'coordination/next.json')


def test_successor_ledger_keeps_old_rows_and_registers_fresh_fit(tmp_path):
    from admit_real_engine_candidate import successor_ledger
    from speech_aware_evidence_acquisition.contracts.ledger import _parse_ledger_tables
    from speech_aware_evidence_acquisition.contracts import FrozenCoreGate, ExecutionPlan
    keys=['date','run/experiment id','execution profile','scope','model/tool','speech carrier + split',
          'split role','split identity hash','consumed','calls','speech audio seconds','gpu hours',
          'inherited exposure','artifacts','notes']
    def row(run,calls):
        values=['2026-01-01',run,'bounded-discovery-probe','fixture','model','slurp train','discovery',
                'c'*64,'no',str(calls),'10','1','historical','docs/old.json','Plan SHA256 '+'a'*64]
        return '| '+' | '.join(values)+' |\n'
    original='| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+row('RUN',8)+row('OLD-FIT',4)
    result=successor_ledger(original,'RUN','OLD-FIT','NEW-FIT','docs/fit.json','b'*64)
    assert result.startswith(original)
    rows=_parse_ledger_tables(result)
    assert [r['run/experiment id'] for r in rows] == ['RUN','OLD-FIT','NEW-FIT']
    ledger=tmp_path/'ledger.md';ledger.write_text(result)
    gate=FrozenCoreGate(e0_receipt=tmp_path/'e0',runtime_receipt=tmp_path/'runtime',data_root=tmp_path,
                        lock_path=tmp_path/'lock',runtime_root=tmp_path,exposure_ledger=ledger,source_root=tmp_path)
    lock={'datasets':[{'kind':'dataset','name':'slurp'}]}
    for run,calls in [('RUN',8),('NEW-FIT',4)]:
        plan=ExecutionPlan(run_id=run,attempt_id='next',execution_profile='bounded-discovery-probe',
            carrier_lock_key='slurp',split_role='discovery',split_identity_hash='c'*64,
            planned_model_calls=calls,planned_gpu_hours=1,planned_speech_audio_seconds=10,protocol_hash='d'*64)
        gate._require_exposure_preregistration(plan,lock)
    with pytest.raises(ValueError,match='already registered'):
        successor_ledger(result,'RUN','OLD-FIT','NEW-FIT','docs/fit.json','b'*64)


def test_staged_receipt_readiness_is_hash_bound_without_publication(tmp_path):
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignCell, declared_readiness
    receipt=json.dumps({'schema':'saea-campaign-decision-v1','status':'decided','decision_id':'d','eligible_run_ids':['RUN']}).encode()
    rel='coordination/next.json'
    cell=CampaignCell.from_document({'cell_id':'c','model_fragment':'m','dataset_fragment':'d','baseline_fragment':'b',
        'run_id':'RUN','attempt_id':'next','readiness':{'state':'ready','requirements':[{'kind':'decision','relpath':rel,
        'sha256':hashlib.sha256(receipt).hexdigest(),'expected_decision_id':'d'}]}})
    assert declared_readiness(cell,tmp_path).state=='held'
    assert declared_readiness(cell,tmp_path,decision_documents={rel:receipt}).state=='ready'
    assert declared_readiness(cell,tmp_path,decision_documents={rel:receipt+b' '}).state=='held'
    assert not (tmp_path/rel).exists()
    doc=dict(cell.document);doc['readiness']={'state':'ready','requirements':[{'kind':'preparation','relpath':rel,'sha256':hashlib.sha256(receipt).hexdigest()}]}
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignError
    with pytest.raises(CampaignError,match='only satisfy a decision'):
        declared_readiness(CampaignCell.from_document(doc),tmp_path,decision_documents={rel:receipt})


def test_invalid_external_config_fails_before_any_publication(tmp_path):
    from admit_real_engine_candidate import admit_external
    repo=tmp_path/'repo'; repo.mkdir()
    data=tmp_path/'data'; data.mkdir()
    with pytest.raises(ValueError, match='exact external admission'):
        admit_external(repo,data,{'schema':'wrong'})
    assert list(data.iterdir()) == []
    assert not (tmp_path/'workspace').exists()


def test_external_entry_restores_workspace_environment_on_failure(monkeypatch, tmp_path):
    import os
    import admit_real_engine_candidate as admission
    keys = ('SAEA_EXPERIMENT_WORKSPACE', 'SAEA_EXPERIMENT_WORKSPACE_MANIFEST',
            'SAEA_EXPERIMENT_WORKSPACE_SHA256')
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(keys[1], 'previous-value')

    def fail(*args):
        for key in keys:
            os.environ[key] = 'temporary-value'
        raise RuntimeError('injected failure')

    monkeypatch.setattr(admission, '_admit_external', fail)
    with pytest.raises(RuntimeError, match='injected failure'):
        admission.admit_external(tmp_path, tmp_path, {})
    assert os.environ.get(keys[0]) is None
    assert os.environ.get(keys[1]) == 'previous-value'
    assert os.environ.get(keys[2]) is None


@pytest.mark.parametrize('fault', ['receipt', 'marker'])
def test_interrupted_publication_resumes_exact_bytes(tmp_path, monkeypatch, fault):
    import admit_real_engine_candidate as admission
    from speech_aware_evidence_acquisition.core import atomic_file
    workspace, data = tmp_path/'workspace', tmp_path/'data'
    manifest = workspace/'manifests/inventory.json'; manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'schema':'saea-experiment-workspace-v1','files':{}}))
    receipt = b'{"fixture": true}\n'
    completion = {'schema':'saea-external-admission-completion-v1', 'candidate':'fixture',
                  'receipt':{'relpath':'coordination/new.json', 'sha256':hashlib.sha256(receipt).hexdigest()},
                  'workspace_manifest':{'path':str(manifest), 'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}, 'binding':{}}
    original = atomic_file.publish_new
    def interrupted(path, raw):
        if Path(path).name == ('new.json' if fault == 'receipt' else 'completion.json'):
            raise OSError('injected interruption')
        return original(path, raw)
    with monkeypatch.context() as patch:
        patch.setattr(atomic_file, 'publish_new', interrupted)
        with pytest.raises(OSError, match='injected interruption'):
            admission.publish_external_completion(workspace, data, completion, receipt)
    assert not (workspace/'completion.json').exists()
    assert admission.resume_external_completion(workspace,data) == completion
    assert admission.resume_external_completion(workspace,data) == completion
    assert (data/'coordination/new.json').read_bytes() == receipt
    (data/'coordination/new.json').write_bytes(b'different')
    with pytest.raises(ValueError, match='existing publication differs'):
        admission.resume_external_completion(workspace,data)


def test_launch_consumer_requires_pinned_marker_receipt_and_manifest(tmp_path):
    import importlib.util
    module_path = ROOT/'scripts/environment/configured_job_launch.py'
    spec = importlib.util.spec_from_file_location('launch_fixture', module_path)
    launch = importlib.util.module_from_spec(spec); spec.loader.exec_module(launch)
    data = tmp_path/'data'; receipt = data/'coordination/next.json'
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({'schema':'saea-campaign-decision-v1','status':'decided'}))
    manifest = tmp_path/'manifest.json'; manifest.write_text('{}')
    marker = tmp_path/'completion.json'
    marker.write_text(json.dumps({'schema':'saea-external-admission-completion-v1','candidate':'c',
        'receipt':{'relpath':'coordination/next.json','sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()},
        'workspace_manifest':{'path':'/fixture/manifest.json','sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}, 'binding':{}}))
    pin = {'marker':str(marker),'sha256':hashlib.sha256(marker.read_bytes()).hexdigest(),
           'data_root':str(data),'workspace_manifest':str(manifest)}
    assert launch.validate_external_admission(pin)['candidate'] == 'c'
    receipt.write_text('{}')
    with pytest.raises(ValueError, match='receipt absent or changed'):
        launch.validate_external_admission(pin)
    marker.unlink()
    with pytest.raises(ValueError, match='marker absent or changed'):
        launch.validate_external_admission(pin)


def test_longest_mount_for_receipt_and_workspace_files(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('launch_mounts', ROOT/'scripts/environment/configured_job_launch.py')
    launch = importlib.util.module_from_spec(spec); spec.loader.exec_module(launch)
    root = tmp_path/'root'; root.mkdir()
    child = tmp_path/'child'; child.mkdir()
    (child/'successor.json').write_text('right')
    (root/'coordination/B').mkdir(parents=True)
    (root/'coordination/B/successor.json').write_text('wrong')
    doc = {'mounts':[{'source':str(root),'target':'/data','readonly':True},
                     {'source':str(child),'target':'/data/coordination/B','readonly':False}]}
    assert launch.mapped_input(doc, '/data/coordination/B/successor.json') == child/'successor.json'
    with pytest.raises(ValueError, match='readonly'):
        launch.mapped_input(doc, '/data/coordination/B/successor.json', readonly=True)


def test_resume_rejects_changed_workspace_asset(tmp_path):
    import admit_real_engine_candidate as admission
    workspace, data = tmp_path/'workspace', tmp_path/'data'
    asset = workspace/'configs/a.json'; asset.parent.mkdir(parents=True); asset.write_text('old')
    manifest = workspace/'manifests/m.json'; manifest.parent.mkdir()
    manifest.write_text(json.dumps({'schema':'saea-experiment-workspace-v1','files':{'configs/a.json':hashlib.sha256(asset.read_bytes()).hexdigest()}}))
    receipt = b'fixture'
    completion = {'schema':'saea-external-admission-completion-v1','candidate':'c','binding':{},'receipt':{'relpath':'coordination/r.json','sha256':hashlib.sha256(receipt).hexdigest()},
                  'workspace_manifest':{'path':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}}
    admission.publish_external_completion(workspace,data,completion,receipt)
    asset.write_text('changed')
    with pytest.raises(ValueError, match='workspace asset differs'):
        admission.resume_external_completion(workspace,data)


@pytest.mark.parametrize('name', ['publication-intent.json', 'receipt.json', 'completion.json'])
def test_exclusive_same_repairs_directory_fsync_after_link_failure(tmp_path, monkeypatch, name):
    import os
    import admit_real_engine_candidate as admission
    path = tmp_path/name
    real_fsync = os.fsync
    def fail_directory(fd):
        if Path(f'/proc/self/fd/{fd}').resolve() == tmp_path:
            assert path.read_bytes() == b'exact'
            raise OSError('directory fsync failed')
        real_fsync(fd)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'fsync', fail_directory)
        with pytest.raises(OSError, match='directory fsync failed'):
            admission._exclusive_same(path, b'exact')
        with pytest.raises(OSError, match='directory fsync failed'):
            admission._exclusive_same(path, b'exact')
    admission._exclusive_same(path, b'exact')
    admission._exclusive_same(path, b'exact')
    with pytest.raises(ValueError, match='existing publication differs'):
        admission._exclusive_same(path, b'different')
    assert path.read_bytes() == b'exact'
