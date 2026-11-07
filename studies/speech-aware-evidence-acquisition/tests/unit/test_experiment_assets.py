import hashlib,json
from pathlib import Path
import pytest
from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
from speech_aware_evidence_acquisition.core.config import compose

def setup(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir();assets=tmp_path/'assets';assets.mkdir()
    doc={'schema':'saea-experiment-assets-v1','files':{}}
    for kind,value in [('model',{'transport':'vllm'}),('dataset',{'count':1080}),('baseline',{'policy':'native'}),('experiment',{'experiment_id':'external'})]:
        path=assets/(kind+'.json');path.write_text(json.dumps(value));doc['files'][f'configs/{kind}/external.json']={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest=assets/'manifest.json';manifest.write_text(json.dumps(doc));monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST',str(manifest));monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256',hashlib.sha256(manifest.read_bytes()).hexdigest());return repo,assets,manifest,doc

def test_external_fragments_reach_real_compose_without_source_overlay(tmp_path,monkeypatch):
    repo,assets,_,_=setup(tmp_path,monkeypatch)
    result=compose(repo,'external','external','external','external',data_root=tmp_path)
    assert result.values['count']==1080 and result.values['experiment_id']=='external'
    assert not (repo/'configs').exists()

@pytest.mark.parametrize('fault',['pin','manifest','source','missing'])
def test_external_pins_fail_closed(tmp_path,monkeypatch,fault):
    repo,assets,manifest,doc=setup(tmp_path,monkeypatch)
    if fault=='pin':(assets/'model.json').write_text('changed')
    elif fault=='missing':(assets/'model.json').unlink()
    elif fault=='source':
        doc['files']['src/inject.py']=doc['files']['configs/model/external.json'];manifest.write_text(json.dumps(doc));monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256',hashlib.sha256(manifest.read_bytes()).hexdigest())
    else:monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256','a'*64)
    with pytest.raises(ValueError):compose(repo,'external','external','external','external',data_root=tmp_path)


def test_external_template_and_roster_are_pinned_without_overwriting_code(tmp_path,monkeypatch):
    repo,assets,manifest,doc=setup(tmp_path,monkeypatch)
    for rel,text in [('configs/probes/template.txt','External frozen template'),('docs/receipts/rosters/fold.json','{"count":1080}')]:
        path=assets/Path(rel).name;path.write_text(text);doc['files'][rel]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps(doc));monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256',hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert resolve_asset(repo,'configs/probes/template.txt').read_text()=='External frozen template'
    assert json.loads(resolve_asset(repo,'docs/receipts/rosters/fold.json').read_bytes())['count']==1080


def test_workspace_outputs_are_separate_and_pin_drift_rejected(tmp_path,monkeypatch):
    import subprocess
    from speech_aware_evidence_acquisition.core.experiment_assets import output_asset,seal_workspace
    repo=tmp_path/'repo';repo.mkdir();subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
    (repo/'README').write_text('source')
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@local','commit','-m','source'],check=True,capture_output=True)
    workspace=tmp_path/'workspace';monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE',str(workspace))
    out=output_asset(repo,'configs/model/external.json');out.parent.mkdir(parents=True);out.write_text('{"transport":"vllm"}')
    pin=seal_workspace(repo);monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_MANIFEST',pin['path']);monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_SHA256',pin['sha256'])
    assert resolve_asset(repo,'configs/model/external.json')==out
    assert not (repo/'configs').exists()
    out.write_text('changed')
    with pytest.raises(ValueError,match='changed'):resolve_asset(repo,'configs/model/external.json')


def test_already_mapped_path_named_configs_is_not_remapped(tmp_path,monkeypatch):
    import sys
    sys.path.insert(0,str(Path(__file__).parents[2]/'scripts/analysis'))
    from tfrl_hardset_a0_workload import input_path
    repo,assets,_,_=setup(tmp_path,monkeypatch)
    path=assets/'configs/model/external.json';path.parent.mkdir(parents=True);path.write_text('actual external bytes')
    assert input_path(path)==path


def test_external_lock_receipt_reaches_real_lock_parser(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.rails.adapters import slurp_module_lock_v2_2_npu_port_r01c as lock
    repo, assets, manifest, doc = setup(tmp_path, monkeypatch)
    receipt = {key: f"frozen-{key}" for _, key in lock.R01C_SOURCES}
    path = assets / "minus-r01.json"
    path.write_text(json.dumps(receipt))
    doc["files"][lock.R01C_RECEIPT_RELPATH] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps(doc))
    monkeypatch.setenv("SAEA_EXPERIMENT_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert lock.r01c_pins(repo) == {leaf: receipt[key] for leaf, key in lock.R01C_SOURCES}
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        lock.r01c_pins(repo)


def test_external_receipt_reaches_real_preparation_source_gate(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.core.preparation import verify_source_receipts
    repo, assets, manifest, doc = setup(tmp_path, monkeypatch)
    path = assets / 'receipt.json'; path.write_text('{"frozen":true}')
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    rel = 'docs/receipts/knowledge/source.json'
    doc['files'][rel] = {'path':str(path), 'sha256':checksum}
    manifest.write_text(json.dumps(doc)); monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256',hashlib.sha256(manifest.read_bytes()).hexdigest())
    pins = [{'root':'repository','relpath':rel,'sha256':checksum}]
    assert verify_source_receipts(pins,repo_root=repo,data_root=tmp_path)==pins
    path.write_text('changed')
    with pytest.raises(ValueError):verify_source_receipts(pins,repo_root=repo,data_root=tmp_path)


@pytest.mark.parametrize('fault',['missing','dangling-symlink'])
def test_declared_workspace_missing_never_falls_back(tmp_path,monkeypatch,fault):
    from speech_aware_evidence_acquisition.core.experiment_assets import output_asset,seal_workspace
    repo,assets,manifest,doc=setup(tmp_path,monkeypatch)
    monkeypatch.setattr('speech_aware_evidence_acquisition.core.bundle.git_commit.require_clean_study_repo',lambda root:'b'*40)
    workspace=tmp_path/'workspace';monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE',str(workspace))
    relative='configs/model/external.json';out=output_asset(repo,relative);out.parent.mkdir(parents=True);out.write_text('{"transport":"new"}')
    pin=seal_workspace(repo);monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_MANIFEST',pin['path']);monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_SHA256',pin['sha256'])
    out.unlink()
    if fault=='dangling-symlink':out.symlink_to(tmp_path/'absent')
    with pytest.raises(ValueError,match='absent from pinned inventory'):resolve_asset(repo,relative)


def test_uncovered_workspace_falls_back_only_after_identity_verification(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition.core.experiment_assets import output_asset,seal_workspace
    repo,assets,manifest,doc=setup(tmp_path,monkeypatch)
    monkeypatch.setattr('speech_aware_evidence_acquisition.core.bundle.git_commit.require_clean_study_repo',lambda root:'b'*40)
    workspace=tmp_path/'workspace';workspace.mkdir();monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE',str(workspace))
    pin=seal_workspace(repo);monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_MANIFEST',pin['path']);monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_SHA256',pin['sha256'])
    assert resolve_asset(repo,'configs/model/external.json')==assets/'model.json'
    monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_SHA256','a'*64)
    with pytest.raises(ValueError,match='inventory digest'):resolve_asset(repo,'configs/model/external.json')


def test_frozen_gate_uses_pinned_workspace_ledger_and_rejects_tampering(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition.contracts import FrozenCoreGate
    from speech_aware_evidence_acquisition.core.experiment_assets import output_asset,seal_workspace
    repo,_,_,_=setup(tmp_path,monkeypatch)
    monkeypatch.setattr('speech_aware_evidence_acquisition.core.bundle.git_commit.require_clean_study_repo',lambda root:'b'*40)
    workspace=tmp_path/'workspace';monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE',str(workspace))
    ledger=output_asset(repo,'docs/exposure-ledger.md');ledger.parent.mkdir(parents=True);ledger.write_text('fixture ledger')
    pin=seal_workspace(repo);monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_MANIFEST',pin['path']);monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_SHA256',pin['sha256'])
    assert FrozenCoreGate.for_study_repo(repo,tmp_path/'data',tmp_path/'runtime').exposure_ledger==ledger
    ledger.write_text('tampered')
    with pytest.raises(ValueError,match='changed'):
        FrozenCoreGate.for_study_repo(repo,tmp_path/'data',tmp_path/'runtime')


@pytest.mark.parametrize('supplement',['media','evidence'])
def test_frozen_gate_resolves_pinned_external_receipts_and_preserves_source(tmp_path,monkeypatch,supplement):
    from speech_aware_evidence_acquisition.contracts import FrozenCoreGate
    repo,assets,manifest,doc=setup(tmp_path,monkeypatch)
    relatives=['docs/receipts/e0-closure.json','docs/receipts/runtime.json','docs/receipts/supplement.json']
    for rel in relatives:
        path=assets/Path(rel).name;path.write_text('{"frozen":true}')
        doc['files'][rel]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps(doc));monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256',hashlib.sha256(manifest.read_bytes()).hexdigest())
    kwargs={f'd1_{supplement}_supplement_relpath':'receipts/supplement.json',f'd1_{supplement}_supplement_sha256':doc['files'][relatives[-1]]['sha256']}
    gate=FrozenCoreGate.for_study_repo(repo,tmp_path/'data',tmp_path/'runtime',**kwargs)
    assert gate.source_root==repo and gate.e0_receipt==assets/'e0-closure.json'
    assert gate.runtime_receipt==assets/'runtime.json'
    assert getattr(gate,f'd1_{supplement}_supplement')==assets/'supplement.json'
    (assets/'e0-closure.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        FrozenCoreGate.for_study_repo(repo,tmp_path/'data',tmp_path/'runtime',**kwargs)


def test_external_e0_keeps_real_artifact_and_scoring_freeze_checks(tmp_path,monkeypatch):
    import sys
    sys.path.insert(0,str(Path(__file__).parents[1]/'contract'))
    from gate_fixture import GateFixture
    from speech_aware_evidence_acquisition.contracts import FrozenCoreGate,GateClosed
    fixture=GateFixture(tmp_path);fixture.gate()
    closure=fixture.receipts/'e0-closure.json'
    manifest=tmp_path/'assets.json';manifest.write_text(json.dumps({'schema':'saea-experiment-assets-v1',
        'files':{'docs/receipts/e0-closure.json':{'path':str(closure),'sha256':hashlib.sha256(closure.read_bytes()).hexdigest()}}}))
    monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST',str(manifest))
    monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256',hashlib.sha256(manifest.read_bytes()).hexdigest())
    gate=FrozenCoreGate.for_study_repo(tmp_path,fixture.data_root,fixture.runtime_root)
    gate._validate_e0()
    source=tmp_path/fixture.SCORING_REL;source.write_text('def score(): return 999\n')
    with pytest.raises(GateClosed,match='scoring freeze drift'):gate._validate_e0()
    source.write_bytes(fixture.SCORING_SOURCE)
    artifact=fixture.receipts/fixture.deliverables['d1_identity']['artifact'];artifact.write_text('{}')
    with pytest.raises(GateClosed,match='artifact hash drift'):gate._validate_e0()


def test_content_cache_revalidates_pins_and_selected_files(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.core import experiment_assets as m
    repo, assets, manifest, doc = setup(tmp_path, monkeypatch)
    m._external_inventory.cache_clear()
    rel = 'configs/model/external.json'
    for _ in range(3):
        assert resolve_asset(repo, rel) == assets / 'model.json'
    assert m._external_inventory.cache_info().misses == 1
    assert m._external_inventory.cache_info().hits == 2
    with pytest.raises(TypeError):
        m._external_inventory(manifest.read_bytes())[rel] = ('/tmp/x', '0' * 64)
    (assets / 'model.json').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        resolve_asset(repo, rel)
    doc['files'][rel]['sha256'] = hashlib.sha256((assets / 'model.json').read_bytes()).hexdigest()
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='manifest digest'):
        resolve_asset(repo, rel)
    monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256', hashlib.sha256(manifest.read_bytes()).hexdigest())
    assert resolve_asset(repo, rel) == assets / 'model.json'
    doc['files']['src/bad.py'] = doc['files'][rel]
    manifest.write_text(json.dumps(doc))
    monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256', hashlib.sha256(manifest.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='executable source'):
        resolve_asset(repo, rel)


def test_inventory_cache_bounds_and_workspace_identity():
    from speech_aware_evidence_acquisition.core import experiment_assets as m
    m._external_inventory.cache_clear()
    m._workspace_inventory.cache_clear()
    for i in range(8):
        # Distinct valid content, retaining exactly the approved schema.
        raw = json.dumps({'schema': 'saea-experiment-assets-v1', 'files': {},}, indent=i).encode()
        m._external_inventory(raw)
    assert m._external_inventory.cache_info().currsize == 4
    for i in range(8):
        workspace_raw = json.dumps({'schema': 'saea-experiment-workspace-v1',
                                    'source_commit': str(i), 'source_manifest_sha256': 'b',
                                    'files': {}}).encode()
        m._workspace_inventory(workspace_raw, str(i), 'b')
    assert m._workspace_inventory.cache_info().currsize == 4
    raw = json.dumps({'schema': 'saea-experiment-workspace-v1', 'source_commit': 'a',
                      'source_manifest_sha256': 'b', 'files': {}}).encode()
    index = m._workspace_inventory(raw, 'a', 'b')
    with pytest.raises(TypeError):
        index['configs/x.json'] = '0' * 64
    with pytest.raises(ValueError, match='identity'):
        m._workspace_inventory(raw, 'different', 'b')
    with pytest.raises(ValueError, match='identity'):
        m._workspace_inventory(raw, 'a', 'different')
