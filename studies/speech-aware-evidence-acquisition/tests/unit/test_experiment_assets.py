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
