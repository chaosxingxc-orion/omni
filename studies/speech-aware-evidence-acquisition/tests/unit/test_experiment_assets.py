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
