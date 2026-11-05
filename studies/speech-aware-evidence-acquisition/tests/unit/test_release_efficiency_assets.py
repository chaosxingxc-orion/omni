"""External release readers honor content pins without private data."""
import hashlib
import json
from pathlib import Path
import pytest
from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells
from speech_aware_evidence_acquisition.core.deployment import resolve_profile_path


def cell(run):
    return {'cell_id':'cell', 'model_fragment':'m', 'dataset_fragment':'d',
            'baseline_fragment':'b', 'run_id':run, 'attempt_id':'attempt',
            'readiness':{'state':'held'}}


def test_external_campaign_and_profile_pin_refuse_missing_or_mutation(tmp_path, monkeypatch):
    repo=tmp_path/'repo'; repo.mkdir()
    inner=repo/'configs/experiment/cell.json'; inner.parent.mkdir(parents=True)
    inner.write_text(json.dumps({'cells':[cell('old')]}))
    profile=repo/'configs/deployment/profile.json';profile.parent.mkdir(parents=True)
    profile.write_text('{"old":true}')
    ext=tmp_path/'external';ext.mkdir()
    exp=ext/'experiment.json';exp.write_text(json.dumps({'cells':[cell('new')]}))
    prof=ext/'profile.json';prof.write_text('{"new":true}')
    doc={'schema':'saea-experiment-assets-v1','files':{
        'configs/experiment/cell.json':{'path':str(exp),'sha256':hashlib.sha256(exp.read_bytes()).hexdigest()},
        'configs/deployment/profile.json':{'path':str(prof),'sha256':hashlib.sha256(prof.read_bytes()).hexdigest()}}}
    manifest=tmp_path/'assets.json';manifest.write_text(json.dumps(doc))
    monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST',str(manifest))
    monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256',hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.delenv('SAEA_EXPERIMENT_WORKSPACE',raising=False)
    assert load_campaign_cells(repo,'cell')[0].run_id=='new'
    assert resolve_profile_path('profile',repo)==prof
    assert resolve_profile_path(str(profile),repo)==profile
    exp.write_text('{}')
    with pytest.raises(ValueError,match='absent or changed'):load_campaign_cells(repo,'cell')
    prof.unlink()
    with pytest.raises(ValueError,match='absent or changed'):resolve_profile_path('profile',repo)
    monkeypatch.delenv('SAEA_EXPERIMENT_MANIFEST')
    monkeypatch.delenv('SAEA_EXPERIMENT_MANIFEST_SHA256')
    assert load_campaign_cells(repo,'cell')[0].run_id=='old'
    assert resolve_profile_path('profile',repo)==profile


def test_external_manifest_cannot_escape_repository_namespace(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest';manifest.write_text(json.dumps({'schema':'saea-experiment-assets-v1','files':{}}))
    monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST',str(manifest))
    monkeypatch.setenv('SAEA_EXPERIMENT_MANIFEST_SHA256',hashlib.sha256(manifest.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='repository relative'):
        resolve_asset(tmp_path,'../secret')
