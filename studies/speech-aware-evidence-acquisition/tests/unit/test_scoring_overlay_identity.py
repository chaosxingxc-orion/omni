"""Score-time overlay replay and separation of generation/tool identities."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from speech_aware_evidence_acquisition.core.bundle import BundleError
from speech_aware_evidence_acquisition.core.bundle.delivery import _recorded_delivery_config_hash, recorded_phase_delivery_declarations
from speech_aware_evidence_acquisition.core.config import compose
from speech_aware_evidence_acquisition.core.flight.factory import load_external_runtime_config
from speech_aware_evidence_acquisition.core.flight.runtime_overlay import apply_runtime_overlay
from speech_aware_evidence_acquisition.core.scoring_batch import ScoreJob, ScoringContext, run_official_score_job
from test_external_workspace_score import _world


def _overlay(path, workers=2):
    raw = json.dumps({'schema':'saea-native-efficiency-v1',
        'dispatch':{'schema':'saea-dispatch-slot-fill-v1','role_input_offload':True,
                    'ready_request':True,'prepare_workers':workers,'validation_workers':2,
                    'admitted_trajectories':2,'model_inflight':2},
        'audio_cache':{'enabled':True,'max_entries':2,'max_bytes':1024,'preload_workers':1}}).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_old_off_and_on_saved_hashes_from_real_fragments(monkeypatch):
    """Replay saved fragment bytes for both arms without reading their traces."""
    import os
    fixture = os.environ.get('SAEA_TEST_PSD_DIAGNOSTIC')
    if not fixture:
        pytest.skip('optional saved real PSD config fixture')
    doc=json.loads(Path(fixture).read_bytes())
    assert doc['plain_composed_hash'] != doc['recorded_config_hash']
    assert doc['overlay_effective_hash'] == doc['recorded_config_hash']


def test_overlay_replay_and_tampering(tmp_path, monkeypatch):
    repo, workspace, inv, bundle = _world(tmp_path, monkeypatch)
    path = workspace/'configs/baseline/b.json'
    d=json.loads(path.read_bytes());d.update(native_episode={'schema':'test'},policy='dynamic',transport='vllm')
    path.write_text(json.dumps(d)); inventory=json.loads(inv.read_bytes())
    inventory['files']['configs/baseline/b.json']=hashlib.sha256(path.read_bytes()).hexdigest()
    raw=json.dumps(inventory,sort_keys=True).encode();inv.write_bytes(raw)
    monkeypatch.setenv('SAEA_EXPERIMENT_WORKSPACE_SHA256',hashlib.sha256(raw).hexdigest())
    names=bundle.manifest['config']['fragments']
    monkeypatch.setenv('SPEECHRL_DATA_DIR',str(tmp_path/'data'))
    original=compose(repo,**names)
    assert _recorded_delivery_config_hash(repo,names)==original.config_hash
    overlay=tmp_path/'runtime.json'; pin=_overlay(overlay)
    monkeypatch.setenv('SAEA_RUNTIME_CONFIG',str(overlay));monkeypatch.setenv('SAEA_RUNTIME_CONFIG_SHA256',pin)
    values=apply_runtime_overlay(original.values,load_external_runtime_config(overlay))
    effective=hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    bundle.manifest['config']['config_hash']=effective
    assert recorded_phase_delivery_declarations(bundle,repo)==()
    monkeypatch.setenv('SAEA_RUNTIME_CONFIG_SHA256','0'*64)
    with pytest.raises(Exception,match='SHA256'):recorded_phase_delivery_declarations(bundle,repo)
    monkeypatch.setenv('SAEA_RUNTIME_CONFIG_SHA256',pin)
    copy=tmp_path/'copy.json';copy.write_bytes(overlay.read_bytes())
    monkeypatch.setenv('SAEA_RUNTIME_CONFIG',str(copy))
    with pytest.raises(BundleError,match='differs'):recorded_phase_delivery_declarations(bundle,repo)
    monkeypatch.setenv('SAEA_RUNTIME_CONFIG',str(overlay))
    changed=_overlay(overlay,workers=3);monkeypatch.setenv('SAEA_RUNTIME_CONFIG_SHA256',changed)
    with pytest.raises(BundleError,match='differs'):recorded_phase_delivery_declarations(bundle,repo)
    monkeypatch.delenv('SAEA_RUNTIME_CONFIG_SHA256')
    with pytest.raises(BundleError,match='requires an explicit SHA256'):recorded_phase_delivery_declarations(bundle,repo)


def test_score_subprocess_selects_tool_and_preserves_source_context(tmp_path, monkeypatch):
    source=tmp_path/'generation'; tool=tmp_path/'tool'; source.mkdir();tool.mkdir()
    manifest=tmp_path/'run.run-manifest.json';manifest.write_text('{}')
    calls=[]
    def run(argv,**kwargs):
        calls.append((argv,kwargs))
        return subprocess.CompletedProcess(argv,0,stdout='{}',stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    context=ScoringContext(source,tmp_path/'data',tmp_path/'lock',tool_repo_root=tool)
    run_official_score_job(ScoreJob('x',manifest),context=context)
    args,kw=calls[0]
    assert kw['cwd']==tool and kw['env']['PYTHONPATH'].split(':')[0]==str(tool/'src')
    assert args[args.index('--repo-root')+1]==str(source)
