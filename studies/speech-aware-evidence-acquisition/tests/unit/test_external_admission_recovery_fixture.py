"""Offline recovery using exact assets from the successful A/B workspaces."""
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT/'scripts/analysis'))
import admit_real_engine_candidate as admission

spec = importlib.util.spec_from_file_location('recovery_launch', ROOT/'scripts/environment/configured_job_launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.mark.parametrize('arm', ['A', 'B'])
@pytest.mark.parametrize('fault', ['receipt', 'marker'])
def test_exact_workspace_recovery_and_consumer(tmp_path, monkeypatch, arm, fault):
    fixture = Path(os.environ.get('SAEA_RECOVERY_FIXTURES', '/nonexistent'))
    if not fixture.is_dir():
        pytest.skip('archived A/B fixtures not mounted')
    workspace = tmp_path/'workspace'
    shutil.copytree(fixture/arm/'workspace', workspace)
    intent_path = workspace/'publication-intent.json'
    intent = json.loads(intent_path.read_bytes())
    completion = intent['completion']
    manifest = workspace/'manifests'/Path(completion['workspace_manifest']['path']).name
    assert sha(manifest) == completion['workspace_manifest']['sha256']
    completion['workspace_manifest']['path'] = str(manifest)
    intent_path.write_text(json.dumps(intent, sort_keys=True, indent=2)+'\n')
    (workspace/'completion.json').unlink()
    data = tmp_path/'data'; receipt = data/completion['receipt']['relpath']
    receipt.parent.mkdir(parents=True)
    original = base64.b64decode(intent['receipt_base64'], validate=True)
    assert hashlib.sha256(original).hexdigest() == completion['receipt']['sha256']
    from speech_aware_evidence_acquisition.core import atomic_file
    import thinking_system_mint_receipt as mint_module
    from speech_aware_evidence_acquisition.core.flight import cli
    def forbidden(*args, **kwargs):
        raise AssertionError('resume must not mint or prepare')
    original_fsync = os.fsync
    target = receipt if fault == 'receipt' else workspace/'completion.json'
    failed = []
    def interrupt(fd):
        if not failed and Path(f'/proc/self/fd/{fd}').resolve() == target.parent and target.exists():
            failed.append(True)
            raise OSError('after link directory fsync')
        return original_fsync(fd)
    with patch.object(mint_module, 'mint', forbidden), patch.object(cli, 'assert_prepared_artifacts_current', forbidden):
        with monkeypatch.context() as p:
            p.setattr(os, 'fsync', interrupt)
            with pytest.raises(OSError, match='after link'):
                admission.resume_external_completion(workspace, data)
        assert failed and target.is_file()
        for _ in range(2):
            assert admission.resume_external_completion(workspace, data) == completion
    assert receipt.read_bytes() == original
    marker = workspace/'completion.json'
    pin = {'marker':str(marker), 'sha256':sha(marker), 'workspace_manifest':str(manifest), 'data_root':str(data)}
    logical_data = '/mnt/cephfs/xingchao/omni'
    child = logical_data+'/coordination/'+completion['candidate']
    doc = {'mounts':[{'source':str(data),'target':logical_data,'readonly':True},
                     {'source':str(receipt.parent),'target':child,'readonly':False},
                     {'source':os.environ['SAEA_RECOVERY_PREPARED'],'target':logical_data+'/prepared','readonly':True},
                     {'source':str(workspace),'target':'/workspace','readonly':True}],
           'environment':{'SPEECHRL_DATA_DIR':logical_data,'SAEA_EXPERIMENT_WORKSPACE':'/workspace',
                          'SAEA_EXPERIMENT_WORKSPACE_SHA256':sha(manifest)}}
    assert launch.mapped_input(doc, logical_data+'/'+completion['receipt']['relpath']) == receipt
    assert launch.validate_external_admission(pin,doc) == completion
    # W1 inventory mounted over W2: the business asset must be checked at its actual mount.
    w2 = tmp_path/'workspace2'; shutil.copytree(workspace,w2)
    asset = completion['binding']['candidate']; (w2/asset).write_text('{}')
    doc['mounts'][-1]['source'] = str(w2)
    doc['mounts'].append({'source':str(manifest),'target':'/workspace/manifests/'+manifest.name,'readonly':True})
    with pytest.raises(ValueError, match='workspace asset absent or changed'):
        launch.validate_external_admission(pin,doc)


def test_real_b_receipt_mount_shape(tmp_path):
    evidence = Path(os.environ.get('SAEA_B_MOUNT_EVIDENCE', '/nonexistent'))
    if not evidence.is_file():
        pytest.skip('read-only B mount evidence not mounted')
    doc = json.loads(evidence.read_bytes())
    target = '/mnt/cephfs/xingchao/omni/coordination/slurp-fold-a-split-p-off-tp4-v1'
    mount, = [m for m in doc['mounts'] if m['target'] == target]
    assert mount['readonly'] is False
    assert mount['source'].endswith('/coordination-root/slurp-fold-a-split-p-off-tp4-v1')
    observed = json.loads(Path(os.environ['SAEA_B_DOCKER_MOUNTS']).read_bytes())
    live, = [m for m in observed['Mounts'] if m['Destination'] == target]
    assert (live['Source'], live['RW']) == (mount['source'], not mount['readonly'])
    child = tmp_path/'child'; child.mkdir()
    root = tmp_path/'root'; root.mkdir()
    mapped = {'mounts':[{'source':str(root),'target':'/mnt/cephfs/xingchao/omni','readonly':True},
                        {**mount, 'source':str(child)}]}
    logical = target+'/successor-v2.json'
    assert launch.mapped_input(mapped, logical) == child/'successor-v2.json'
