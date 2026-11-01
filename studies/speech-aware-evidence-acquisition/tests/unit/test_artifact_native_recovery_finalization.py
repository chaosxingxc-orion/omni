"""Artifact finalization retains the exact preflight native-recovery ancestry."""
import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from test_tfgrpo_runtime import setup
from speech_aware_evidence_acquisition.core.bundle.finalize import _artifact_native_replay, _validate_artifact_finalization_binding
from speech_aware_evidence_acquisition.core.bundle import BundleError
from speech_aware_evidence_acquisition.core.contact_journal import NativeContactReplay, canonical, digest


def fixture(tmp_path):
    caller, client, _, transport, _, _ = setup(tmp_path)
    source = client._adapter._plan; data = client._binding.data_root
    client.close(); client._adapter._attempt_handle.mark_aborted('precontact failure')
    target = replace(source, attempt_id='recovered-artifact')
    replay = NativeContactReplay(data, source, target, client.arm)
    receipt = data/'recovery.json'
    receipt.write_bytes(canonical({'schema':'saea-native-recovery-v1',
        'source':replay.source_binding(),'target_plan':asdict(target)}))
    run_dir = data/'runs'/target.run_id/target.attempt_id; run_dir.mkdir(parents=True)
    declaration = {'schema':'saea-resume-declaration-v1','run_id':target.run_id,
        'attempt_id':target.attempt_id,'predecessors':[{'attempt_id':source.attempt_id,
            'mode':'native-contact-journal','pin_sha256':replay.source_binding()['attempt_sha256']}],
        'native_recovery_receipt':{'path':str(receipt),'sha256':digest(receipt.read_bytes())}}
    path = run_dir/'resume-declaration.json'; path.write_bytes(canonical(declaration))
    bundle = SimpleNamespace(manifest_path=run_dir/'manifest.json')
    values = {'tfgrpo_learning_registration':{'present':True},'arm':client.arm}
    return bundle, data, source, target, values, path, receipt


def test_recovered_artifact_loads_real_zero_contact_ancestry(tmp_path):
    bundle,data,source,target,values,_,_=fixture(tmp_path)
    replay=_artifact_native_replay(bundle,data_root=data,plan=target,values=values)
    assert replay.scientific_source_plan==source
    assert replay.request_ids==frozenset()
    replay.assert_target(data,target,values['arm'])


@pytest.mark.parametrize('fault',['target','predecessor','receipt','foreign_directory','symlink','source_state'])
def test_recovered_artifact_rejects_changed_custody(tmp_path,fault):
    bundle,data,source,target,values,path,receipt=fixture(tmp_path)
    if fault in ('target','predecessor'):
        doc=json.loads(path.read_bytes())
        if fault=='target':doc['attempt_id']='foreign'
        else:doc['predecessors'][0]['pin_sha256']='0'*64
        path.write_bytes(canonical(doc))
    elif fault=='receipt':receipt.write_text('{}')
    elif fault=='foreign_directory':
        other=data/'foreign';other.mkdir();(other/path.name).write_bytes(path.read_bytes())
        bundle.manifest_path=other/'manifest.json'
    elif fault=='symlink':
        other=data/'redirect.json';other.write_bytes(path.read_bytes());path.unlink();path.symlink_to(other)
    else:
        attempt=data/'attempts'/source.run_id/(source.attempt_id+'.json')
        doc=json.loads(attempt.read_bytes());doc['state']='OPENED';attempt.write_bytes(canonical(doc))
    with pytest.raises((ValueError,RuntimeError)):
        _artifact_native_replay(bundle,data_root=data,plan=target,values=values)


def test_plain_artifact_keeps_no_replay(tmp_path):
    bundle,data,source,target,values,path,_=fixture(tmp_path)
    path.unlink()
    assert _artifact_native_replay(bundle,data_root=data,plan=target,values=values) is None
    assert _artifact_native_replay(bundle,data_root=data,plan=target,values={}) is None


def test_finalization_passes_verified_replay_without_rewriting_plan(tmp_path,monkeypatch):
    bundle,data,source,target,values,_,_=fixture(tmp_path)
    from speech_aware_evidence_acquisition.core import config
    from speech_aware_evidence_acquisition.core.flight import runtime
    from speech_aware_evidence_acquisition.core.bundle import finalize
    binding={'role':'training-feedback-proposal'}
    bundle.manifest={'config':{'fragments':dict(model='m',dataset='d',baseline='b',experiment='e'),'config_hash':'hash'},
        'plan':asdict(target),'artifact_result':binding,'result_kind':'artifact'}
    bundle.split_ids=frozenset(['sample']);bundle.outputs_rows=({'result_kind':'artifact','artifact_role':binding['role']},)
    monkeypatch.setattr(finalize,'require_clean_load_bearing_surface',lambda _:None)
    monkeypatch.setattr(config,'compose',lambda *a,**k:SimpleNamespace(values=values,config_hash='hash'))
    def validate(actual,**kwargs):
        assert kwargs['plan']==target
        assert kwargs['native_contact_replay'].scientific_source_plan==source
        return binding
    monkeypatch.setattr(runtime,'validate_artifact_result_binding',validate)
    _validate_artifact_finalization_binding(bundle,tmp_path,data)
    bundle.manifest['plan']['attempt_id']='tampered'
    with pytest.raises(BundleError,match='artifact finalization binding'):
        _validate_artifact_finalization_binding(bundle,tmp_path,data)
