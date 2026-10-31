import copy,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from speech_aware_evidence_acquisition.core.bundle import tfrl_bank as m,BundleError
from speech_aware_evidence_acquisition.core.config import _canonical_hash
from speech_aware_evidence_acquisition.rails import tfrl_v2_consumption as rail
from test_tfrl_v2_consumption import make_bank,SCOPE,FOLD_B_ID,FOLD_B_ROWS
# Autouse fixture required by the actual runner harness.
from test_llama_server_transport import _spawn_live_matching_process


def setup(tmp_path):
    data=tmp_path/'data';path,receipt=make_bank(data)
    block=rail.declaration_block(iteration=1,consumer='FOLD-B',bank_kind='union',
        receipt_relpath=str(path.relative_to(data)),receipt_sha=rail.receipt_sha256(receipt),source_scope=SCOPE)
    values={'native_episode':{'roles':{'perception':{'phase_id':'perception'}},'consumption_module':{'tfrl_v2_bank':block}}}
    cfg=SimpleNamespace(values=values,config_hash=_canonical_hash(values));split=SimpleNamespace(ids=FOLD_B_ROWS,identity_hash=FOLD_B_ID)
    directory=data/'runs/RUN/attempt';directory.mkdir(parents=True)
    manifest={'run_id':'RUN','config':{'config_hash':cfg.config_hash},
        'attempt':{'attempt_id':'attempt','state_path':str(data/'attempts/RUN/attempt.json')}}
    manifest['tfrl_bank_config']=m.seal(cfg,data,split,directory)
    return data,path,manifest,directory,split


def test_sealed_snapshot_reconstructs_bank_and_preserves_ordinary_manifest(tmp_path):
    data,path,manifest,directory,split=setup(tmp_path)
    context=m.load(manifest,directory/'RUN.run-manifest.json',split.ids,split.identity_hash)
    assert context['bank'].rule_text()=='Consumption rules:'
    assert m.load({},directory/'plain.json',split.ids,split.identity_hash) is None
    assert m.seal(SimpleNamespace(values={}),data,split,directory) is None


@pytest.mark.parametrize('fault',['snapshot','manifest_hash','root','bank','traversal','missing_decl'])
def test_bank_snapshot_and_files_fail_closed(tmp_path,fault):
    data,path,manifest,directory,split=setup(tmp_path)
    snapshot=Path(manifest['tfrl_bank_config']['path'])
    if fault=='manifest_hash':manifest['config']['config_hash']='a'*64
    elif fault=='bank':(path.parent/'rule-bank.json').write_text('{}')
    elif fault=='root':manifest['attempt']['state_path']=str(tmp_path/'elsewhere.json')
    elif fault=='traversal':
        receipt=json.loads(path.read_text());receipt['rule_bank']['file']='../rule-bank.json';path.write_text(json.dumps(receipt))
    else:
        doc=json.loads(snapshot.read_text())
        if fault=='snapshot':doc['values']['changed']=True
        else:doc['values']['native_episode']['consumption_module']={}
        snapshot.write_text(json.dumps(doc));manifest['tfrl_bank_config']['sha256']=hashlib.sha256(snapshot.read_bytes()).hexdigest()
        if fault=='missing_decl':manifest['config']['config_hash']=_canonical_hash(doc['values'])
    with pytest.raises((BundleError,rail.TfrlV2BankRefusal)):
        m.load(manifest,directory/'RUN.run-manifest.json',split.ids,split.identity_hash)


def test_actual_banked_runner_passes_bundle_arm_binding_only_with_verified_context(tmp_path):
    pytest.importorskip('openjiuwen')
    from test_tfrl_current_bank_audit import actual
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_supply_variant_trace_binding
    result,trace,bank=actual(tmp_path,with_trace=True)
    episode=result.native_episode_receipt
    contact=[v for v in episode['role_visits'] if v['stage']=='solver'][-1]['execution_receipt']['final_contact']
    row={'sample_id':result.sample_id,'native_episode_receipt':episode,'supply_variants':result.supply_variants,
         'request_id':contact['request_id'],'request_sha256':contact['request_sha256'],'response_sha256':contact['response_sha256']}
    with pytest.raises(BundleError,match='context'):_verify_supply_variant_trace_binding(row,trace)
    _verify_supply_variant_trace_binding(row,trace,bank_context={'bank':bank,'perception_phase':'perception'})
    bad=copy.deepcopy(trace);bad[contact['request_id']+'/response']['text']='tampered'
    with pytest.raises(BundleError,match='trace'):
        _verify_supply_variant_trace_binding(row,bad,bank_context={'bank':bank,'perception_phase':'perception'})


# These fixtures exercise real manifest construction/verification with a fake
# transport. They establish custody, never an experimental performance result.
from test_bundle import bundled_repo


def test_full_bundle_seals_config_and_rejects_changed_bank(synthetic_world,bundled_repo,tmp_path,monkeypatch):
    import test_bundle as b
    from speech_aware_evidence_acquisition.core.config import ComposedConfig
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    data=synthetic_world.data_root
    path,receipt=make_bank(data)
    block=rail.declaration_block(iteration=1,consumer='FOLD-B',bank_kind='union',
        receipt_relpath=str(path.relative_to(data)),receipt_sha=rail.receipt_sha256(receipt),source_scope=SCOPE)
    values=dict(b._composed().values)
    values['native_episode']={'roles':{'perception':{'phase_id':'perception'}},
        'consumption_module':{'tfrl_v2_bank':block}}
    cfg=ComposedConfig(values=values,config_hash=_canonical_hash(values),fragments=b._composed().fragments)
    monkeypatch.setattr(b,'_composed',lambda:cfg)
    manifest_path,_,_,_=b._run_completed(synthetic_world,bundled_repo,tmp_path)
    manifest=json.loads(manifest_path.read_text())
    assert manifest['tfrl_bank_config']['sha256']
    verify_bundle(manifest_path)
    (path.parent/'rule-bank.json').write_text('{}')
    with pytest.raises((BundleError,rail.TfrlV2BankRefusal)):
        verify_bundle(manifest_path)


def test_aborted_manifest_survives_missing_bank(synthetic_world,bundled_repo,tmp_path,monkeypatch):
    import test_bundle as b
    from speech_aware_evidence_acquisition.core.bundle import manifest as manifest_module
    def refuse_seal(*args,**kwargs):
        raise AssertionError('aborted manifest must not depend on live bank files')
    monkeypatch.setattr(m,'seal',refuse_seal)
    lock,repo=bundled_repo
    ids,split_hash,_=b.load_split(repo,'discovery',lock,synthetic_world.data_root)
    split=b.load_verified_split(repo,'discovery',lock,synthetic_world.data_root)
    cfg=b._composed()
    target=manifest_module.build_manifest(repo_root=repo,data_root=synthetic_world.data_root,
        composed=cfg,plan=b._plan(split_hash),verified_split=split,session_receipt_path=None,
        run_dir=tmp_path/'aborted',status=b.STATUS_ABORTED,failure_reason='bank unavailable')
    manifest=json.loads(target.read_text())
    assert manifest['status']==b.STATUS_ABORTED
    assert 'tfrl_bank_config' not in manifest


def test_bank_projection_refuses_unregistered_arm_order(tmp_path):
    from speech_aware_evidence_acquisition.rails.tfrl_bank_request_audit import verify_group
    with pytest.raises(ValueError,match='registered K4 arm order'):
        verify_group('row',[{'arm_id':'legacy'}],[],bank=None)
