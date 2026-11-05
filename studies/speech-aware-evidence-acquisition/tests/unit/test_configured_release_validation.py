"""Offline configuration gate tests using the real transport renderer."""
import hashlib
import importlib.util
import json
import socket
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.core.config import compose
from speech_aware_evidence_acquisition.core.preparation import canonical_bytes
from speech_aware_evidence_acquisition.core.session.receipt import ModelFileRef, SessionReceipt, write_session_receipt
from test_raw_native_transport import _payload

path = Path(__file__).parents[2] / 'scripts/environment/configured_release_validation.py'
spec = importlib.util.spec_from_file_location('configured_release_validation', path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def pin(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def render_config(tmp_path, monkeypatch):
    repo, data = tmp_path/'repo', tmp_path/'data'
    repo.mkdir(); data.mkdir()
    docs = {'model': {'base_url': 'http://127.0.0.1:1', 'prompt_order': 'native-raw-speech-v1', 'transport': 'vllm'},
            'dataset': {'dataset_name': 'test'}, 'baseline': {'arm': 'test'},
            'experiment': {'experiment_name': 'test'}}
    fragments = {}
    for kind, obj in docs.items():
        target = write(repo/'configs'/kind/'x.json', obj)
        fragments[kind] = {'name': 'x', 'sha256': pin(target)['sha256']}
    session = SessionReceipt(schema='saea-session-receipt-v1', study='speech-aware-evidence-acquisition',
        session_id='offline', binary_path='bin/python', binary_sha256='a'*64, argv=('python',), pid=1,
        host='127.0.0.1', port=1, model_files=(ModelFileRef(path='model/test', sha256='b'*64),),
        started_at='2026-01-01T00:00:00Z', startup_stdout_sha256='c'*64)
    session_file=tmp_path/'session.json';write_session_receipt(session_file,session)
    payload,_ = _payload(data)
    intent=write(tmp_path/'intent.json', {'agent_contact_id':'offline:perception:0', 'payload':payload})
    cfg = {'schema':m.SCHEMA,'mode':'render','repo':str(repo),'data':str(data),'fragments':fragments,
           'config_sha256':compose(repo,**{k:'x' for k in docs},data_root=data).config_hash,'session':pin(session_file),
           'requests':[{'role':'perception','intent':pin(intent),'expected_body_sha256':'0'*64,'expected_receipt_sha256':'0'*64}]}
    config=write(tmp_path/'config.json',cfg)
    def no_network(*args,**kwargs): raise AssertionError('network attempted')
    monkeypatch.setattr(socket.socket,'connect',no_network)
    result=m.execute(config);assert not result['ok']
    cfg['requests'][0]['expected_body_sha256']=result['results'][0]['body_sha256']
    cfg['requests'][0]['expected_receipt_sha256']=result['results'][0]['receipt_sha256']
    write(config,cfg)
    return config,cfg,intent


def test_real_renderer_offline_compares_body_receipt(render_config):
    config,_,_=render_config;result=m.execute(config)
    assert result['ok'] and result['results'][0]['role']=='perception'
    assert result['new_model_calls']==result['fit_calls']==result['tokenizer_calls']==0
    assert 'stubbed' in result['scope']


@pytest.mark.parametrize('fault',['missing','intent_hash','fragment_hash','body','receipt','role'])
def test_render_fails_closed_on_source_or_output_drift(render_config,fault):
    config,cfg,intent=render_config
    if fault=='missing':intent.unlink()
    elif fault=='intent_hash':intent.write_text('{}')
    elif fault=='fragment_hash':cfg['fragments']['model']['sha256']='f'*64
    elif fault=='role':cfg['requests'][0]['role']='slot'
    else:cfg['requests'][0]['expected_'+fault+'_sha256']='f'*64
    write(config,cfg)
    if fault in ('body','receipt'):assert not m.execute(config)['ok']
    elif fault == 'fragment_hash':
        with pytest.raises(ValueError):m.execute(config)
    else:
        result=m.execute(config);assert not result['ok'] and result['results'][0]['role']==cfg['requests'][0]['role'] and 'error' in result['results'][0]


def test_bridge_requires_pinned_release_review_and_never_touches_ledger(render_config, tmp_path):
    from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter
    config,cfg,_=render_config;data=Path(cfg['data']);writer=RowCheckpointWriter(tmp_path/'ckpt','run','old',identity={'split_identity_hash':'a'*64},fsync=False);writer.close()
    bridge=write(tmp_path/'bridge.json',{'schema':'invalid','evidence':{key:pin(config) for key in ('commit_object','tree_listing','source_archive','release_manifest','source_review','reference_pins')}})
    cfg.update(mode='bridge-precheck',bridge=pin(bridge),checkpoint=pin(writer.path),run_id='run',predecessor_attempt_id='old',successor_attempt_id='new')
    del cfg['session'],cfg['requests'];write(config,cfg)
    with pytest.raises(ValueError):m.execute(config)
    assert not (data/'attempts').exists()
    bridge.unlink()
    with pytest.raises(ValueError,match='missing'):m.execute(config)
    assert not (data/'attempts').exists()


def test_bridge_precheck_dispatches_original_review_and_checkpoint_without_verify(render_config, tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.core.flight import checkpoint_compatibility as bridge_module
    from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter
    config,cfg,_=render_config
    writer=RowCheckpointWriter(tmp_path/'ckpt','run','old',identity={'split_identity_hash':'a'*64},fsync=False);writer.close()
    evidence_file=write(tmp_path/'evidence.json', {'review':'fixture'})
    evidence={key:pin(evidence_file) for key in ('commit_object','tree_listing','source_archive','release_manifest','source_review','reference_pins')}
    evidence['startup_data_root']=str(Path(cfg['data']))
    bridge={'schema':bridge_module.RELEASE_SCHEMA,'predecessor_commit':'b'*40,
        'predecessor_fragments':{k:'x' for k in cfg['fragments']},'successor_attempt_id':'new',
        'change':'image-release-operational-v1','evidence':evidence}
    bridge_file=write(tmp_path/'bridge.json',bridge)
    cfg.update(mode='bridge-precheck',bridge=pin(bridge_file),checkpoint=pin(writer.path),run_id='run',predecessor_attempt_id='old',successor_attempt_id='new')
    del cfg['session'],cfg['requests'];write(config,cfg)
    for rel in bridge_module.IMPLEMENTATION.values():
        target=Path(cfg['repo'])/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('source')
    seen={}
    def precheck(self, **kwargs):
        seen['review']=self.evidence['source_review'];seen['declaration']=kwargs['declaration']
        return {'activation_authorized':False,'reviewed_source_count':1}
    def no_verify(*args, **kwargs):raise AssertionError('formal verify must not run')
    monkeypatch.setattr(bridge_module.ImageReleaseCompatibility,'precheck',precheck)
    monkeypatch.setattr(bridge_module.ImageReleaseCompatibility,'verify',no_verify)
    result=m.execute(config)
    assert result['ok'] and result['activation_authorized'] is False
    assert seen['review']==evidence['source_review'] and seen['declaration']['checkpoint_sha256']==pin(writer.path)['sha256']
    assert not (Path(cfg['data'])/'attempts').exists()
    bridge_file.write_text('{}')
    with pytest.raises(ValueError,match='SHA256 differs'):m.execute(config)


def test_render_uses_vllm_forced_close_knob_and_refuses_other_backend(render_config, monkeypatch):
    from speech_aware_evidence_acquisition.core import reasoning_knobs
    config,cfg,_=render_config
    called=[];original=reasoning_knobs.vllm_forced_close_message_from_server_args
    def recording(args):
        result=original(args);called.append(result);return result
    monkeypatch.setattr(reasoning_knobs,'vllm_forced_close_message_from_server_args',recording)
    assert m.execute(config)['ok'] and called and all(item is None for item in called)
    target=Path(cfg['repo'])/'configs/model/x.json';model=json.loads(target.read_bytes());model['transport']='llama-server';target.write_text(json.dumps(model));cfg['fragments']['model']['sha256']=pin(target)['sha256'];cfg['config_sha256']=compose(Path(cfg['repo']),**{k:'x' for k in cfg['fragments']},data_root=Path(cfg['data'])).config_hash;write(config,cfg)
    with pytest.raises(ValueError,match='only vllm'):m.execute(config)


def test_bridge_torn_checkpoint_and_missing_review_are_refused(render_config, tmp_path):
    from speech_aware_evidence_acquisition.core.flight import checkpoint_compatibility as bridge_module
    from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter
    config,cfg,_=render_config
    writer=RowCheckpointWriter(tmp_path/'torn','run','old',identity={'split_identity_hash':'a'*64},fsync=False);writer.close()
    with writer.path.open('ab') as out:out.write(b'{"partial"')
    evidence_file=write(tmp_path/'evidence.json', {'review':'fixture'})
    evidence={key:pin(evidence_file) for key in ('commit_object','tree_listing','source_archive','release_manifest','source_review','reference_pins')};evidence['startup_data_root']=cfg['data']
    bridge={'schema':bridge_module.RELEASE_SCHEMA,'predecessor_commit':'b'*40,'predecessor_fragments':{k:'x' for k in cfg['fragments']},'successor_attempt_id':'new','change':'image-release-operational-v1','evidence':evidence}
    bridge_file=write(tmp_path/'bridge.json',bridge)
    cfg.update(mode='bridge-precheck',bridge=pin(bridge_file),checkpoint=pin(writer.path),run_id='run',predecessor_attempt_id='old',successor_attempt_id='new');del cfg['session'],cfg['requests'];write(config,cfg)
    with pytest.raises(ValueError,match='torn tail'):m.execute(config)
    evidence_file.unlink()
    with pytest.raises(ValueError,match='missing'):m.execute(config)
    assert not (Path(cfg['data'])/'attempts').exists()
