import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

p = Path(__file__).parents[2] / 'scripts/environment/configured_queue_auto_bind.py'
s = importlib.util.spec_from_file_location('auto_bind', p)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


def pin(path, value):
    path.write_text(json.dumps(value)); return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(tmp_path):
    launch={'path':'/producer.json','sha256':'a'*64}
    src={'state_dir':str(tmp_path),'queue_id':'source','jobs':[{'id':'generation','launch':launch}]}
    state={'queue_id':'source','status':'running','identity':{'x':1},'jobs':[{'id':'generation','status':'running','container_id':'b'*64}]}
    (tmp_path/'state.json').write_text(json.dumps(state))
    frag={'after_without_container_id':{'launch':launch,'condition':'generation-released'},'handoff_verifier':{'path':'/verifier','sha256':'c'*64}}
    ready={'jobs':[{'after':None}]}
    cfg={'source_queue':pin(tmp_path/'source.json',src),'source_binding':None,'source_job':'generation','dependency':pin(tmp_path/'fragment.json',frag),'successor_queue':pin(tmp_path/'ready.json',ready),'bound_queue':str(tmp_path/'bound.json'),'hold_path':str(tmp_path/'hold')}
    calls=[]
    q=SimpleNamespace(identities=lambda _: {'x':1},observe=lambda *a: ({},{'container_id':'b'*64,'state':'running'}),execute=lambda p,mode: calls.append(mode) or {'status':'waiting'})
    return cfg,q,state,calls


def test_binds_real_id_and_honors_hold_then_resumes(tmp_path):
    cfg,q,_,calls=fixture(tmp_path);Path(cfg['hold_path']).touch()
    assert m.step(cfg,q)=='held'
    assert json.loads(Path(cfg['bound_queue']).read_text())['jobs'][0]['after']['container_id']=='b'*64
    assert calls==['plan']
    Path(cfg['hold_path']).unlink();assert m.step(cfg,q)=='waiting'
    assert calls==['plan','plan','once']


def test_missing_or_failed_or_changed_owner_refused(tmp_path):
    cfg,q,state,calls=fixture(tmp_path);state['jobs'][0].pop('container_id');(tmp_path/'state.json').write_text(json.dumps(state))
    assert m.step(cfg,q)=='waiting-container' and not calls
    state['jobs'][0].update(container_id='b'*64,status='unknown');(tmp_path/'state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='unknown'):m.step(cfg,q)
    state['jobs'][0]['status']='running';(tmp_path/'state.json').write_text(json.dumps(state));q.observe=lambda *a: ({},{'container_id':'d'*64,'state':'running'})
    with pytest.raises(ValueError,match='container'):m.step(cfg,q)


def test_existing_binding_and_pins_cannot_change(tmp_path):
    cfg,q,_,_=fixture(tmp_path);m.step(cfg,q)
    Path(cfg['bound_queue']).write_text('{}')
    with pytest.raises(ValueError,match='output differs'):m.step(cfg,q)
    Path(cfg['dependency']['path']).write_text('{}')
    with pytest.raises(ValueError,match='pinned input changed'):m.step(cfg,q)


def test_chained_binder_waits_for_pinned_upstream(tmp_path):
    cfg,q,_,_=fixture(tmp_path)
    upstream_dir=tmp_path/'upstream';upstream_dir.mkdir()
    source=cfg['source_queue']['path']
    upstream={'state_dir':str(upstream_dir),'bound_queue':source}
    up_pin=pin(tmp_path/'upstream.json',upstream)
    cfg['source_binding']=up_pin;cfg['source_queue']={'path':source}
    assert m.step(cfg,q)=='waiting-source'
    digest=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    (upstream_dir/'status.json').write_text(json.dumps({'status':'waiting','config_sha256':up_pin['sha256'],'bound_queue_sha256':digest}))
    assert m.step(cfg,q)=='waiting'
    (upstream_dir/'status.json').write_text(json.dumps({'status':'failed','config_sha256':up_pin['sha256'],'bound_queue_sha256':digest}))
    with pytest.raises(ValueError,match='upstream'):m.step(cfg,q)
