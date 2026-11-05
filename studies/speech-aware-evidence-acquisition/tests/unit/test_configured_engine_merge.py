import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('merge',Path(__file__).parents[2]/'scripts/environment/configured_engine_merge.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def profile(cards,port):
    return SimpleNamespace(cards=cards,instances=[SimpleNamespace(port=port)],image='sha256:'+'a'*64,
        model_path='/models/frozen',effective_namespace_policy='shared-pid-host')


@pytest.mark.parametrize('fault',['missing-card','protected-card','protected-port','overlap','foreign-image'])
def test_geometry_refuses_unsafe_merge(fault):
    a,b,target=profile([4,5],8202),profile([6,7],8203),profile([4,5,6,7],8202)
    if fault=='missing-card':target.cards=[4,5,6]
    if fault=='protected-card':a.cards=[0,5];target.cards=[0,5,6,7]
    if fault=='protected-port':target.instances[0].port=8200
    if fault=='overlap':b.cards=[5,6]
    if fault=='foreign-image':b.image='sha256:'+'b'*64
    with pytest.raises(ValueError):m.geometry([a,b],target,[0,1,2,3],[8200])


def test_exact_merge_geometry():
    m.geometry([profile([4,5],8202),profile([6,7],8203)],profile([4,5,6,7],8202),[0,1,2,3],[8200])


def test_unknown_journal_blocks_before_stop(tmp_path,monkeypatch):
    root=tmp_path/'journals/RUN/generation-v2';root.mkdir(parents=True)
    (root/('a'*64+'.intent.json')).write_text('{}')
    ap=tmp_path/'attempt.json';ap.write_text(json.dumps({'state':'ABORTED','run_id':'RUN','attempt_id':'generation-v2','actual':{'calls':1}}))
    monkeypatch.setattr(m.engine.settlement,'mapped',lambda *a:ap)
    s={'producer':{'mounts':[{'source':str(root.parent),'target':'/data/contact-journals/RUN','readonly':False}]}}
    with pytest.raises(ValueError,match='unknown or orphan'):m.custody(s,Path('/data'))
    assert len(list(root.iterdir()))==1


def test_lease_settlement_keeps_exact_bytes_and_guard(tmp_path):
    locks=tmp_path/'locks';locks.mkdir();out=tmp_path/'out';out.mkdir()
    guard=locks/'generation.guard';guard.touch();inode=guard.stat().st_ino
    doc={'pid':999999999,'session_id':'old','study_head':'head','run_id':'R','attempt_id':'a','token':'private'}
    blob=json.dumps(doc).encode();(locks/'flight.lock').write_bytes(blob)
    s={'locks':str(locks),'session':'old','producer':{'expected_commit':'head'},'name':'b'}
    proof={'custody':[{'run_id':'R','attempt_id':'a','contacts':[{}]}]}
    result=m.settle_lease(s,proof,out)
    assert result['observed_contacts']==1 and not (locks/'flight.lock').exists()
    assert Path(result['preserved_path']).read_bytes()==blob and guard.stat().st_ino==inode


def test_foreign_lease_not_removed(tmp_path):
    (tmp_path/'flight.lock').write_text(json.dumps({'pid':999999999,'session_id':'foreign','study_head':'head'}))
    s={'locks':str(tmp_path),'session':'old','producer':{'expected_commit':'head'},'name':'b'}
    with pytest.raises(ValueError,match='session differs'):m.settle_lease(s,{},tmp_path)
    assert (tmp_path/'flight.lock').exists()
