"""Prove handoff ordering without server operations or inference."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

REPO=Path(__file__).resolve().parents[2]
HOST=REPO.parents[1]/'scripts/npu'
if not (HOST/'a0_selected_handoff.py').exists():
    pytest.skip('public umbrella host driver required',allow_module_level=True)
sys.path.insert(0,str(HOST))
import a0_selected_handoff as m


@pytest.fixture
def setup(tmp_path,monkeypatch):
    data=tmp_path/'data';tree=data/'tree';source=data/'source'
    study=tree/'studies/speech-aware-evidence-acquisition'
    pkg=study/'docs/readiness';(pkg/m.NAME).mkdir(parents=True);(pkg/m.PREVIOUS).mkdir()
    (pkg/m.NAME/'candidate.json').write_text(json.dumps({'profile':'tp4'}))
    (pkg/m.PREVIOUS/'candidate.json').write_text(json.dumps({'profile':'tp8'}))
    terminal=data/'work/codex/real500-successor-queue-v1'/m.PREVIOUS/'run.exit'
    terminal.parent.mkdir(parents=True);terminal.write_text('0')
    selection=data/'selection.json';selection.write_text(json.dumps({'selected_run_id':'SAEA-SLURP-REAL500-TP4-S128-V1'}))
    monkeypatch.setattr(m,'D',data);monkeypatch.setattr(m,'U',data/'public')
    operations=[]
    def git(study,*args):
        operations.append(('git',args))
        return '' if 'status' in args else 'commit'
    def formal(tree,args,log,profile=None):operations.append(('formal',args))
    def call(args,log,env=None):operations.append(('call',args))
    monkeypatch.setattr(m,'git',git);monkeypatch.setattr(m,'formal',formal);monkeypatch.setattr(m,'call',call)
    monkeypatch.setattr(m,'capture',lambda *a:operations.append(('capture',[])))
    return tree,source,selection,data/'work/codex/handoff',operations,terminal


def test_select_before_switch_and_preflight_before_dispatch(setup):
    tree,source,selection,out,ops,_=setup
    m.handoff(tree,source,selection,out)
    names=[(' '.join(args) if kind!='capture' else 'capture') for kind,args in ops]
    index=lambda token:next(i for i,line in enumerate(names) if token in line)
    assert index('bind_a0_selected_profile')<index('topology_switch.py')<index('capture')
    assert index('refresh_a0_resident_binding')<index('register_real_engine_resident')<index('admit_real_engine_candidate')
    assert index('--read-only')<index('run_real_engine_candidate')
    assert json.loads((out/'status.json').read_text())['state']=='complete353_full1080_score_pending'
    with pytest.raises(ValueError,match='fresh owned'):m.handoff(tree,source,selection,out)


def test_failed_binding_never_stops_resident(setup,monkeypatch):
    tree,source,selection,out,ops,_=setup
    def fail(*a,**kw):raise ValueError('bad complete selection')
    monkeypatch.setattr(m,'formal',fail)
    with pytest.raises(ValueError):m.handoff(tree,source,selection,out)
    assert not any(kind in ('call','capture') for kind,_ in ops)
    assert json.loads((out/'status.json').read_text())['automatic_retry'] is False


def test_incomplete_predecessor_never_mutates(setup):
    tree,source,selection,out,ops,terminal=setup
    terminal.unlink()
    with pytest.raises(ValueError,match='complete successful'):m.handoff(tree,source,selection,out)
    assert not out.exists()
