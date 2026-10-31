"""Admission queue gate tests: no stage starts without exact upstream custody."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/flights/nx'))
import fold_a_current_admit_queued as q


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


def test_queue_done_requires_successful_identity(tmp_path):
    state=tmp_path/'state.json'
    put(state, {'schema':'saea-card-queue-state-v1','queue_id':'q','jobs':{'prep':{'status':'done','exit':0}}})
    q.queue_done(state,'q','prep')
    for bad in ({'status':'running','exit':None},{'status':'failed','exit':1},{'status':'done','exit':1}):
        put(state, {'schema':'saea-card-queue-state-v1','queue_id':'q','jobs':{'prep':bad}})
        with pytest.raises(ValueError): q.queue_done(state,'q','prep')


def test_changed_detects_untracked_and_tracked(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init','-q'],cwd=repo,check=True)
    subprocess.run(['git','config','user.name','Test'],cwd=repo,check=True)
    subprocess.run(['git','config','user.email','test@example.invalid'],cwd=repo,check=True)
    (repo/'tracked').write_text('a');subprocess.run(['git','add','tracked'],cwd=repo,check=True)
    subprocess.run(['git','commit','-qm','base'],cwd=repo,check=True)
    assert q.changed(repo)==set()
    (repo/'tracked').write_text('b');(repo/'untracked').write_text('x')
    assert q.changed(repo)=={'tracked','untracked'}


def test_admit_rejects_incomplete_preparation_before_output(tmp_path, monkeypatch):
    repo,source,data=tmp_path/'repo',tmp_path/'source',tmp_path/'data'
    repo.mkdir();source.mkdir();data.mkdir()
    queue=data/'state.json';put(queue, {'schema':'saea-card-queue-state-v1','queue_id':'q','jobs':{'fold-a-current-prepare':{'status':'running','exit':None}}})
    prepared=data/'prepared.json';put(prepared, {'rows':1080,'model_contacts':0,'launchable':False,'source_commit':'head'})
    for p in (data/'lock.json',data/'binding.json') : put(p,{})
    with pytest.raises(ValueError,match='successful upstream'):
        q.admit(repo,source,data,data/'lock.json',data/'binding.json',prepared,queue,'q','base',data/'out')
    assert not (data/'out').exists()


@pytest.mark.parametrize('fault',[None,'registration','mint','supply','unrelated','drift'])
def test_complete_admission_chain_real_git(tmp_path,monkeypatch,fault):
    repo,source,data=[tmp_path/n for n in ('repo','source','data')]
    for p in (repo,source,data):p.mkdir()
    subprocess.run(['git','init','-q'],cwd=repo,check=True)
    for key,value in [('user.name','Test'),('user.email','test@example.invalid')]:
        subprocess.run(['git','config',key,value],cwd=repo,check=True)
    (repo/'seed').write_text('base')
    subprocess.run(['git','add','seed'],cwd=repo,check=True)
    subprocess.run(['git','commit','-qm','base'],cwd=repo,check=True)
    base=q.preparation.clean(repo)
    names={f'configs/{kind}/{cell}.json' for cell in (q.v.CELL,q.v.comparator.CELL) for kind in ('baseline','experiment')}
    names.update({q.v.PACKAGE+'/derivation.json',q.v.comparator.PKG+'/derivation.json'})
    for n in names:put(repo/n,{'held':True})
    head=q.preparation.commit_exact(repo,names,'prepare')
    # Source checkout clean check only; all generated mutations stay in repo.
    real_clean=q.preparation.clean
    monkeypatch.setattr(q.preparation,'clean',lambda tree,expected=None: 'source' if tree==source else real_clean(tree,expected))
    prepared,queue,lock,binding=[data/n for n in ('prepared.json','queue.json','lock.json','binding.json')]
    put(prepared,{'rows':1080,'model_contacts':0,'launchable':False,'source_commit':head})
    put(queue,{'schema':'saea-card-queue-state-v1','queue_id':'q','jobs':{'fold-a-current-prepare':{'status':'done','exit':0}}})
    put(lock,{});put(binding,{})
    real_run=subprocess.run;calls=[]
    def run(args,**kwargs):
        if args[0]=='git':return real_run(args,**kwargs)
        assert kwargs['env']['PYTHONPATH'] == str(repo/'src')
        stage=Path(args[1]).stem.removeprefix('fold_a_current_');calls.append(stage)
        if stage==fault:return subprocess.CompletedProcess(args,2)
        if stage=='registration':
            n=f'configs/experiment/{q.v.CELL}.json';put(repo/n,{'registered':True})
            archive=Path(args[args.index('--archive')+1]);put(archive/'intent.json',{'outputs':{n:q.sha(repo/n)}})
        elif stage=='mint':put(data/q.v.DECISION_PATH,{'status':'decided'})
        else:
            put(repo/f'configs/baseline/{q.v.CELL}.json',{'supply':'own'})
            if fault=='unrelated':(repo/'unrelated').write_text('preserve')
            if fault=='drift':put(binding,{'changed':True})
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(q.subprocess,'run',run)
    out=data/'out';args=(repo,source,data,lock,binding,prepared,queue,'q',base,out)
    if fault:
        with pytest.raises(ValueError):q.admit(*args)
        assert (out/'failed.json').exists() and not (out/'complete.json').exists()
    else:
        result=q.admit(*args)
        assert result['source_commit']==real_clean(repo)
        assert q.read(out/'complete.json')['launchable'] is False
    expected={'registration':['registration'],'mint':['registration','mint']}.get(fault,['registration','mint','supply'])
    assert calls==expected
    before=list(calls)
    with pytest.raises(ValueError):q.admit(*args)
    assert calls==before


@pytest.mark.parametrize('fault',[None,'input','head','upstream'])
def test_dispatch_requires_admitted_commit_and_inputs(tmp_path,monkeypatch,fault):
    repo,data=tmp_path/'repo',tmp_path/'data';repo.mkdir();data.mkdir()
    queue,admitted,lock=[data/n for n in ('queue.json','admitted.json','lock.json')]
    put(queue,{'schema':'saea-card-queue-state-v1','queue_id':'q','jobs':{'fold-a-current-admit':{'status':'done','exit':0}}})
    put(lock,{});put(admitted,{'source_commit':'ready','run_id':q.v.RUN,'rows':1080,'model_contacts':0,'inputs':{str(lock):q.sha(lock)}})
    if fault=='input':put(lock,{'changed':True})
    if fault=='upstream':put(queue,{'schema':'saea-card-queue-state-v1','queue_id':'q','jobs':{}})
    def clean(*args):
        if fault=='head':raise ValueError('changed commit')
    calls=[]
    monkeypatch.setattr(q.preparation,'clean',clean)
    def run(args, **kwargs):
        assert kwargs['env']['PYTHONPATH'] == str(repo/'src')
        calls.append(args)
    monkeypatch.setattr(q.subprocess,'run',run)
    if fault:
        with pytest.raises(ValueError):q.dispatch(repo,data,lock,admitted,queue,'q',data/'campaign')
        assert not calls
    else:
        q.dispatch(repo,data,lock,admitted,queue,'q',data/'campaign')
        assert len(calls)==1 and calls[0][1].endswith('fold_a_current_campaign.py')
        assert calls[0][-2:]==['--source-commit','ready']
