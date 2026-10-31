"""Queued preparation verifies the real transform and commits only its outputs."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/flights/nx'))
import fold_a_current_prepare_queued as q
from test_fold_a_current_comparator import source


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(q.c.raw(value))


def init(repo):
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    for key, value in [('user.name', 'Test'), ('user.email', 'test@example.invalid')]:
        subprocess.run(['git', 'config', key, value], cwd=repo, check=True)
    (repo/'seed').write_text('seed')
    subprocess.run(['git', 'add', 'seed'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-qm', 'seed'], cwd=repo, check=True)
    return q.clean(repo)


@pytest.mark.parametrize('fault', [None, 'untracked', 'staged', 'tracked'])
def test_exact_commit_real_git(tmp_path, fault):
    repo = tmp_path/'repo'; before = init(repo)
    put(repo/'owned/file.json', {'owned': True})
    if fault == 'tracked': (repo/'seed').write_text('changed')
    elif fault:
        (repo/'unrelated').write_text('preserve')
        if fault == 'staged': subprocess.run(['git','add','unrelated'],cwd=repo,check=True)
    if fault:
        with pytest.raises(ValueError, match='unexpected changes'):
            q.commit_exact(repo, ['owned/file.json'], 'owned')
        assert q.git(repo,'rev-parse','HEAD') == before
    else:
        assert q.commit_exact(repo, ['owned/file.json'], 'owned') != before
        assert q.git(repo,'show','--format=','--name-only','HEAD') == 'owned/file.json'


@pytest.fixture
def comparator(source, tmp_path, monkeypatch):
    base, exp, _ = source
    repo, src, producer, data = [tmp_path/n for n in ('repo','source','producer','data')]
    shutil.copytree(REPO/'configs', repo/'configs')
    c=q.c
    for n in (c.inputs.ROSTER,'docs/results/2026-09-30-slurp-a2-comp1b-dev2000-read.md','scripts/flights/nx/l338_port_pins.json'):
        target=repo/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/n,target)
    spec=c.read(REPO/c.SPEC);put(src/c.SPEC,spec)
    b=f"configs/baseline/{spec['baseline']}.json";e=f"configs/experiment/{spec['cell']}.json"
    m='configs/model/'+exp['cells'][0]['model_fragment']+'.json'
    put(src/b,base);put(src/e,exp)
    (src/m).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/m,src/m)
    pins={}
    for n in (c.inputs.RECEIPT,c.read(repo/c.inputs.PROTOCOL)['query_rows_relpath'],c.inputs.ALIAS_RECEIPT,c.inputs.SEGMENTS,c.inputs.CONTENT,c.inputs.PROVENANCE):
        put(data/n,{'fixture':n});pins[n]=c.sha(data/n)
    sd=data/'parent/complete.json';rd=data/'retrieval/complete.json'
    put(sd,{'rows':1080,'score_surface':'sealed-final-delivery'})
    put(rd,{'rows':1080,'values_unchanged':True,'pins':pins})
    binding={'source_commit':'source-head','source_run':spec['run_id'],'queue_state':'queue.json'}
    put(data/'queue.json',{'jobs':{'fold-a-comparator':{'status':'done','exit':0}}})
    outputs={f'configs/{k}/{c.CELL}.json':c.raw(v) for k,v in zip(('baseline','experiment'),c.derive(repo,base,exp,pins))}
    record={'parent_binding':binding,'source_commit':'source-head','source_run':spec['run_id'],
        'source_files':{n:c.sha(src/n) for n in (b,e,m)},'agent_completion_sha256':c.sha(sd),
        'retrieval_completion_sha256':c.sha(rd),'allowed_execution_stages':['check'],'model_contacts':0,
        'claims_system_lock':False,'output_hashes':{n:hashlib.sha256(v).hexdigest() for n,v in outputs.items()}}
    complete=data/'comparator/complete.json';put(complete,record)
    outputs[c.PKG+'/derivation.json']=c.raw(record)
    for n,blob in outputs.items():
        (producer/n).parent.mkdir(parents=True,exist_ok=True);(producer/n).write_bytes(blob)
    monkeypatch.setattr(q,'clean',lambda *a:'source-head')
    monkeypatch.setattr(q,'git',lambda *a:'producer-head')
    monkeypatch.setattr(c,'bound_parents',lambda *a:(src/c.SPEC,sd,rd))
    return repo,src,producer,data,binding,complete,outputs


@pytest.mark.parametrize('fault',[None,'queue','vector','output-and-hash','producer-bytes','existing'])
def test_import_rebuilds_actual_transform_before_writes(comparator,fault):
    repo,src,producer,data,binding,complete,outputs=comparator;c=q.c
    if fault=='queue':put(data/'queue.json',{'jobs':{'fold-a-comparator':{'status':'running'}}})
    elif fault=='vector':(data/c.inputs.CONTENT).write_text('changed')
    elif fault=='output-and-hash':
        record=c.read(complete);n=next(iter(record['output_hashes']));record['output_hashes'][n]='a'*64;put(complete,record)
    elif fault=='producer-bytes':(producer/next(iter(outputs))).write_text('{}')
    elif fault=='existing':put(repo/next(iter(outputs)),{})
    args=(repo,src,producer,data,binding,complete,'producer-head')
    if fault:
        with pytest.raises(ValueError):q.verified_comparator(*args)
    else:assert q.verified_comparator(*args)==outputs


@pytest.mark.parametrize('failure',[None,'agreement','unrelated'])
def test_queue_pipeline_real_commits_no_model_calls(tmp_path,monkeypatch,failure):
    repo,src,producer,data=[tmp_path/n for n in ('repo','source','producer','data')]
    head=init(repo);src.mkdir();producer.mkdir();data.mkdir()
    binding=data/'binding.json';complete=data/'comparator.json';lock=data/'lock.json'
    for p in (binding,complete,lock):put(p,{})
    imported={'configs/comparator.json':q.c.raw({'held':True})}
    monkeypatch.setattr(q,'verified_comparator',lambda *a:imported)
    calls=[]
    def agreement(repo,data,lock,completion,out,head):
        q.clean(repo,head);calls.append('agreement')
        if failure=='agreement':raise ValueError('builder failed')
        put(out/'complete.json',{'rows':1080})
    def derive(repo,*args):
        calls.append('derive')
        for kind in ('baseline','experiment'):put(repo/f'configs/{kind}/{q.evaluation.CELL}.json',{'held':True})
        put(repo/q.evaluation.PACKAGE/'derivation.json',{})
        if failure=='unrelated':(repo/'unrelated').write_text('preserve')
    monkeypatch.setattr(q.evaluation.agreement,'prepare',agreement)
    monkeypatch.setattr(q.evaluation,'write',derive)
    out=data/'out';args=(repo,src,producer,data,lock,binding,complete,out,head,'producer')
    if failure:
        with pytest.raises(ValueError):q.execute(*args)
        assert (out/'failed.json').exists() and not (out/'complete.json').exists()
    else:
        result=q.execute(*args)
        assert result['launchable'] is False and result['model_contacts']==0
        assert q.git(repo,'rev-list','--count','HEAD')=='3'
    before=list(calls)
    with pytest.raises(ValueError):q.execute(*args)
    assert calls==before
