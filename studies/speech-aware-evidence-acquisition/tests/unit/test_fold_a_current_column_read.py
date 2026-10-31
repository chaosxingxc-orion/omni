"""Fold A diagnostic registration, source custody and score-only execution gates."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/analysis'))
import fold_a_current_column_read as c


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(c.reg.raw(value))


def commit(repo):
    subprocess.run(['git','add','.'],cwd=repo,check=True)
    subprocess.run(['git','commit','-qm','fixture'],cwd=repo,check=True)
    return c.clean(repo)


@pytest.fixture
def prepared(tmp_path):
    repo, source, data = (tmp_path/n for n in ('reader','source','data'))
    for p in (repo,source):
        p.mkdir();subprocess.run(['git','init','-q'],cwd=p,check=True)
        subprocess.run(['git','config','user.name','Test'],cwd=p,check=True)
        subprocess.run(['git','config','user.email','test@example.invalid'],cwd=p,check=True)
        (p/'docs').mkdir();(p/'docs/exposure-ledger.md').write_text('# Ledger\n')
    data.mkdir()
    ids = [f'row-{i}' for i in range(1080)]
    put(source/c.ROSTER, {'items':[{'sample_id':sid} for sid in ids],'split_identity_hash':'fold-a'})
    put(source/f'configs/experiment/{c.v.CELL}.json', {'cells':[{'run_id':c.v.RUN,
        'attempt_id':'generation-v1','readiness':{'state':'ready'}}],
        'populations':[{'ids':ids}], 'population':{'split_identity_hash':'fold-a'}})
    for p in (source/f'configs/baseline/{c.v.CELL}.json', source/c.reg.PKG/'analysis-recipe-v1.json'):
        put(p,{})
    script=source/'scripts/analysis/stage3_rollout_score.py';script.parent.mkdir(parents=True);script.write_text('# pinned scorer')
    head=commit(source);commit(repo)
    lock=data/'lock.json';put(lock,{'version':1})
    admitted,completed=data/'admitted.json',data/'completed.json'
    put(admitted,{'source_commit':head,'run_id':c.v.RUN,'rows':1080,'model_contacts':0,'inputs':{str(lock):c.reg.sha(lock)}})
    put(completed,{'commit':head,'run_id':c.v.RUN,'rows':1080,'score_surface':'sealed-final-delivery'})
    return repo,source,data,admitted,completed


def registered(prepared):
    repo,source,data,admitted,completed=prepared
    result=c.register(*prepared)
    assert set(result['files'])=={c.RECIPE,'docs/exposure-ledger.md'}
    commit(repo)
    return c.require_recipe(repo,data)


def test_register_real_ledger_and_commit(prepared):
    recipe=registered(prepared)
    assert recipe['roster']==c.ROSTER
    assert recipe['identity']['split_identity_hash']=='fold-a'
    assert recipe['official_complete_scoring_jobs']==6
    assert recipe['new_model_calls']==0 and not recipe['historical_fold_b_parity_applies']
    assert recipe['replicate_copies']==list(c.reader.pf.REPLICATE_SET)
    with pytest.raises(ValueError,match='already registered'):c.register(*prepared)


@pytest.mark.parametrize('fault',['rows','run','commit','score_surface','roster','lock','source_dirty'])
def test_bad_parent_refused_before_registration(prepared,fault):
    repo,source,data,admitted,completed=prepared
    if fault in ('rows','run','commit','score_surface'):
        doc=c.reg.read(completed)
        doc[{'run':'run_id'}.get(fault,fault)]={'rows':4,'run':'other','commit':'other','score_surface':'partial'}[fault]
        put(completed,doc)
    elif fault=='roster':
        doc=c.reg.read(source/c.ROSTER);doc['items'][0]['sample_id']=doc['items'][1]['sample_id'];put(source/c.ROSTER,doc)
    elif fault=='lock':put(data/'lock.json',{'version':2})
    else:(source/'unrelated').write_text('dirty')
    with pytest.raises(ValueError):c.register(*prepared)
    assert not (data/'analysis'/c.READ).exists()


@pytest.mark.parametrize('fault',['recipe','parent','implementation','ledger'])
def test_registered_pins_and_rules_cannot_drift(prepared,fault,monkeypatch):
    repo,source,data,admitted,completed=prepared
    registered(prepared)
    if fault=='parent':put(completed,{'changed':True})
    elif fault=='implementation':monkeypatch.setattr(c.reader.hist,'DRAWS',100)
    else:
        p=repo/(c.RECIPE if fault=='recipe' else 'docs/exposure-ledger.md')
        p.write_text(p.read_text().replace('six complete CPU','seven complete CPU')+'\n')
        commit(repo)
    if fault=='ledger':
        # Whitespace alone does not invalidate a valid ledger; remove the unique entry.
        p.write_text('\n'.join(line for line in p.read_text().splitlines() if '| '+c.READ+' |' not in line)+'\n');commit(repo)
    with pytest.raises((ValueError,KeyError)):c.require_recipe(repo,data)


def test_score_command_uses_fold_a_full_roster(prepared):
    repo,source,data,*_=prepared
    cmd=c.command(source,data,data/'root')
    for key,value in [('--roster',c.ROSTER),('--run',c.v.RUN),('--stage','all'),('--repo-root',str(source))]:
        assert cmd[cmd.index(key)+1]==value
    assert '--samples' not in cmd


@pytest.mark.parametrize('stage,fault',[('columns',None),('columns','subprocess'),('columns','verify'),('read',None),('read','verify')])
def test_stages_keep_failure_evidence_and_never_replay(prepared,monkeypatch,stage,fault):
    repo,source,data,*_=prepared;registered(prepared)
    root=data/'analysis'/c.READ
    if stage=='read':put(root/'columns.complete.json',{'rows':1080})
    monkeypatch.setattr(c.delivered,'build',lambda *a:{'inputs':{'final':'hash'}})
    real_run=subprocess.run;calls=[]
    def execute(args,**kw):
        if args[0]=='git':return real_run(args,**kw)
        assert kw['env']['PYTHONPATH']==str(source/'src')
        assert kw['env']['SAEA_ROLLOUT_TRACE_READ_ID']==c.READ
        calls.append(args)
        if fault=='subprocess':raise subprocess.CalledProcessError(1,args)
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(c.subprocess,'run',execute)
    def verify(*args):
        if fault=='verify':raise ValueError('bad scores')
        return {'score':'hash'}
    monkeypatch.setattr(c,'verify_columns',verify)
    monkeypatch.setattr(c,'summarize',lambda *a:{'rows':1080,'claims_system_gain':False})
    if fault:
        with pytest.raises((ValueError,subprocess.CalledProcessError)):c.run(repo,data,stage)
        assert (root/(stage+'.failed.json')).exists()
        assert not (root/(stage+'.complete.json')).exists()
    else:
        c.run(repo,data,stage)
        assert (root/(stage+'.complete.json')).exists()
    assert len(calls)==(1 if stage=='columns' else 0)
    with pytest.raises(ValueError,match='already started'):c.run(repo,data,stage)
    assert len(calls)==(1 if stage=='columns' else 0)


def test_read_does_not_start_before_complete_columns(prepared,monkeypatch):
    repo,source,data,*_=prepared;registered(prepared)
    monkeypatch.setattr(c.delivered,'build',lambda *a:{'inputs':{}})
    with pytest.raises(ValueError,match='complete column stage'):c.run(repo,data,'read')
    assert not (data/'analysis'/c.READ/'read.intent.json').exists()


@pytest.mark.parametrize('fault',[None,'roster','parent','missing','wrong_ids'])
def test_columns_verify_exact_roster_and_cached_source_binding(prepared,monkeypatch,fault):
    repo,source,data,*_=prepared;root=data/'analysis'/c.READ;directory=root/'columns'
    parent=data/'runs'/c.v.RUN/'generation-v1'/(c.v.RUN+'.run-manifest.json');put(parent,{'sealed':True})
    manifest={'rows':1080,'branch_order':list(c.reader.pf.ARM_ORDER),
        'stream_order':c.reader.hist.stream_column_names(),'copies':{},
        'source':{'run_id':c.v.RUN,'attempt_id':'generation-v1','manifest_path':str(parent),
            'manifest_sha256':c.reg.sha(parent),'roster_sha256':c.reg.sha(source/c.ROSTER)}}
    for name in c.NAMES:
        copy=directory/(name+'.jsonl');put(copy,{'test':'copy'})
        manifest['copies'][name]={'path':str(copy),'sha256':c.reg.sha(copy)}
        score=directory/c.reader.hist.COLUMN_SCORE_NAME.format(arm=name,scorer=c.reader.hist.SCORER)
        put(score,{'test':'score'});receipt=score.with_name(score.name+'.receipt.json');put(receipt,{'receipt':True})
    if fault=='roster':manifest['source']['roster_sha256']='other'
    if fault=='parent':manifest['source']['run_id']='other'
    if fault=='missing':receipt.unlink()
    put(directory/'render-manifest.json',manifest)
    ids=[r['sample_id'] for r in c.reg.read(source/c.ROSTER)['items']]
    if fault=='wrong_ids':ids[-1]='not-fold-a'
    calls=[]
    def cache(**kwargs):
        assert kwargs['cache_only'] is True and kwargs['parent_manifest']==parent
        assert kwargs['column_binding']['visit']=='first-solver-level2'
        assert kwargs['column_binding']['roster_sha256']==c.reg.sha(source/c.ROSTER)
        calls.append(kwargs)
        return {'per_sample':[{'sample_id':sid} for sid in ids], 'aggregate':{'identity_check':{'passed':True}}}
    monkeypatch.setattr(c,'score_complete_derived_copy',cache)
    if fault:
        with pytest.raises(ValueError):c.verify_columns(source,data,root)
    else:
        pins=c.verify_columns(source,data,root)
        assert len(calls)==6 and len(pins)==19


def test_queue_requires_completed_campaign_then_commits_registration(prepared,monkeypatch):
    repo,source,data,admitted,completed=prepared
    queue=data/'state.json';put(queue,{'schema':'saea-card-queue-state-v1','queue_id':'q','jobs':{}})
    with pytest.raises(ValueError,match='successful upstream'):
        c.queued(*prepared,queue,'q')
    assert not (data/'analysis'/c.READ).exists()
    put(queue,{'schema':'saea-card-queue-state-v1','queue_id':'q',
        'jobs':{'fold-a-current-campaign':{'status':'done','exit':0}}})
    stages=[]
    def run(r,d,stage):
        c.require_recipe(r,d);stages.append(stage);return {'stage':stage}
    monkeypatch.setattr(c,'run',run)
    assert c.queued(*prepared,queue,'q')=={'stage':'read'}
    assert stages==['columns','read']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()


def test_summary_runs_real_paired_statistics_without_fold_b_parity(tmp_path,monkeypatch):
    from test_l338_port_column_read import port_arms_scores
    ids=[f's{i}' for i in range(8)]
    scored=port_arms_scores(ids,gain=0)
    for name in c.NAMES:
        scored.setdefault(name,dict(scored['lock']))
    monkeypatch.setattr(c.reader.base,'roster_maps',lambda p:(ids,lambda sid:sid,lambda sid:'scenario'))
    monkeypatch.setattr(c.reader,'load_port_column_rows',lambda p:scored)
    result=c.summarize(tmp_path,tmp_path)
    assert result['rows']==8 and result['claims_system_gain'] is False
    assert set(result['contrast']['levels'])==set(c.NAMES)
    assert result['contrast']['versus_delivering']['aa']['slu_f1']['point_pp']==0
    assert 'per_scenario' in result and 'parity' not in result
