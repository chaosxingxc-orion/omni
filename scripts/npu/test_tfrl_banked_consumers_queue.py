import pytest
from tfrl_banked_consumers_queue import ready,identities,require_live_upstream


def results(iteration):
    return {h:{'run_id':f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}-{h.upper()}-V1',
        'attempt_id':'generation-v1','iteration':iteration,'rows':n,
        'official_scoring_jobs':7,'new_model_calls':0} for h,n in [('h1',313),('h2',312)]}


@pytest.mark.parametrize('iteration',[2,3])
def test_queue_only_accepts_matching_complete_halves(iteration):
    r=results(iteration);assert ready(r,iteration)
    r['h1']=None;assert not ready(r,iteration)
    r['h2']['iteration']=1
    with pytest.raises(ValueError):ready(r,iteration)


@pytest.mark.parametrize('value',[1,4,True])
def test_queue_rejects_nonbanked_iteration(value):
    with pytest.raises(ValueError):identities(value)


def test_verified_wait_requires_live_handle():
    calls=[]
    require_live_upstream({'state':'generation.log','pid':123},lambda p,s:calls.append((p,s)))
    assert calls==[(123,0)]
    for state in ('complete','failed'):
        with pytest.raises(ValueError):require_live_upstream({'state':state,'pid':123})
    def missing(pid,signal):raise ProcessLookupError()
    with pytest.raises(ValueError,match='process missing'):
        require_live_upstream({'state':'generation.log','pid':123},missing)


def test_queue_launches_registered_consumer_once_after_results(tmp_path,monkeypatch):
    import json,hashlib
    import tfrl_banked_consumers_queue as m
    monkeypatch.setattr(m,'D',tmp_path)
    pool,bank,rel=m.identities(2);tree=tmp_path/'tree';study=tree/'studies/speech-aware-evidence-acquisition'
    p=study/rel;p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'iteration':2,'bank_run_id':bank,'sources':{
        h:{'repo_relpath':h} for h in ('h1','h2')}}))
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    for half,result in results(2).items():
        path=tmp_path/'analysis'/(result['run_id']+'-READ')/'complete.json'
        path.parent.mkdir(parents=True);path.write_text(json.dumps(result))
    def git(command,**kw):
        return '' if command[-1]=='--porcelain' else str(study) if command[-1]=='--show-toplevel' else 'a'*40
    monkeypatch.setattr(m.subprocess,'check_output',git)
    calls=[]
    class Child:
        pid=999
        def wait(self):
            path=tmp_path/'analysis'/bank/'complete.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'rows':625,'recipe_sha256':digest,
                'accounting':{'generation_calls':0,'unresolved_tokenizer_requests':0}}))
            return 0
    def launch(command,**kwargs):
        calls.append((command,kwargs));return Child()
    monkeypatch.setattr(m.subprocess,'Popen',launch)
    out=tmp_path/'work/codex/queue'
    m.run(tree,tmp_path/'q1',tmp_path/'q2',out,2)
    assert len(calls)==1 and calls[0][0][-2:]==['--iteration','2']
    assert calls[0][1]['start_new_session']
    assert json.loads((out/'status.json').read_bytes())['state']=='complete'
    with pytest.raises(ValueError,match='fresh owned'):
        m.run(tree,tmp_path/'q1',tmp_path/'q2',out,2)
    assert len(calls)==1
