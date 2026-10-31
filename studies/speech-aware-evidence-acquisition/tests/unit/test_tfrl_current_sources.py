import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
import tfrl_current_sources as m
from test_tfrl_current_half_result import result


def put(root,rel,obj):
    p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return p


@pytest.fixture
def source_fixture(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition.core import bundle,config
    source=tmp_path/'source';data=tmp_path/'data';run='SAEA-SLURP-TFRL-CURRENT-SEED-I1-H1-V1';n=313
    ids=[f'fixture:{i}' for i in range(n)];items=[{'sample_id':sid} for sid in ids]
    rp=put(source,'roster.json',{'items':items,'split_identity_hash':'split'})
    plan={'half':'R01C-H1','run_id':run,'attempt_id':'generation-v1','iteration':1,'row_plan':[{'row_id':i} for i in ids],
        'roster':{'file_sha256':m.sha(rp),'split_identity_hash':'split'}}
    pp=put(source,'plan.json',plan)
    recipe={'run_id':run,'attempt_id':'generation-v1','rows':n,'split_identity_hash':'split',
        'sampling_plan':{'relpath':'plan.json','sha256':m.sha(pp)},'roster':{'relpath':'roster.json','sha256':m.sha(rp)}}
    recipe_path=put(source,'recipe.json',recipe);recipe_pin={'relpath':'recipe.json','sha256':m.sha(recipe_path)}
    monkeypatch.setattr(m,'validate_recipe',lambda *a:recipe)
    mp=put(data,f'runs/{run}/generation-v1/{run}.run-manifest.json',{'identity':'fixture'})
    b=SimpleNamespace(attempt_state='FINALIZED',manifest={'status':'COMPLETED','run_id':run,'attempt':{'attempt_id':'generation-v1'},
        'config':{'config_hash':'cfg','fragments':dict.fromkeys(('model','dataset','baseline','experiment'),'fragment')},'outputs':{'sha256':'output'}},
        outputs_rows=[],trace_records={})
    monkeypatch.setattr(bundle,'verify_bundle',lambda _:b)
    monkeypatch.setattr(config,'compose',lambda *a:SimpleNamespace(config_hash='cfg'))
    check={'rows':n};monkeypatch.setattr(m.audit,'verify',lambda *a:check)
    directory=data/'analysis'/(run+'-READ');columns=directory/'columns';copies={};declarations=[]
    for arm in m.ARMS:
        p=columns/f'arm-{arm}-first-visit-outputs.jsonl';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}\n')
        copies[arm]={'path':str(p),'sha256':m.sha(p)};declarations.append({'arm_id':arm})
    render_path=put(columns,'render-manifest.json',{'source':{'manifest_sha256':m.sha(mp),'roster_sha256':m.sha(rp),'config_hash':'cfg'},
        'rows':n,'arm_order':list(m.ARMS),'copies':copies,'arm_declarations':declarations})
    vote=columns/'arm-vote-outputs.jsonl';vote.write_text('{}\n');results={}
    for arm in (*m.ARMS,'vote'):
        r=result();r['per_sample']=[dict(r['per_sample'][0],sample_id=sid) for sid in ids];r['aggregate']['samples']=n
        results[arm]=r;sp=put(columns,f'arm-{arm}-{m.SCORER}-scores.json',{m.SCORER:r})
        cp=vote if arm=='vote' else Path(copies[arm]['path'])
        column={'arm':arm,'plan_sha256':m.sha(pp),'render_manifest_sha256':m.sha(render_path),'recipe_sha256':m.sha(recipe_path),
            'rule':'plurality intent; support>=2/4 entities; lowest-k intent tie'} if arm=='vote' else {
            'arm':arm,'arm_declaration':{'arm_id':arm},'render_manifest_sha256':m.sha(render_path),'roster_sha256':m.sha(rp),'visit':'first-solver'}
        receipt={'schema':'saea-derived-complete-score-receipt-v1','parent_manifest':str(mp.resolve()),'parent_manifest_sha256':m.sha(mp),
            'parent_run_id':run,'parent_attempt_id':'generation-v1','parent_outputs_sha256':'output','roster_count':n,
            'roster_ids_sha256':hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest(),
            'outputs_copy':str(cp.resolve()),'outputs_copy_sha256':m.sha(cp),'scorer_id':m.SCORER,'column_binding':column,'result_sha256':m.sha(sp)}
        put(columns,sp.name+'.receipt.json',receipt)
    credit,summary=m.analyze(plan,results)
    cp=directory/'credit.jsonl';cp.write_text('\n'.join(json.dumps(r) for r in credit)+'\n')
    complete={**summary,'schema':'saea-tfrl-current-half-result-v1','rows':n,'run_id':run,'attempt_id':'generation-v1','half':'R01C-H1',
        'iteration':1,'recipe_sha256':m.sha(recipe_path),'plan_sha256':m.sha(pp),'manifest_sha256':m.sha(mp),'credit_sha256':m.sha(cp),
        'new_model_calls':0,'official_scoring_jobs':7,'request_audit':check,
        'artifacts':{str(p.relative_to(directory)):m.sha(p) for p in columns.iterdir()}}
    put(directory,'complete.json',complete)
    return source,data,recipe_pin,directory,b


@pytest.mark.parametrize('fault',[None,'score','receipt','missing','credit','summary','unfinished','request','escape'])
def test_complete_source_revalidation(source_fixture,fault,monkeypatch):
    source,data,pin,directory,b=source_fixture
    if fault in ('score','receipt','missing'):
        rel=f'columns/arm-s1-{m.SCORER}-scores.json'+('.receipt.json' if fault=='receipt' else '')
        p=directory/rel;c=m.read(directory/'complete.json')
        if fault=='missing':c['artifacts'].pop(rel)
        else:
            v=m.read(p)
            if fault=='receipt':v['parent_run_id']='other'
            else:v[m.SCORER]['aggregate']['official']['slu_f1']=.5
            p.write_text(json.dumps(v));c['artifacts'][rel]=m.sha(p)
        put(directory,'complete.json',c)
    elif fault=='credit':
        p=directory/'credit.jsonl';p.write_text('[]\n');c=m.read(directory/'complete.json');c['credit_sha256']=m.sha(p);put(directory,'complete.json',c)
    elif fault=='summary':
        c=m.read(directory/'complete.json');c['vote_minus_lock_pp']=99;put(directory,'complete.json',c)
    elif fault=='unfinished':b.attempt_state='OPENED'
    elif fault=='request':
        def reject(*a):raise ValueError('request differs')
        monkeypatch.setattr(m.audit,'verify',reject)
    elif fault=='escape':
        c=m.read(directory/'complete.json');c['artifacts']['../outside']='0'*64;put(directory,'complete.json',c)
    if fault:
        with pytest.raises(ValueError):m.load_half(source,data,'h1',pin)
    else:
        loaded=m.load_half(source,data,'h1',pin)
        assert len(loaded['credit'])==313 and loaded['source']['run_id']==b.manifest['run_id']


def test_pool_refuses_sources_before_any_output(tmp_path,monkeypatch):
    import tfrl_current_pool_read as p
    import flight_registration_check as reg
    from speech_aware_evidence_acquisition import contracts
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    recipe={'schema':'saea-tfrl-current-pool-read-recipe-v1','read_run_id':p.RUN,'rows':625,'new_model_calls':0,'official_scoring_jobs':0,
        'threshold_pp':1.0,'claims_benefit':False,'next_iteration_authorized':False,'source_recipes':{'h1':{},'h2':{}}}
    path=put(tmp_path,p.REL,recipe)
    monkeypatch.setattr(reg,'parse_ledger',lambda *a:{p.RUN:[{'artifacts':p.REL,'notes':p.sha(path)}]})
    monkeypatch.setattr(reg,'is_countersigned',lambda *a:True)
    def refuse(*a):raise ValueError('incomplete half')
    monkeypatch.setattr(p,'load_half',refuse)
    with pytest.raises(ValueError,match='incomplete'):p.run(tmp_path,tmp_path,tmp_path)
    assert not (tmp_path/'analysis').exists()
