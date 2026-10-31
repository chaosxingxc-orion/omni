from pathlib import Path
import copy,json,sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
import tfrl_h1_union as m


def plans():
    parent={'run_id':'old','roster':{},'budget':{},'decoding':{'temperature':0.7},'row_plan':[
        {'row_id':str(i),'contacts':[{'contact_id':'old'+str(i),'request_identity_sha256':'old','seed':i}]} for i in range(313)]}
    new=copy.deepcopy(parent);new.update(run_id='new',roster={'rows':310},budget={'calls':2170})
    new['row_plan']=new['row_plan'][3:]
    for row in new['row_plan']:
        row['contacts'][0].update(contact_id='new'+row['row_id'],request_identity_sha256='new')
    return parent,new,['0','1','2']


def test_exact_partition_preserves_seed_and_rejects_policy_or_population_changes():
    p,n,k=plans();m.validate_plans(p,n,k)
    for mutation in ('seed','decoding','duplicate','overlap'):
        changed=copy.deepcopy(n)
        if mutation=='seed':changed['row_plan'][0]['contacts'][0]['seed']+=1
        if mutation=='decoding':changed['decoding']['temperature']=0.1
        if mutation=='duplicate':changed['row_plan'][-1]=changed['row_plan'][0]
        if mutation=='overlap':changed['row_plan'][0]['row_id']='0'
        with pytest.raises(ValueError,match='partition'):m.validate_plans(p,changed,k)


def test_union_never_drops_affected_rows_or_averages_subsets():
    p,n,k=plans();ids=[r['row_id'] for r in p['row_plan']]
    old={sid:{'sample_id':sid} for sid in k};new={r['row_id']:{'sample_id':r['row_id']} for r in n['row_plan']}
    rows=m.complete_union(ids,old,list(new),new)
    assert len(rows)==313
    new.pop('312')
    with pytest.raises(ValueError):m.complete_union(ids,old,[r['row_id'] for r in n['row_plan']],new)


def test_score_intent_refuses_ambiguous_replay_and_changed_source(tmp_path,monkeypatch):
    import speech_aware_evidence_acquisition.reproduction.slurp_official_eval as official
    import tfrl_current_half_result as half
    calls=[]
    def score(*a):calls.append(1);return {'verified':True}
    monkeypatch.setattr(official,'score_slurp_outputs_official_train',score)
    monkeypatch.setattr(half,'score_table',lambda result,ids:None)
    p=tmp_path/'outputs.jsonl';p.write_text('{}\n');out=tmp_path/'scores.json'
    kw=dict(binding={'source':'a'},ids=['one'],lock={},data=tmp_path)
    assert m.score_column(p,out,**kw)=={'verified':True}
    assert m.score_column(p,out,**kw)=={'verified':True};assert len(calls)==1
    with pytest.raises(ValueError,match='receipt'):m.score_column(p,out,**{**kw,'binding':{'source':'b'}})
    out.with_suffix('.receipt.json').unlink()
    with pytest.raises(ValueError,match='incomplete'):m.score_column(p,out,**kw)
    assert len(calls)==1


def test_union_verification_never_starts_missing_official_score(tmp_path,monkeypatch):
    import speech_aware_evidence_acquisition.reproduction.slurp_official_eval as official
    monkeypatch.setattr(official,'score_slurp_outputs_official_train',lambda *a:pytest.fail('must not score'))
    p=tmp_path/'outputs.jsonl';p.write_text('{}\n')
    with pytest.raises(ValueError,match='receipt missing'):
        m.score_column(p,tmp_path/'scores.json',binding={},ids=['one'],lock={},data=tmp_path,verify_only=True)
    assert not (tmp_path/'scores.intent.json').exists()


def test_distillation_source_namespaces_and_exact_full_population(monkeypatch):
    import tfrl_current_distill as d
    def extract(rows,plan,resolver):
        sid=plan['row_plan'][0]['row_id']
        return {(sid,'s1'):{'answer':sid}},resolver,[{'row_id':sid,'request_id':'same-local-request'}]
    monkeypatch.setattr(d,'sealed_answers',extract)
    loaded={'plan':{'row_plan':[{'row_id':'a'},{'row_id':'b'}]},'segments':[
        {'namespace':'old','plan':{'row_plan':[{'row_id':'a'}]},'rows':object()},
        {'namespace':'new','plan':{'row_plan':[{'row_id':'b'}]},'rows':object()}]}
    answers,_,trace=d.answers_for_loaded(loaded,resolver=object())
    assert set(answers)=={('a','s1'),('b','s1')}
    assert [r['source_namespace'] for r in trace]==['old','new']
    loaded['segments'][1]['plan']['row_plan'][0]['row_id']='a'
    with pytest.raises(ValueError,match='repeats'):d.answers_for_loaded(loaded,resolver=object())
