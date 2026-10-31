import copy
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
import tfrl_current_half_result as m


def result(correct=True):
    row={'sample_id':'row','scenario_action_exact':int(correct)}
    for kind in ('word','char'):
        row.update({f'slu_f1_{kind}_tp':int(correct),f'slu_f1_{kind}_fp':int(not correct),f'slu_f1_{kind}_fn':int(not correct)})
    return {'per_sample':[row],'aggregate':{'samples':1,'identity_check':{'passed':True},'official':{'slu_f1':float(correct)},
        'evaluator_commit':'official','scorer_version':'v1','split_id':'train','projection':None,'filler_normalization':None}}


@pytest.mark.parametrize('fault',[None,'missing_arm','duplicate','bad_count','nan','scorer','intent','micro','official_nan'])
def test_full_half_credit_requires_all_official_columns(fault):
    plan={'iteration':1,'row_plan':[{'row_id':'row'}]}
    results={a:result(a not in ('aa','lock')) for a in (*m.ARMS,'vote')}
    if fault=='missing_arm':results.pop('s4')
    elif fault=='duplicate':results['s1']['per_sample']*=2
    elif fault=='bad_count':results['s1']['per_sample'][0]['slu_f1_word_tp']=-1
    elif fault=='nan':results['s1']['per_sample'][0]['slu_f1_word_tp']=float('nan')
    elif fault=='scorer':results['s1']['aggregate']['evaluator_commit']='other'
    elif fault=='intent':results['s1']['per_sample'][0]['scenario_action_exact']=.4
    elif fault=='official_nan':results['s1']['aggregate']['official']['slu_f1']=float('nan')
    elif fault=='micro':results['s1']['aggregate']['official']['slu_f1']=.5
    if fault:
        with pytest.raises(ValueError):m.analyze(plan,results)
    else:
        credit,summary=m.analyze(plan,results)
        assert credit[0]['positive'] and summary['vote_minus_lock_pp']==100
        assert summary['next_iteration_authorized'] is False and summary['claims_benefit'] is False


def test_unstable_twin_prevents_positive_credit():
    results={a:result(a!='lock') for a in (*m.ARMS,'vote')}
    credit,_=m.analyze({'iteration':1,'row_plan':[{'row_id':'row'}]},results)
    assert not credit[0]['positive'] and credit[0]['reason']=='baseline-unstable'


def test_vote_uses_real_copy_answers_and_rejects_tamper(tmp_path):
    manifest={'rows':1,'copies':{}}
    for i,arm in enumerate(m.ARMS[:4]):
        answer={'scenario':'alarm' if i<2 else 'calendar','action':'set','entities':[{'type':'time','filler':'six'}] if i<2 else []}
        path=tmp_path/(arm+'.jsonl');path.write_text(json.dumps({'sample_id':'row','text':json.dumps(answer)})+'\n')
        manifest['copies'][arm]={'path':str(path),'sha256':m.sha(path)}
    vote=m.votes_from_copies(manifest)
    assert json.loads(vote[0]['text'])=={'scenario':'alarm','action':'set','entities':[{'type':'time','filler':'six'}]}
    path.write_text('changed')
    with pytest.raises(ValueError,match='copy changed'):m.votes_from_copies(manifest)


def test_request_audit_failure_precedes_scoring(tmp_path,monkeypatch):
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core import bundle,scoring_batch
    from types import SimpleNamespace
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    plan={'run_id':'run','attempt_id':'attempt','roster':{'rows':313,'split_identity_hash':'a'*64}}
    rp=tmp_path/'roster.json';rp.write_text('{}');plan['roster']['file_sha256']=m.sha(rp)
    pp=tmp_path/'plan.json';pp.write_text(json.dumps(plan));recipe={'run_id':'run','attempt_id':'attempt','rows':313,'split_identity_hash':'a'*64,
        'sampling_plan':{'relpath':'plan.json','sha256':m.sha(pp)},'roster':{'relpath':'roster.json','sha256':m.sha(rp)}}
    monkeypatch.setattr(m,'validate_recipe',lambda *a:recipe)
    monkeypatch.setattr(bundle,'verify_bundle',lambda *a:SimpleNamespace(attempt_state='FINALIZED',
        manifest={'status':'COMPLETED','run_id':'run','attempt':{'attempt_id':'attempt'}},outputs_rows=[],trace_records={}))
    def refuse(*a):raise ValueError('request mismatch')
    monkeypatch.setattr(m.audit,'verify',refuse)
    def no_score(**kw):pytest.fail('scorer called before request audit')
    monkeypatch.setattr(scoring_batch,'score_complete_derived_copy',no_score)
    out=tmp_path/'analysis/result'
    with pytest.raises(ValueError,match='request mismatch'):m.run(tmp_path,tmp_path,tmp_path/'recipe.json',out)
    assert not out.exists()


@pytest.mark.parametrize('fault',[None,'plan','spent','not_held','duplicate'])
def test_read_registration_binds_both_halves_without_admission(tmp_path,monkeypatch,fault):
    import tfrl_current_read_registration as reg
    from speech_aware_evidence_acquisition import contracts
    from speech_aware_evidence_acquisition.core.bundle import git_commit
    monkeypatch.setattr(contracts,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(git_commit,'require_clean_study_repo',lambda _: 'a'*40)
    repo=tmp_path/'repo';data=tmp_path/'data';(repo/'docs').mkdir(parents=True);(repo/'docs/exposure-ledger.md').write_text('# Ledger\n')
    hashes={}
    for half,n in [('h1',313),('h2',312)]:
        name=f'slurp-tfrl-current-seed-i1-{half}-v1';run=f'RUN-{half}';pkg=repo/'docs/readiness'/name;pkg.mkdir(parents=True)
        rp=repo/(half+'.json');rp.write_text('{}')
        plan={'run_id':run,'roster':{'rows':n,'file_sha256':reg.sha(rp)}}
        if fault=='plan' and half=='h2':plan['roster']['rows']=1
        (pkg/'sampling-plan.json').write_text(json.dumps(plan))
        e={'population':{'count':n,'roster_receipt':half+'.json','split_identity_hash':'b'*64},
            'cells':[{'run_id':run,'attempt_id':'gen','readiness':{'state':'ready' if fault=='not_held' else 'held'},'allowed_execution_stages':['check']}]}
        ep=repo/'configs/experiment'/(name+'.json');ep.parent.mkdir(parents=True,exist_ok=True);ep.write_text(json.dumps(e));hashes[str(ep.relative_to(repo))]=reg.sha(ep)
    if fault=='spent':(data/'attempts/RUN-h1').mkdir(parents=True)
    if fault=='duplicate':(repo/'docs/exposure-ledger.md').write_text('| RUN-h1-READ |')
    binding=tmp_path/'binding.json';binding.write_text(json.dumps({'schema':'saea-tfrl-current-supply-binding-v1','rows':625,'new_model_calls':0,'readiness':'held','output_hashes':hashes}))
    out=data/'work/codex/registration'
    if fault:
        with pytest.raises(ValueError):reg.register(repo,data,binding,out)
        assert not out.exists()
    else:
        result=reg.register(repo,data,binding,out)
        assert len(result['recipes'])==2 and result['new_scoring_jobs']==0 and result['generation_readiness']=='held'
        for rel in result['recipes']:
            doc=m.validate_recipe(repo,repo/rel)
            assert doc['official_scoring_jobs']==7
