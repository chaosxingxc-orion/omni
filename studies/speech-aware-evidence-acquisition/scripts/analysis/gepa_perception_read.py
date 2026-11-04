"""Official Perception GEPA results and exact own-P to S input provenance."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from tfrl_odd40_cells import read,raw,sha,confined
from tfrl_hardset_a0_registration import pinned
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.rails.gepa_perception_request_audit import verify_inputs,ARMS


def audit_inputs(repo,values,bundle,ids,predecessor_records=None):
    from speech_aware_evidence_acquisition.rails.gepa_prompt_material import validate_pair_declaration,pair_node,independent_perception_repeat
    import stage3_rollout_score as scorer
    declaration=validate_pair_declaration(values['native_episode']['consumption_module'])
    if pair_node(declaration)!='perception':raise ValueError('P declaration required')
    if not independent_perception_repeat(values['native_episode']['consumption_module']):
        raise ValueError('four-independent-P reader requires explicit v2 repeat scope; legacy twin reuses lock P')
    original=confined(repo/'configs',values['phase_material']['perception']['instruction_template']).read_text()
    solver=confined(repo/'configs',values['phase_material']['solver']['instruction_template']).read_text()
    templates={'aa':original,'lock':original}
    for arm in ('parent','child'):
        pin=declaration[arm+'_template'];path=confined(repo/'configs',pin['relpath'])
        if sha(path)!=pin['sha256']:raise ValueError('P template changed')
        templates[arm]=path.read_text()
    by_id={r['sample_id']:r for r in bundle.outputs_rows};audits=[];absent=[]
    for sid in ids:
        receipt=by_id[sid]['native_episode_receipt']
        visits=receipt['role_visits']
        if not any(v['stage']=='solver' for v in visits):absent.append(sid);continue
        contacts={c['agent_contact_id']:c for v in visits for c in v['execution_receipt']['contacts']}
        trace=scorer._custody_trace_source(bundle,predecessor_records,sid)
        requests={};responses={}
        for branch in receipt['perception_branches']['branches']:
            arm=branch['branch_id']
            if branch.get('solver_absent'):
                raise ValueError('partial P branch Solver requires explicit abstention audit')
            pc=contacts[branch['contact_id']];sc=contacts[branch['solver_contact_id']]
            requests[arm]={'perception':trace[pc['request_id']+'/request']['payload'],
                           'solver':trace[sc['request_id']+'/request']['payload']}
            responses[arm]=trace[pc['request_id']+'/response']
        audits.append(verify_inputs(sid,requests,responses,templates=templates,solver_template=solver))
    return {'audited_rows':len(audits),'absent_rows':absent,'rows':audits,'new_model_calls':0}


def run(repo,data,out,name):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.data.lock import load_lock,umbrella_lock_path
    from registered_workload import validate_workload
    from tfgrpo_h2_read import registered_scorer
    from gepa_probe_read import verify_delivered_scores
    from gepa_pair_read import summarize
    from h1_thinking_comparison import native_costs
    import stage3_rollout_score as scorer
    assert_execution_environment()
    if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):raise ValueError('confined name required')
    if not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('owned analysis output required')
    pkg=repo/'docs/readiness'/name;spec=read(pkg/'candidate.json');recipe=read(pkg/'analysis-recipe-v1.json')
    if spec['workload_kind']!='gepa-perception-template-pair':raise ValueError('registered P workload required')
    validate_workload(repo,data,name,spec)
    expected={'schema':'saea-gepa-perception-pair-read-v1','generation_run_id':spec['run_id'],'attempt_id':spec['attempt_id'],
        'rows':spec['rows'],'roster':spec['roster'],'arms':list(ARMS),'delivered_arm':'lock','pair_material':spec['material']}
    if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('P read recipe changed')
    rid=spec['run_id'];directory=data/'runs'/rid/spec['attempt_id'];bundle=verify_bundle(directory/(rid+'.run-manifest.json'))
    items=pinned(repo,spec['roster'])['items'];ids=[r['sample_id'] for r in items]
    if bundle.attempt_state!='FINALIZED' or set(bundle.split_ids)!=set(ids) or len(bundle.split_ids)!=len(ids):raise ValueError('complete finalized roster required')
    fragments=bundle.manifest['config']['fragments'];cfg=compose(repo,*(fragments[k] for k in ('model','dataset','baseline','experiment')))
    if cfg.config_hash!=bundle.manifest['config']['config_hash']:raise ValueError('sealed config changed')
    group=scorer.parse_perception_arm_group(cfg.values,cfg.values['native_episode']['roles']['perception'],cfg.values['knowledge_stores'])
    extra={}
    if bundle.manifest.get('predecessor_attempt') or bundle.manifest.get('predecessor_attempts'):
        from speech_aware_evidence_acquisition.core.bundle.outputs import verify_predecessor_attempts
        extra['predecessor_records'],_=verify_predecessor_attempts(bundle.manifest,split_ids=bundle.split_ids,attempt_id=spec['attempt_id'])
    out.mkdir(parents=True,exist_ok=True)
    rendered=scorer.render_perception_branch_columns(bundle,group,{r['sample_id']:r['stratum'] for r in items},out,
        run_id=rid,attempt_id=spec['attempt_id'],roster_sha256=spec['roster']['sha256'],config_hash=cfg.config_hash,**extra)
    if rendered['branch_order']!=list(ARMS):raise ValueError('exact P branch order required')
    inputs=audit_inputs(repo,cfg.values,bundle,ids,**extra)
    scorer_id=registered_scorer(cfg.values,bundle.manifest)
    scorer.score_perception_branch_copies(scorer_id,rendered,out,load_lock(umbrella_lock_path(repo)),data,{r['sample_id']:r for r in bundle.outputs_rows})
    results={a:read(out/f'arm-{a}-{scorer_id}-scores.json')[scorer_id] for a in ARMS}
    verify_delivered_scores(results['lock'],scorer.sealed_per_sample(directory,rid,scorer_id))
    result=summarize({a:results[a] for a in ('parent','child','lock')},items)
    aa=summarize({'parent':results['lock'],'child':results['aa'],'lock':results['lock']},items)
    result.update(schema='saea-gepa-perception-pair-result-v1',source=rendered['source'],input_audit=inputs,
        repeat_control={'scope':'independent-perception','official':results['aa']['aggregate']['official'],'vs_lock':aa['comparisons']['child_vs_parent']},
        native_role_costs=native_costs(bundle.outputs_rows,bundle.trace_records),total_cost=bundle.manifest['cost'],
        new_model_calls=0,quality_scope='independent P-to-S arms; development or held-out status follows registered partition')
    target=out/'complete.json'
    if target.exists():
        if target.read_bytes()!=raw(result):raise ValueError('existing analysis changed')
    else:publish_new(target,raw(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--name',required=True);print(json.dumps(run(**vars(p.parse_args())),sort_keys=True))
