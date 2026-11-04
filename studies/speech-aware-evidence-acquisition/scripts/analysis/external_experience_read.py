"""Official learned/lock scores with actual shared-P request custody."""
import argparse,json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from tfrl_hardset_a0_workload import read,raw,sha,confined
from tfrl_hardset_a0_registration import pin,pinned
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import intent_slot_reward
from speech_aware_evidence_acquisition.rails.gepa_pair_request_audit import _project
from speech_aware_evidence_acquisition.rails.tfrl_bank_request_audit import checked_control
from tfrl_current_half_result import score_table
ARMS=('learned','lock')


def summarize(results,items):
    ids=[r['sample_id'] for r in items]
    if not ids or len(set(ids))!=len(ids) or set(results)!=set(ARMS):raise ValueError('complete learned/lock pair required')
    tables={a:score_table(results[a],ids) for a in ARMS}
    keys=('evaluator_commit','scorer_version','split_id','projection','filler_normalization')
    if any(results['learned']['aggregate'][k]!=results['lock']['aggregate'][k] for k in keys):raise ValueError('scorer identity differs')
    for arm in ARMS:
        for sid in ids:
            row=tables[arm][sid]
            for key in ('slot_f1_exact_tp','slot_f1_exact_fp','slot_f1_exact_fn'):
                value=row[key]
                if type(value) not in (int,float) or not math.isfinite(value) or value<0 or int(value)!=value:raise ValueError('finite nonnegative exact counts required')
            if type(row['scenario_action_exact']) not in (int,float) or row['scenario_action_exact'] not in (0,1):raise ValueError('binary intent score required')
    success={a:{s:int(tables[a][s]['scenario_action_exact']==1 and tables[a][s]['slot_f1_exact_fp']==tables[a][s]['slot_f1_exact_fn']==0) for s in ids} for a in ARMS}
    rewards={a:[intent_slot_reward(tables[a][s]) for s in ids] for a in ARMS}
    return {'rows':len(ids),'sample_ids':ids,'official':{a:results[a]['aggregate']['official'] for a in ARMS},'strict_success':{a:sum(success[a].values()) for a in ARMS},'proxy_scores':rewards,'comparisons':{'learned_vs_lock':{'strict_repairs':sum(success['learned'][s]>success['lock'][s] for s in ids),'strict_breaks':sum(success['learned'][s]<success['lock'][s] for s in ids),'proxy_mean_delta':(sum(rewards['learned'])-sum(rewards['lock']))/len(ids)}},'candidate_accepted':False,'claims_stable_gain':False}


def audit(repo,values,bundle,ids):
    components=values['native_episode']['consumption_module']['components']
    params=[c['params'] for c in components if c['component_id']=='tfgrpo-solver-experience' and c['state']=='on']
    if len(params)!=1 or params[0].get('target_arm')!='learned':raise ValueError('exact reviewed learned component required')
    params=params[0];templates={a:confined(repo/'configs',params['template' if a=='learned' else 'base_template']['relpath']).read_text() for a in ARMS}
    byid={r['sample_id']:r for r in bundle.outputs_rows};audits=[];abstentions=[]
    for sid in ids:
        visits=byid[sid]['native_episode_receipt']['role_visits'];sv=[v for v in visits if v['stage']=='solver']
        if not sv:abstentions.append(sid);continue
        if len(sv)!=1:raise ValueError('one Solver visit required')
        contacts=sv[0]['execution_receipt']['contacts']
        if len(contacts)!=2:raise ValueError('one learned and one lock primary contact required')
        req={a:bundle.trace_records[c['request_id']+'/request']['payload'] for a,c in zip(ARMS,contacts)}
        lock=checked_control(req['lock'])['messages'][1]['content'];prefix=templates['lock']+'\n\nNative episode limits: '
        if not lock.startswith(prefix):raise ValueError('lock template differs')
        suffix=lock[len(templates['lock']):];limits=json.loads(lock[len(prefix):])
        if json.dumps(limits,sort_keys=True)!=lock[len(prefix):]:raise ValueError('noncanonical limits')
        projections={}
        for a,c in zip(ARMS,contacts):
            if req[a]['sample_id']!=sid or req[a]['request_id']!=c['request_id'] or req[a]['history'][-1]['contact_id']!=c['agent_contact_id']:raise ValueError('contact custody differs')
            projections[a],_= _project(req[a],templates[a]+suffix)
        if projections['learned']!=projections['lock']:raise ValueError('inputs differ outside learned template')
        ps=[v for v in visits if v['stage']=='perception']
        if len(ps)!=1:raise ValueError('one shared P required')
        pc=ps[0]['execution_receipt']['final_contact'];state=[h for h in req['lock']['history'] if h.get('role')=='agent_state']
        if len(state)!=1 or state[0].get('source_response_sha256')!=pc['response_text_sha256']:raise ValueError('state differs from actual P')
        audits.append({'sample_id':sid,'perception_contact':pc,'request_ids':{a:req[a]['request_id'] for a in ARMS},'verified_primary_solver_requests':2})
    return {'audited_rows':len(audits),'abstention_rows':abstentions,'rows':audits}



def shared_learning_cost(values,data):
    """Reference acquisition is a shared batch cost, never an extra per-arm evaluation charge."""
    components=values['native_episode']['consumption_module']['components']
    params=next(c['params'] for c in components if c['component_id']=='tfgrpo-solver-experience' and c['state']=='on')
    if params.get('source_kind')!='external-tfgrpo-reference-v1':return None
    candidate=pinned(data,params['candidate'])
    return {'usage_identity':candidate['usage_identity'], 'learning_usage':candidate['learning_usage'],
            'accounting':candidate['accounting'], 'scope':candidate['usage_scope'],
            'included_in_evaluation_total_cost':False}

def run(repo,data,out,name):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.data.lock import load_lock,umbrella_lock_path
    from registered_workload import validate_workload
    from tfgrpo_h2_read import registered_scorer
    from gepa_probe_read import verify_delivered_scores
    from h1_thinking_comparison import native_costs
    import stage3_rollout_score as scorer
    assert_execution_environment()
    if Path(name).name!=name or name in ('.','..'):raise ValueError('safe name required')
    if not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('owned output required')
    pkg=repo/'docs/readiness'/name;spec=read(pkg/'candidate.json');recipe=read(pkg/'analysis-recipe-v1.json')
    if spec.get('workload_kind')!='external-experience-pair':raise ValueError('external experience registration required')
    validate_workload(repo,data,name,spec)
    expected={'schema':'saea-external-experience-pair-read-v1','rows':spec['rows'],'generation_run_id':spec['run_id'],'attempt_id':spec['attempt_id'],'roster':spec['roster'],'arms':list(ARMS),'delivered_arm':'lock','pair_material':spec['material']}
    if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('recipe differs')
    rid=spec['run_id'];directory=data/'runs'/rid/spec['attempt_id'];bundle=verify_bundle(directory/(rid+'.run-manifest.json'))
    items=pinned(repo,spec['roster'])['items'];ids=[r['sample_id'] for r in items]
    if bundle.attempt_state!='FINALIZED' or set(bundle.split_ids)!=set(ids) or len(bundle.split_ids)!=len(ids):raise ValueError('finalized complete subset required')
    f=bundle.manifest['config']['fragments'];cfg=compose(repo,*(f[k] for k in ('model','dataset','baseline','experiment')))
    if cfg.config_hash!=bundle.manifest['config']['config_hash']:raise ValueError('sealed config differs')
    roles=cfg.values['native_episode']['roles'];group=scorer.parse_solver_arm_group(cfg.values,roles['solver'],roles['perception'],cfg.values['knowledge_stores']);scorer_id=registered_scorer(cfg.values,bundle.manifest)
    out.mkdir(parents=True,exist_ok=True)
    rendered=scorer.render_supply_variant_first_visit(bundle,group,{r['sample_id']:r['stratum'] for r in items},out,run_id=rid,attempt_id=spec['attempt_id'],roster_sha256=spec['roster']['sha256'],config_hash=cfg.config_hash)
    if rendered['arm_order']!=list(ARMS):raise ValueError('learned/lock arm order required')
    inputs=audit(repo,cfg.values,bundle,ids)
    scorer.score_supply_variant_copies(scorer_id,rendered,out,load_lock(umbrella_lock_path(repo)),data,scorer.sealed_per_sample(directory,rid,scorer_id))
    results={a:read(out/f'arm-{a}-{scorer_id}-scores.json')[scorer_id] for a in ARMS};verify_delivered_scores(results['lock'],scorer.sealed_per_sample(directory,rid,scorer_id))
    result=summarize(results,items);result.update(schema='saea-external-experience-pair-result-v1',new_model_calls=0,source=rendered['source'],input_audit=inputs,analysis_recipe_sha256=sha(pkg/'analysis-recipe-v1.json'),score_receipts={a:pin(data,str((out/f'arm-{a}-{scorer_id}-scores.json.receipt.json').relative_to(data))) for a in ARMS},native_role_costs=native_costs(bundle.outputs_rows,bundle.trace_records),total_cost=bundle.manifest['cost'])
    shared=shared_learning_cost(cfg.values,data)
    if shared is not None:result['shared_learning_cost']=shared
    target=out/'complete.json'
    if target.exists():
        if target.read_bytes()!=raw(result):raise ValueError('analysis changed')
    else:publish_new(target,raw(result))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--name',required=True);print(json.dumps(run(**vars(p.parse_args())),sort_keys=True))
