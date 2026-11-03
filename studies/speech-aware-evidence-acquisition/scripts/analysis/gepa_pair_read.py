"""Official complete parent/child/lock scores plus actual same-input custody."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from tfrl_odd40_cells import read,raw,sha,confined
from tfrl_hardset_a0_registration import pin,pinned
from tfrl_current_half_result import score_table
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.rails import solver_prompt_candidate as prompt
from speech_aware_evidence_acquisition.rails.gepa_pair_request_audit import verify_inputs,ARMS
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import intent_slot_reward


def summarize(results,items):
    ids=[r['sample_id'] for r in items]
    if not ids or len(set(ids))!=len(ids) or set(results)!=set(ARMS):
        raise ValueError('complete unique parent child lock roster required')
    tables={a:score_table(results[a],ids) for a in ARMS}
    keys=('evaluator_commit','scorer_version','split_id','projection','filler_normalization')
    identity=lambda a:{k:results[a]['aggregate'][k] for k in keys}
    if any(identity(a)!=identity('lock') for a in ARMS):raise ValueError('official scorer identities differ')
    success={}; rewards={}
    for arm in ARMS:
        success[arm]={};rewards[arm]=[]
        for sid in ids:
            row=tables[arm][sid]
            for k in ('slot_f1_exact_tp','slot_f1_exact_fp','slot_f1_exact_fn'):
                v=row[k]
                if type(v) not in (int,float) or not math.isfinite(v) or v<0 or int(v)!=v:
                    raise ValueError('finite nonnegative exact slot counts required')
            if type(row['scenario_action_exact']) not in (int,float) or row['scenario_action_exact'] not in (0,1):
                raise ValueError('binary intent score required')
            success[arm][sid]=int(row['scenario_action_exact']==1 and row['slot_f1_exact_fp']==row['slot_f1_exact_fn']==0)
            rewards[arm].append(intent_slot_reward(row))
    pairs={}
    for a,b in [('child','parent'),('parent','lock'),('child','lock')]:
        pairs[a+'_vs_'+b]={'strict_repairs':sum(success[a][s]>success[b][s] for s in ids),
            'strict_breaks':sum(success[a][s]<success[b][s] for s in ids),
            'proxy_mean_delta':(sum(rewards[a])-sum(rewards[b]))/len(ids)}
    return {'rows':len(ids),'sample_ids':ids,'official':{a:results[a]['aggregate']['official'] for a in ARMS},
        'strict_success':{a:sum(success[a].values()) for a in ARMS},'proxy_scores':rewards,'comparisons':pairs,
        'minibatch_strictly_improved':sum(rewards['child'])>sum(rewards['parent']),
        'candidate_accepted':False,'claims_stable_gain':False}


def audit_inputs(repo,values,bundle,ids):
    from speech_aware_evidence_acquisition.rails.gepa_prompt_material import validate_pair_declaration,pair_frozen_identity
    from speech_aware_evidence_acquisition.rails.tfrl_bank_request_audit import checked_control
    declaration=validate_pair_declaration(values['native_episode']['consumption_module'])
    if declaration['frozen_sha256']!=pair_frozen_identity(values):raise ValueError('paired frozen graph changed')
    phase=values['phase_material']['solver']
    if (phase.get('instruction_fill') or phase.get('state_fill') or phase.get('fill_params')
            or phase.get('predecessor_binding','none')!='none'):
        raise ValueError('first paired reader requires declared static Solver template')
    original=confined(repo/'configs',phase['instruction_template']).read_text();templates={'lock':original}
    for arm in ARMS[:2]:
        p=declaration[arm+'_template']; path=confined(repo/'configs',p['relpath'])
        if sha(path)!=p['sha256']:raise ValueError('paired template bytes changed')
        templates[arm]=path.read_text()
    by_id={r['sample_id']:r for r in bundle.outputs_rows};audits=[];fallbacks=[]
    for sid in ids:
        row=by_id[sid];visits=row['native_episode_receipt']['role_visits']
        solver=[v for v in visits if v['stage']=='solver']
        if not solver:
            # Official renderer independently verifies canonical abstention.
            fallbacks.append(sid);continue
        if len(solver)!=1:raise ValueError('paired graph must have one Solver visit')
        contacts=solver[0]['execution_receipt']['contacts']
        if len(contacts)!=3:raise ValueError('three first Solver contacts required')
        requests={a:bundle.trace_records[c['request_id']+'/request']['payload'] for a,c in zip(ARMS,contacts)}
        lock=checked_control(requests['lock'])['messages'][1]['content'];prefix=original+'\n\nNative episode limits: '
        if not lock.startswith(prefix):raise ValueError('lock instruction differs from source template')
        suffix=lock[len(original):];limits=json.loads(lock[len(prefix):])
        if json.dumps(limits,sort_keys=True)!=lock[len(prefix):]:raise ValueError('noncanonical native limit suffix')
        audit=verify_inputs(sid,dict(zip(ARMS,contacts)),requests,rendered_instructions={a:templates[a]+suffix for a in ARMS})
        perception=[v for v in visits if v['stage']=='perception']
        if len(perception)!=1:raise ValueError('one shared native perception visit required')
        pc=perception[0]['execution_receipt']['final_contact']
        state=[h for h in requests['lock']['history'] if h.get('role')=='agent_state']
        if len(state)!=1 or state[0].get('source_response_sha256')!=pc['response_text_sha256']:
            raise ValueError('Solver state is not its exact native P response')
        audit['perception_contact']=pc;audits.append(audit)
    return {'audited_rows':len(audits),'abstention_rows':fallbacks,'rows':audits,
        'shared_perception_sha256':prompt.sha([{'sample_id':a['sample_id'],'state':a['shared_agent_state_sha256'],'P':a['perception_contact']} for a in audits]),
        'shared_supply_sha256':prompt.sha([{'sample_id':a['sample_id'],'supply':a['shared_supply_sha256']} for a in audits])}


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
    if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):raise ValueError('confined name required')
    if not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('owned analysis output required')
    pkg=repo/'docs/readiness'/name;spec=read(pkg/'candidate.json');recipe=read(pkg/'analysis-recipe-v1.json')
    if spec.get('workload_kind')!='gepa-solver-template-pair':raise ValueError('registered paired workload required')
    validate_workload(repo,data,name,spec)
    expected={'schema':'saea-gepa-pair-read-v1','generation_run_id':spec['run_id'],'attempt_id':spec['attempt_id'],
        'rows':spec['rows'],'roster':spec['roster'],'arms':list(ARMS),'delivered_arm':'lock','pair_material':spec['material']}
    if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('paired read recipe changed')
    rid=spec['run_id']; directory=data/'runs'/rid/spec['attempt_id'];bundle=verify_bundle(directory/(rid+'.run-manifest.json'))
    items=pinned(repo,spec['roster'])['items'];ids=[r['sample_id'] for r in items]
    if bundle.attempt_state!='FINALIZED' or set(bundle.split_ids)!=set(ids) or len(bundle.split_ids)!=len(ids):raise ValueError('complete finalized paired roster required')
    f=bundle.manifest['config']['fragments'];cfg=compose(repo,*(f[k] for k in ('model','dataset','baseline','experiment')))
    if cfg.config_hash!=bundle.manifest['config']['config_hash']:raise ValueError('sealed configuration changed')
    roles=cfg.values['native_episode']['roles'];group=scorer.parse_solver_arm_group(cfg.values,roles['solver'],roles['perception'],cfg.values['knowledge_stores'])
    scorer_id=registered_scorer(cfg.values,bundle.manifest);out.mkdir(parents=True,exist_ok=True)
    rendered=scorer.render_supply_variant_first_visit(bundle,group,{r['sample_id']:r['stratum'] for r in items},out,
        run_id=rid,attempt_id=spec['attempt_id'],roster_sha256=spec['roster']['sha256'],config_hash=cfg.config_hash)
    if rendered['arm_order']!=list(ARMS):raise ValueError('exact paired arm order required')
    inputs=audit_inputs(repo,cfg.values,bundle,ids)
    scorer.score_supply_variant_copies(scorer_id,rendered,out,load_lock(umbrella_lock_path(repo)),data,scorer.sealed_per_sample(directory,rid,scorer_id))
    results={}
    for a in ARMS:
        doc=read(out/f'arm-{a}-{scorer_id}-scores.json')
        if set(doc)!={scorer_id}:raise ValueError('unexpected official scorer')
        results[a]=doc[scorer_id]
    verify_delivered_scores(results['lock'],scorer.sealed_per_sample(directory,rid,scorer_id))
    result=summarize(results,items)
    result.update(schema='saea-gepa-pair-result-v1',new_model_calls=0,source=rendered['source'],
        input_audit=inputs,analysis_recipe_sha256=sha(pkg/'analysis-recipe-v1.json'),
        score_receipts={a:pin(data,str((out/f'arm-{a}-{scorer_id}-scores.json.receipt.json').relative_to(data))) for a in ARMS},
        native_role_costs=native_costs(bundle.outputs_rows,bundle.trace_records),total_cost=bundle.manifest['cost'])
    target=out/'complete.json'
    if target.exists():
        if target.read_bytes()!=raw(result):raise ValueError('existing paired analysis changed')
    else:publish_new(target,raw(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--name',required=True);print(json.dumps(run(**vars(p.parse_args())),sort_keys=True))
