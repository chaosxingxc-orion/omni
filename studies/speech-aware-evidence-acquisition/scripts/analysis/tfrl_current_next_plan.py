"""Prepare exact cross-fit successor sampling plans from measured real banks.

Plans stay held: supply allocation, actual rendering proof, generation/FIT
registration and independent confirmation are separate requirements. No model,
tokenizer, scorer, admission or readiness mutation occurs here.
"""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
import tfrl_v2_sampler as sampler
from tfrl_current_seed_cells import discard_synthetic_cost_forecasts
from tfrl_current_sources import read,sha,pinned
from tfrl_current_pool import pool
from tfrl_current_distill import distiller
from tfrl_recovered_consumers import BANK,POOL,REL as RECOVERED_REL
import stage3_rollout_score as s3


def derive(rosters,receipts,*,iteration,control_bounds=None):
    if type(iteration)is not int or iteration not in (2,3):raise ValueError('at most three iterations')
    if set(rosters)!={'h1','h2'} or set(receipts)!={'h1','h2'}:raise ValueError('two halves required')
    ids={h:[r['sample_id'] for r in d['items']] for h,d in rosters.items()}
    if (len(ids['h1'])!=313 or len(ids['h2'])!=312 or set(ids['h1'])&set(ids['h2'])
            or any(len(set(v))!=len(v) for v in ids.values())):raise ValueError('exact disjoint313/312 required')
    result={}
    for h in ('h1','h2'):
        other='h2' if h=='h1' else 'h1';receipt=receipts[other];half='R01C-'+h.upper()
        if (receipt['iteration']!=iteration-1 or receipt['kind']!='half'
                or receipt['source_halves']!=['R01C-'+other.upper()]
                or receipt['source_split_identities']!={'R01C-'+other.upper():rosters[other]['split_identity_hash']}
                or not set(receipt['source_row_ids'])<=set(ids[other])):
            raise ValueError('bank is not the exact previous opposite-half source')
        run=f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}-{h.upper()}-V1'
        plan=sampler.build_plan(rosters[h],half,iteration,bank_receipt=receipt,
            execution_run_id=run,execution_attempt_id='generation-v1')
        discard_synthetic_cost_forecasts(plan)
        # Legacy sampler has a hardcoded51/43 item forecast, not the current
        # actual55-item module. Do not silently present it as an executable cap.
        plan['banks']['supply_bound_items']=None
        plan['banks']['supply_allocation_status']='unresolved: current source module bounds required'
        if control_bounds is not None:
            from speech_aware_evidence_acquisition.rails.tfrl_supply_allocation import declaration
            if set(control_bounds)!=set(rosters):raise ValueError('two actual control bounds required')
            allocation=declaration(control_bounds[h])
            plan['banks']['supply_allocation']=allocation
            plan['banks']['supply_bound_items']=allocation['candidate_max_items']
            plan['banks']['supply_allocation_status']='declared; actual source rendering and FIT required'
        sampler.execution_columns(plan)
        result[h]=plan
    return result



def first_construction_decision(record):
    """Existing i1 kill rule; a pass allows preparation, not system acceptance."""
    import math
    gain=record.get('vote_minus_lock_pp')
    if (record.get('rows')!=625 or record.get('iteration')!=1
            or type(gain) not in (int,float) or not math.isfinite(gain)
            or record.get('threshold_pp')!=1.0
            or record.get('construction_gate_passed') is not (round(gain,6)>=1.0)):
        raise ValueError('complete consistent first construction gate required')
    return {'construction_gain_pp':gain,'preparation_gate_passed':record['construction_gate_passed'],
            'decision':'prepare-held-successor' if record['construction_gate_passed'] else 'stop-family',
            'system_accepted':False,'independent_confirmation_required':True}

def require_first_construction_gate(record):
    decision=first_construction_decision(record)
    if not decision['preparation_gate_passed']:
        raise ValueError('iteration1 construction gain below1pp: stop family under registered rule')
    return decision


def decide(repo,data,out):
    """Seal a verified first-family decision without waiting for tokenizer work."""
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from tfrl_recovered_consumers import load as load_recovered
    assert_execution_environment();head=require_clean_study_repo(repo)
    _,_,halves=load_recovered(repo,data)
    verified=pool({v['plan']['half']:(v['plan'],v['results']) for v in halves.values()})
    path=data/'analysis'/POOL/'complete.json';recorded=read(path)
    sources={h:v['source'] for h,v in halves.items()}
    if any(recorded.get(k)!=v for k,v in verified.items()) or recorded['sources']!=sources:
        raise ValueError('first construction pool differs from verified sources')
    result={'schema':'saea-tfrl-first-construction-decision-v1','implementation_commit':head,
        'pool_sha256':sha(path),'sources':sources,'rows':625,**first_construction_decision(verified),
        'new_model_calls':0,'new_tokenizer_calls':0,'new_scoring_jobs':0,
        'generation_authorized':False}
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned decision output required')
    s3._immutable_json(out,result);return result


def stage_cells(repo,data,sources,rosters,plans,receipts,pins):
    """Build reviewable real-bank successor configs, without touching the repo."""
    from tfrl_current_banked_cells import derive as derive_cell,rail
    if any(set(mapping)!={'h1','h2'} for mapping in (sources,rosters,plans,receipts,pins)):
        raise ValueError('two complete source halves required for staging')
    scope={'R01C-'+h.upper():r['split_identity_hash'] for h,r in rosters.items()}
    result={}
    for h in ('h1','h2'):
        other='h2' if h=='h1' else 'h1'
        # Verify the exact receipt bytes before deriving the canonical content pin.
        path=pinned(data,pins[other])
        if read(path)!=receipts[other]:raise ValueError('bank receipt document differs')
        content_pin={'relpath':pins[other]['relpath'],
                     'sha256':rail.receipt_sha256(receipts[other])}
        original,experiment=sources[h]
        name,base,exp=derive_cell(repo,original,experiment,rosters[h],plans[h],
                                  receipts[other],content_pin,scope)
        run=exp['cells'][0]['run_id']
        for rid in (run,run+'-FIT'):
            if any((data/k/rid).exists() for k in ('attempts','runs','prepared')):
                raise ValueError('successor identity already used; inspect before resuming')
        for kind,doc in (('baseline',base),('experiment',exp)):
            rel=f'configs/{kind}/{name}.json'
            if (repo/rel).exists():raise ValueError('successor config already exists')
            result[rel]=doc
    return result


def run(repo,data,out,iteration=2):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();head=require_clean_study_repo(repo)
    if type(iteration) is not int or iteration not in (2,3):raise ValueError('successor iteration2/3 required')
    from tfrl_recovered_consumers import identities
    pool_id,bank_id,recovered_rel=identities(iteration-1)
    bankroot=data/'analysis'/bank_id;completion=read(bankroot/'complete.json')
    pooled=read(data/'analysis'/pool_id/'complete.json')
    if (completion['rows']!=625 or pooled['rows']!=625 or completion['sources']!=pooled['sources']):
        raise ValueError('matching complete recovered625 bank and score sources required')
    # Recompute from exact registered source halves; a completion flag alone
    # cannot authorize successor preparation.
    from tfrl_recovered_consumers import load as load_recovered
    _,_,verified_halves=load_recovered(repo,data,iteration=iteration-1)
    verified_pool=pool({v['plan']['half']:(v['plan'],v['results']) for v in verified_halves.values()})
    if any(pooled.get(k)!=v for k,v in verified_pool.items()):raise ValueError('recorded pool differs from verified source scores')
    if pooled['sources']!={h:v['source'] for h,v in verified_halves.items()}:raise ValueError('pool source custody differs')
    if verified_pool['iteration']!=iteration-1:raise ValueError('previous iteration source required')
    if iteration==2:
        gate=require_first_construction_gate(verified_pool)
    else:
        # The first-family kill rule remains bound to verified first results;
        # a later pool never substitutes for that evidence.
        _,_,first_halves=load_recovered(repo,data,iteration=1)
        first_pool=pool({v['plan']['half']:(v['plan'],v['results']) for v in first_halves.values()})
        gate=require_first_construction_gate(first_pool)
        gate['previous_iteration']=2
        gate['later_acceptance_pending']=True
    rosters={h:read(repo/sampler.cm.HALF_ROSTER_RELPATH['R01C-'+h.upper()]) for h in ('h1','h2')}
    receipts={};pins={}
    for h in ('h1','h2'):
        path=bankroot/h/'bank-receipt.json'
        if sha(path)!=completion['banks'][h]['receipt_sha256']:raise ValueError('bank receipt changed')
        distiller.load_bank(path)  # Rehash all bank artifacts before planning.
        receipts[h]=read(path);pins[h]={'relpath':str(path.relative_to(data)),'sha256':sha(path)}
    # Bind the allocation to the exact source trees that produced the banks.
    recovered=read(repo/recovered_rel)
    if completion['recipe_sha256']!=sha(repo/recovered_rel):raise ValueError('bank source recipe differs')
    bounds={};allocation_sources={};source_documents={}
    from speech_aware_evidence_acquisition.core.config import compose
    for h,source in recovered['sources'].items():
        root=(data/source['repo_relpath']).resolve()
        if not root.is_relative_to(data.resolve()):raise ValueError('source escapes data root')
        actual=require_clean_study_repo(root)
        if actual!=source['commit']:raise ValueError('allocation source commit differs')
        pinned(root,source['recipe'])
        name='slurp-tfrl-current-seed-i1-h1-context-repair-v1' if h=='h1' else 'slurp-tfrl-current-seed-i1-h2-v1'
        if iteration>2:name=f'slurp-tfrl-current-bank-i{iteration-1}-{h}-v1'
        exp_path=root/f'configs/experiment/{name}.json';exp=read(exp_path);cell,=exp['cells']
        cfg=compose(root,cell['model_fragment'],cell['dataset_fragment'],cell['baseline_fragment'],name)
        module=cfg.values['native_episode']['consumption_module']
        baseline_path=root/f"configs/baseline/{cell['baseline_fragment']}.json"
        source_documents[h]=(read(baseline_path),exp)
        bounds[h]=module['decoding_frame']['max_supplied_items']
        allocation_sources[h]={'commit':actual,'experiment_sha256':sha(exp_path),
            'config_hash':cfg.config_hash,'module_sha256':sampler.cm.content_sha256(module),
            'baseline_sha256':sha(baseline_path), 'repo_relpath':source['repo_relpath']}
    plans=derive(rosters,receipts,iteration=iteration,control_bounds=bounds)
    # Compile both halves before publishing any output. Pins in executable
    # declarations use canonical receipt content hashes, not file hashes.
    staged=stage_cells(repo,data,source_documents,rosters,plans,receipts,pins)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):raise ValueError('fresh owned plan output required')
    for h,p in plans.items():
        p['roster']['file_sha256']=sha(repo/sampler.cm.HALF_ROSTER_RELPATH['R01C-'+h.upper()])
        p['banks']['allocation_source']=allocation_sources[h]
        s3._immutable_json(out/f'{h}-sampling-plan.json',p)
    for rel,doc in staged.items():s3._immutable_json(out/rel,doc)
    result={'schema':'saea-tfrl-current-next-held-plans-v1','source_commit':head,'iteration':iteration,'rows':625,
        'plans':{h:sha(out/f'{h}-sampling-plan.json') for h in plans},
        'held_cell_files':{rel:sha(out/rel) for rel in staged},'bank_receipts':pins,'construction_gate':gate,
        'parent_pool_sha256':sha(data/'analysis'/pool_id/'complete.json'),'parent_banks_sha256':sha(bankroot/'complete.json'),
        'new_model_calls':0,'readiness':'held','generation_authorized_by_this_file':False,
        'remaining':['review complete construction gain and stop rule','verify declared allocation on actual source inputs while preserving controls',
            'compile and actual shared-render bank request proof','own supply and exact FIT','register/admit/preflight',
            'independent confirmation before improvement claim']}
    s3._immutable_json(out/'complete.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=2)
    p.add_argument('--decision-only',action='store_true')
    args=vars(p.parse_args());decision_only=args.pop('decision_only')
    if decision_only:
        if args.pop('iteration')!=2:raise ValueError('first decision applies to successor2')
        result=decide(**args)
    else:result=run(**args)
    print(json.dumps(result))
