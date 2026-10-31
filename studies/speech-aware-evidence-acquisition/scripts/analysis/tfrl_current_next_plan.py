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



def require_first_construction_gate(record):
    """Existing i1 kill rule; a pass allows preparation, not system acceptance."""
    import math
    gain=record.get('vote_minus_lock_pp')
    if (record.get('rows')!=625 or record.get('iteration')!=1
            or type(gain) not in (int,float) or not math.isfinite(gain)
            or record.get('threshold_pp')!=1.0
            or record.get('construction_gate_passed') is not (round(gain,6)>=1.0)):
        raise ValueError('complete consistent first construction gate required')
    if not record['construction_gate_passed']:
        raise ValueError('iteration1 construction gain below1pp: stop family under registered rule')
    return {'construction_gain_pp':gain,'preparation_gate_passed':True,
            'system_accepted':False,'independent_confirmation_required':True}

def run(repo,data,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();head=require_clean_study_repo(repo)
    bankroot=data/'analysis'/BANK;completion=read(bankroot/'complete.json')
    pooled=read(data/'analysis'/POOL/'complete.json')
    if (completion['rows']!=625 or pooled['rows']!=625 or completion['sources']!=pooled['sources']):
        raise ValueError('matching complete recovered625 bank and score sources required')
    # Recompute from exact registered source halves; a completion flag alone
    # cannot authorize successor preparation.
    from tfrl_recovered_consumers import load as load_recovered
    _,_,verified_halves=load_recovered(repo,data)
    verified_pool=pool({v['plan']['half']:(v['plan'],v['results']) for v in verified_halves.values()})
    if any(pooled.get(k)!=v for k,v in verified_pool.items()):raise ValueError('recorded pool differs from verified source scores')
    if pooled['sources']!={h:v['source'] for h,v in verified_halves.items()}:raise ValueError('pool source custody differs')
    gate=require_first_construction_gate(verified_pool)
    rosters={h:read(repo/sampler.cm.HALF_ROSTER_RELPATH['R01C-'+h.upper()]) for h in ('h1','h2')}
    receipts={};pins={}
    for h in ('h1','h2'):
        path=bankroot/h/'bank-receipt.json'
        if sha(path)!=completion['banks'][h]['receipt_sha256']:raise ValueError('bank receipt changed')
        distiller.load_bank(path)  # Rehash all bank artifacts before planning.
        receipts[h]=read(path);pins[h]={'relpath':str(path.relative_to(data)),'sha256':sha(path)}
    # Bind the allocation to the exact source trees that produced the banks.
    recovered=read(repo/RECOVERED_REL)
    if completion['recipe_sha256']!=sha(repo/RECOVERED_REL):raise ValueError('bank source recipe differs')
    bounds={};allocation_sources={}
    from speech_aware_evidence_acquisition.core.config import compose
    for h,source in recovered['sources'].items():
        root=(data/source['repo_relpath']).resolve()
        if not root.is_relative_to(data.resolve()):raise ValueError('source escapes data root')
        actual=require_clean_study_repo(root)
        if actual!=source['commit']:raise ValueError('allocation source commit differs')
        pinned(root,source['recipe'])
        name='slurp-tfrl-current-seed-i1-h1-context-repair-v1' if h=='h1' else 'slurp-tfrl-current-seed-i1-h2-v1'
        exp_path=root/f'configs/experiment/{name}.json';exp=read(exp_path);cell,=exp['cells']
        cfg=compose(root,cell['model_fragment'],cell['dataset_fragment'],cell['baseline_fragment'],name)
        module=cfg.values['native_episode']['consumption_module']
        bounds[h]=module['decoding_frame']['max_supplied_items']
        allocation_sources[h]={'commit':actual,'experiment_sha256':sha(exp_path),
            'config_hash':cfg.config_hash,'module_sha256':sampler.cm.content_sha256(module)}
    plans=derive(rosters,receipts,iteration=2,control_bounds=bounds)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):raise ValueError('fresh owned plan output required')
    for h,p in plans.items():
        p['roster']['file_sha256']=sha(repo/sampler.cm.HALF_ROSTER_RELPATH['R01C-'+h.upper()])
        p['banks']['allocation_source']=allocation_sources[h]
        s3._immutable_json(out/f'{h}-sampling-plan.json',p)
    result={'schema':'saea-tfrl-current-next-held-plans-v1','source_commit':head,'iteration':2,'rows':625,
        'plans':{h:sha(out/f'{h}-sampling-plan.json') for h in plans},'bank_receipts':pins,'construction_gate':gate,
        'parent_pool_sha256':sha(data/'analysis'/POOL/'complete.json'),'parent_banks_sha256':sha(bankroot/'complete.json'),
        'new_model_calls':0,'readiness':'held','generation_authorized_by_this_file':False,
        'remaining':['review complete construction gain and stop rule','verify declared allocation on actual source inputs while preserving controls',
            'compile and actual shared-render bank request proof','own supply and exact FIT','register/admit/preflight',
            'independent confirmation before improvement claim']}
    s3._immutable_json(out/'complete.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args()))))
