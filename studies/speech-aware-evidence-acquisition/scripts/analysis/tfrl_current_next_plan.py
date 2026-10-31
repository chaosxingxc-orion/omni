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
from tfrl_recovered_consumers import BANK,POOL
import stage3_rollout_score as s3


def derive(rosters,receipts,*,iteration):
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
        plan['banks']['supply_allocation_status']='unresolved: reserve bank budget without altering aa/lock controls'
        sampler.execution_columns(plan)
        result[h]=plan
    return result


def run(repo,data,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();head=require_clean_study_repo(repo)
    bankroot=data/'analysis'/BANK;completion=read(bankroot/'complete.json')
    pooled=read(data/'analysis'/POOL/'complete.json')
    if (completion['rows']!=625 or pooled['rows']!=625 or completion['sources']!=pooled['sources']):
        raise ValueError('matching complete recovered625 bank and score sources required')
    rosters={h:read(repo/sampler.cm.HALF_ROSTER_RELPATH['R01C-'+h.upper()]) for h in ('h1','h2')}
    receipts={};pins={}
    for h in ('h1','h2'):
        path=bankroot/h/'bank-receipt.json'
        if sha(path)!=completion['banks'][h]['receipt_sha256']:raise ValueError('bank receipt changed')
        distiller.load_bank(path)  # Rehash all bank artifacts before planning.
        receipts[h]=read(path);pins[h]={'relpath':str(path.relative_to(data)),'sha256':sha(path)}
    plans=derive(rosters,receipts,iteration=2)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):raise ValueError('fresh owned plan output required')
    for h,p in plans.items():
        p['roster']['file_sha256']=sha(repo/sampler.cm.HALF_ROSTER_RELPATH['R01C-'+h.upper()])
        s3._immutable_json(out/f'{h}-sampling-plan.json',p)
    result={'schema':'saea-tfrl-current-next-held-plans-v1','source_commit':head,'iteration':2,'rows':625,
        'plans':{h:sha(out/f'{h}-sampling-plan.json') for h in plans},'bank_receipts':pins,
        'parent_pool_sha256':sha(data/'analysis'/POOL/'complete.json'),'parent_banks_sha256':sha(bankroot/'complete.json'),
        'new_model_calls':0,'readiness':'held','generation_authorized_by_this_file':False,
        'remaining':['review complete construction gain and stop rule','resolve actual bank supply allocation while preserving controls',
            'compile and actual shared-render bank request proof','own supply and exact FIT','register/admit/preflight',
            'independent confirmation before improvement claim']}
    s3._immutable_json(out/'complete.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args()))))
