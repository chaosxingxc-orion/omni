"""Register/consume exact H1 source union plus complete H2 for625 pool and banks.

Previous failed consumer recipes remain unchanged. New source trees and result
receipts are explicit; no scoring or inference occurs in verification/pooling.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/checks')]
from tfrl_current_sources import read,sha,pinned,load_half
import tfrl_h1_union as h1union
import tfrl_current_bank_run as bank
from tfrl_current_pool import pool
import stage3_rollout_score as s3
AUTH=bank.AUTH
POOL='SAEA-SLURP-TFRL-CURRENT-SEED-I1-RECOVERED-POOL-READ-V1'
BANK='SAEA-SLURP-TFRL-CURRENT-SEED-I1-RECOVERED-BANK-V1'
REL='docs/readiness/tfrl-current-recovered-consumers/analysis-recipe-v1.json'


def identities(iteration):
    """Keep recovered i1 bytes; later iterations have independent consumer IDs."""
    if type(iteration) is not int or iteration not in (1,2,3):
        raise ValueError('consumer iteration must be1..3')
    if iteration==1:return POOL,BANK,REL
    prefix=f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}'
    return (prefix+'-POOL-READ-V1',prefix+'-BANK-V1',
            f'docs/readiness/tfrl-current-bank-i{iteration}-consumers/analysis-recipe-v1.json')


def register(repo,data,h1,h2,iteration=1):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from tfrl_current_half_result import validate_recipe
    assert_execution_environment();require_clean_study_repo(repo)
    pool_id,bank_id,recipe_rel=identities(iteration)
    sources={};models=[];runtimes=[];sessions=[];stores=[]
    source_specs= [('h1',h1,h1union.NAME,h1union.REL),
        ('h2',h2,'slurp-tfrl-current-seed-i1-h2-v1','docs/readiness/slurp-tfrl-current-seed-i1-h2-v1/analysis-recipe-v1.json')]
    if iteration>1:
        source_specs=[(half,root,f'slurp-tfrl-current-bank-i{iteration}-{half}-v1',
            f'docs/readiness/slurp-tfrl-current-bank-i{iteration}-{half}-v1/analysis-recipe-v1.json')
            for half,root in [('h1',h1),('h2',h2)]]
    for half,root,name,rel in source_specs:
        head=require_clean_study_repo(root)
        if half=='h1' and iteration==1:r=h1union.validate_recipe(root,root/rel)
        else:r=validate_recipe(root,root/rel)
        if r['rows']!=(313 if half=='h1' else 312):raise ValueError('exact complete half required')
        if iteration>1 and (r['run_id']!=f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}-{half.upper()}-V1'
                or r['attempt_id']!='generation-v1'):
            raise ValueError('registered banked source iteration differs')
        exp=read(root/f'configs/experiment/{name}.json');cell,=exp['cells']
        modelrel=f"configs/model/{cell['model_fragment']}.json";model=read(root/modelrel)
        runtimerel='docs/'+model['runtime_receipt']
        models.append(sha(root/modelrel));runtimes.append(sha(root/runtimerel));sessions.append(exp['service_session'])
        stores.append(read(root/f"configs/baseline/{cell['baseline_fragment']}.json")['knowledge_stores'])
        sources[half]={'repo_relpath':str(root.resolve().relative_to(data.resolve())),'commit':head,
            'recipe':{'relpath':rel,'sha256':sha(root/rel)},
            'model':{'relpath':modelrel,'sha256':models[-1]},'runtime':{'relpath':runtimerel,'sha256':runtimes[-1]}}
    if any(v[0]!=v[1] for v in (models,runtimes,sessions,stores)):raise ValueError('common model, runtime, resident and knowledge required')
    recipe={'schema':'saea-tfrl-recovered-consumers-v1','read_run_id':pool_id,'bank_run_id':bank_id,'rows':625,
        'sources':sources,'new_model_calls':0,'official_scoring_jobs':0,'tokenizer_contact_cap':bank.CAP,
        'service_session':sessions[0],'threshold_pp':1.0,'authority':AUTH,'claims_benefit':False,
        'next_iteration_authorized':False,'entrypoint':'scripts/analysis/tfrl_recovered_consumers.py',
        'bank_method':'Existing measured contrast-rule and evidence-exemplar distiller; cross-fit checked',
        'exemplar_token_cap':2000,'rule_token_cap':400,'rule_support_rows':3}
    if iteration>1:recipe['iteration']=iteration
    ledger=repo/'docs/exposure-ledger.md';text=ledger.read_text()
    if (repo/recipe_rel).exists() or any('| '+rid+' |' in text for rid in (pool_id,bank_id)):raise ValueError('fresh consumer identities required')
    s3._immutable_json(repo/recipe_rel,recipe)
    lines=[]
    for rid,calls,scope in [(pool_id,0,'Source-bound complete625 official counts'),(bank_id,bank.CAP,'Source-bound625 measured cross-fit banks')]:
        fields=['2026-10-31',rid,'model-free-check' if calls==0 else 'bounded-discovery-probe',scope,
            'no model or scorer' if calls==0 else 'tokenizer only; no generation','slurp TRAIN construction625','discovery',
            'two exact source identities','no',str(calls),'0','0','All original and recovery costs retained',recipe_rel,
            f'Recipe SHA256 {sha(repo/recipe_rel)}; root countersigned under {AUTH}, coordinator Codex; no test tuning; independent confirmation required']
        lines.append('| '+' | '.join(fields)+' |')
    ledger.write_text(text+'\n## Recovered K4 consumers\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join(lines)+'\n')
    return recipe


def validate(repo,iteration=1):
    import flight_registration_check as reg
    pool_id,bank_id,recipe_rel=identities(iteration)
    r=read(repo/recipe_rel);expected={'schema':'saea-tfrl-recovered-consumers-v1','read_run_id':pool_id,'bank_run_id':bank_id,
        'rows':625,'new_model_calls':0,'official_scoring_jobs':0,'tokenizer_contact_cap':bank.CAP,'threshold_pp':1.0,
        'claims_benefit':False,'next_iteration_authorized':False,'exemplar_token_cap':2000,'rule_token_cap':400,'rule_support_rows':3}
    if iteration>1:expected['iteration']=iteration
    if any(r.get(k)!=v for k,v in expected.items()) or set(r['sources'])!={'h1','h2'}:raise ValueError('consumer scope changed')
    ledger=reg.parse_ledger(repo,Path('docs/exposure-ledger.md'))
    for rid in (pool_id,bank_id):
        rows=ledger.get(rid,[])
        if (len(rows)!=1 or not reg.is_countersigned(rows[0]) or rows[0]['artifacts']!=recipe_rel
                or sha(repo/recipe_rel) not in rows[0].get('notes','')):raise ValueError('registered consumer receipt required')
    return r


def load(repo,data,iteration=1):
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    identities(iteration)
    r=validate(repo,iteration);roots={};halves={}
    for half,spec in r['sources'].items():
        root=(data/spec['repo_relpath']).resolve()
        if not root.is_relative_to(data.resolve()) or require_clean_study_repo(root)!=spec['commit']:raise ValueError('frozen source tree changed')
        pinned(root,spec['recipe']);pinned(root,spec['model']);pinned(root,spec['runtime']);roots[half]=root
        if half=='h1' and iteration==1:
            halves[half]=h1union.run(root,data,data/'analysis'/h1union.READ,verify_only=True)
        else:halves[half]=load_half(root,data,half,spec['recipe'],iteration=iteration)
    return r,roots,halves


def run(repo,data,mode,iteration=1):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();require_clean_study_repo(repo)
    if mode not in ('pool','bank'):raise ValueError('explicit pool or bank mode required')
    pool_id,bank_id,recipe_rel=identities(iteration)
    recipe,roots,halves=load(repo,data,iteration)
    result=pool({x['plan']['half']:(x['plan'],x['results']) for x in halves.values()})
    if result['iteration']!=iteration:raise ValueError('pooled iteration differs from registered consumer')
    result.update(read_run_id=pool_id,recipe_sha256=sha(repo/recipe_rel),sources={h:x['source'] for h,x in halves.items()},
                  new_model_calls=0,official_scoring_jobs=0)
    s3._immutable_json(data/'analysis'/pool_id/'complete.json',result)
    if mode=='pool':return result
    spec=recipe['sources']['h2']
    bank_recipe={**recipe,'source_model':spec['model'],'runtime_receipt':spec['runtime']}
    return bank.produce(repo,roots['h2'],data,bank_recipe,halves,out=data/'analysis'/bank_id,recipe_path=repo/recipe_rel)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','pool','bank'])
    for k in ('repo','data'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('h1','h2'):p.add_argument('--'+k,type=Path)
    p.add_argument('--iteration',type=int,default=1)
    a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='register':result=register(**a)
    else:a.pop('h1');a.pop('h2');result=run(**a,mode=mode)
    print(json.dumps(result))
