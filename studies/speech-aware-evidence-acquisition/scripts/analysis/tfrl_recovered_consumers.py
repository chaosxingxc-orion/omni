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


def register(repo,data,h1,h2):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from tfrl_current_half_result import validate_recipe
    assert_execution_environment();require_clean_study_repo(repo)
    sources={};models=[];runtimes=[];sessions=[];stores=[]
    for half,root,name,rel in [('h1',h1,h1union.NAME,h1union.REL),
        ('h2',h2,'slurp-tfrl-current-seed-i1-h2-v1','docs/readiness/slurp-tfrl-current-seed-i1-h2-v1/analysis-recipe-v1.json')]:
        head=require_clean_study_repo(root)
        if half=='h1':r=h1union.validate_recipe(root,root/rel)
        else:r=validate_recipe(root,root/rel)
        if r['rows']!=(313 if half=='h1' else 312):raise ValueError('exact complete half required')
        exp=read(root/f'configs/experiment/{name}.json');cell,=exp['cells']
        modelrel=f"configs/model/{cell['model_fragment']}.json";model=read(root/modelrel)
        runtimerel='docs/'+model['runtime_receipt']
        models.append(sha(root/modelrel));runtimes.append(sha(root/runtimerel));sessions.append(exp['service_session'])
        stores.append(read(root/f"configs/baseline/{cell['baseline_fragment']}.json")['knowledge_stores'])
        sources[half]={'repo_relpath':str(root.resolve().relative_to(data.resolve())),'commit':head,
            'recipe':{'relpath':rel,'sha256':sha(root/rel)},
            'model':{'relpath':modelrel,'sha256':models[-1]},'runtime':{'relpath':runtimerel,'sha256':runtimes[-1]}}
    if any(v[0]!=v[1] for v in (models,runtimes,sessions,stores)):raise ValueError('common model, runtime, resident and knowledge required')
    recipe={'schema':'saea-tfrl-recovered-consumers-v1','read_run_id':POOL,'bank_run_id':BANK,'rows':625,
        'sources':sources,'new_model_calls':0,'official_scoring_jobs':0,'tokenizer_contact_cap':bank.CAP,
        'service_session':sessions[0],'threshold_pp':1.0,'authority':AUTH,'claims_benefit':False,
        'next_iteration_authorized':False,'entrypoint':'scripts/analysis/tfrl_recovered_consumers.py',
        'bank_method':'Existing measured contrast-rule and evidence-exemplar distiller; cross-fit checked',
        'exemplar_token_cap':2000,'rule_token_cap':400,'rule_support_rows':3}
    ledger=repo/'docs/exposure-ledger.md';text=ledger.read_text()
    if (repo/REL).exists() or any('| '+rid+' |' in text for rid in (POOL,BANK)):raise ValueError('fresh consumer identities required')
    s3._immutable_json(repo/REL,recipe)
    lines=[]
    for rid,calls,scope in [(POOL,0,'Source-bound complete625 official counts'),(BANK,bank.CAP,'Source-bound625 measured cross-fit banks')]:
        fields=['2026-10-31',rid,'model-free-check' if calls==0 else 'bounded-discovery-probe',scope,
            'no model or scorer' if calls==0 else 'tokenizer only; no generation','slurp TRAIN construction625','discovery',
            'two exact source identities','no',str(calls),'0','0','All original and recovery costs retained',REL,
            f'Recipe SHA256 {sha(repo/REL)}; root countersigned under {AUTH}, coordinator Codex; no test tuning; independent confirmation required']
        lines.append('| '+' | '.join(fields)+' |')
    ledger.write_text(text+'\n## Recovered K4 consumers\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join(lines)+'\n')
    return recipe


def validate(repo):
    import flight_registration_check as reg
    r=read(repo/REL);expected={'schema':'saea-tfrl-recovered-consumers-v1','read_run_id':POOL,'bank_run_id':BANK,
        'rows':625,'new_model_calls':0,'official_scoring_jobs':0,'tokenizer_contact_cap':bank.CAP,'threshold_pp':1.0,
        'claims_benefit':False,'next_iteration_authorized':False,'exemplar_token_cap':2000,'rule_token_cap':400,'rule_support_rows':3}
    if any(r.get(k)!=v for k,v in expected.items()) or set(r['sources'])!={'h1','h2'}:raise ValueError('consumer scope changed')
    ledger=reg.parse_ledger(repo,Path('docs/exposure-ledger.md'))
    for rid in (POOL,BANK):
        rows=ledger.get(rid,[])
        if (len(rows)!=1 or not reg.is_countersigned(rows[0]) or rows[0]['artifacts']!=REL
                or sha(repo/REL) not in rows[0].get('notes','')):raise ValueError('registered consumer receipt required')
    return r


def load(repo,data):
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    r=validate(repo);roots={};halves={}
    for half,spec in r['sources'].items():
        root=(data/spec['repo_relpath']).resolve()
        if not root.is_relative_to(data.resolve()) or require_clean_study_repo(root)!=spec['commit']:raise ValueError('frozen source tree changed')
        pinned(root,spec['recipe']);pinned(root,spec['model']);pinned(root,spec['runtime']);roots[half]=root
        if half=='h1':
            halves[half]=h1union.run(root,data,data/'analysis'/h1union.READ,verify_only=True)
        else:halves[half]=load_half(root,data,half,spec['recipe'])
    return r,roots,halves


def run(repo,data,mode):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();require_clean_study_repo(repo)
    recipe,roots,halves=load(repo,data)
    result=pool({x['plan']['half']:(x['plan'],x['results']) for x in halves.values()})
    result.update(read_run_id=POOL,recipe_sha256=sha(repo/REL),sources={h:x['source'] for h,x in halves.items()},
                  new_model_calls=0,official_scoring_jobs=0)
    s3._immutable_json(data/'analysis'/POOL/'complete.json',result)
    if mode=='pool':return result
    spec=recipe['sources']['h2']
    bank_recipe={**recipe,'source_model':spec['model'],'runtime_receipt':spec['runtime']}
    return bank.produce(repo,roots['h2'],data,bank_recipe,halves,out=data/'analysis'/BANK,recipe_path=repo/REL)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','pool','bank'])
    for k in ('repo','data'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('h1','h2'):p.add_argument('--'+k,type=Path)
    a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='register':result=register(**a)
    else:a.pop('h1');a.pop('h2');result=run(**a,mode=mode)
    print(json.dumps(result))
