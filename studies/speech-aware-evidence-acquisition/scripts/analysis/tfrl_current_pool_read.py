"""Register or consume a source-bound, model-free pooled real K4 analysis."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/checks')]
from tfrl_current_sources import read,sha,pinned,load_half
from tfrl_current_pool import pool
from register_real_engine_candidate import write
RUN='SAEA-SLURP-TFRL-CURRENT-SEED-I1-POOL-READ-V1'
REL='docs/readiness/tfrl-current-seed-i1-pool-read/analysis-recipe-v1.json'
AUTH='docs/decisions/2026-10-31-owner-lock-baseline-and-cost-repair.md'


def register(repo,source):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from tfrl_current_half_result import validate_recipe
    assert_execution_environment();require_clean_study_repo(repo);require_clean_study_repo(source)
    ledger=repo/'docs/exposure-ledger.md';text=ledger.read_text()
    if (repo/REL).exists() or '| '+RUN+' |' in text:raise ValueError('pool read already registered')
    pins={}
    for half in ('h1','h2'):
        rel=f'docs/readiness/slurp-tfrl-current-seed-i1-{half}-v1/analysis-recipe-v1.json';p=source/rel
        r=validate_recipe(source,p)
        if r['run_id']!=f'SAEA-SLURP-TFRL-CURRENT-SEED-I1-{half.upper()}-V1':raise ValueError('source half differs')
        pins[half]={'relpath':rel,'sha256':sha(p)}
    recipe={'schema':'saea-tfrl-current-pool-read-recipe-v1','read_run_id':RUN,'source_recipes':pins,
        'rows':625,'official_scoring_jobs':0,'new_model_calls':0,'authority':AUTH,
        'entrypoint':'scripts/analysis/tfrl_current_pool_read.py','aggregation':'sum official TP/FP/FN across disjoint313+312',
        'threshold_pp':1.0,'next_iteration_authorized':False,'claims_benefit':False}
    write(repo/REL,recipe)
    ledger.write_text(text+'\n## Current real K4 pooled read\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n| '+' | '.join(['2026-10-31',RUN,'model-free-check','Source-bound complete625 K4 counts','no model or scorer','slurp TRAIN construction625','discovery','two source recipe identities','no','0','0','0','All generation and scoring costs retained',REL,f'Recipe SHA256 {sha(repo/REL)}; root countersigned under {AUTH}, coordinator Codex; independent confirmation required'])+' |\n')
    return recipe


def run(repo,source,data):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    import flight_registration_check as reg
    assert_execution_environment();recipe=read(repo/REL)
    entries=reg.parse_ledger(repo,Path('docs/exposure-ledger.md')).get(RUN,[])
    if (len(entries)!=1 or not reg.is_countersigned(entries[0]) or entries[0]['artifacts']!=REL
            or sha(repo/REL) not in entries[0].get('notes','')):raise ValueError('exact registered pool recipe required')
    if (recipe['schema']!='saea-tfrl-current-pool-read-recipe-v1' or recipe['read_run_id']!=RUN
            or recipe['rows']!=625 or recipe['new_model_calls']!=0 or recipe['official_scoring_jobs']!=0
            or recipe['threshold_pp']!=1.0 or recipe['claims_benefit'] is not False
            or recipe['next_iteration_authorized'] is not False or set(recipe['source_recipes'])!={'h1','h2'}):
        raise ValueError('pool recipe differs')
    halves={h:load_half(source,data,h,recipe['source_recipes'][h]) for h in ('h1','h2')}
    result=pool({r['plan']['half']:(r['plan'],r['results']) for r in halves.values()})
    result.update(read_run_id=RUN,recipe_sha256=sha(repo/REL),sources={h:r['source'] for h,r in halves.items()},
        new_model_calls=0,official_scoring_jobs=0)
    target=data/'analysis'/RUN/'complete.json'
    if target.exists():
        if read(target)!=result:raise ValueError('existing pool result differs')
    else:write(target,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run'])
    for k in ('repo','source'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--data',type=Path);a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='register':a.pop('data');result=register(**a)
    else:result=run(**a)
    print(json.dumps(result))
