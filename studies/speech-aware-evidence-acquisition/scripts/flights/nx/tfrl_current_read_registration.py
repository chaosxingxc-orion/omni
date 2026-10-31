"""Register result recipes for the two held, prepared real K4 halves."""
import argparse
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'src'),str(Path(__file__).parent)]
from tfrl_current_supply_binding import read,sha,raw
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
AUTHORITY='docs/decisions/2026-10-31-owner-lock-baseline-and-cost-repair.md'


def register(repo,data,binding,out,iteration=1,first_decision=None):
    if type(iteration) is not int or iteration not in (1,2,3):raise ValueError("iteration1..3 required")
    prefix="seed" if iteration==1 else "bank"
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();head=require_clean_study_repo(repo)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned registration record required')
    completed=read(binding)
    if (completed['schema']!='saea-tfrl-current-supply-binding-v1' or completed['rows']!=625
            or completed['new_model_calls']!=0 or completed['readiness']!='held'):
        raise ValueError('completed held real supply binding required')
    for rel,digest in completed['output_hashes'].items():
        if sha(repo/rel)!=digest:raise ValueError('bound configuration changed')
    ledger=repo/'docs/exposure-ledger.md';text=ledger.read_text();pending=[];rows=[]
    for half in ('h1','h2'):
        name=f'slurp-tfrl-current-{prefix}-i{iteration}-{half}-v1';exp=read(repo/f'configs/experiment/{name}.json');cell,=exp['cells']
        run=cell['run_id'];read_id=run+'-READ';pkg=repo/'docs/readiness'/name
        if (cell['readiness']['state']!='held' or cell['allowed_execution_stages']!=['check']
                or any((data/k/run).exists() for k in ('runs','attempts'))):raise ValueError('unspent held half required')
        if iteration>1:
            if first_decision is None:raise ValueError('banked read requires first decision')
            sys.path.insert(0,str(repo/'scripts/analysis'))
            from tfrl_banked_supply import validate_preparation_inputs
            validate_preparation_inputs(repo,data,name,iteration,first_decision,allow_bound=True)
            from tfrl_current_sources import pinned
            base=read(repo/f"configs/baseline/{cell['baseline_fragment']}.json")
            pinned(data,base['prepared_phase_supply'])
            needed={f'configs/experiment/{name}.json',f"configs/baseline/{cell['baseline_fragment']}.json"}
            if not needed<=set(completed['output_hashes']):raise ValueError('binding omits source configs')
        pp=pkg/'sampling-plan.json';plan=read(pp);rp=repo/exp['population']['roster_receipt'];n=exp['population']['count']
        if plan['run_id']!=run or plan['roster']['rows']!=n or sha(rp)!=plan['roster']['file_sha256']:
            raise ValueError('exact bound sampling plan and roster required')
        recipe={'schema':'saea-tfrl-current-half-result-recipe-v1','read_run_id':read_id,'run_id':run,
            'attempt_id':cell['attempt_id'],'rows':n,'split_identity_hash':exp['population']['split_identity_hash'],
            'sampling_plan':{'relpath':str(pp.relative_to(repo)),'sha256':sha(pp)},
            'roster':{'relpath':str(rp.relative_to(repo)),'sha256':sha(rp)},
            'scorer':'slurp-slu-official-train','arms':['s1','s2','s3','s4','aa','lock'],
            'official_scoring_jobs':7,'new_model_calls':0,'claims_benefit':False,
            'source_binding_sha256':sha(binding),'authority':AUTHORITY,
            'entrypoint':'scripts/analysis/tfrl_current_half_result.py',
            'request_gate':'verified complete bundle; exact plan/receipt/actual request seeds and common context',
            'credit':'row F1 minus lock; best intent-correct; positive only with stable aa/lock',
            'vote':'plurality intent, lowest-k tie, entity support >=2/4',
            'decision_scope':'Descriptive half only. Pool both halves and evaluate independent confirmation before improvement claim.',
            'prerequisite':'Full A0 gain lock before construction admission; this READ does not admit generation.'}
        if iteration>1:
            recipe.update(iteration=iteration,
                first_construction_decision={'relpath':str(first_decision.resolve().relative_to(data.resolve())),
                    'sha256':sha(first_decision)},
                request_gate='verified complete banked bundle; exact recorded bank augmentation/allocation; unchanged aa/lock; cross-fit source and per-row seeds')
        path=pkg/'analysis-recipe-v1.json'
        if path.exists() or '| '+read_id+' |' in text:raise ValueError('read already registered')
        pending.append((path,recipe))
    out.mkdir(parents=True)
    for path,recipe in pending:
        publish_new(path,raw(recipe));rid=recipe['read_run_id']
        rows.append(['2026-10-31',rid,'model-free-check','Complete real K4 half credit and derived vote','no model; official CPU scorer',
            'slurp TRAIN construction '+str(recipe['rows']),'discovery',recipe['split_identity_hash'],'no','0','0','0',
            'All generation and FIT costs retained',str(path.relative_to(repo)),
            f'Recipe SHA256 {sha(path)}; root countersigned under {AUTHORITY}, coordinator Codex; 7 whole-roster scoring jobs; generation remains held'])
    section='\n## Real K4 half result reads\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'
    section+='\n'.join('| '+' | '.join(row)+' |' for row in rows)+'\n'
    ledger.write_text(text+section)
    publish_new(out/'ledger-append.md',section.encode())
    result={'schema':'saea-tfrl-current-read-registration-v1','source_commit':head,'rows':625,
        'recipes':{str(p.relative_to(repo)):sha(p) for p,_ in pending},'new_model_calls':0,'new_scoring_jobs':0,
        'generation_readiness':'held','official_jobs_when_complete':14,'source_binding_sha256':sha(binding)}
    publish_new(out/'complete.json',raw(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','binding','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=1)
    p.add_argument('--first-decision',type=Path)
    print(json.dumps(register(**vars(p.parse_args()))))
