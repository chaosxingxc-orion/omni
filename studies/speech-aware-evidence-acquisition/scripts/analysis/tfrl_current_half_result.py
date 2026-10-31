"""Consume a complete real K4 half through request audit, official scores and credit.

No inference. The registered recipe pins the sampling plan and roster. Shared
scorers checkpoint complete columns; an incomplete scorer invocation is never
replayed automatically. No reward is derived before source/request validation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/analysis'),str(REPO/'scripts/flights/nx'),str(REPO/'scripts/checks')]
import stage3_rollout_score as s3
import tfrl_seed_request_audit as audit
import tfrl_v2_sampler as sampler
import paired_ci
from slurp_tfrl_v2_port_read import vote_answer
SCORER='slurp-slu-official-train'
ARMS=('s1','s2','s3','s4','aa','lock')


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def score_table(result,ids):
    rows=result['per_sample'];keys=[r['sample_id'] for r in rows];a=result['aggregate']
    if (len(keys)!=len(ids) or len(set(keys))!=len(keys) or set(keys)!=set(ids)
            or a['samples']!=len(ids) or a['identity_check']['passed'] is not True):
        raise ValueError('official score must cover the whole half exactly once')
    table={r['sample_id']:r for r in rows}
    for row in rows:
        paired_ci.row_counts(row)
        if row['scenario_action_exact'] not in (0,1):raise ValueError('invalid intent score')
    official=a['official']['slu_f1']
    if type(official) not in (int,float) or not math.isfinite(official) or not 0 <= official <= 1:
        raise ValueError('invalid official micro F1')
    if abs(paired_ci.pooled_slu_f1(table,ids)-official)>1e-6:
        raise ValueError('official micro F1 differs from per-row counts')
    return table


def analyze(plan, results):
    ids=[r['row_id'] for r in plan['row_plan']]
    if set(results)!=set(ARMS)|{'vote'}:raise ValueError('all six arms and vote required')
    tables={arm:score_table(result,ids) for arm,result in results.items()}
    identity=lambda a:{k:a[k] for k in ('evaluator_commit','scorer_version','split_id','projection','filler_normalization')}
    if any(identity(r['aggregate'])!=identity(results['lock']['aggregate']) for r in results.values()):
        raise ValueError('official scoring identities differ')
    credit=sampler.credit_rows(ids,plan['iteration'],4,sampler.scores_from_counts(tables),twin_arm='aa')
    gain=100*(paired_ci.pooled_slu_f1(tables['vote'],ids)-paired_ci.pooled_slu_f1(tables['lock'],ids))
    return credit,{'rows':len(ids),'credit':sampler.credit_summary(credit),'vote_minus_lock_pp':gain,
        'official':{a:r['aggregate']['official'] for a,r in results.items()},
        'scope':'Construction-half diagnostics; pool both halves before any iteration decision.',
        'next_iteration_authorized':False,'claims_benefit':False}


def votes_from_copies(manifest):
    copies={}
    for arm in ARMS[:4]:
        entry=manifest['copies'][arm];path=Path(entry['path'])
        if sha(path)!=entry['sha256']:raise ValueError('candidate copy changed')
        rows=list(s3.iter_jsonl(path));ids=[r['sample_id'] for r in rows]
        if len(ids)!=manifest['rows'] or len(ids)!=len(set(ids)):raise ValueError('incomplete vote source')
        copies[arm]={r['sample_id']:json.loads(r['text']) for r in rows}
    ids=set(copies['s1'])
    if any(set(c)!=ids for c in copies.values()):raise ValueError('vote sources differ')
    return [{'sample_id':sid,'text':s3.render_answer(vote_answer([copies[a][sid] for a in ARMS[:4]]))}
            for sid in sorted(ids)]


def validate_recipe(repo,path):
    import flight_registration_check as registration
    path=path.resolve()
    if not path.is_relative_to(repo.resolve()):raise ValueError('repository recipe required')
    recipe=read(path)
    if (recipe['schema']!='saea-tfrl-current-half-result-recipe-v1' or recipe['scorer']!=SCORER
            or recipe['arms']!=list(ARMS) or recipe['new_model_calls']!=0
            or recipe['official_scoring_jobs']!=7 or recipe['claims_benefit'] is not False):
        raise ValueError('exact real half read recipe required')
    entries=registration.parse_ledger(repo,Path('docs/exposure-ledger.md')).get(recipe['read_run_id'],[])
    if (len(entries)!=1 or not registration.is_countersigned(entries[0])
            or entries[0]['artifacts']!=str(path.relative_to(repo.resolve()))
            or sha(path) not in entries[0].get('notes','')):
        raise ValueError('countersigned exact recipe required before result consumption')
    return recipe


def run(repo,data,recipe_path,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.data.lock import load_lock,umbrella_lock_path
    from speech_aware_evidence_acquisition.core.scoring_batch import score_complete_derived_copy
    assert_execution_environment();recipe=validate_recipe(repo,recipe_path)
    if not out.resolve().is_relative_to(data.resolve()/'analysis'):raise ValueError('owned analysis output required')
    def pinned(key):
        pin=recipe[key];path=(repo/pin['relpath']).resolve()
        if not path.is_relative_to(repo.resolve()) or sha(path)!=pin['sha256']:raise ValueError('recipe input pin differs')
        return path
    pp=pinned('sampling_plan');rp=pinned('roster');plan=read(pp)
    if (plan['run_id']!=recipe['run_id'] or plan['attempt_id']!=recipe['attempt_id']
            or plan['roster']['file_sha256']!=sha(rp) or plan['roster']['rows']!=recipe['rows']
            or plan['roster']['split_identity_hash']!=recipe['split_identity_hash']):
        raise ValueError('recipe and plan identity differ')
    directory=data/'runs'/plan['run_id']/plan['attempt_id'];mp=directory/(plan['run_id']+'.run-manifest.json')
    bundle=verify_bundle(mp)
    if (bundle.attempt_state!='FINALIZED' or bundle.manifest['status']!='COMPLETED'
            or bundle.manifest['run_id']!=plan['run_id']
            or bundle.manifest['attempt']['attempt_id']!=plan['attempt_id']):raise ValueError('finalized planned half required')
    f=bundle.manifest['config']['fragments'];cfg=compose(repo,*(f[k] for k in ('model','dataset','baseline','experiment')))
    if cfg.config_hash!=bundle.manifest['config']['config_hash']:raise ValueError('sealed config changed')
    from tfrl_current_bank_audit import verify_for_values
    check=verify_for_values(plan,bundle.outputs_rows,bundle.trace_records,values=cfg.values,data=data)
    roles=cfg.values['native_episode']['roles']
    group=s3.parse_solver_arm_group(cfg.values,roles['solver'],roles['perception'],cfg.values['knowledge_stores'])
    roster,roster_sha,_=s3.load_roster(repo,str(rp.relative_to(repo)))
    out.mkdir(parents=True,exist_ok=True)
    s3._immutable_json(out/'request-audit.json',{**check,'plan_sha256':sha(pp),'manifest_sha256':sha(mp)})
    render=s3.render_supply_variant_first_visit(bundle,group,roster,out/'columns',run_id=plan['run_id'],
        attempt_id=plan['attempt_id'],roster_sha256=roster_sha,config_hash=cfg.config_hash)
    lock=load_lock(umbrella_lock_path(repo));delivered=s3.sealed_per_sample(directory,plan['run_id'],SCORER)
    s3.score_supply_variant_copies(SCORER,render,out/'columns',lock,data,delivered)
    results={a:read(out/'columns'/f'arm-{a}-{SCORER}-scores.json')[SCORER] for a in ARMS}
    vote_path=out/'columns/arm-vote-outputs.jsonl';vote_sha=s3._immutable_jsonl(vote_path,votes_from_copies(render))
    results['vote']=score_complete_derived_copy(parent_manifest=mp,outputs_copy=vote_path,expected_copy_sha256=vote_sha,
        scorer_id=SCORER,result_path=out/'columns'/f'arm-vote-{SCORER}-scores.json',lock=lock,data_root=data,
        column_binding={'arm':'vote','plan_sha256':sha(pp),'render_manifest_sha256':sha(out/'columns/render-manifest.json'),
                        'recipe_sha256':sha(recipe_path),'rule':'plurality intent; support>=2/4 entities; lowest-k intent tie'})
    credit,summary=analyze(plan,results)
    s3._immutable_jsonl(out/'credit.jsonl',credit)
    pins={str(p.relative_to(out)):sha(p) for p in sorted((out/'columns').glob('*')) if p.is_file() and p.suffix!='.lock'}
    summary.update(schema='saea-tfrl-current-half-result-v1',run_id=plan['run_id'],attempt_id=plan['attempt_id'],
        half=plan['half'],iteration=plan['iteration'],request_audit=check,recipe_sha256=sha(recipe_path),
        plan_sha256=sha(pp),manifest_sha256=sha(mp),credit_sha256=sha(out/'credit.jsonl'),artifacts=pins,
        new_model_calls=0,official_scoring_jobs=7)
    s3._immutable_json(out/'complete.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','recipe_path','out'):p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args()))))
