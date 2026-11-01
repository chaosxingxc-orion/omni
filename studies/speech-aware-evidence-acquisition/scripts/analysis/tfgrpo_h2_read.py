"""Official full-H2 learned/lock comparison; no generation or candidate promotion."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
import tfgrpo_h2_job as job
import stage3_rollout_score as score
import paired_ci as pc
from tfrl_current_half_result import score_table
from speech_aware_evidence_acquisition.analysis.paired import resampled_statistic
from speech_aware_evidence_acquisition.analysis.parallel import map_by_key
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
SEED=20261101
DRAWS=10000


def statistics(results,items,*,draws=DRAWS):
    ids=[r['sample_id'] for r in items]
    if len(ids)!=312 or len(set(ids))!=312 or set(results)!={'learned','lock'}:
        raise ValueError('complete exact312 learned/lock result pair required')
    if type(draws)is not int or draws<2:raise ValueError('at least two paired draws required')
    tables={a:score_table(r,ids) for a,r in results.items()}
    identity=lambda a:{k:a[k] for k in ('evaluator_commit','scorer_version','split_id','projection','filler_normalization')}
    if identity(results['learned']['aggregate'])!=identity(results['lock']['aggregate']):
        raise ValueError('official scorer identities differ')
    clusters={r['sample_id']:str(r['utterance_id']) for r in items}
    strata={r['sample_id']:r['stratum'] for r in items}
    if any(not x for x in [*clusters.values(),*strata.values()]):raise ValueError('utterance and scenario required')
    metrics={'slu_f1':pc.pooled_slu_f1,'intent':pc.intent_mean,'per_row_slu_f1':pc.per_row_f1_mean}
    def interval(metric):
        result=resampled_statistic(sample_ids=ids,statistic=lambda selected:metric(tables['learned'],selected)-metric(tables['lock'],selected),seed=SEED,n_replicates=draws,ci_level=.95,cluster_of=clusters.__getitem__,strata_of=strata.__getitem__)
        return pc._interval(result.replicates,result.point_estimate)
    contrasts=map_by_key({k:(lambda metric=f:interval(metric)) for k,f in metrics.items()},workers=3)
    scenarios={}
    for label in sorted(set(strata.values())):
        selected=[sid for sid in ids if strata[sid]==label]
        scenarios[label]={'rows':len(selected),'delta_pp':{k:100*(f(tables['learned'],selected)-f(tables['lock'],selected)) for k,f in metrics.items()}}
    gain=contrasts['slu_f1']['point_pp'];intent=contrasts['intent']['point_pp']
    return {'schema':'saea-tfgrpo-h2-paired-result-v1','rows':312,'contrast':'learned minus lock','metrics':contrasts,
        'official_aggregates':{a:r['aggregate']['official'] for a,r in results.items()},'scenario_table':scenarios,
        'seed':SEED,'draws':draws,'cluster_unit':'original-utterance','strata':'scenario',
        'numeric_gate_passed':gain>=1.0 and intent>=0.0,'gate':{'slu_gain_min_pp':1.0,'intent_gain_min_pp':0.0},
        'candidate_accepted':False,'claims_stable_gain':False,
        'limitations':['Development comparison, not untouched test.','Bootstrap quantifies roster sampling uncertainty conditional on observed decodes; no independent decoding variance estimate.','Numeric gate alone does not establish stable cross-task improvement.']}


def read(repo,data,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.data.lock import load_lock,umbrella_lock_path
    from registered_workload import validate_workload
    assert_execution_environment()
    if not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('owned analysis directory required')
    pkg=repo/'docs/readiness'/job.NAME;spec=job.read(pkg/'candidate.json');validate_workload(repo,data,job.NAME,spec)
    recipe=job.read(pkg/'analysis-recipe-v1.json')
    if (recipe['schema']!='saea-tfgrpo-h2-paired-read-v1' or recipe['rows']!=312 or recipe['generation_run_id']!=job.RUN
            or recipe['attempt_id']!='generation-v1' or recipe['arms']!=['learned','lock'] or recipe['delivered_arm']!='lock'
            or recipe['roster']!=spec['roster']):raise ValueError('exact preregistered H2 comparison required')
    directory=data/'runs'/job.RUN/'generation-v1';manifest=directory/(job.RUN+'.run-manifest.json')
    bundle=verify_bundle(manifest)
    roster=job.pinned(repo,spec['roster']);ids=[r['sample_id'] for r in roster['items']]
    if bundle.attempt_state!='FINALIZED' or len(bundle.split_ids)!=312 or set(bundle.split_ids)!=set(ids):raise ValueError('complete finalized full H2 required')
    fragments=bundle.manifest['config']['fragments']
    cfg=compose(repo,*(fragments[k] for k in ('model','dataset','baseline','experiment')))
    roles=cfg.values['native_episode']['roles'];group=score.parse_solver_arm_group(cfg.values,roles['solver'],roles['perception'],cfg.values['knowledge_stores'])
    if cfg.config_hash!=bundle.manifest['config']['config_hash']:raise ValueError('sealed H2 configuration changed')
    out.mkdir(parents=True,exist_ok=True)
    rendered=score.render_supply_variant_first_visit(bundle,group,{r['sample_id']:r['stratum'] for r in roster['items']},out,
        run_id=job.RUN,attempt_id='generation-v1',roster_sha256=spec['roster']['sha256'],config_hash=cfg.config_hash)
    if rendered['arm_order']!=['learned','lock']:raise ValueError('only exact paired arms required')
    summary=score.score_supply_variant_copies(score.DEFAULT_SCORER_ID,rendered,out,load_lock(umbrella_lock_path(repo)),data,
        score.sealed_per_sample(directory,job.RUN,score.DEFAULT_SCORER_ID))
    results={}
    for arm in ('learned','lock'):
        path=out/f'arm-{arm}-{score.DEFAULT_SCORER_ID}-scores.json';doc=job.read(path)
        if set(doc)!={score.DEFAULT_SCORER_ID}:raise ValueError('registered official scorer required')
        results[arm]=doc[score.DEFAULT_SCORER_ID]
    result=statistics(results,roster['items'])
    result.update(new_model_calls=0,new_tokenizer_endpoint_calls=0,official_arm_scoring_jobs=2,source=rendered['source'],
        inputs={str(pkg/'analysis-recipe-v1.json'):job.sha(pkg/'analysis-recipe-v1.json'),str(out/'score-summary.json'):job.sha(out/'score-summary.json')},
        score_receipts={a:job.pin(data,str((out/f'arm-{a}-{score.DEFAULT_SCORER_ID}-scores.json.receipt.json').relative_to(data))) for a in results})
    path=out/'paired-result.json';raw=job.raw(result)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('previous paired result differs')
    else:publish_new(path,raw)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(read(a.repo,a.data,a.out),sort_keys=True))
