"""Official full-H2 long-ID/short-ID/lock comparison; no candidate promotion."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
import tfgrpo_render_h2_job as job
import stage3_rollout_score as score
import paired_ci as pc
from tfrl_current_half_result import score_table
from speech_aware_evidence_acquisition.analysis.paired import resampled_statistic
from speech_aware_evidence_acquisition.analysis.parallel import map_by_key
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
SEED=20261101
DRAWS=10000


from tfgrpo_h2_read import statistics as paired_statistics, label_membership, label_diagnostics, registered_scorer


def statistics(results,items,*,draws=DRAWS):
    if set(results)!={'learned','compact','lock'}:raise ValueError('all three complete arms required')
    def pair(a,b):return paired_statistics({'learned':results[a],'lock':results[b]},items,draws=draws)
    return {'schema':'saea-tfgrpo-render-h2-result-v1','rows':312,
        'compact_vs_lock':pair('compact','lock'),'compact_vs_learned':pair('compact','learned'),
        'learned_vs_lock':pair('learned','lock'),'candidate_accepted':False,'claims_stable_gain':False,
        'latency_causal_claim':False,'order':['learned','compact','lock'],
        'primary_contrast':'compact_vs_lock','representation_contrast':'compact_vs_learned'}


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
    if (recipe['schema']!='saea-tfgrpo-render-h2-read-v1' or recipe['rows']!=312 or recipe['generation_run_id']!=job.RUN
            or recipe['attempt_id']!='generation-v1' or recipe['arms']!=['learned','compact','lock'] or recipe['delivered_arm']!='lock'
            or recipe['roster']!=spec['roster']):raise ValueError('exact preregistered H2 comparison required')
    directory=data/'runs'/job.RUN/'generation-v1';manifest=directory/(job.RUN+'.run-manifest.json')
    bundle=verify_bundle(manifest)
    roster=job.pinned(repo,spec['roster']);ids=[r['sample_id'] for r in roster['items']]
    if bundle.attempt_state!='FINALIZED' or len(bundle.split_ids)!=312 or set(bundle.split_ids)!=set(ids):raise ValueError('complete finalized full H2 required')
    fragments=bundle.manifest['config']['fragments']
    cfg=compose(repo,*(fragments[k] for k in ('model','dataset','baseline','experiment')))
    roles=cfg.values['native_episode']['roles'];group=score.parse_solver_arm_group(cfg.values,roles['solver'],roles['perception'],cfg.values['knowledge_stores'])
    if cfg.config_hash!=bundle.manifest['config']['config_hash']:raise ValueError('sealed H2 configuration changed')
    scorer_id=registered_scorer(cfg.values,bundle.manifest)
    out.mkdir(parents=True,exist_ok=True)
    rendered=score.render_supply_variant_first_visit(bundle,group,{r['sample_id']:r['stratum'] for r in roster['items']},out,
        run_id=job.RUN,attempt_id='generation-v1',roster_sha256=spec['roster']['sha256'],config_hash=cfg.config_hash)
    if rendered['arm_order']!=['learned','compact','lock']:raise ValueError('only exact paired arms required')
    summary=score.score_supply_variant_copies(scorer_id,rendered,out,load_lock(umbrella_lock_path(repo)),data,
        score.sealed_per_sample(directory,job.RUN,scorer_id))
    results={}
    for arm in ('learned','compact','lock'):
        path=out/f'arm-{arm}-{scorer_id}-scores.json';doc=job.read(path)
        if set(doc)!={scorer_id}:raise ValueError('registered official scorer required')
        results[arm]=doc[scorer_id]
    result=statistics(results,roster['items'])
    from speech_aware_evidence_acquisition.data.slurp import load_slurp_split
    labels=label_membership(roster['items'],load_slurp_split(load_lock(umbrella_lock_path(repo)),data,'train'))
    result['label_diagnostics']={name:label_diagnostics({'learned':results[a],'lock':results[b]},roster['items'],labels)
        for name,a,b in [('compact_vs_lock','compact','lock'),('compact_vs_learned','compact','learned')]}
    result['numeric_gate_passed']=result['compact_vs_lock']['numeric_gate_passed']
    result.update(scorer_id=scorer_id,new_model_calls=0,new_tokenizer_endpoint_calls=0,official_arm_scoring_jobs=3,source=rendered['source'],
        inputs={str(pkg/'analysis-recipe-v1.json'):job.sha(pkg/'analysis-recipe-v1.json'),str(out/'score-summary.json'):job.sha(out/'score-summary.json')},
        score_receipts={a:job.pin(data,str((out/f'arm-{a}-{scorer_id}-scores.json.receipt.json').relative_to(data))) for a in results})
    path=out/'three-arm-result.json';raw=job.raw(result)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('previous paired result differs')
    else:publish_new(path,raw)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(read(a.repo,a.data,a.out),sort_keys=True))
