"""Read sealed native-node development arms, coverage and cost; no model calls."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from tfrl_hardset_a0_workload import read,raw,sha
from tfrl_hardset_a0_registration import pinned
from tfrl_h1_observation_read import summarize_scores,assert_registered_bundle
from h1_thinking_comparison import native_costs
from intent_slot_workload import KIND,ARMS


def run(repo,data,out,name):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.data.slurp import load_slurp_split
    from speech_aware_evidence_acquisition.data.lock import load_lock,umbrella_lock_path
    from tfgrpo_h2_read import label_membership
    from registered_workload import validate_workload
    assert_execution_environment()
    if not out.resolve().is_relative_to((data/'work/codex').resolve()):raise ValueError('owned result output required')
    pkg=repo/'docs/readiness'/name;spec=read(pkg/'candidate.json')
    if spec['workload_kind']!=KIND:raise ValueError('node registration required')
    validate_workload(repo,data,name,spec)
    rid=spec['run_id'];aid=spec['attempt_id'];directory=data/'runs'/rid/aid;mp=directory/(rid+'.run-manifest.json')
    bundle=verify_bundle(mp);roster=pinned(repo,spec['roster']);ids=[r['sample_id'] for r in roster['items']]
    if bundle.attempt_state!='FINALIZED' or bundle.manifest['status']!='COMPLETED' or len(bundle.split_ids)!=80 or set(bundle.split_ids)!=set(ids):raise ValueError('complete finalized80 required')
    assert_registered_bundle(bundle.manifest,read(pkg/'generation-plan-registered-v1.json'),name,rid,aid)
    f=bundle.manifest['config']['fragments'];cfg=compose(repo,*(f[k] for k in ('model','dataset','baseline','experiment')))
    if cfg.config_hash!=bundle.manifest['config']['config_hash']:raise ValueError('sealed config changed')
    sp=directory/(rid+'.scores.json');scores=read(sp);scorer='slurp-slu-official-train'
    if set(scores)!={scorer,'cost'}:raise ValueError('official TRAIN and cost only')
    result=summarize_scores(scores[scorer],roster['items'],expected_rows=80)
    labels=label_membership(roster['items'],load_slurp_split(load_lock(umbrella_lock_path(repo)),data,'train'))
    covered=0;missing_p=0;failures={};coverage=[]
    for row in bundle.outputs_rows:
        receipt=row['native_episode_receipt'];visits=receipt['role_visits'];ps=[v for v in visits if v['stage']=='perception']
        reason=receipt.get('terminal_disposition',{}).get('reason')
        if reason:failures[reason]=failures.get(reason,0)+1
        candidates=[]
        if ps:
            contact=ps[0]['execution_receipt']['final_contact'];text=bundle.trace_records[contact['request_id']+'/response']['text']
            try:candidates=[v['intent_id'] for v in json.loads(text)['candidates']]
            except (ValueError,KeyError,TypeError):missing_p+=1
        else:missing_p+=1
        hit=labels[row['sample_id']]['intent'] in candidates;covered+=hit
        coverage.append({'sample_id':row['sample_id'],'candidate_count':len(candidates),'gold_in_candidates':hit})
    result.update(schema='saea-intent-slot-arm-result-v1',node_arm=spec['job_config']['node_arm'],run_id=rid,
        candidate_coverage={'covered':covered,'total':80,'invalid_or_absent_perception':missing_p,'rows':coverage},
        terminal_dispositions=failures,native_role_costs=native_costs(bundle.outputs_rows,bundle.trace_records),
        cost=bundle.manifest['cost'],new_model_calls=0,new_scoring_jobs=0,candidate_accepted=False,
        scorer_identity={k:scores[scorer]['aggregate'][k] for k in ('evaluator_commit','scorer_version','split_id','projection','filler_normalization')},
        source={'manifest_sha256':sha(mp),'scores_sha256':sha(sp),'roster_sha256':spec['roster']['sha256'],'config_hash':cfg.config_hash})
    out.mkdir(parents=True,exist_ok=True);path=out/'complete.json'
    if path.exists():
        if path.read_bytes()!=raw(result):raise ValueError('result changed')
    else:publish_new(path,raw(result))
    return result

def compare(results):
    if set(results)!=set(ARMS):raise ValueError('three complete native node arms required')
    reference=results[ARMS[0]]
    ids=[r['sample_id'] for r in reference['candidate_coverage']['rows']]
    for arm,result in results.items():
        if (result['node_arm']!=arm or result['rows']!=80 or result['scorer_identity']!=reference['scorer_identity']
                or [r['sample_id'] for r in result['candidate_coverage']['rows']]!=ids):
            raise ValueError('node arms differ in identity, scorer or population')
    return {'schema':'saea-intent-slot-comparison-v1','rows':80,'arms':results,
            'contrasts':{arm:{'slu_f1_delta_pp':100*(r['official']['slu_f1']-reference['official']['slu_f1']),
                             'intent_delta_pp':100*(r['intent_accuracy']-reference['intent_accuracy']),
                             'strict_delta_rows':r['task_correct_rows']-reference['task_correct_rows']}
                         for arm,r in results.items() if arm!=ARMS[0]},
            'candidate_accepted':False,'limitations':['Fixed development comparison; no automatic promotion.',
              'Separate runs do not share Perception contacts.','Candidate outside recovery disabled in split graph.']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--name',required=True);print(json.dumps(run(**vars(p.parse_args())),sort_keys=True))
