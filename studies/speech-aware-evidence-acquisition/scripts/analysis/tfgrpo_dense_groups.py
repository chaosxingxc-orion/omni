"""Export fixed H1 binary-allwrong partial-credit contrasts; never read H2 labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from tfgrpo_seed_groups import group_from_row, gold_train, file_sha
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (
    DENSE_REWARD, canonical, sha, empty_experience, intent_slot_reward, group_has_contrast)


def select_ids(tables,ids):
    if set(tables)!={'s1','s2','s3','s4'} or any(set(t)!=set(ids) for t in tables.values()):
        raise ValueError('complete same-roster four-arm counts required')
    selected=[];tied=[];excluded=[]
    for sid in ids:
        rows=[tables[a][sid] for a in ('s1','s2','s3','s4')]
        rewards=[intent_slot_reward(r) for r in rows]
        if any(r==1 for r in rewards):excluded.append(sid)
        elif group_has_contrast([{'reward':r} for r in rewards],DENSE_REWARD):selected.append(sid)
        else:tied.append(sid)
    return selected,tied,excluded


def export(source,data,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.data.lock import load_lock,umbrella_lock_path
    from tfrl_current_half_result import score_table
    import tfrl_h1_union as union
    assert_execution_environment();source_head=require_clean_study_repo(source)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned export directory required')
    part=union.run(source,data,data/'analysis'/union.READ,verify_only=True)
    ids=[r['row_id'] for r in part['plan']['row_plan']]
    if len(ids)!=313 or len(set(ids))!=313:raise ValueError('fixed full H1 required')
    tables={a:score_table(part['results'][a],ids) for a in ('s1','s2','s3','s4')}
    scored_answers={}
    for arm in tables:
        path=data/'analysis'/union.READ/'columns'/f'arm-{arm}-outputs.jsonl'
        records=[json.loads(x) for x in path.read_text().splitlines()]
        scored_answers[arm]={r['sample_id']:r['text'] for r in records}
        if len(records)!=313 or set(scored_answers[arm])!=set(ids):raise ValueError('complete scored answers required')
    selected,tied,excluded=select_ids(tables,ids)
    if (len(selected),len(tied),len(excluded))!=(51,123,139):raise ValueError('preregistered H1 reward distribution differs')
    # H2 identities only, no labels or outcomes. Keep the source halves disjoint.
    h2=json.loads((source/'docs/receipts/rosters/slurp-stage3-construction-r01c-h2.json').read_bytes())
    if set(ids)&{r['sample_id'] for r in h2['items']}:raise ValueError('H1/H2 sample overlap')
    wanted=set(selected);gold,gold_pin=gold_train(load_lock(umbrella_lock_path(source)),data,wanted)
    cfg=part['config'];values=cfg.values
    # Read exact configured model name from the frozen continuation experiment.
    exp=json.loads((source/f'configs/experiment/{union.NAME}.json').read_bytes());cell,=exp['cells'];model_id=cell['model_fragment']
    modelrel=f'configs/model/{model_id}.json'
    policy={'model':{'relpath':modelrel,'sha256':file_sha(source/modelrel)},'policy':values['policy'],
        'native_episode':values['native_episode'],'knowledge_stores':values['knowledge_stores'],
        'model_decoding':values['decoding_params'],'task_instruction_template':values['task_instruction_template'],
        'rollout_kind':'shared-perception-adaptation'}
    parent=empty_experience(model_id);ph=sha(policy);groups=[];custody=[];seen=set()
    for segment in part['segments']:
        for row in segment['rows'].outputs_rows:
            sid=row['sample_id']
            if sid not in wanted:continue
            if sid in seen:raise ValueError('duplicate selected source')
            seen.add(sid)
            g,pins=group_from_row(row,segment['rows'].trace_records,tables,gold[sid],model_id=model_id,
                policy_sha256=ph,experience_sha256=parent['sha256'],namespace=segment['namespace'],reward_mode=DENSE_REWARD,scored_answers=scored_answers)
            if any(r['reward']==1 for r in g['rollouts']):raise ValueError('full-success trajectory in allwrong selection')
            groups.append(g);custody.extend(pins)
    if seen!=wanted or len(custody)!=204:raise ValueError('missing source trajectories')
    groups.sort(key=lambda g:g['query_id'])
    roster={r['sample_id']:r for r in part['roster']['items']}
    document={'schema':'saea-tfgrpo-dense-h1-groups-v1','source_half':'h1','evaluation_half':'h2',
        'model_id':model_id,'rollout_kind':'shared-perception-adaptation','policy':policy,'policy_sha256':ph,
        'parent':parent,'groups':groups,'source':part['source'],'gold':gold_pin,'custody':custody,
        'reward_mode':DENSE_REWARD,'roster_items':[roster[g['query_id']] for g in groups]}
    coverage={'intents':len({gold[s]['scenario']+'_'+gold[s]['action'] for s in selected}),
        'slot_types':len({e['type'] for s in selected for e in gold[s]['entities']}),
        'observed_intent_slot_relations':len({(gold[s]['scenario']+'_'+gold[s]['action'],e['type']) for s in selected for e in gold[s]['entities']})}
    result={'schema':'saea-tfgrpo-dense-export-v1','source_commit':source_head,'source':part['source'],
        'population_rows':313,'selected_groups':51,'trajectories':204,'remaining_allwrong_tied':123,
        'excluded_any_correct_groups':139,'normal_learning_calls':307,'reward_mode':DENSE_REWARD,
        'selected_coverage':coverage,'new_model_calls':0,'new_scoring_jobs':0,'test_labels_read':False,'H2_labels_read':False,
        'source_costs_retained':True,'artifact_sha256':hashlib.sha256(canonical(document)).hexdigest(),
        'selection_ids_sha256':sha(sorted(selected)),'tied_ids_sha256':sha(sorted(tied)),
        'limits':['Historical trajectories use empty experience; no new-policy rollout claim.',
                  'Partial reward is for normalized arm answer, not raw Solver text alone.',
                  '51 groups do not cover the full hard distribution; 123 tied allwrong retained.']}
    out.mkdir(parents=True);publish_new(out/'h1.json',canonical(document));publish_new(out/'complete.json',canonical(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','data','out'):p.add_argument('--'+k,type=Path,required=True)
    print(json.dumps(export(**vars(p.parse_args())),sort_keys=True))
