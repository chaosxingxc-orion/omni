"""Diagnose fixed K4 vote losses from verified official counts; no new scoring.

Oracle comparisons describe candidate headroom only and cannot select a runtime
answer. Existing construction results remain the decision authority.
"""
import argparse,json,math,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from tfrl_recovered_consumers import load,POOL
from tfrl_current_sources import read,sha
from tfrl_current_pool import pool
from tfrl_current_half_result import score_table,ARMS
import paired_ci as ci


def diagnose(plan,results):
    ids=[r['row_id'] for r in plan['row_plan']]
    if not ids or len(ids)!=len(set(ids)) or set(results)!={*ARMS,'vote'}:
        raise ValueError('complete unique roster and seven columns required')
    tables={a:score_table(v,ids) for a,v in results.items()}
    counts=Counter();strata={};deltas={k:Counter() for k in ('word','char')}
    for sid in ids:
        rows={a:t[sid] for a,t in tables.items()};lock=rows['lock'];vote=rows['vote']
        li=lock['scenario_action_exact'];vi=vote['scenario_action_exact']
        candidates=[rows[a] for a in ARMS[:4]]
        counts[f'intent_lock{li}_vote{vi}']+=1
        correct=sum(r['scenario_action_exact'] for r in candidates)
        counts[f'candidate_correct_intents_{correct}']+=1
        lf=ci.row_f1(lock);vf=ci.row_f1(vote);best=max(ci.row_f1(r) for r in candidates)
        counts['vote_row_f1_better_lock']+=vf>lf+1e-12
        counts['vote_row_f1_worse_lock']+=vf<lf-1e-12
        counts['vote_row_f1_equal_lock']+=abs(vf-lf)<=1e-12
        counts['oracle_candidate_better_lock']+=best>lf+1e-12
        counts['oracle_candidate_better_vote']+=best>vf+1e-12
        counts['oracle_improvement_missed_by_vote']+=best>lf+1e-12 and vf<=lf+1e-12
        counts['correct_candidate_intent_lost_by_vote']+=correct>0 and not vi
        counts['all_candidates_wrong_intent']+=correct==0
        for r in rows.values():
            for key in ('slot_f1_exact_tp','slot_f1_exact_fp','slot_f1_exact_fn'):
                value=r[key]
                if type(value) not in (int,float) or value<0 or not math.isfinite(value):
                    raise ValueError('invalid exact slot count')
        counts['vote_loses_exact_slot_tp_vs_lock']+=vote['slot_f1_exact_tp']<lock['slot_f1_exact_tp']
        counts['vote_adds_exact_slot_fp_vs_lock']+=vote['slot_f1_exact_fp']>lock['slot_f1_exact_fp']
        counts['vote_adds_exact_slot_fn_vs_lock']+=vote['slot_f1_exact_fn']>lock['slot_f1_exact_fn']
        label=f'intent_lock{li}_vote{vi}'
        strata.setdefault(label,[]).append(sid)
        for kind in deltas:
            for stat in ('tp','fp','fn'):
                key=f'slu_f1_{kind}_{stat}'
                deltas[kind][stat]+=vote[key]-lock[key]
    groups={}
    for label,subset in strata.items():
        groups[label]={'rows':len(subset),'lock_slu_f1':ci.pooled_slu_f1(tables['lock'],subset),
            'vote_slu_f1':ci.pooled_slu_f1(tables['vote'],subset),
            'exact_slot_counts':{arm:{stat:sum(tables[arm][sid]['slot_f1_exact_'+stat] for sid in subset)
                for stat in ('tp','fp','fn')} for arm in ('lock','vote')}}
    return {'rows':len(ids),'counts':dict(counts),'vote_minus_lock_slu_counts':{k:dict(v) for k,v in deltas.items()},
        'intent_strata':groups,'metrics':{a:{'slu_f1':ci.pooled_slu_f1(t,ids),'intent':ci.intent_mean(t,ids)} for a,t in tables.items()}}


def run(repo,data,source,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    assert_execution_environment();head=require_clean_study_repo(repo);source_head=require_clean_study_repo(source)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned output required')
    _,_,halves=load(source,data)
    verified=pool({v['plan']['half']:(v['plan'],v['results']) for v in halves.values()})
    path=data/'analysis'/POOL/'complete.json';record=read(path)
    if any(record.get(k)!=v for k,v in verified.items()) or record['sources']!={h:v['source'] for h,v in halves.items()}:
        raise ValueError('recorded pool differs')
    # Retain official sufficient counts; do not average the two half F1 values.
    merged={};plan={'row_plan':sum([v['plan']['row_plan'] for v in halves.values()],[])}
    for arm in (*ARMS,'vote'):
        rows=sum([v['results'][arm]['per_sample'] for v in halves.values()],[])
        ids=[r['sample_id'] for r in rows]
        merged[arm]={'per_sample':rows,'aggregate':{'samples':625,'identity_check':{'passed':True},
            'official':{'slu_f1':ci.pooled_slu_f1({r['sample_id']:r for r in rows},ids)}}}
    result=diagnose(plan,merged)
    result.update(schema='saea-tfrl-first625-failure-diagnosis-v1',implementation_commit=head,source_commit=source_head,
        pool_sha256=sha(path),sources=record['sources'],new_model_calls=0,new_tokenizer_calls=0,new_scoring_jobs=0,
        generation_authorized=False,system_accepted=False,
        limitations=['Post-hoc construction diagnostics; no independent gain claim.',
            'Oracle uses gold and is not a deployable answer selector.',
            'Shared perception and evidence errors cannot be distinguished causally from solver errors by final scores alone.',
            'Stratum F1 values are not additive contributions to the full micro F1.'])
    publish_new(out,(json.dumps(result,indent=2)+'\n').encode());return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','source','out'):p.add_argument('--'+k,type=Path,required=True)
    result=run(**vars(p.parse_args()));print(json.dumps({'rows':result['rows'],'counts':result['counts']}))
