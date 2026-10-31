"""Pool two complete real construction halves using official sufficient counts.

No scoring or model calls. The caller must first verify each registered half
result and its artifacts. Do not average half F1 or promote this construction
vote into a held-out system-gain claim.
"""
from tfrl_current_half_result import analyze,score_table,ARMS
import paired_ci


def pool(halves):
    if set(halves)!={'R01C-H1','R01C-H2'}:raise ValueError('both construction halves required')
    merged={arm:{} for arm in (*ARMS,'vote')};all_ids=set();iterations=set();summaries={}
    for half,(plan,results) in halves.items():
        if plan['half']!=half:raise ValueError('half plan identity differs')
        ids=[r['row_id'] for r in plan['row_plan']]
        if len(ids)!=(313 if half=='R01C-H1' else 312) or len(ids)!=len(set(ids)) or all_ids & set(ids):
            raise ValueError('exact disjoint 313/312 halves required')
        _,summaries[half]=analyze(plan,results)
        iterations.add(plan['iteration']);all_ids.update(ids)
        for arm in merged:merged[arm].update(score_table(results[arm],ids))
    if len(iterations)!=1:raise ValueError('construction iterations differ')
    identity=lambda a:tuple(a[k] for k in ('evaluator_commit','scorer_version','split_id','projection','filler_normalization'))
    if identity(halves['R01C-H1'][1]['lock']['aggregate'])!=identity(halves['R01C-H2'][1]['lock']['aggregate']):
        raise ValueError('halves have different official scorer identities')
    ids=sorted(all_ids);metrics={a:{'slu_f1':paired_ci.pooled_slu_f1(rows,ids),
        'intent_accuracy':paired_ci.intent_mean(rows,ids)} for a,rows in merged.items()}
    gain=100*(metrics['vote']['slu_f1']-metrics['lock']['slu_f1'])
    return {'schema':'saea-tfrl-current-pooled-construction-v1','rows':625,'iteration':iterations.pop(),
        'halves':summaries,'official_reaggregated':metrics,'vote_minus_lock_pp':gain,
        'construction_gate_passed':round(gain,6)>=1.0,'threshold_pp':1.0,
        'next_iteration_authorized':False,'claims_benefit':False,
        'remaining':'Independent system comparison and iteration acceptance required; construction gate alone is insufficient.'}
