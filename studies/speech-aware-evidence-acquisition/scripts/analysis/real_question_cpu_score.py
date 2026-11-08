"""Score cached candidates against explicitly supplied development gold."""
import argparse
import json
import resource
import time
from pathlib import Path
from real_question_cpu_prepare import checked, save


def main():
    p=argparse.ArgumentParser(); p.add_argument('cache'); p.add_argument('gold_pin'); p.add_argument('output'); p.add_argument('--reference'); p.add_argument('--baseline'); p.add_argument('--expected',type=int,required=True); a=p.parse_args()
    start=time.monotonic(); cache=json.loads(Path(a.cache).read_text()); labels=json.loads(checked(json.loads(Path(a.gold_pin).read_text())).read_text())
    gold={x['sample_id']:x['gold_intent'] for x in labels}; rows=cache['rows']
    if len(gold)!=len(labels) or len(gold)!=a.expected or set(gold)!=set(rows): raise ValueError('gold/cache identity or count mismatch')
    reference=json.loads(Path(a.reference).read_text()) if a.reference else None
    if reference is not None:
        reference={r['sample_id']:r for r in reference}
        if set(reference)!=set(gold): raise ValueError('reference identity mismatch')
    baseline=json.loads(Path(a.baseline).read_text())['rows'] if a.baseline else None
    if baseline is not None and set(baseline)!=set(gold): raise ValueError('baseline identity mismatch')
    if baseline is not None and reference is not None: raise ValueError('choose one paired baseline')
    metrics={'n':len(gold),'recall3_count':0,'recall5_count':0,'mrr':0.,'missing':0}; per={}; paired={str(k):{'repair':0,'regression':0,'both_hit':0,'both_miss':0} for k in (3,5)}; differences=0
    for sid,g in gold.items():
        ranking=[x[0].split(':',1)[-1] for x in rows[sid]]; rank=ranking.index(g)+1 if g in ranking else None
        m=per.setdefault(g,{'n':0,'recall3_count':0,'recall5_count':0,'mrr':0.,'missing':0})
        for item in (metrics,m):
            item['n'] += (item is m); item['recall3_count'] += rank is not None and rank<=3; item['recall5_count'] += rank is not None and rank<=5; item['mrr'] += 1/rank if rank else 0; item['missing'] += rank is None
        if reference is not None or baseline is not None:
            if reference is not None:
                ref=reference[sid]; rr=ref['routes']['question_nn1_by_intent']; old_rank=rr['gold_rank']; differences += ranking[:5]!=rr['top5'] or rank!=old_rank or ref['gold_intent']!=g
            else:
                old_order=[x[0].split(':',1)[-1] for x in baseline[sid]]; old_rank=old_order.index(g)+1 if g in old_order else None
            for k in (3,5):
                old=old_rank is not None and old_rank<=k; new=rank is not None and rank<=k
                paired[str(k)]['both_hit' if old and new else 'repair' if new else 'regression' if old else 'both_miss']+=1
    for item in [metrics,*per.values()]: item['mrr']/=item['n']
    save(a.output,dict(metrics=metrics,per_intent=per,paired_vs_reference=paired if reference is not None or baseline is not None else None,reference_row_differences=differences if reference else None,score_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

if __name__=='__main__': main()
