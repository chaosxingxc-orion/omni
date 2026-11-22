"""Analyze shape warmup and mixed-step sizes in two adjacent isolated trace windows."""
import argparse,json,math
from pathlib import Path
from statistics import median

def summarize(warm,measured,n,bin_size):
    if n<=0 or bin_size<=0:raise ValueError('invalid denominators')
    rows=warm['rows']+measured['rows']
    if not warm['rows'] or not measured['rows'] or any(b['index']!=a['index']+1 for a,b in zip(rows,rows[1:])):
        raise ValueError('non-adjacent iteration windows')
    for w in (warm,measured):
        if sum(r['generation_tokens'] for r in w['rows'])!=w['intervals']:raise ValueError('interval coverage differs')
    first=[];reused=[];seen=set()
    for r in rows:
        if r['kind']=='decode':
            size=r['generation_tokens']
            (reused if size in seen else first).append(r);seen.add(size)
    first_timed=[r for r in first if r['index']>=measured['rows'][0]['index']]
    def weighted(rs):return sum(r['generation_tokens']*r['elapsed_ms'] for r in rs)/1000
    bins={}
    for r in measured['rows']:
        if r['kind']=='mixed':bins.setdefault(math.ceil(r['context_tokens']/bin_size)*bin_size,[]).append(r)
    return dict(round_hashes=[w['round_sha256'] for w in (warm,measured)],
        first_shapes=[dict(step=r['index'],batch_size=r['generation_tokens'],ms=r['elapsed_ms']) for r in first],
        reused_steps=len(reused),reused_ms_range=[min(r['elapsed_ms'] for r in reused),max(r['elapsed_ms'] for r in reused)] if reused else None,
        first_shape_timed_seconds_per_request=weighted(first_timed)/n,
        mixed_bins=[dict(context_tokens_above=k-bin_size,context_tokens_at_most=k,n_steps=len(rs),
            step_p50_ms=median(r['elapsed_ms'] for r in rs),weighted_seconds=weighted(rs)) for k,rs in sorted(bins.items())],
        scope='first observed pure-decode shapes, not a measurement of graph capture; source-round hashes should be checked against the lane')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('warm',type=Path);p.add_argument('measured',type=Path)
    p.add_argument('--requests',type=int,required=True);p.add_argument('--bin-size',type=int,required=True);a=p.parse_args()
    print(json.dumps(summarize(json.loads(a.warm.read_bytes()),json.loads(a.measured.read_bytes()),a.requests,a.bin_size),indent=2))
