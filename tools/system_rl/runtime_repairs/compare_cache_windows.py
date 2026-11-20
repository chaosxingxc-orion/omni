"""Compare actual cold, prime, partial-hit TP traces in absolute token order."""
import argparse
import json
from pathlib import Path


def compare(prime, warm, cold):
    def ranks(trace):
        rows = {w['rank']: w['prefix_trace'] for w in trace['workers']}
        if len(trace['workers']) != 4 or set(rows) != set(range(4)):
            raise ValueError('expected four distinct TP ranks')
        return rows
    p, w, c = map(ranks, (prime, warm, cold))
    cut = w[0]['first_computed']
    n = len(cold['prompt_token_ids'])
    if (not 0 < cut < n or warm['prompt_token_ids'] != cold['prompt_token_ids']
            or prime['prompt_token_ids'][:cut] != cold['prompt_token_ids'][:cut]
            or prime['prompt_token_ids'] == cold['prompt_token_ids']):
        raise ValueError('target or shared prefix identity differs')
    result = []
    for i in sorted(w):
        for obj, offset, trace in ((p, 0, prime), (w, cut, warm), (c, 0, cold)):
            if (obj[i]['first_computed'], obj[i]['prompt_length']) != (offset, len(trace['prompt_token_ids'])):
                raise ValueError('unexpected observed scheduler window')
        def sites(obj, a, b):
            suffix = f':{a}:{b}'
            return {k.removesuffix(suffix): v for k, v in obj[i]['rows'].items() if k.endswith(suffix)}
        def diff(x, y):
            if not x or x.keys() != y.keys():
                raise ValueError('incomplete paired trace sites')
            return sorted(k for k in x if x[k] != y[k])
        pr = sites(p, 0, cut)
        su = sites(w, cut, n)
        if pr.keys() != su.keys():
            raise ValueError('different operator coverage')
        result.append(dict(rank=i, sites=len(su), prefix_diff=diff(pr, sites(c, 0, cut)),
                           suffix_diff=diff(su, sites(c, cut, n))))
    # For the expanded trace, compare actual dataflow, including both
    # residual inputs and the independent Q/K/V branches. A candidate is a
    # first observed boundary, not an assertion about an unobserved kernel.
    for i,r in enumerate(result):
        keys={k.rsplit(':',2)[0] for k in w[i]['rows']}
        layers=sorted(int(k.split('.')[0][2:]) for k in keys if k.endswith('.in.hidden'))
        changed=set(r['suffix_diff'])
        if not layers:continue
        first=[]
        for layer in layers:
            b=f'op{layer}.';nin=f'{layer}.in'
            dep={b+'in.sum':[b+'in.hidden',b+'in.residual'],nin:[b+'in.sum']}
            for q in ('q','k','v'):
                s=b+q
                dep.update({s+'.base':[nin],s+'.a':[nin],s+'.b':[s+'.a'],s:[s+'.base',s+'.b']})
            dep[b+'attn']=[b+q for q in ('q','k','v')]+['positions']
            if b+'base' in keys:
                dep.update({b+'full_input':[b+'attn'],b+'base':[b+'full_input'],b+'lora_a':[b+'full_input'],
                            b+'lora_b':[b+'lora_a'],b+'o':[b+'base',b+'lora_b',b+'indices']})
            dep.update({b+'post.hidden':[b+'o'],b+'post.residual':[b+'in.sum'],
                        b+'post.sum':[b+'post.hidden',b+'post.residual'],f'{layer}.post':[b+'post.sum']})
            for k,parents in dep.items():
                if k == b+'full_input' and any(b+'attn' in peer['suffix_diff'] for peer in result):
                    continue  # all-gather consumes every TP rank's attention output
                if k in changed and not changed.intersection(parents):
                    if not set(parents)<=keys:raise ValueError('missing paired operands: '+k)
                    first.append(k)
        r['first_observed_with_equal_local_inputs']=first
    return dict(prompt_tokens=n, cached_tokens=cut, actions_equal=warm['response_token_ids']==cold['response_token_ids'], ranks=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prime', 'warm', 'cold'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(*(json.loads(getattr(args, k).read_bytes()) for k in ('prime','warm','cold'))), indent=2))
