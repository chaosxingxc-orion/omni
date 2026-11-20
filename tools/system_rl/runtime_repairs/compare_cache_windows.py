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
    return dict(prompt_tokens=n, cached_tokens=cut, actions_equal=warm['response_token_ids']==cold['response_token_ids'], ranks=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prime', 'warm', 'cold'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(*(json.loads(getattr(args, k).read_bytes()) for k in ('prime','warm','cold'))), indent=2))
