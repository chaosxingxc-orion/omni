"""Pair batch latency from independent, warmed service instances on identical inputs."""
import argparse
import json
from pathlib import Path
import statistics
from io_utils import doc, pin, require
from batch_window import analyze


def collect(cfg_path, gold, parser, lane):
    c = doc(pin(cfg_path))
    result = doc(doc(pin(Path(c['output'])/'RESULT.json'))['rounds'][0])
    s, = [s for s in c['services'] if s['name'] == lane]
    w, = [w for w in analyze(cfg_path, gold, parser) if w['lane'] == lane]
    weights = doc(pin(Path(c['output']).parents[1]/(c['measurement_id'].removesuffix('-latency')+'-run.local.json')))['adapter_alias']
    rank_rows = [r for r in result['requests'] if r['service'] == lane]
    ordered = sorted(rank_rows,key=lambda r:(r['group'],r['replicate']))
    m = result['metrics'][lane]
    body = [json.loads(doc(r['receipt_ref'])['raw']) for r in ordered]
    require(m['mm_hits']==0 and m['mm_queries']==len(ordered) and w['output_budget']['reason']==['stop'],
            'audio replay or non-stop outcomes; no matched comparison')
    return dict(inputs=[r['sha256'] for r in result['input_refs']], container=doc(c['start_receipts'][lane] if isinstance(c['start_receipts'][lane],dict) else pin(c['start_receipts'][lane]))['container_id'],
                weights=weights, guards=c['guards'], concurrency=result['concurrency_per_service'][lane],
                salt_namespace=c['measurement_id'], metrics=m, window=w, prompts=[r['usage']['prompt_tokens'] for r in ordered],
                actions=[x['choices'][0]['token_ids'] for x in body],
                scores=[[t['logprob'] for t in x['choices'][0]['logprobs']['content']] for x in body])


def compare(parent, candidate, gold, parser, lane):
    # Warm-up inputs must be identical and disjoint from measured inputs.
    def warm(cfg):
        d=doc(pin(cfg));base=Path(d['output']).parents[1]
        wr=base/(d['measurement_id'].removesuffix('-latency')+'-warm')
        require(doc(pin(wr/'rounds/DRAINED.json'))['guards_releasable'],'warm not sealed')
        v=doc(doc(pin(wr/'rounds/RESULT.json'))['rounds'][0])
        wc=doc(pin(wr/'CONFIG.local.json'))
        require(wc['start_receipts'][lane]==d['start_receipts'][lane] and
                v['admission']=='group_drained' and
                v['concurrency_per_service'][lane]==d['lane_concurrency'][0][lane],'warm service/config drift')
        return [r['sha256'] for r in v['input_refs']]
    a=collect(parent,gold,parser,lane); b=collect(candidate,gold,parser,lane)
    require(a['container']!=b['container'] and a['salt_namespace']!=b['salt_namespace'],'shared serving state')
    for k in ('inputs','weights','guards','concurrency','prompts'):
        require(a[k]==b[k],'mismatched '+k)
    require(warm(parent)==warm(candidate) and not set(warm(parent)).intersection(a['inputs']),'warm panel drift')
    for k in ('prefix_queries','prefix_hits'):
        require(a['metrics'][k]==b['metrics'][k],'prefix cache state differs')
    rows=[]
    for x,y in zip(a['window']['batches'],b['window']['batches'],strict=True):
        require((x['batch'],x['size'])==(y['batch'],y['size']),'batch drift')
        rows.append(dict(batch=x['batch'],parent_s=x['release_to_last_s'],candidate_s=y['release_to_last_s'],
                         saved_s=x['release_to_last_s']-y['release_to_last_s'],
                         parent_tokens=x['generated'],candidate_tokens=y['generated']))
    diffs=[]; seq=[]; common=0
    for ax,bx,ap,bp in zip(a['actions'],b['actions'],a['scores'],b['scores'],strict=True):
        delta=[]
        for x,y,u,v in zip(ax,bx,ap,bp):
            if x!=y:break
            delta.append(v-u)
        common+=len(delta);diffs.extend(delta)
        if ax==bx:seq.append(abs(sum(delta)))
    return dict(batches=rows,mean_batch_saved_s=statistics.mean(r['saved_s'] for r in rows),
                relative_batch_reduction=1-b['window']['mean_batch_s']/a['window']['mean_batch_s'],
                prefix=[a['metrics']['prefix_hits'],a['metrics']['prefix_queries']],
                output_actions=dict(common=common,common_max_abs=max(map(abs,diffs),default=None),
                                    identical_sequences=len(seq),sequence_max_abs=max(seq,default=None)),
                parent_valid=a['window']['valid'],candidate_valid=b['window']['valid'],
                parent_round=a['window']['round_sha256'],candidate_round=b['window']['round_sha256'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','candidate','gold','parser'):
        p.add_argument(name,type=Path)
    p.add_argument('lane')
    print(json.dumps(compare(**vars(p.parse_args())),indent=2))
