"""Join device-event intervals to measured, nonoverlapping scheduler steps."""
import argparse,csv,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path
from io_utils import doc,pin,require
from report_batch_stage import report

TILED=('input_norm','qkv_projection','qk_norm_rope','output_projection',
       'norm_router_topk','expert_routing','expert_matmuls_activation','expert_restore')
WHOLE=('attention_kv','attention_gather','expert_collective_and_cat','weighted_sum')

def analyze(cfg,state,abi,lane,tile):
    r=report(cfg,state,lane); steps=r['steps']; layers=doc(pin(abi))['timed_layers']
    expected={s['step']:s for s in steps if len(s['decode'])!=len(s['alloc'])}
    totals=defaultdict(list); detail=[]; ranks=0
    for path in state.glob('stages-*.jsonl'):
        events=[json.loads(line) for line in path.read_bytes().splitlines()]
        fw=[e for e in events if e['kind']=='worker_forward']
        if not fw:continue
        interval=defaultdict(list)
        for e in events:
            if e['kind']=='prefill_interval':
                interval[e['step']].append(e)
        matches={}
        for f in fw:
            hits=[s for s in steps if s['start_ns']<=f['start']<=f['t']<=s['end_ns']]
            if not hits:continue
            require(len(hits)==1,'overlapping steps')
            s=hits[0]
            require(s['step'] not in matches,'duplicate worker frame')
            matches[s['step']]=f
        require(len(matches)==len(steps),'missing timing worker frame')
        for i,s in expected.items():
            seen=Counter();values=defaultdict(list)
            for e in interval[matches[i]['start']]:
                key=e['layer'],e['phase'];seen[key]+=1
                require(math.isfinite(e['device_ms']) and e['device_ms']>=0,'invalid interval')
                values[key].append(e['device_ms'])
            tiles=math.ceil(s['scheduled_tokens']/tile)
            target=Counter({(layer,phase):tiles if phase in TILED else 1
                            for layer in layers for phase in (*TILED,*WHOLE)})
            require(seen==target,f'incomplete step {i}: shortages {target-seen}, extras {seen-target}')
            for (layer,p),arr in values.items():
                t=sum(arr)
                totals[i,layer,p].append(t)
        ranks+=1
    require(ranks==4 and expected,'missing measured ranks or prefill')
    for (i,layer,p),v in sorted(totals.items()):
        require(len(v)==ranks,'missing interval aggregate')
        s=expected[i]
        detail.append(dict(step=i,kind='mixed' if s['decode'] else 'prefill',layer=layer,phase=p,
                           tokens=s['scheduled_tokens'],tp_median_ms=statistics.median(v),
                           tp_min_ms=min(v),tp_max_ms=max(v)))
    return detail

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('config','state','abi'):p.add_argument(k,type=Path)
    p.add_argument('lane');p.add_argument('--tile',required=True,type=int)
    args=p.parse_args();rows=analyze(args.config,args.state,args.abi,args.lane,args.tile)
    import sys
    w=csv.DictWriter(sys.stdout,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
