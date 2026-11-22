"""Read closed paired Prometheus windows; return exact bucket counts and quantile bounds."""
import argparse,json,math,re
from pathlib import Path
from io_utils import doc,pin,require

def bins(raw,name):
    values={}
    for labels,value in re.findall(r'^vllm:'+re.escape(name)+r'_bucket\{([^}\n]+)\} ([\d.eE+\-]+)$',raw,re.M):
        bound=re.search(r'(?:^|,)le="([^"]+)"(?:,|$)',labels)
        require(bound is not None,'bucket without limit')
        bucket=float(bound[1]);require(bucket not in values,'ambiguous engine/model label')
        values[bucket]=int(float(value))
    require(math.inf in values,'no complete histogram')
    return values

def summary(before,after,name,expected=None):
    a,b=bins(before,name),bins(after,name)
    require(a.keys()==b.keys(),'schema changed')
    rows=[];prev=0;lo=0
    for bound in sorted(a):
        cumulative=b[bound]-a[bound]
        require(cumulative>=prev,'counter reset or mixed series')
        rows.append(dict(lower=lo,upper=bound if math.isfinite(bound) else None,count=cumulative-prev))
        lo=bound;prev=cumulative
    require(prev>0 and (expected is None or prev==expected),'histogram denominator mismatch')
    def quantile(q):
        target=math.ceil(q*prev);total=0
        for r in rows:
            total+=r['count']
            if total>=target:return [r['lower'],r['upper']]
    return dict(n=prev,p50_bounds=quantile(.5),p95_bounds=quantile(.95),bins=rows)

def audit(root):
    done=doc(pin(root/'RESULT.json'));require(done['passed'] and doc(pin(root/'DRAINED.json'))['guards_releasable'],'unsealed')
    out=[]
    for ref in done['rounds']:
        round_=doc(ref)
        for lane in round_['metrics']:
            p=Path(ref['path']).parent/lane
            before,after=[doc(pin(p/(s+'.json')))['metrics'] for s in ('BEFORE','AFTER')]
            requests=[r for r in round_['requests'] if r['service']==lane]
            gaps=sum(r['usage']['completion_tokens']-1 for r in requests)
            out.append(dict(lane=lane,block=round_['block'],round_sha256=ref['sha256'],
              itl=summary(before,after,'inter_token_latency_seconds',gaps),
              decode=summary(before,after,'request_decode_time_seconds',len(requests)),
              # Output-side prompt + generation accounting; not SchedulerOutput's token budget.
              reported_iteration_tokens=summary(before,after,'iteration_tokens_total')))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('windows',nargs='+',type=Path)
    print(json.dumps({str(w):audit(w) for w in p.parse_args().windows},indent=2))
