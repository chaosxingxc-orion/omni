"""Anonymized latency receipts from a completed guarded two-arm study."""
from io_utils import doc,pin,save,require
from pathlib import Path
import json,statistics,math,sys

def summarize(path):
    root=Path(path);report=doc(pin(root/'RESULT.json'))
    require(report['passed'] and doc(pin(root/'DRAINED.json'))['guards_releasable'],'study incomplete')
    samples=json.loads((root.parent/'NPU-WINDOW.json').read_bytes())['samples'] if (root.parent/'NPU-WINDOW.json').is_file() else []
    result=[]
    for ref in report['rounds']:
        d=doc(ref)
        for k,totals in d['per_service'].items():
            rows=[r for r in d['requests'] if r['service']==k]; m=d['metrics'][k]; e=sorted(r['elapsed_seconds'] for r in rows)
            starts=[];ends=[]; outputs=[]
            for r in rows:
                rec=doc(r['receipt_ref']); intent=doc(pin(Path(r['receipt_ref']['path']).parent/'INTENT.json'))
                starts.append(intent['started_epoch']);ends.append(intent['started_epoch']+rec['elapsed_seconds'])
                resp=json.loads(rec['raw']);outputs.append(resp['choices'][0]['message']['content'])
            active=[s for s in samples if min(starts)<=s['at']<=max(ends) and s['lanes'][k].get('num_requests_running',0)>0]
            vals={}
            if active:
                cards={'a':range(4),'b':range(4,8)}[k]
                for field in ('Aicore Usage Rate','NPU Utilization','HBM Bandwidth Usage Rate'):
                    v=[s['devices'][str(i)][field] for s in active for i in cards if field in s['devices'][str(i)]]
                    vals[field]=statistics.mean(v) if v else None
                for field in ('num_requests_running','num_requests_waiting'):
                    vals[field]=statistics.mean(s['lanes'][k][field] for s in active)
                vals['samples']=len(active)
            result.append(dict(block=d['block'],arm=d['arm'],lane=k,admission=d['admission'],concurrency=d['concurrency_per_service'],
                n=len(e),http_mean=statistics.mean(e),http_p50=statistics.median(e),http_p95=e[math.ceil(.95*len(e))-1],
                queue=m['mean_queue_seconds'],prefill=m['mean_prefill_seconds'],decode=m['mean_decode_seconds'],ttft=m['mean_ttft_seconds'],
                tpot=m['mean_request_tpot_seconds'],output_tokens=totals['completion_tokens'],rps=len(e)/totals['wall_seconds'],
                prefix=[m['prefix_hits'],m['prefix_queries']],mm=[m['mm_hits'],m['mm_queries']],stop=sum(r['finish_reasons']==['stop'] for r in rows),
                device=vals,round_sha256=ref['sha256']))
    return result
if __name__=='__main__':print(json.dumps(summarize(sys.argv[1]),indent=2))
