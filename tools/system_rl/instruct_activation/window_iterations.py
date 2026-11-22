"""Attribute sealed output intervals to native scheduler step classes on a single TP engine."""
from pathlib import Path
import argparse,json,math,re,subprocess
from io_utils import doc,pin,require

LINE=re.compile(r'Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, (\d+) generation requests, (\d+) generation tokens, iteration elapsed time: ([\d.]+) ms( \(dummy\))?,')
def report(root,lane,receipt):
    cfg=doc(pin(root/'CONFIG.local.json'));rw=Path(cfg['output'])
    begun=doc(pin(rw/'STARTED.json'));done=doc(pin(rw/'DRAINED.json'))
    result=doc(pin(rw/'RESULT.json'));ref,=result['rounds']
    require(result['passed'] and done['guards_releasable'] and doc(begun['config_ref'])==cfg,'unsealed window')
    arm,=[s for s in cfg['services'] if s['name']==lane];r=doc(ref)
    cid=doc(pin(receipt))['container_id']
    state=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
    require(state['Name'].lstrip('/')==arm['container'] and state['State']['Pid']==arm['pid'] and int((Path('/proc')/str(arm['pid'])/'stat').read_text().rsplit(')',1)[1].split()[19])==arm['start_ticks'],'container identity differs')
    output=subprocess.check_output(['docker','logs','--since',str(begun['started_epoch']),'--until',str(done['finished_epoch']),cid],stderr=subprocess.STDOUT,text=True)
    rows=[]
    for line in output.splitlines():
        if 'Iteration(' not in line:continue
        m=LINE.search(line);require(m is not None,'unknown iteration format')
        index,c,ct,g,gt=map(int,m.groups()[:5]);ms=float(m[6])
        require(not rows or index==rows[-1]['index']+1,'skipped or repeated iteration')
        require(gt==g and ((c==0)==(ct==0)),'not ordinary one-token decode')
        kind='mixed' if c and g else 'decode' if g else 'prefill' if c else 'idle'
        rows.append(dict(index=index,context_requests=c,context_tokens=ct,generation_tokens=gt,elapsed_ms=ms,kind=kind))
    gaps=int(r['metrics'][lane]['itl_intervals'])
    require(rows and sum(x['generation_tokens'] for x in rows)==gaps,'native steps do not cover returned token intervals')
    def quant(v,q):return sorted(v)[math.ceil(q*len(v))-1]
    groups={}
    for kind in ('decode','mixed','prefill','idle'):
        sub=[x for x in rows if x['kind']==kind];v=[x['elapsed_ms'] for x in sub]
        if sub:groups[kind]=dict(steps=len(sub),tokens=sum(x['generation_tokens'] for x in sub),
            ctx=sum(x['context_tokens'] for x in sub),step_ms_sum=sum(v),step_p50_ms=quant(v,.5),step_p95_ms=quant(v,.95),
            tokens_in_steps_over_1s=sum(x['generation_tokens'] for x in sub if x['elapsed_ms']>1000),
            token_weighted_step_seconds=sum(x['elapsed_ms']*x['generation_tokens'] for x in sub)/1000)
    return dict(round_sha256=ref['sha256'],intervals=gaps,request_gap_seconds=r['metrics'][lane]['itl_total_seconds'],
                groups=groups,rows=rows,scope='engine result wait; step classes do not identify individual requests or measure device kernels')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('window',type=Path);p.add_argument('lane');p.add_argument('receipt',type=Path)
    a=p.parse_args();print(json.dumps(report(a.window,a.lane,a.receipt),indent=2))
