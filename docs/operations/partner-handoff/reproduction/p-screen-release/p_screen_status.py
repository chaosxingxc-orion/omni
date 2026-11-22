"""Report service throughput and roster yield separately; use the seal for completion."""
import argparse,datetime,json,math,os,sqlite3,time
from pathlib import Path
from p_screen_run import apport

def write_status(path,result):
    tmp=path.with_suffix('.new')
    with tmp.open('w') as f:
        json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def delta_report(config):
    root = Path(config['output'])
    manifest = json.loads((root/'DELTA.json').read_bytes())
    now = time.time()
    with sqlite3.connect((root/'p-screen.sqlite').as_uri()+'?mode=ro', uri=True) as db:
        db.execute('begin')
        first, last = db.execute('select min(issued), max(finished) from calls').fetchone()
        statuses = dict(db.execute('select status,count(*) from calls group by status'))
        lanes = dict(db.execute("select lane,count(*) from calls where status='valid' group by lane"))
        cells = {d:{'error':0,'success':0} for d in config['sampling']['domain_mix']}
        for domain, hit, n in db.execute(
                "select domain,rank>0,count(*) from calls where status='valid' group by domain,rank>0"):
            cells[domain]['success' if hit else 'error'] = n
        start = max(now-config['progress_seconds'], first or now)
        finished = sum(statuses.get(status,0) for status in ('valid','invalid'))
        elapsed = max(0,(last or now)-start)
        recent = db.execute("select count(*) from calls where finished>=? and status in ('valid','invalid')",(start,)).fetchone()[0]
        rate = recent/elapsed if elapsed >= 30 else None
        seal = root/'COMPLETE.json'
        receipt = json.loads(seal.read_bytes()) if seal.is_file() else None
        if receipt and (receipt['screened'] != manifest['rows'] or receipt['candidate_sha256'] != manifest['queue_sha256']):
            raise ValueError('delta receipt differs from queue')
        remaining = manifest['rows']-finished
        def timestamp(t):
            return datetime.datetime.fromtimestamp(t).astimezone().isoformat(timespec='seconds') if t is not None else None
        return dict(at=timestamp(now), total=manifest['rows'], finished=finished,
                    by_domain=manifest['by_domain'], status=statuses, valid=sum(lanes.values()),
                    by_lane=lanes, cells=cells, first_submit=timestamp(first), last_return=timestamp(last),
                    completed_at=timestamp(receipt['finished']) if receipt else None,
                    samples_per_second=rate, seconds_per_sample=1/rate if rate else None,
                    estimated_completion=timestamp(now+remaining/rate) if rate and remaining else None,
                    actual_complete=receipt is not None)


def report(config):
    if config['sampling'].get('completion')=='exhaust_queue':
        return delta_report(config)
    root=Path(config['output']); s=config['sampling']; now=time.time()
    q={d:{o:apport(n,s['domain_mix'])[d] for o,n in apport(s['sample_count'],s['error_success']).items()} for d in s['domain_mix']}
    recall=json.loads((Path(config['recall'])/'RECALL-COMPLETE.json').read_bytes())
    populations=dict(real=recall['hits_by_domain']['train_real'],synthetic=recall['hits_by_domain']['train_synth'])
    priority=json.loads((root/'HISTORY.json').read_bytes())['incorrect']
    with sqlite3.connect((root/'p-screen.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        db.execute('begin')
        first,last=db.execute('select min(issued),max(finished) from calls').fetchone()
        statuses=dict(db.execute('select status,count(*) from calls group by status'))
        lanes=dict(db.execute("select lane,count(*) from calls where status='valid' group by lane"))
        counts={d:dict(error=0,success=0) for d in q}
        for d,hit,n in db.execute("select domain,rank>0,count(*) from calls where status='valid' group by domain,rank>0"):
            counts[d]['success' if hit else 'error']=n
        # Issued order can differ inside one in-flight window. Discard the entire
        # boundary window after preferential real records for the yield estimate.
        guard=priority+config['concurrency_per_lane']*sum(len(h['local_ports']) for h in config['hosts'].values())
        start=max(now-config['progress_seconds'],first or now)
        finished_n=db.execute('select count(*) from calls where finished>=?',(start,)).fetchone()[0]
        rate=finished_n/max(1,now-start)
        valid_rate=db.execute("select count(*) from calls where status='valid' and finished>=?",(start,)).fetchone()[0]/max(1,now-start)
        detail={};remaining=0;reasons=[]
        for d in q:
            used=db.execute('select count(*) from calls where domain=? and finished is not null',(d,)).fetchone()[0]
            n,k,j=db.execute("""select count(*),coalesce(sum(status='valid' and rank=0),0),
                coalesce(sum(status='valid' and rank>0),0) from (select * from calls where domain=? order by issued limit -1 offset ?)
                where finished is not null""",(d,guard if d=='real' and priority else 0)).fetchone()
            left=populations[d]-used
            rem={o:max(0,q[d][o]-counts[d][o]) for o in q[d]}
            rates={o:(x/n if n else None) for o,x in [('error',k),('success',j)]}
            detail[d]=dict(population=populations[d],remaining_candidates_including_inflight=left,remaining_quota=rem,
                yield_denominator=n,observed_yield=rates,required_yield={o:(v/left if left else None) for o,v in rem.items()})
            if not any(rem.values()):continue
            if sum(rem.values())>left:reasons.append(d+': remaining population insufficient');continue
            if n<200:reasons.append(d+': insufficient random observations');continue
            if any(v and (rates[o] or 0)*left<v for o,v in rem.items()):
                reasons.append(d+': projected population yield below quota');continue
            remaining+=max(v/rates[o] if v else 0 for o,v in rem.items())
        seal=root/'COMPLETE.json'
        receipt=json.loads(seal.read_bytes()) if seal.exists() else None
        if receipt and (receipt['rows']!=s['sample_count'] or any(counts[d][o]<q[d][o] for d in q for o in q[d])):
            raise ValueError('completion receipt and sampled counts differ; retry status on a consistent snapshot')
        def fmt(t):return datetime.datetime.fromtimestamp(t).astimezone().isoformat(timespec='seconds') if t is not None else None
        closed=receipt['finished'] if receipt else None
        return dict(at=fmt(now),first_submit=fmt(first),last_return=fmt(last),completed_at=fmt(closed),
            elapsed_seconds=(closed or now)-first if first else None,status=statuses,valid=sum(lanes.values()),by_lane=lanes,
            cells=counts,quotas=q,admitted=sum(min(q[d][o],counts[d][o]) for d in q for o in q[d]),valid_per_second=valid_rate,
            population_yield=detail,eta_unavailable_reasons=[] if receipt else reasons,
            estimated_completion=None if receipt or reasons or not rate else fmt(now+remaining/rate),actual_complete=receipt is not None)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);parser.add_argument('--watch',action='store_true')
    parser.add_argument('--completion-report',action='store_true',help='write terminal time from the actual sealed roster immediately')
    args=parser.parse_args();c=json.loads(args.config.read_bytes())
    while True:
        result=report(c)
        if args.completion_report:
            if not result['actual_complete']:raise ValueError('screen not sealed')
            write_status(Path(c['output'])/'COMPLETION-TIME.json',result);break
        if not args.watch:print(json.dumps(result,indent=2));break
        write_status(Path(c['output'])/'PROGRESS.json',result)
        if result['actual_complete']:break
        time.sleep(c['progress_seconds'])
