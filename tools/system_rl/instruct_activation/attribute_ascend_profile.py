"""Map device tasks through actual Kineto connections to enclosing decoder calls."""
import argparse,bisect,collections,json,re,sqlite3,statistics
from pathlib import Path

def reduce(db,layers):
    con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
    labels=dict(con.execute('select id,value from STRING_IDS'))
    sql='select startNs,endNs,globalTid,name from PYTORCH_API where type=50003'
    calls=collections.defaultdict(list); logits=collections.defaultdict(list)
    for a,b,tid,name in con.execute(sql):
        m=re.fullmatch(r'nn.Module: Qwen3MoeDecoderLayer_(\d+)',labels[name])
        if labels[name].startswith("nn.Module: LogitsProcessor"):logits[tid].append(int(a))
        if m:calls[tid].append((int(a),int(b),int(m[1])))
    call={};bounds={};runs=collections.defaultdict(list)
    for tid,ls in calls.items():
        ls.sort()
        if [x[2] for x in ls]!=list(range(layers))*(len(ls)//layers):raise ValueError('partial/unordered decoder traversal')
        for i,(a,b,l) in enumerate(ls):
            next_starts=[x for x in logits[tid] if x>a]
            if i+1<len(ls):next_starts.append(ls[i+1][0])
            limit=min(next_starts,default=b)
            bounds[tid,i]=min(limit,b)
            j=i//layers; runs[j].append((a,b,l,b<=limit));call[(tid,i)]=(j,l)
    if not runs:raise ValueError('no decoder calls')
    starts={t:[a for a,b,l in ls] for t,ls in calls.items()}
    targets=collections.defaultdict(set)
    for a,b,t,c in con.execute('select p.startNs,p.endNs,p.globalTid,g.connectionId from PYTORCH_API p join CONNECTION_IDS g on g.id=p.connectionId where p.type=50001'):
        if t not in starts:continue
        i=bisect.bisect_right(starts[t],int(a))-1
        if i>=0 and int(b)<=bounds[t,i]:targets[c].add(call[t,i])
    groups=collections.defaultdict(lambda:collections.defaultdict(list)); norms=collections.defaultdict(collections.Counter)
    shapes=collections.defaultdict(collections.Counter); other=0; tasks=0
    for a,b,c,kind,sh in con.execute('select t.startNs,t.endNs,t.connectionId,k.opType,k.inputShapes from TASK t join COMPUTE_TASK_INFO k using(globalTaskId)'):
        tasks+=1;s=targets.get(c)
        if not s or len(s)!=1:other+=1;continue
        rid,l=next(iter(s)); kind=labels.get(kind,str(kind));groups[rid,l][kind].append((int(a),int(b)))
        if l==0:shapes[rid][(kind,str(labels.get(sh,sh)))]+=1
        if l==0 and 'norm' in kind.lower():norms[rid][str(labels.get(sh,sh))]+=1
    out=[]
    for rid,ls in runs.items():
        per=[]
        for a,b,l,valid_end in ls:
            g=groups[rid,l]
            per.append(dict(layer=l,host_ms=(b-a)/1e6 if valid_end else None,ops={k:dict(count=len(v),device_sum_ms=sum(b-a for a,b in v)/1e6) for k,v in sorted(g.items())}))
        out.append(dict(traversal=rid,norm_shapes=dict(norms[rid]),layer0_shapes=[dict(op=k,shape=s,count=n) for (k,s),n in shapes[rid].items()],layers=per))
    whole={}
    for k,n,ns in con.execute('select k.opType,count(*),sum(t.endNs-t.startNs) from TASK t join COMPUTE_TASK_INFO k using(globalTaskId) group by k.opType'):
        whole[labels.get(k,str(k))]=dict(count=n,device_sum_ms=ns/1e6)
    comm=[dict(name=labels.get(k,str(k)),count=n,interval_sum_ms=ns/1e6) for k,n,ns in con.execute('select opName,count(*),sum(endNs-startNs) from COMMUNICATION_OP group by opName')]
    assert sum(x['count'] for x in whole.values())==tasks
    return dict(whole_window=whole,communication_ops=comm,device_compute_tasks=tasks,outside_or_unattributed=other,traversals=out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('database',type=Path);p.add_argument('--layers',type=int,required=True);c=p.parse_args(); print(json.dumps(reduce(c.database,c.layers)))
