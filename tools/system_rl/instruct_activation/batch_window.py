"""Measure closed finite RL batches from producer release through final HTTP receipt."""
import argparse,json,statistics,importlib.util,hashlib
from pathlib import Path
from io_utils import canonical,doc,pin,require


def analyze(cfgfile,goldfile,parser):
    c=doc(pin(cfgfile));root=Path(c['output'])
    require(doc(pin(root/'DRAINED.json'))['guards_releasable'],'not drained')
    result=doc(pin(root/'RESULT.json'));require(result['passed'],'failed window')
    labels={r['wire']:r for r in doc(pin(goldfile))}
    spec=importlib.util.spec_from_file_location('ranked_candidate_indices',parser)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    out=[]
    for ref in result['rounds']:
        rr=doc(ref);require(rr['admission']=='group_drained','not batch input')
        n,=c['grouped_concurrency']
        salt=hashlib.sha256(canonical([c['measurement_id'],rr['block']])).hexdigest()
        for name,width in rr['concurrency_per_service'].items():
            service,=[v for v in c['services'] if v['name']==name]
            require(n<=width and len(rr['input_refs'])%n==0,'batch not concurrently admitted')
            rows=[x for x in rr['requests'] if x['service']==name]
            prev=-1; waves=[]; generation=[]
            for g in range(len(rr['input_refs'])//n):
                rs=sorted((x for x in rows if x['group']==g),key=lambda x:x['replicate'])
                require([x['replicate'] for x in rs]==list(range(n)),'missing batch member')
                start,={x['batch_release_from_round_seconds'] for x in rs}
                sends=[x['ready_to_http_seconds'] for x in rs]
                ends=[x['finished_from_round_seconds'] for x in rs]
                require(prev<=start<=min(sends) and max(sends)<=min(ends),'not single wave or overlapping waves')
                valid=0
                for x in rs:
                    raw=doc(x['receipt_ref']);require(raw['status']==200,'failed request')
                    choice,=json.loads(raw['raw'])['choices']
                    src=rr['input_refs'][g*n+x['replicate']]
                    wire=dict(doc(src),cache_salt=salt)
                    wire['model']=service.get('stock_adapter_alias',wire['model'])
                    intent=doc(pin(Path(x['receipt_ref']['path']).with_name('INTENT.json')))
                    require(intent['wire_sha256']==hashlib.sha256(canonical(wire)).hexdigest()
                            and hashlib.sha256(raw['raw'].encode()).hexdigest()==raw['response_sha256'],'unmatched receipt')
                    max_tokens=wire.get('max_completion_tokens')
                    if max_tokens is None:
                        max_tokens=wire.get('max_tokens')
                    generation.append(dict(max_tokens=max_tokens,min_tokens=wire.get('min_tokens',0),
                                           ignore_eos=wire.get('ignore_eos',False),reason=choice['finish_reason'],
                                           **{k:x['usage'][k] for k in ('completion_tokens','prompt_tokens','total_tokens')}))
                    label=labels[src['sha256']]
                    try:ranking=mod.parse(choice['message']['content'],label['ranking'],3)
                    except ValueError:ranking=[]
                    if ranking and choice['finish_reason']=='stop':valid+=1
                last=rs[ends.index(max(ends))]
                prev=max(ends);waves.append(dict(batch=g,size=n,valid=valid,release_s=start,last_http_s=prev,
                    http_finish_from_release_s=sorted(e-start for e in ends),
                    last_member=dict(index=last['replicate'],prompt_tokens=last['usage']['prompt_tokens'],
                                     output_tokens=last['usage']['completion_tokens']),
                    prompt_range=[min(x['usage']['prompt_tokens'] for x in rs),max(x['usage']['prompt_tokens'] for x in rs)],
                    last_only_s=max(ends)-sorted(ends)[-2] if n>1 else 0,
                    release_to_last_s=prev-start,send_spread_s=max(sends)-min(sends),
                    first_to_last_finish_s=prev-min(ends),generated=sum(x['usage']['completion_tokens'] for x in rs)))
            require(len(rows)==len(waves)*n,'unexpected request')
            total=waves[-1]['last_http_s']-waves[0]['release_s']
            out.append(dict(lane=name,round_sha256=ref['sha256'],batches=waves,
                output_budget={k:sorted({row[k] for row in generation}, key=str) for k in ('max_tokens','min_tokens','ignore_eos','reason')},
                token_ranges={k:[min(row[k] for row in generation),max(row[k] for row in generation)]
                              for k in ('completion_tokens','prompt_tokens','total_tokens')},
                mean_batch_s=statistics.mean(w['release_to_last_s'] for w in waves),all_valid_batches=sum(w['valid']==n for w in waves),
                valid=sum(w['valid'] for w in waves),requests=len(rows),wall_s=total,
                requests_per_second=len(rows)/total,valid_per_second=sum(w['valid'] for w in waves)/total))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('config','gold','parser'):p.add_argument(k,type=Path)
    a=p.parse_args();print(json.dumps(analyze(a.config,a.gold,a.parser),indent=2))
