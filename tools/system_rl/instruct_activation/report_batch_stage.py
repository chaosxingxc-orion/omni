"""Join actual engine steps to closed batch receipts using wire and request IDs."""
from collections import defaultdict
import argparse,csv,json,statistics
from pathlib import Path
from io_utils import pin,doc,require

def report(cfg,state,lane):
    c=doc(pin(cfg));root=Path(c['output'])
    require(doc(pin(root/'DRAINED.json'))['guards_releasable'],'window busy')
    ref,=doc(pin(root/'RESULT.json'))['rounds'];r=doc(ref)
    require(r['admission']=='group_drained','not fixed batch')
    ev=[]
    for f in sorted(state.glob('stages-*.jsonl')):
        for line in f.read_bytes().splitlines():
            ev.append(dict(json.loads(line),pid=f.stem.split('-')[-1]))
    ev.sort(key=lambda e:e['t'])
    def only(kind,k,v):
        hits=[e for e in ev if e['kind']==kind and e.get(k)==v]
        require(len(hits)==1,f'{kind} binding count {len(hits)}')
        return hits[0]
    selected={};requests=[];bybatch=defaultdict(list)
    for row in r['requests']:
        if row['service']!=lane:continue
        rr=doc(row['receipt_ref']);require(rr['status']==200,'failed request')
        resp=json.loads(rr['raw']);i=doc(pin(Path(row['receipt_ref']['path']).with_name('INTENT.json')))
        h=only('http','wire',i['wire_sha256']); sent=only('sent','wire',i['wire_sha256'])
        rid=resp['id'];v=only('render','rid',rid);m=only('map','rid',rid);key=m['internal']
        ipc=only('ipc_send','rid',key);queued=only('enqueue','rid',key)
        s=[e for e in ev if e['kind']=='scheduled' and key in e['alloc']]
        u=[e for e in ev if e['kind']=='update' and e['out'].get(key,0)]
        require(s and u and sum(e['out'][key] for e in u)==row['usage']['completion_tokens'],'missing step or output')
        ends=[h['start'],h['body_start'],h['t'],v['start'],v['t'],m['t'],ipc['t'],queued['t'],s[0]['start'],u[0]['t'],u[-1]['t'],sent['t']]
        require(ends==sorted(ends),'cross-process monotonic timeline inconsistent')
        cols=['middleware','body_hash','parse_to_render','render','prepare_assign','register','ipc_admit','queue','scheduled_to_first_token','later_tokens','http_finish']
        require(key not in selected,'duplicate input')
        code=f"{row['group']}.{row['replicate']}";selected[key]=code
        d=dict(slot=code,batch=row['group'],ms={k:(b-a)/1e6 for k,a,b in zip(cols,ends,ends[1:])},
               first_entry_ns=h['start'],enqueue_ns=queued['t'],first_schedule_ns=s[0]['start'],sent_ns=sent['t'],
               prompt=queued['prompt'],output=row['usage']['completion_tokens'],scheduled_steps=len(s))
        requests.append(d);bybatch[d['batch']].append(d)
    steps=[];active={}; workers=defaultdict(list)
    for e in ev:
        kind=e['kind'];t=e['t']
        if kind=='scheduled':
            if not selected.keys() & e['alloc'].keys():continue
            require(set(e['alloc'])<=selected.keys(),'foreign request in batch')
            names=lambda ks:[selected[k] for k in ks if k in selected]
            row=dict(step=len(steps),start_ns=e['start'],end_ns=None,budget=e['budget'],seq_limit=e['seq_limit'],encoder_budget=e['encoder_budget'],waiting=e['waiting'],
                     alloc={selected[k]:n for k,n in e['alloc'].items()},scheduled_tokens=sum(e['alloc'].values()),
                     decode=names(k for k in e['alloc'] if e['gen'][k]>0),unscheduled_decodes=names(k for k in e['gen'] if e['gen'][k]>0 and k not in e['alloc']),
                     encoder=names(e['enc']),ms=dict(schedule=(e['end']-e['start'])/1e6))
            require(row['scheduled_tokens']<=row['budget'],'invalid scheduler count')
            require(e['key'] not in active,'overlapping step key')
            groups={slot.partition('.')[0] for slot in row['alloc']}
            require(len(groups)==1,'cross-batch step')
            row['batch']=int(groups.pop())
            active[e['key']]=row;steps.append(row)
        elif kind=='worker_forward':
            if set(e['alloc'])&selected.keys():workers[e['pid']].append(e)
        elif e.get('key') in active and kind in ('submit','wait','sample','update'):
            row=active[e['key']];row['ms'][kind]=(t-e['start'])/1e6
            if kind=='update':
                row['end_ns']=t;row['ms']['total']=(t-row['start_ns'])/1e6
                row['ms']['other']=row['ms']['total']-sum(v for k,v in row['ms'].items() if k!='total')
                row['out']={selected[k]:v for k,v in e['out'].items() if v and k in selected}
                del active[e['key']]
    require(not active and steps and len(workers)==4,'missing core or TP ranks')
    for arr in workers.values():
        require(len(arr)==len(steps),'incomplete worker steps')
        for row,e in zip(steps,arr):
            require({selected[k]:v for k,v in e['alloc'].items()}==row['alloc'] and row['start_ns']<=e['entry']<=e['start']<=e['t']<=row['end_ns'],'worker step not attributable')
            row.setdefault('worker_host_ms',[]).append(dict(input=(e['start']-e['entry'])/1e6,forward=(e['t']-e['start'])/1e6))
    waves=[]
    for b,ls in bybatch.items():
        first=min(ls,key=lambda x:x['sent_ns'])
        waves.append(dict(batch=b,first_completed=first['slot'],stages_ms=first['ms'],
             entry_spread_ms=(max(x['first_entry_ns'] for x in ls)-min(x['first_entry_ns'] for x in ls))/1e6,
             enqueue_spread_ms=(max(x['enqueue_ns'] for x in ls)-min(x['enqueue_ns'] for x in ls))/1e6,
             first_schedule_spread_ms=(max(x['first_schedule_ns'] for x in ls)-min(x['first_schedule_ns'] for x in ls))/1e6))
    return dict(round_sha256=ref['sha256'],requests=requests,steps=steps,waves=waves,request_mean_ms={k:statistics.mean(d['ms'][k] for d in requests) for k in cols},
             totals_ms={k:sum(s['ms'].get(k,0) for s in steps) for k in ('schedule','submit','wait','sample','update','other','total')},
             budget_full_steps=sum(s['scheduled_tokens']==s['budget'] for s in steps),
             missed_already_generating=sum(len(s['unscheduled_decodes']) for s in steps))

def export_csv(result, directory):
    """Publish relative timings and slot IDs without payloads or process IDs."""
    directory.mkdir(parents=True,exist_ok=True)
    def write(name, rows):
        with (directory/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    def intervals(values):
        return {k:round(v,3) for k,v in values.items()}
    def slots(values):
        return ' '.join(f'{s}={n}' for s,n in values.items())
    requests=result['requests']
    start={b:min(r['first_entry_ns'] for r in requests if r['batch']==b)
           for b in {r['batch'] for r in requests}}
    rows=[]
    for r in requests:
        b=r['batch']
        rows.append(dict(slot=r['slot'],batch=b,prompt_tokens=r['prompt'],output_tokens=r['output'],
                         entry_from_first_ms=round((r['first_entry_ns']-start[b])/1e6,3),
                         **intervals(r['ms'])))
    write('batch-request-stages-ms.csv',rows)
    rows=[]
    for s in result['steps']:
        b=s['batch']
        phase=('prefill' if not s['decode'] else
               'decode' if len(s['decode'])==len(s['alloc']) else 'mixed')
        rows.append(dict(step=s['step'],batch=b,start_from_first_request_ms=round((s['start_ns']-start[b])/1e6,3),
                         phase=phase,tokens_used=s['scheduled_tokens'],tokens_available=s['budget'],
                         encoder_budget=s['encoder_budget'],max_seqs=s['seq_limit'],waiting=s['waiting'],
                         allocated_by_slot=slots(s['alloc']),outputs_by_slot=slots(s['out']),
                         decode_slots=' '.join(s['decode']),missed_decode_slots=' '.join(s['unscheduled_decodes']),
                         encoder_slots=' '.join(s['encoder']),
                         **intervals({k:s['ms'].get(k,0) for k in ('schedule','submit','wait','sample','update','other','total')}),
                         max_tp_worker_input=round(max(w['input'] for w in s['worker_host_ms']),3),
                         max_tp_worker_host_forward=round(max(w['forward'] for w in s['worker_host_ms']),3)))
    write('batch-engine-steps-ms.csv',rows)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('config',type=Path)
    p.add_argument('state',type=Path)
    p.add_argument('lane')
    p.add_argument('--csv-dir',type=Path)
    a=p.parse_args()
    result=report(a.config,a.state,a.lane)
    if a.csv_dir:
        export_csv(result,a.csv_dir)
    print(json.dumps(result,indent=2))
