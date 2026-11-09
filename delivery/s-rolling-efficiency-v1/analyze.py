#!/usr/bin/env python3
"""本地只读效率审计。仅向本脚本目录写输出；不导入/执行被审计运行时代码。
运行: PYTHONDONTWRITEBYTECODE=1 python delivery/s-rolling-efficiency-v1/analyze.py
可选 --root 指定包含 delivery 的仓库；--self-test 执行增量/重置算例。
归档只读取 LAUNCH/TERMINAL/resource-preparation 和 rows，小对象逐条聚合，不解包。
"""
import argparse, collections, csv, datetime, hashlib, io, json, math, re, statistics, tarfile
from pathlib import Path
OUT=Path(__file__).resolve().parent
UTC=datetime.timezone.utc

def utc(t):
    return datetime.datetime.fromtimestamp(t,UTC).isoformat() if t is not None else None
def div(a,b):
    return a/b if a is not None and b is not None and b else None
def mean(v):
    v=[x for x in v if x is not None]; return statistics.mean(v) if v else None
def quant(v,p):
    v=sorted(v)
    if not v:return None
    x=(len(v)-1)*p;i=int(x);j=min(i+1,len(v)-1)
    return v[i]+(v[j]-v[i])*(x-i)
def describe(v):
    v=[x for x in v if isinstance(x,(int,float)) and math.isfinite(x)]
    return dict(n=len(v),sum=sum(v),mean=mean(v),p50=quant(v,.5),p95=quant(v,.95),min=min(v) if v else None,max=max(v) if v else None)
def read(p):
    return json.loads(Path(p).read_text())
def dump(name,d):
    p=OUT/name;p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');return p
def summetric(m,k):
    v=m.get('counters',{}).get(k,{})
    return sum(v.values()) if isinstance(v,dict) and v else None
COUNTERS=['prompt_tokens','generation_tokens','prefix_queries','prefix_hits','queue_time_sum','queue_time_count','prefill_time_sum','prefill_time_count','decode_time_sum','decode_time_count','preemptions']
def deltas(a,b):
    missing=[k for k in COUNTERS if summetric(a,k) is None or summetric(b,k) is None]
    d={k:summetric(b,k)-summetric(a,k) for k in COUNTERS if k not in missing}
    negative={k:v for k,v in d.items() if v < -1e-7}
    return d,missing,negative
def identity(rt):
    keys=['boot_id','service_id','service_pid','session_id']
    return {k:rt.get(k) for k in keys}
def identkey(rt):return json.dumps(identity(rt),sort_keys=True)
def points_smi(s):
    # 数据行形式: | chip | bus | AICore | Memory | HBM |；不把标题或功率行当AI Core。
    rows=[]
    for line in s.splitlines():
        parts=[x.strip() for x in line.strip().strip('|').split('|')]
        if len(parts)>=3 and re.fullmatch(r'[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]',parts[1]):
            try:rows.append(float(parts[2].split()[0]))
            except ValueError:pass
    return rows

def main(root):
    b=root/'delivery/s-single-intent-execution-v1';state=b/'rolling-state'; code=b/'rolling-code'
    manifest={}
    def capture(p):
        p=Path(p);raw=p.read_bytes();manifest[str(p.relative_to(root))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());return json.loads(raw)
    snapshots=[]
    for p in sorted((state/'resources').glob('snapshot-*.json')):
        d=capture(p);d['_source']=str(p.relative_to(root));snapshots.append(d)
    snapshots.sort(key=lambda d:d['at'])
    initial=capture(state/'initial-lanes.json')
    taskdocs=[capture(p) for p in sorted((state/'tasks').glob('*.json'))]
    records={}; archive_stats=[]
    # 首先读取现有小回执，绝不全量打印 archive/raw。
    for d in taskdocs:
        if d.get('launch'):
            l=d['launch'];key=l.get('task_id',d.get('task',{}).get('task_id')); records[key]=dict(task_id=key,lane=d.get('lane',l.get('lane')),launch=l,terminal=d.get('terminal'),status=d.get('status'),task=d.get('task',{}))
    oldcontainers={v.get('live_launch',{}).get('container_id'):lane for lane,v in initial.items() if v.get('live_launch')}
    for ap in sorted((state/'archives').glob('*.tar')):
        # tar不压缩；只顺序读取需用成员，不抽取文件，不保存原始轨迹。
        agg=collections.defaultdict(lambda:dict(branches=0,prompt_tokens=0,completion_tokens=0,reasoning_tokens=0,total_tokens=0,content_chars=0,whitespace_chars=0,trailing_whitespace_chars=0,empty_content=0))
        times=collections.defaultdict(list);row_times=[]; first_finish=None;last_finish=None;first_http=None; rows=0;launch=None;terminal=None;prep=[];finish=collections.Counter();error_categories=collections.Counter();blank_length=0;invalid_long_ws=0
        h=hashlib.sha256()
        with ap.open('rb') as f:
            while True:
                chunk=f.read(1024*1024)
                if not chunk:break
                h.update(chunk)
        manifest[str(ap.relative_to(root))]=dict(bytes=ap.stat().st_size,sha256=h.hexdigest())
        with tarfile.open(ap,mode='r|') as tf:
            for m in tf:
                if not m.isfile():continue
                name=m.name.removeprefix('./')
                isrow=name.startswith('run/rows/') and name.endswith('.json')
                if name not in ['LAUNCH.json','run/TERMINAL.json'] and not isrow and not name.startswith('run/resource-preparation/'):continue
                f=tf.extractfile(m)
                if m.size>2_000_000:raise ValueError('unexpected selected member size: '+name)
                d=json.load(f)
                if name=='LAUNCH.json':launch=d;continue
                if name=='run/TERMINAL.json':terminal=d;continue
                if name.startswith('run/resource-preparation/'):prep.append(d);continue
                rows+=1;reason=d.get('finish_reason');finish[str(reason)]+=1
                content=d.get('wire_content',d.get('content'));content=content if isinstance(content,str) else ''
                empty=not content.strip();valid=d.get('valid') is True
                if not valid:
                    err=d.get('error') or '';error_categories['finite_length' if reason=='length' else ('nonempty_entity_field' if 'nonempty text' in err else ('empty_content' if empty else 'other_invalid'))]+=1
                group='valid' if valid else ('length' if reason=='length' else ('empty' if empty else 'other_invalid'))
                usage=d.get('usage') or {};detail=usage.get('completion_tokens_details') or {};reasoning=detail.get('reasoning_tokens',0) or 0
                trailing=len(content)-len(content.rstrip());ws=sum(c.isspace() for c in content)
                for key in ['all',group]:
                    a=agg[key];a['branches']+=1
                    for token in ['prompt_tokens','completion_tokens','total_tokens']:a[token]+=usage.get(token,0) or 0
                    a['reasoning_tokens']+=reasoning;a['content_chars']+=len(content);a['whitespace_chars']+=ws;a['trailing_whitespace_chars']+=trailing;a['empty_content']+=int(empty)
                if not valid and trailing>=64:invalid_long_ws+=1
                if reason=='length' and empty:blank_length+=1
                tm=d.get('timing') or {}
                for k in ['prepare_seconds','executor_queue_seconds','admission_seconds','postprocess_seconds','new_http_seconds']:
                    if isinstance(tm.get(k),(int,float)):times[k].append(tm[k])
                rt=tm.get('row_ready_epoch')
                if rt is not None:
                    row_times.append(rt)
                    if first_finish is None or rt<first_finish:first_finish=rt;first_http=tm.get('new_http_seconds')
                    last_finish=rt if last_finish is None else max(last_finish,rt)
        if not launch or not terminal:raise ValueError('archive missing launch/terminal '+str(ap))
        lane=launch.get('lane') or oldcontainers.get(launch.get('container_id'));tid=launch.get('task_id') or ('historical-'+str(lane)+'-545')
        if lane is None:raise ValueError('cannot anchor archive lane')
        if not launch.get('runtime'):launch['runtime']=initial[lane]['runtime']
        rec=records.get(tid,dict(task_id=tid,lane=lane,launch=launch,terminal=terminal,status=terminal['status'],task={}))
        rec.update(first_row_ready=first_finish,last_row_ready=last_finish,archive=str(ap.relative_to(root)),first_http_seconds=first_http)
        records[tid]=rec
        wall=terminal.get('wall_seconds');allg=agg['all'];invalid={k:sum(v[k] for g,v in agg.items() if g not in ('all','valid')) for k in ['branches','prompt_tokens','completion_tokens','reasoning_tokens','total_tokens','content_chars','whitespace_chars','trailing_whitespace_chars']}
        for g,v in agg.items():v['nonreasoning_tokens']=v['completion_tokens']-v['reasoning_tokens']
        a=dict(task_id=tid,lane=lane,source=str(ap.relative_to(root)),start_epoch=terminal.get('original_launch_epoch',launch.get('at')),end_epoch=terminal['at'],rows=rows,total=terminal['total'],valid=terminal['valid'],unknown=terminal['unknown'],concurrency=terminal['concurrency'],wall_seconds=wall,seconds_per_branch=div(wall,rows),seconds_per_valid_branch=div(wall,terminal['valid']),prompt_tokens_per_second=div(allg['prompt_tokens'],wall),generation_tokens_per_second=div(allg['completion_tokens'],wall),groups=dict(agg),finish_reasons=dict(finish),error_categories=dict(error_categories),invalid=invalid,invalid_completion_token_share=div(invalid['completion_tokens'],allg['completion_tokens']),invalid_total_token_share=div(invalid['total_tokens'],allg['total_tokens']),invalid_long_trailing_whitespace_branches=invalid_long_ws,blank_and_length=blank_length,timing={k:describe(v) for k,v in times.items()},preparation_seconds=terminal.get('preparation_seconds'),first_response_after_launch_seconds=first_finish-launch.get('at',first_finish) if first_finish else None,first_http_seconds=first_http,resource_preparation=prep,usage_matches_terminal=all(allg[k]==terminal['usage'][k] for k in ['prompt_tokens','completion_tokens','total_tokens']))
        archive_stats.append(a)
    train=capture(state/'TRAIN96-REAL-OPERATION.json'); trainid=train['launch']['task_id']
    trainrec=records[trainid];trainrec['first_row_ready']=train['first_response_event']['at']
    train_config=capture(root/trainrec['task']['config'])
    # 96初期回执只作启动证据，不能覆盖后来的 FAILED1246/3240。
    live=[]
    for lane in 'ABCD':
        curr=capture(state/f'lanes/{lane}-CURRENT.json');d=capture(state/f'lanes/{lane}-LIVE.json');l=curr['launch']
        if l.get('task_id') and l['task_id'] not in records:records[l['task_id']]=dict(task_id=l['task_id'],lane=lane,launch=l,terminal=None,status=curr['status'],task={})
        live.append({k:d.get(k) for k in ['at','run_id','attempt_id','completed_branches','total_branches','completed_samples','total_samples','invalid','usage','finish_reasons','invalid_reasons','elapsed_seconds','seconds_per_branch','inflight_or_unknown','scheduler','interval']}|dict(lane=lane,task_id=l.get('task_id'),cards=4,instances=1,TP=4))
    reserve=capture(state/'RESERVE.json');audit=capture(state/'SERVICE-CONFIG-AUDIT.json')
    reference_interval=capture(root/'delivery/s-rolling-12h-v1/ENGINE-INTERVAL-ANALYSIS.json');reference_cumulative=capture(root/'delivery/s-rolling-12h-v1/ENGINE-CUMULATIVE-ANALYSIS.json')
    by_lane={l:sorted([r for r in records.values() if r['lane']==l],key=lambda r:r['launch'].get('at',0)) for l in 'ABCD'}
    for lane,rs in by_lane.items():
        for r in rs:
            r['start']=r['launch'].get('at',(r.get('terminal') or {}).get('original_launch_epoch'))
            r['end']=(r.get('terminal') or {}).get('at',float('inf'))
            r['concurrency']=r['launch'].get('concurrency') or (r.get('terminal') or {}).get('concurrency') or r.get('task',{}).get('resources',{}).get('concurrency',64)
    def task_at(lane,t):
        rs=[r for r in by_lane[lane] if r['start']<=t<=r['end']]
        return rs[-1] if rs else None
    # AI Core逐卡点样本；四卡映射来自launch.cards。
    samples=[];point_summary={}
    for snap in snapshots:
        for host,hd in snap['hosts'].items():
            values=points_smi(hd.get('npu_smi',''))
            for lane,m in hd.get('metrics',{}).items():
                idx=range(0,4) if lane in 'AC' else range(4,8);vs=[values[i] for i in idx if i<len(values)]
                r=task_at(lane,snap['at']);samples.append(dict(at=snap['at'],host_at=hd.get('at'),source=snap['_source'],lane=lane,AI_Core=vs,running=m.get('running'),waiting=m.get('waiting'),task_id=r['task_id'] if r else None,concurrency=r['concurrency'] if r else None,counter_available=bool(m.get('counters'))))
    for l in 'ABCD':
        ss=[s for s in samples if s['lane']==l];vs=[v for s in ss for v in s['AI_Core']]
        saturated=[s for s in ss if s.get('concurrency') and s['running'] is not None and s['running']>=s['concurrency']-1 and s['waiting']==0];sv=[v for s in saturated for v in s['AI_Core']]
        point_summary[l]=dict(point_count=len(ss),card_samples=len(vs),AI_Core_mean=mean(vs),AI_Core_max=max(vs) if vs else None,at_least80=sum(v>=80 for v in vs),saturated_endpoint_points=len(saturated),saturated_card_samples=len(sv),saturated_AI_Core_mean=mean(sv),running_zero_waiting_zero_points=sum(s['running']==0 and s['waiting']==0 for s in ss),waiting_positive_points=sum(s['waiting'] is not None and s['waiting']>0 for s in ss),missing_card_points=sum(len(s['AI_Core'])!=4 for s in ss))
    interval=[]
    for prev,nxt in zip(snapshots,snapshots[1:]):
        for host,hd in nxt['hosts'].items():
            for lane,endm in hd.get('metrics',{}).items():
                startm=prev['hosts'].get(host,{}).get('metrics',{}).get(lane,{})
                d,missing,negative=deltas(startm,endm);dt=nxt['at']-prev['at'];a=task_at(lane,prev['at']);brec=task_at(lane,nxt['at']);same=a is not None and brec is not None and a['task_id']==brec['task_id'];idents=[identkey(r['launch'].get('runtime',{})) for r in by_lane[lane] if r['start']<=nxt['at'] and r['end']>=prev['at']];idstable=len(set(idents))==1 and bool(idents) and all(all(v is not None for v in json.loads(i).values()) for i in idents)
                service_change=len(set(idents))>1;usable=not missing and not negative and not service_change
                cap=a['concurrency'] if same else None;rs=startm.get('running');re_=endm.get('running');ws=startm.get('waiting');we=endm.get('waiting');plateau=same and rs is not None and re_ is not None and min(rs,re_)>=cap-1 and ws==we==0 and prev['at']>=(a.get('first_row_ready') or a['start'])
                if missing:classification='missing_counters'
                elif negative or service_change:classification='reset_or_service_change'
                elif not same:classification='task_boundary_or_terminal_gap'
                elif not plateau:classification='same_task_nonsaturated_or_startup_tail'
                else:classification='same_task_saturated_endpoints'
                row=dict(lane=lane,start_epoch=prev['at'],end_epoch=nxt['at'],start_utc=utc(prev['at']),end_utc=utc(nxt['at']),seconds=dt,sources=[prev['_source'],nxt['_source']],task_id=a['task_id'] if same else None,task_start=a['task_id'] if a else None,task_end=brec['task_id'] if brec else None,concurrency=cap,classification=classification,delta_usable=usable,missing_counters=missing,negative_deltas=negative,service_identity_status='receipt_anchored_no_reset_observed' if idstable else ('receipt_identity_change' if service_change else 'unknown'),scrape_service_identity_available=False,monotonicity_is_not_restart_proof=True,running_start=rs,running_end=re_,waiting_start=ws,waiting_end=we,kv_fraction_start=summetric(startm,'cache_usage'),kv_fraction_end=summetric(endm,'cache_usage'),delta=d if usable else None)
                if usable:
                    row.update(completion_count=d['decode_time_count'],requests_per_second=div(d['decode_time_count'],dt),seconds_per_branch=div(dt,d['decode_time_count']),prompt_tokens_per_second=d['prompt_tokens']/dt,generation_tokens_per_second=d['generation_tokens']/dt,prefix_hit_ratio=div(d['prefix_hits'],d['prefix_queries']),preemptions_delta=d['preemptions'])
                    for name in ['queue','prefill','decode']:row[name+'_mean_seconds']=div(d[name+'_time_sum'],d[name+'_time_count'])
                    row['completion_count_consistent']=len(set(d[x+'_time_count'] for x in ['queue','prefill','decode']))==1
                    for k in ['queue','prefill','decode']:row[k+'_count']=d[k+'_time_count']
                interval.append(row)
    def aggregate(rows):
        rows=[r for r in rows if r['delta_usable']];duration=sum(r['seconds'] for r in rows);d={k:sum(r['delta'][k] for r in rows) for k in COUNTERS};n=d['decode_time_count'];kv=[r['kv_fraction_end'] for r in rows if r['kv_fraction_end'] is not None]
        result=dict(intervals=len(rows),seconds=duration,start_utc=utc(min(r['start_epoch'] for r in rows)) if rows else None,end_utc=utc(max(r['end_epoch'] for r in rows)) if rows else None,non_contiguous_windows=True,completed_requests=n,seconds_per_branch=div(duration,n),prompt_tokens_per_second=div(d['prompt_tokens'],duration),generation_tokens_per_second=div(d['generation_tokens'],duration),prompt_tokens_per_completed_request_ratio=div(d['prompt_tokens'],n),generation_tokens_per_completed_request_ratio=div(d['generation_tokens'],n),token_per_request_note='counter flow ratio; in-flight token inventory crosses boundaries; not exact completed-request mean length',prefix_hit_ratio=div(d['prefix_hits'],d['prefix_queries']),preemptions_delta=d['preemptions'],kv_end_min=min(kv) if kv else None,kv_end_max=max(kv) if kv else None,delta_totals=d)
        for k in ['queue','prefill','decode']:result[k+'_mean_seconds']=div(d[k+'_time_sum'],d[k+'_time_count'])
        return result
    steady_by_lane={l:aggregate([r for r in interval if r['lane']==l and r['classification']=='same_task_saturated_endpoints']) for l in 'ABCD'}
    trainwindows=[r for r in interval if r['task_id']==trainid and r['classification']=='same_task_saturated_endpoints'];d64windows=[r for r in interval if r['lane']=='D' and r['concurrency']==64 and r['classification']=='same_task_saturated_endpoints'];oldD=[r for r in interval if r['task_id']=='historical-D-545' and r['classification']=='same_task_saturated_endpoints']
    comparison=dict(D96_TRAIN=aggregate(trainwindows),D64_all545=aggregate(d64windows),D64_historical545=aggregate(oldD),causal=False,confounders=['TRAIN1080 vs fixed545','prompt/output length distributions','temperature/seed/prompt policy','completion selection','different start and end phases','D96 failed, one unknown'],train_terminal={k:trainrec['terminal'].get(k) for k in ['at','status','completed','total','valid','unknown','absent','wall_seconds','preparation_seconds','usage','newly_admitted','logical_HTTP_seconds','executor_queue_seconds']},train_client_config={k:train_config.get(k) for k in ['concurrency','timeout_seconds','expected_rows','full_denominator','dataset_partition']},initial_train_observation={k:train['latest'].get(k) for k in ['at','completed_branches','total_branches','elapsed_seconds','seconds_per_branch']},first_train_response_after_launch_seconds=train['first_response_event']['at']-train['launch']['at'])
    for name,rows in [('D96_TRAIN',trainwindows),('D64_all545',d64windows),('D64_historical545',oldD)]:
        endpoints=set((r['lane'],t) for r in rows for t in [r['start_epoch'],r['end_epoch']]);vs=[v for s in samples if (s['lane'],s['at']) in endpoints for v in s['AI_Core']];comparison[name].update(AI_Core_sample_count=len(vs),AI_Core_point_mean=mean(vs))
    # 累计与相邻差分分开保存：绝不平均ratio，也不对KV gauge求累计差。
    cumulative={}
    for host,hd in snapshots[-1]['hosts'].items():
        for lane,m in hd['metrics'].items():
            cumulative[lane]=dict(at=snapshots[-1]['at'],scope='engine lifetime, not task totals',completion_count=summetric(m,'decode_time_count'),prefix_hit_ratio=div(summetric(m,'prefix_hits'),summetric(m,'prefix_queries')),kv_fraction=summetric(m,'cache_usage'),preemptions=summetric(m,'preemptions'))
            for k in ['queue','prefill','decode']:cumulative[lane][k+'_mean_seconds']=div(summetric(m,k+'_time_sum'),summetric(m,k+'_time_count'))
    gaps=[]
    for lane,rs in by_lane.items():
        for a,brec in zip(rs,rs[1:]):
            if not math.isfinite(a['end']):continue
            t=brec['start']-a['end'];gaps.append(dict(lane=lane,predecessor=a['task_id'],successor=brec['task_id'],start_epoch=a['end'],end_epoch=brec['start'],gap_seconds=t,predecessor_status=(a.get('terminal') or {}).get('status'),unknown=(a.get('terminal') or {}).get('unknown'),meaning='terminal-to-launch; excludes first-serial-response and run initialization; not entire idle time'))
    stages=[]
    for r in records.values():
        p=state/'stage'/(r['task_id']+'-'+('npu2' if r['lane'] in 'AB' else 'npu')+'.json')
        if p.exists():
            s=capture(p);stages.append(dict(task_id=r['task_id'],lane=r['lane'],stage_at=s['at'],launch_at=r['start'],lead_seconds=r['start']-s['at'],archive_bytes=s.get('archive_bytes')))
    # 逐配置只读结构性统计；不打印私有policy custody，也不拷贝prompt/wire。
    cfg_summary=[]
    for p in sorted((root/'delivery/s-tfrl-rolling-v1/config').glob('*.json')):
        d=capture(p);cfg_summary.append(dict(path=str(p.relative_to(root)),keys=list(d) if isinstance(d,dict) else [],task_count=len(d.get('tasks',[])) if isinstance(d,dict) else None))
    source_pins={}
    for rel in ['scheduler.py','host_driver.py','train_runtime.py','bind_train.py','train-runtime-v1/run_s_native.py']:
        p=code/rel;raw=p.read_bytes();pin=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());manifest[str(p.relative_to(root))]=pin;source_pins[rel]=pin
    checks=[]
    for l,v in reference_interval['lanes'].items():
        ours=next(r for r in interval if r['lane']==l and abs(r['start_epoch']-reference_interval['start_epoch'])<.01 and abs(r['end_epoch']-reference_interval['end_epoch'])<.01)
        mapping={'completion_count':'completion_count','prompt_tokens_per_second':'prompt_tokens_per_second','generation_tokens_per_second':'generation_tokens_per_second','prefix_cache_hit_ratio':'prefix_hit_ratio','queue_mean_seconds_of_completions':'queue_mean_seconds','prefill_mean_seconds_of_completions':'prefill_mean_seconds','decode_mean_seconds_of_completions':'decode_mean_seconds'}
        checks.append(dict(lane=l,checks={k:abs(v[k]-ours[k2])<1e-8 for k,k2 in mapping.items()}))
    result=dict(schema='rolling-efficiency-local-analysis-v1',generated_utc=datetime.datetime.now(UTC).isoformat(),scope=dict(no_SSH=True,no_API=True,no_NPU=True,no_subagents=True,no_runtime_mutation=True,no_git_commit=True,output_only=str(OUT.relative_to(root))),data_range=dict(first_resource_utc=utc(snapshots[0]['at']),last_resource_utc=utc(snapshots[-1]['at']),resource_span_seconds=snapshots[-1]['at']-snapshots[0]['at'],resource_snapshots=len(snapshots),host_snapshots=sum(len(x['hosts']) for x in snapshots),metric_points=len(samples),adjacent_intervals=len(interval),counter_missing_intervals=sum(bool(r['missing_counters']) for r in interval),reset_or_service_change_intervals=sum(r['classification']=='reset_or_service_change' for r in interval),sample_intervals_seconds=describe([n['at']-p['at'] for p,n in zip(snapshots,snapshots[1:])]),actual_monetary_cost=None,card_count=16,instances=4,TP=4),identity_caveat='snapshots lack process_start_time/service_id; identity anchored to launch receipts, counter monotonicity detects visible resets only; unseen restart with counters overtaking old values cannot be excluded',classification_counts=dict(collections.Counter(r['classification'] for r in interval)),AI_Core_points=point_summary,steady_by_lane=steady_by_lane,comparison=comparison,latest_engine_cumulative=cumulative,terminals_and_archives=archive_stats,current_lanes=live,transitions=gaps,prestaging=stages,CPU_supply={k:reserve.get(k) for k in ['at','ready_tasks','ready_branches','cpu_preparing','total_running','estimated_queue_wall_hours','coverage_12h']}|dict(next_staged={k:v.get('next_staged') for k,v in reserve['lanes'].items()},configuration_summary=cfg_summary),engine_config=dict(at=audit.get('at'),max_num_seqs=audit['shared_argv']['--max-num-seqs'],max_num_batched_tokens=audit['shared_argv']['--max-num-batched-tokens'],gpu_memory_utilization=audit['shared_argv']['--gpu-memory-utilization'],TP=4,scope='argv audit receipt, not measured achieved batch size'),source_pins=source_pins,reference_crosscheck=checks,unknowns=['continuous AI Core time integral','per-step achieved decode batch / NPU kernel time / HCCL / HBM bandwidth','process start metric on each scrape','TRAIN96 full3240 result and exact failed-request token charge','isolated bind/preflight duration and timing after preparation_seconds ends','causal concurrency gain at same IDs/wire','exact whitespace token count and semantic-correct throughput for all tasks','cash cost','future queue guarantee'])
    dump('INPUT-MANIFEST.json',manifest);dump('INTERVALS.json',interval);dump('RESOURCE-POINTS.json',samples);dump('RESULT.json',result)
    cols=['lane','start_utc','end_utc','seconds','task_id','concurrency','classification','delta_usable','completion_count','seconds_per_branch','prompt_tokens_per_second','generation_tokens_per_second','queue_mean_seconds','prefill_mean_seconds','decode_mean_seconds','prefix_hit_ratio','kv_fraction_end','preemptions_delta','running_start','running_end','waiting_start','waiting_end']
    with (OUT/'INTERVALS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(interval)
    print(json.dumps({'data_range':result['data_range'],'classifications':result['classification_counts'],'steady':steady_by_lane,'comparison':comparison,'archive_rows':sum(a['rows'] for a in archive_stats),'crosscheck':checks},ensure_ascii=False,indent=2))

def selftest():
    def m(v):return {'counters':{k:{k:v} for k in COUNTERS}}
    d,missing,negative=deltas(m(2),m(5));assert not missing and not negative and all(v==3 for v in d.values())
    d,missing,negative=deltas(m(5),m(2));assert len(negative)==len(COUNTERS)
    d,missing,negative=deltas({},m(2));assert len(missing)==len(COUNTERS) and not d
    assert div(0,0) is None and div(6,3)==2
    assert points_smi('| 0 | 0000:C1:00.0 | 37 | 0 / 0 | 61 / 65 |')==[37.]
    assert points_smi('| 0 | 0000:C1:00.0 | 37                   0 / 0         61 / 65 |')==[37.]
    print('SELF_TEST_PASS: monotonic delta, reset, missing, denominator, AI Core parser')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=OUT.parents[1]);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:selftest()
    else:main(a.root.resolve())
