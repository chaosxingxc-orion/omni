"""Configured finite HTTP measurement under the original service guards.

No service start/stop, model changes, request retries, training or new scheduler.
Reuses the project's immutable local storage and receipt implementation.
"""
import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from io_utils import doc, pin, save, storage, require, canonical


def metrics(endpoint):
    with urllib.request.urlopen(endpoint+'/metrics', timeout=15) as response:
        return response.read().decode()


def idle(raw):
    for key in ('running','waiting'):
        values=[float(row.rsplit(' ',1)[1]) for row in raw.splitlines()
                if row.startswith('vllm:num_requests_'+key+'{')]
        require(values and sum(values)==0, 'service has in-flight requests')


def service_identity(service):
    fields=(Path('/proc')/str(service['pid'])/'stat').read_text().rsplit(')',1)[1].split()
    require(fields[0]!='Z' and int(fields[19])==service['start_ticks'],'service process identity changed')


def counters(raw, name):
    values=[float(row.split()[1]) for row in raw.splitlines() if row.startswith('vllm:'+name+'{')]
    return sum(values) if values else None


def work_metrics(before, after):
    def delta(name):
        a,b=counters(before,name),counters(after,name)
        if a is None or b is None:return None
        require(math.isfinite(a) and math.isfinite(b) and b>=a,'invalid or reset metric: '+name)
        return b-a
    def mean(name):
        n,total=delta(name+'_count'),delta(name+'_sum')
        return total/n if n and total is not None else None
    return dict(prefix_queries=delta('prefix_cache_queries_total'),prefix_hits=delta('prefix_cache_hits_total'),
        mm_queries=delta('mm_cache_queries_total'),mm_hits=delta('mm_cache_hits_total'),
        mean_ttft_seconds=mean('time_to_first_token_seconds'),
        mean_itl_seconds=mean('inter_token_latency_seconds'),
        mean_request_tpot_seconds=mean('request_time_per_output_token_seconds'),
        tpot_requests=delta('request_time_per_output_token_seconds_count'),
        itl_intervals=delta('inter_token_latency_seconds_count'),
        itl_total_seconds=delta('inter_token_latency_seconds_sum'),
        mean_decode_seconds=mean('request_decode_time_seconds'),
        mean_prefill_seconds=mean('request_prefill_time_seconds'),
        mean_queue_seconds=mean('request_queue_time_seconds'))


def grouped_rounds(config, services):
    """Two complete groups per round; matched wires and isolated cache namespaces."""
    blocks=config['grouped_blocks']
    require(config['schema']=='existing-service-efficiency-v1' and (len(services)==2 or
            len(services)==1 and config.get('decode_replay') is True) and blocks,
            'group measurement needs two instances and paired blocks')
    arms=config['round_order']
    paired=arms in (['paired'],['paired','paired_warm'])
    require(paired or set(arms)=={'split','affinity'} and len(arms)==2,
            'use affinity/split or paired with optional warm repeat')
    require('rounds' not in config and 'cases' not in config and 'wire_ref' not in config,
            'choose one measurement input form')
    roster=[]
    for pair in blocks:
        require(len(pair)==2 and all(1<=len(group)<=4 if paired else len(group)==4 for group in pair),
                'expected two input groups with the configured replicates')
        wires=[[doc(ref) for ref in group] for group in pair]
        for group in wires:
            replay=config.get('decode_replay',False)
            require(type(replay) is bool and (not replay or paired),'replay requires a paired measurement')
            allowed={'seed','max_tokens','vllm_xargs'} if replay else {'seed'}
            if replay:
                for w in group:
                    actions=(w.get('vllm_xargs') or {}).get('decode_action_ids')
                    require(isinstance(actions,list) and actions and len(actions)==w['max_tokens']
                            and set(w['vllm_xargs'])=={'decode_action_ids'}
                            and all(type(t) is int and t>=0 for t in actions)
                            and w['top_logprobs']==2, 'invalid diagnostic action replay')
            common=[{k:v for k,v in w.items() if k not in allowed} for w in group]
            require(all(x==common[0] for x in common) and len({w['seed'] for w in group})==len(group),
                    'expected byte-equivalent conditioning and distinct seeds within group')
            require(all(w['model']=='qwen3-omni-30b-a3b-instruct' and
                w['chat_template_kwargs']['enable_thinking'] is False and 'cache_salt' not in w
                and w['n']==1 and w['stream'] is False for w in group),'unexpected group wire')
        require(wires[0][0]['messages']!=wires[1][0]['messages'],'need distinct inputs across groups')
        roster.append(wires)
    nonce=config['measurement_id']
    require(isinstance(nonce,str) and nonce and len(nonce)<=128,'explicit cache namespace')
    in_flight=config.get('grouped_concurrency',[4])
    require(type(in_flight) is list and in_flight and len(set(in_flight))==len(in_flight)
            and all(type(n) is int and 1<=n<=(8 if paired else 4) for n in in_flight),
            'configured requests in flight per lane')
    for b,wires in enumerate(roster):
        if paired:
            for concurrency in in_flight:
                salt=hashlib.sha256(canonical([nonce,b,'paired',concurrency])).hexdigest()
                for arm in arms:
                    matched=[(g,i,dict(wire,cache_salt=salt))
                             for g,group in enumerate(wires) for i,wire in enumerate(group)]
                    yield b,arm,{s['name']:matched for s in services},concurrency
            continue
        order=config['round_order'] if b%2==0 else config['round_order'][::-1]
        for arm in order:
            for concurrency in (in_flight if b%2==0 else in_flight[::-1]):
                salt=hashlib.sha256(canonical([nonce,b,arm,concurrency])).hexdigest()
                groups={s['name']:[] for s in services}
                for g,group in enumerate(wires):
                    for i,wire in enumerate(group):
                        lane=(b+g+(i%2 if arm=='split' else 0))%2
                        groups[services[lane]['name']].append((g,i,dict(wire,cache_salt=salt)))
                yield b,arm,groups,concurrency


def distinct_rounds(config, services):
    """Paired comparison on different recordings, with roles crossed between panels."""
    width = config['grouped_concurrency']
    require(len(width)==1 and type(width[0]) is int and width[0]>0 and len(services)==2, 'paired concurrency')
    n=width[0]; modes=config.get('admission_order')
    windows=config.get('lane_concurrency')
    require(bool(config['measurement_id']) and (
        modes is not None and sorted(modes)==['continuous','group_drained'] and windows is None or
        modes is None and isinstance(windows,list) and len(windows)==len(config['distinct_panels'])
        and all(set(w)=={s['name'] for s in services} and
                all(type(k) is int and 1<=k<=32 for k in w.values()) for w in windows)), 'paired admission')
    for b,refs in enumerate(config['distinct_panels']):
        wires=[doc(ref) for ref in refs]
        require(len(wires)>=2*(max(windows[b].values()) if windows else n) and len(wires)%n==0 and
                len({hashlib.sha256(canonical(w['messages'])).digest() for w in wires})==len(wires), 'distinct complete waves required')
        salt=hashlib.sha256(canonical([config['measurement_id'],b])).hexdigest()
        require(all(w['model']=='qwen3-omni-30b-a3b-instruct' and 'cache_salt' not in w for w in wires),'unchanged business wires')
        panel=[(i//n,i%n,dict(w,cache_salt=salt)) for i,w in enumerate(wires)]
        for arm in config['round_order']:
            require(arm in ('paired','paired_warm'),'paired cache windows')
            yield b,arm,{v['name']:panel for v in services},windows[b] if windows else n


def measure_grouped(config, root, plan):
    services=config['services']
    reports=[]
    for index,(block,arm,roster,concurrency) in enumerate(plan):
        storage(root,1024**3)
        directory=root/f'round-{index}'
        before={}
        for service in services:
            name=service['name'];before[name]=metrics(service['endpoint']);idle(before[name])
            save(directory/name/'BEFORE.json',dict(metrics=before[name]))
        started=time.monotonic()
        admission = config.get('admission', 'continuous')
        require(admission in ('continuous', 'group_drained') and
                (admission == 'continuous' or arm.startswith('paired')), 'invalid grouped admission')

        def measure_lane(service):
            name = service['name']
            requests = roster[name]
            if 'stock_adapter_alias' in service:
                alias=service['stock_adapter_alias']
                with urllib.request.urlopen(service['endpoint']+'/v1/models',timeout=8) as response:
                    entries=json.load(response)['data']
                require(any(x['id']==alias and x.get('parent')=='qwen3-omni-30b-a3b-instruct' for x in entries),'stock adapter not present')
                requests=[(g,i,dict(w,model=alias)) for g,i,w in requests]
            mode = config['admission_order'][(block+services.index(service))%2] if 'admission_order' in config else admission
            waves = [requests]
            if mode == 'group_drained':
                waves = [[item for item in requests if item[0] == group]
                         for group in dict.fromkeys(item[0] for item in requests)]
            lane_rows = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency[name] if isinstance(concurrency,dict) else concurrency) as pool:
                for wave in waves:
                    released=time.monotonic()-started
                    jobs = {pool.submit(contact, directory/name/f'{g}-{i}', service['endpoint'], wire,
                                        config['request_timeout_seconds'], queued_at=started): (g, i)
                            for g, i, wire in wave}
                    for job in concurrent.futures.as_completed(jobs):
                        g, i = jobs[job]
                        lane_rows.append(dict(service=name, group=g, replicate=i, batch_release_from_round_seconds=released, **job.result()))
            return lane_rows

        with contextlib.ExitStack() as controls:
            if config.get('capture_device_trace') is True:
                def control(s,command):
                    req=urllib.request.Request(s['endpoint']+'/'+command,method='POST')
                    with urllib.request.urlopen(req,timeout=config['request_timeout_seconds']) as r:
                        require(r.status==200,command)
                for s in services:
                    control(s,'start_profile')
                    controls.callback(control,s,'stop_profile')
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(services)) as lanes:
                rows = [row for lane_rows in lanes.map(measure_lane, services) for row in lane_rows]
        elapsed=time.monotonic()-started
        by_service={}
        for service in services:
            name=service['name'];raw=metrics(service['endpoint']);idle(raw);service_identity(service)
            save(directory/name/'AFTER.json',dict(metrics=raw))
            by_service[name]=work_metrics(before[name],raw)
        queries=[v['prefix_queries'] for v in by_service.values()]
        hits=[v['prefix_hits'] for v in by_service.values()]
        prompts=sum(r['usage']['prompt_tokens'] for r in rows)
        require(None not in queries+hits and sum(queries)==prompts and 0<=sum(hits)<=prompts,
                'cache counters do not account for exactly the measured prompts')
        per_service={}
        for name in before:
            service_rows=[r for r in rows if r['service']==name]
            seconds=max(r['finished_from_round_seconds'] for r in service_rows)
            tokens=sum(r['usage']['completion_tokens'] for r in service_rows)
            per_service[name]=dict(requests=len(service_rows),completion_tokens=tokens,wall_seconds=seconds,
                                   output_tokens_per_second=tokens/seconds)
        reports.append(save(directory/'RESULT.json',dict(block=block,arm=arm,wall_seconds=elapsed,
            concurrency_per_service=concurrency, admission=({s['name']:config['admission_order'][(block+i)%2] for i,s in enumerate(services)} if 'admission_order' in config else admission), requests_per_second=len(rows)/elapsed,
            groups_per_minute=None if 'distinct_panels' in config else 120*(len(services) if arm.startswith('paired') else 1)/elapsed,
            prompt_tokens=prompts,cached_tokens=int(sum(hits)),completion_tokens=sum(r['usage']['completion_tokens'] for r in rows),
            http_p50_seconds=statistics.median(r['elapsed_seconds'] for r in rows),metrics=by_service,
            per_service=per_service,
            requests=sorted(rows,key=lambda r:(r['group'],r['replicate'])),
            input_refs=(config['distinct_panels'] if 'distinct_panels' in config else config['grouped_blocks'])[block])))
    return reports


def contact(root, endpoint, wire, timeout, api_path='/v1/chat/completions', *, queued_at=None):
    require(api_path in ('/v1/chat/completions','/v1/completions'),'known diagnostic API route')
    require(not (root/'INTENT.json').exists(), 'existing request: reconcile, never resend')
    payload=canonical(wire)
    save(root/'INTENT.json',dict(endpoint=endpoint,api_path=api_path,wire_sha256=hashlib.sha256(payload).hexdigest(),
         started_epoch=time.time(),training_use=False,automatic_retry=False))
    started=time.monotonic()
    request=urllib.request.Request(endpoint+api_path,data=payload,
        headers={'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            raw=response.read();status=response.status
    except urllib.error.HTTPError as error:
        raw=error.read();status=error.code
    except Exception as error:
        save(root/'UNKNOWN.json',dict(error=repr(error),elapsed=time.monotonic()-started,
            automatic_retry=False,retain_guard_until_service_idle=True))
        raise
    finished=time.monotonic()
    elapsed=finished-started
    receipt=save(root/'RESPONSE.json',dict(status=status,raw=raw.decode(),elapsed_seconds=elapsed,
        response_sha256=hashlib.sha256(raw).hexdigest()))
    require(status==200,'HTTP rejection retained: '+str(status))
    value=json.loads(raw);require(value.get('choices'),'missing choices')
    if wire.get('return_token_ids') is True:
        tokens=value['choices'][0].get('token_ids')
        probs=(value['choices'][0].get('logprobs') or {}).get('content')
        require(isinstance(tokens,list) and tokens and isinstance(probs,list)
                and len(tokens)==len(probs) and all(item['token']==f'token_id:{t}'
                and math.isfinite(item['logprob']) and item['logprob']!=-9999
                for t,item in zip(tokens,probs,strict=True)), 'invalid sampled action probabilities')
    return dict(receipt_ref=receipt,elapsed_seconds=elapsed,usage=value.get('usage',{}),
        ready_to_http_seconds=None if queued_at is None else started-queued_at,
        finished_from_round_seconds=None if queued_at is None else finished-queued_at,
        finish_reasons=[item.get('finish_reason') for item in value['choices']])


def run(config_path):
    config=doc(pin(config_path));root=Path(config['output'])
    require(config['schema']=='existing-service-efficiency-v1','config schema')
    require(config['source_boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'host changed')
    storage(root,1024**3)
    grouped='grouped_blocks' in config or 'distinct_panels' in config
    require(not ('grouped_blocks' in config and 'distinct_panels' in config),'ambiguous panel selection')
    plan=list((distinct_rounds if 'distinct_panels' in config else grouped_rounds)(config,config['services'])) if grouped else None
    cases=[] if grouped else config.get('cases') or [dict(id='default',wire_ref=config['wire_ref'])]
    require(grouped or cases and len({c['id'] for c in cases})==len(cases),'distinct configured case IDs')
    wires={}
    for case in cases:
        require(isinstance(case['id'],str) and case['id'] and
                all(c.isalnum() or c in '-_' for c in case['id']),'safe case ID')
        wire=doc(case['wire_ref'])
        require(wire['model']=='qwen3-omni-30b-a3b-instruct','Instruct required')
        api_path=case.get('api_path','/v1/chat/completions')
        require(api_path in ('/v1/chat/completions','/v1/completions'),'known configured API route')
        if api_path=='/v1/chat/completions':
            require(wire['chat_template_kwargs']['enable_thinking'] is False,'Instruct OFF required')
        else:
            require(case.get('thinking') is False and case.get('text_only') is True and
                    isinstance(wire.get('prompt'),str) and 'messages' not in wire,
                    'rendered text-only OFF diagnostic; not multimodal training acceptance')
        wires[case['id']]=wire
    require(grouped or config['rounds'] and all(type(n) is int and 1<=n<=32 for n in config['rounds']),
            'bounded explicit concurrency')
    require(len({x['endpoint'] for x in config['services']})==len(config['services']),'duplicate endpoints')
    require(len({x['name'] for x in config['services']})==len(config['services']) and
            all(x['name'] and all(c.isalnum() or c in '-_' for c in x['name']) for x in config['services']),
            'unique safe service names')
    require(all(x['endpoint'].startswith('http://127.0.0.1:') for x in config['services']), 'local service only')
    with contextlib.ExitStack() as stack:
        for guard in sorted(config['guards']):
            lock=stack.enter_context(Path(guard).open('rb'))
            wait=config.get('guard_wait_seconds',0)
            require(type(wait) is int and 0<=wait<=3600,'bounded guard wait required')
            deadline=time.monotonic()+wait
            while True:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise
                    time.sleep(min(2,remaining))
        require(not (root/'STARTED.json').exists(),'measurement already started; no replay')
        for service in config['services']:
            service_identity(service);idle(metrics(service['endpoint']))
        save(root/'STARTED.json',dict(config_ref=pin(config_path),started_epoch=time.time(),
            guards=config['guards'],training_use=False))
        rows=[]
        try:
            if grouped:rows=measure_grouped(config,root,plan)
            planned=[(case,concurrency) for case in cases for concurrency in config['rounds']]
            for index,(case,concurrency) in enumerate(planned):
                wire=wires[case['id']]
                for service in config['services']:
                    save(root/f'round-{index}'/service['name']/'BEFORE.json',dict(metrics=metrics(service['endpoint'])))
                begin=time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency*len(config['services'])) as pool:
                    jobs={pool.submit(contact,root/f'round-{index}'/service['name']/str(i),
                          service['endpoint'],wire,config['request_timeout_seconds'],
                          case.get('api_path','/v1/chat/completions')):service['name']
                          for service in config['services'] for i in range(concurrency)}
                    round_rows=[]
                    for future in concurrent.futures.as_completed(jobs):
                        round_rows.append(dict(service=jobs[future],**future.result()))
                elapsed=time.monotonic()-begin
                summary=dict(case_id=case['id'],wire_ref=case['wire_ref'],
                    concurrency_per_service=concurrency,wall_seconds=elapsed,requests=round_rows,
                    aggregate_completion_tokens=sum(r['usage'].get('completion_tokens',0) for r in round_rows),
                    scope='two-service HTTP wall time; includes prefill/queue/decode, not isolated decode')
                rows.append(save(root/f'round-{index}'/'RESULT.json',summary))
                for service in config['services']:
                    raw=metrics(service['endpoint']);idle(raw)
                    save(root/f'round-{index}'/service['name']/'AFTER.json',dict(metrics=raw))
            save(root/'RESULT.json',dict(passed=True,rounds=rows,training_use=False,
                precision_acceptance=False,service_mutations=0))
        finally:
            # On a lost response do not release a guard while generation may
            # still be running. No retry or replacement HTTP request is sent.
            import sys
            failed=sys.exc_info()[0] is not None
            recovered=False
            while True:
                try:
                    for service in config['services']:idle(metrics(service['endpoint']))
                    break
                except Exception:
                    if failed and 'start_receipts' in config and 'distinct_panels' in config:
                        from reconcile_failed_panel import inspect_window
                        try:
                            outcome=inspect_window(config_path,config['start_receipts'])
                        except Exception:
                            pass  # Unknown or incomplete ledger: retain the guard.
                        else:
                            save(root/'DRAINED.json',outcome);recovered=True;break
                    time.sleep(10)
            if not recovered:
                save(root/'DRAINED.json',dict(guards_releasable=True,finished_epoch=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();run(args.config)
