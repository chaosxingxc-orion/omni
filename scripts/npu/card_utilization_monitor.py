#!/usr/bin/env python3
"""Sample all eight NPUs and report rolling observed utilization, not occupancy.

Run detached; output contains aggregate telemetry only. Missing samples never
count as idle or high utilization. p10/mean are sample statistics, not continuous
hardware accounting. This observer never launches or stops experiments.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from card_queue import parse_smi, read_json, write_atomic, iso


def summarize(samples, card_ids, busy_threshold=90):
    result={}
    for card in card_ids:
        values=sorted(s['cards'][card]['ai'] for s in samples if card in s['cards'])
        n=len(values)
        result[str(card)]={'observed_samples':n,'missing_samples':len(samples)-n,
            'mean_ai_percent':sum(values)/n if n else None,
            'p10_ai_percent':values[max(0,(n-1)//10)] if n else None,
            'zero_ai_fraction':sum(v==0 for v in values)/n if n else None,
            'above_threshold_fraction':sum(v>=busy_threshold for v in values)/n if n else None}
    full=[s for s in samples if all(c in s['cards'] for c in card_ids)]
    return {'cards':result,'complete_samples':len(full),
        'all_cards_above_threshold_fraction':sum(all(s['cards'][c]['ai']>=busy_threshold for c in card_ids) for s in full)/len(full) if full else None,
        'busy_threshold_percent':busy_threshold,
        'scope':'rolling observed samples; resident memory is not utilization'}


def sample(queue_dir):
    result={'time':iso(),'cards':{},'endpoints':{},'errors':[]}
    try:
        text=subprocess.check_output(['npu-smi','info'],text=True,timeout=15)
        result['cards']=parse_smi(text)[0]
    except (OSError,subprocess.SubprocessError) as e:result['errors'].append(type(e).__name__)
    for port in range(8200,8204):
        try:
            with urllib.request.urlopen('http://127.0.0.1:%d/metrics'%port,timeout=3) as response:
                lines=response.read().decode().splitlines()
            gauges={}
            for name in ('num_requests_running','num_requests_waiting'):
                found=[float(line.rsplit(' ',1)[1]) for line in lines if line.startswith('vllm:'+name+'{')]
                gauges[name]=sum(found) if found else None
            result['endpoints'][str(port)]=gauges
        except (OSError,ValueError) as e:result['endpoints'][str(port)]={'error':type(e).__name__}
    state=read_json(str(queue_dir/'state.json')) or {}
    definition=read_json(str(queue_dir/'queue.json')) or {}
    result['queue']={'state':state.get('state'),'current':state.get('current'),
        'jobs':{j['id']:state.get('jobs',{}).get(j['id'],{}).get('status','pending') for j in definition.get('jobs',[])}}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-dir',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--interval',type=float,default=30)
    p.add_argument('--window',type=int,default=60)
    a=p.parse_args()
    if a.interval<=0 or a.window<1:p.error('positive interval/window required')
    a.out.mkdir(parents=True,exist_ok=False)
    stop=[False]
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    samples=deque(maxlen=a.window)
    while not stop[0]:
        started=time.monotonic();current=sample(a.queue_dir);samples.append(current)
        report={'updated':current['time'],'interval_seconds':a.interval,
            'window_started':samples[0]['time'],'sample_count':len(samples),
            'latest':current,'rolling':summarize(samples,list(range(8)))}
        with (a.out/'samples.jsonl').open('a') as f:f.write(json.dumps(current)+'\n')
        write_atomic(str(a.out/'summary.json'),json.dumps(report,indent=2)+'\n')
        if current['queue']['state'] in ('done','failed','stopped'):break
        until=started+a.interval
        while not stop[0] and time.monotonic()<until:time.sleep(min(1,until-time.monotonic()))


if __name__=='__main__':main()
