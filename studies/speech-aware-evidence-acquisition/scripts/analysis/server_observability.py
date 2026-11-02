"""Read-only vLLM metrics snapshots and reset-aware window attribution.

Histogram intervals describe completed requests on each instance. They are not
per-request traces and must not be added to client timings as disjoint stages.
"""
import argparse
import sys
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

HISTOGRAMS = ('request_queue_time_seconds', 'request_prefill_time_seconds',
    'request_decode_time_seconds', 'request_inference_time_seconds',
    'e2e_request_latency_seconds', 'time_to_first_token_seconds',
    'inter_token_latency_seconds')
COUNTERS = ('request_success_total', 'prompt_tokens_total', 'generation_tokens_total',
    'num_preemptions_total', 'prefix_cache_queries_total', 'prefix_cache_hits_total',
    'mm_cache_queries_total', 'mm_cache_hits_total', 'prompt_tokens_cached_total')
GAUGES = ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc')
LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^\s]+)(?:\s+[^\s]+)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def parse_metrics(text):
    """Keep label identity privately as a hash, plus histogram bounds only."""
    names = {'vllm:'+k for k in (*COUNTERS,*GAUGES)}
    names |= {'vllm:'+k+s for k in HISTOGRAMS for s in ('_sum','_count','_bucket')}
    result = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):continue
        match=LINE.fullmatch(line)
        if not match or match[1] not in names:continue
        value=float(match[3])
        if not math.isfinite(value):raise ValueError('nonfinite metric value')
        labels=dict(LABEL.findall(match[2] or ''))
        bound=labels.pop('le',None)
        identity=hashlib.sha256(json.dumps(labels,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        key=match[1]+'|'+identity+('|' + bound if bound is not None else '')
        if key in result:raise ValueError('duplicate metric series')
        result[key]={'name':match[1],'series':identity,'value':value}
        if bound is not None:result[key]['upper_bound']=bound
    return result


def snapshot(endpoints, *, window_id, mode='normal', identity=None):
    if mode not in ('normal','debug'):raise ValueError('normal or debug mode required')
    if not window_id:raise ValueError('window identity required')
    result={'schema':'saea-server-observability-v1','window_id':window_id,'mode':mode,
        'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'engine_identity':identity,'instances':{},'new_model_calls':0}
    for label,url in endpoints.items():
        try:
            with urllib.request.urlopen(url,timeout=10) as response:
                body=response.read(16*1024*1024+1)
            if len(body)>16*1024*1024:raise ValueError('metrics payload exceeds bound')
            metrics=parse_metrics(body.decode())
            # Normal retains counters + histogram sums/counts. Debug adds buckets.
            if mode=='normal':metrics={k:v for k,v in metrics.items() if not v['name'].endswith('_bucket')}
            result['instances'][label]={'metrics':metrics,'metrics_sha256':hashlib.sha256(body).hexdigest()}
        except Exception as error:
            result['instances'][label]={'error_type':type(error).__name__,'available':False}
    return result


def difference(before,after):
    if before['window_id']!=after['window_id']:raise ValueError('window identity differs')
    if before['mode']!=after['mode']:raise ValueError('observation mode differs')
    if before.get('engine_identity')!=after.get('engine_identity'):raise ValueError('engine identity changed')
    result={'schema':'saea-server-observability-delta-v1','window_id':before['window_id'],
        'mode':before['mode'],'instances':{},'request_attribution':'aggregate-instance-window-only',
        'isolated_window_verified':False,'engine_identity_supplied':before.get('engine_identity') is not None,
        'limitations':['Concurrent workloads may contribute; match window boundaries and journal counts before attribution.',
            'Histograms may cover overlapping stages and different event populations; never sum as disjoint wall time.',
            'No per-request server trace or raw-audio encoder timing is inferred.']}
    for instance in sorted(set(before['instances'])|set(after['instances'])):
        a=before['instances'].get(instance,{}).get('metrics');b=after['instances'].get(instance,{}).get('metrics')
        if a is None or b is None:
            result['instances'][instance]={'available':False,'reason':'missing snapshot'};continue
        reset=[];changed=set(a)^set(b);delta={}
        for key in set(a)&set(b):
            name=b[key]['name'];d=b[key]['value']-a[key]['value']
            if name in {'vllm:'+k for k in GAUGES}:continue
            if d<0:reset.append(key)
            delta[key]={**b[key],'value':d}
        if reset or changed:
            result['instances'][instance]={'available':False,'reason':'counter reset or series changed','reset_series_count':len(reset),'changed_series_count':len(changed)};continue
        def total(name):
            vals=[v['value'] for v in delta.values() if v['name']=='vllm:'+name]
            return sum(vals) if vals else None
        hist={}
        for name in HISTOGRAMS:
            count=total(name+'_count');summed=total(name+'_sum')
            h={'count':count,'sum_seconds':summed,'mean_seconds':summed/count if count and summed is not None else None}
            buckets={}
            for v in delta.values():
                if v['name']=='vllm:'+name+'_bucket':
                    bound=v['upper_bound'];buckets[bound]=buckets.get(bound,0)+v['value']
            if buckets:h['cumulative_buckets']=buckets
            hist[name]=h
        result['instances'][instance]={'available':True,'histograms':hist,'counters':{k:total(k) for k in COUNTERS},
            'gauges_after':{name:[v['value'] for v in b.values() if v['name']=='vllm:'+name] for name in GAUGES}}
    return result


def write_new(path,doc):
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    publish_new(path,(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())



def _validate_stop_source(stop_file, terminal_status, expected_container_id):
    if terminal_status is None:
        if stop_file is None or expected_container_id is not None:
            raise ValueError('stop-file or paired terminal-status/expected-container-id required')
    elif stop_file is not None or not isinstance(expected_container_id, str) or not re.fullmatch('[a-f0-9]{64}', expected_container_id):
        raise ValueError('terminal-status requires a full 64-character container ID and excludes stop-file')


def _read_terminal_status(path, expected_container_id):
    """Read a configured launcher observation; missing never means complete."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        observed = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise ValueError('corrupt launcher terminal status') from error
    if not isinstance(observed, dict):
        raise ValueError('launcher status must be an object')
    container_id = observed.get('container_id')
    if not isinstance(container_id, str) or not re.fullmatch('[a-f0-9]{64}', container_id):
        raise ValueError('launcher status requires a full container ID')
    if container_id != expected_container_id:
        raise ValueError('launcher container identity changed')
    state = observed.get('state')
    if state not in ('created', 'running', 'paused', 'restarting', 'exited', 'dead') or type(observed.get('running')) is not bool:
        raise ValueError('invalid launcher state or running flag')
    if 'exit_code' in observed and type(observed['exit_code']) is not int:
        raise ValueError('invalid launcher exit code')
    if state in ('exited', 'dead'):
        if observed['running']:
            raise ValueError('terminal launcher state cannot be running')
        return {key: observed[key] for key in ('container_id', 'state', 'running', 'exit_code') if key in observed}
    return None


def collect_windows(endpoints, *, window_id, mode, config, out, max_snapshots, stop_file=None, identity_path=None, terminal_status=None, expected_container_id=None):
    """Background collector with a validated event before the final snapshot.

    Polling the local terminal file once per second never contacts a service or
    model. Only scheduled or terminal events trigger metrics HTTP collection.
    Reaching max_snapshots without a previously observed terminal event is a
    bounded observation, not a claim that the producer or science completed.
    """
    from speech_aware_evidence_acquisition.core.stage_timing import _load_configuration
    _validate_stop_source(stop_file, terminal_status, expected_container_id)
    profile=_load_configuration(mode,str(config),None,None)
    if type(max_snapshots) is not int or max_snapshots<2:raise ValueError('at least two bounded snapshots required')
    def read_terminal():
        if terminal_status is not None:
            return _read_terminal_status(terminal_status, expected_container_id)
        return {'source': 'stop-file'} if stop_file.exists() else None
    terminal=read_terminal()
    out.mkdir(parents=True,exist_ok=False)
    write_new(out/'config.json',{'schema':'saea-server-observation-config-v1','effective':profile,
        'configuration_sha256':hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest(),
        'window_id':window_id,'max_snapshots':max_snapshots,'scope':'instance aggregate; no request payloads',
        'termination': {'source': 'terminal-status', 'expected_container_id': expected_container_id}
            if terminal_status is not None else {'source': 'stop-file'}})
    previous=None
    for index in range(max_snapshots):
        # Revalidate even an event latched during the local wait. An atomic
        # replacement for another container must not authorize this snapshot.
        terminal=read_terminal() or terminal
        identity=json.loads(identity_path.read_bytes()) if identity_path else None
        current=snapshot(endpoints,window_id=window_id,mode=mode,identity=identity)
        write_new(out/f'{index:04d}.snapshot.json',current)
        if previous is not None:write_new(out/f'{index:04d}.delta.json',difference(previous,current))
        previous=current
        read_terminal()  # Reject corruption or identity replacement during collection.
        if terminal is not None:
            break
        if index+1==max_snapshots:break
        deadline=time.monotonic()+profile['server_metrics_interval_seconds']
        while True:
            terminal=read_terminal()
            if terminal is not None:break
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            time.sleep(min(1,remaining))
    completion={'snapshots':index+1,'terminal_observed':terminal is not None,'new_model_calls':0,
        'termination_reason':'terminal-event' if terminal is not None else 'snapshot-limit',
        'final_snapshot_after_terminal':terminal is not None}
    if terminal_status is not None:
        completion['terminal_status']=terminal
        completion['scientific_success']='verify shared runtime artifacts separately'
    write_new(out/'complete.json',completion)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('snapshot');s.add_argument('--endpoint',action='append',required=True,help='label=http://host/metrics')
    s.add_argument('--window-id',required=True);s.add_argument('--mode',choices=('normal','debug'),default='normal');s.add_argument('--identity',type=Path)
    d=sub.add_parser('diff');d.add_argument('--before',type=Path,required=True);d.add_argument('--after',type=Path,required=True)
    w=sub.add_parser('watch');w.add_argument('--endpoint',action='append',required=True);w.add_argument('--window-id',required=True)
    w.add_argument('--mode',choices=('normal','debug'),default='normal');w.add_argument('--config',type=Path,required=True)
    stop=w.add_mutually_exclusive_group(required=True)
    stop.add_argument('--stop-file',type=Path);stop.add_argument('--terminal-status',type=Path)
    w.add_argument('--expected-container-id');w.add_argument('--max-snapshots',type=int,default=145);w.add_argument('--identity',type=Path)
    for x in (s,d,w):x.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.action=='watch':
        try:_validate_stop_source(a.stop_file,a.terminal_status,a.expected_container_id)
        except ValueError as error:p.error(str(error))
        endpoints=dict(x.split('=',1) for x in a.endpoint)
        if len(endpoints)!=len(a.endpoint):raise ValueError('unique instance labels required')
        collect_windows(endpoints,window_id=a.window_id,mode=a.mode,config=a.config,out=a.out,stop_file=a.stop_file,max_snapshots=a.max_snapshots,identity_path=a.identity,terminal_status=a.terminal_status,expected_container_id=a.expected_container_id)
        return
    if a.action=='snapshot':
        endpoints=dict(x.split('=',1) for x in a.endpoint)
        if len(endpoints)!=len(a.endpoint):raise ValueError('unique instance labels required')
        doc=snapshot(endpoints,window_id=a.window_id,mode=a.mode,identity=json.loads(a.identity.read_bytes()) if a.identity else None)
    else:doc=difference(json.loads(a.before.read_bytes()),json.loads(a.after.read_bytes()))
    write_new(a.out,doc)

if __name__=='__main__':main()
