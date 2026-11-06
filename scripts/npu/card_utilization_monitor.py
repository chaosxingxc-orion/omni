#!/usr/bin/env python3
"""Sample all eight NPUs and report rolling observed utilization, not occupancy.

Run detached; output contains aggregate telemetry only. Missing samples never
count as idle or high utilization. p10/mean are sample statistics, not continuous
hardware accounting. This observer never launches or stops experiments.
"""
import argparse
from collections import deque
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from card_queue import parse_smi, read_json, write_atomic, iso


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timezone-aware observation timestamps required')
    return parsed.timestamp()


def assess_windows(samples, config):
    """Assess full, externally evidenced dispatch-to-response windows.

    Linear interpolation between adjacent hardware observations estimates time
    means; it is not continuous hardware accounting. Every window must be
    bracketed, with no missing cards, invalid values or excessive sampling gap.
    Window selection remains the caller's responsibility, not proven by a hash.
    """
    if config.get('schema') != 'saea-utilization-windows-v1':
        raise ValueError('utilization window schema required')
    threshold = config.get('threshold_percent')
    gap = config.get('max_sample_gap_seconds')
    for value in (threshold, gap):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('finite threshold and gap required')
    if not 0 <= threshold <= 100 or gap <= 0:
        raise ValueError('invalid threshold or sampling gap')
    times = [timestamp(row['time']) for row in samples]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('observation times must strictly increase')
    windows = config.get('windows')
    if not isinstance(windows, list) or not windows:
        raise ValueError('nonempty inference windows required')
    reports, assigned, names = [], set(), set()
    for window in windows:
        name, cards = window['id'], window['cards']
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('unique window identity required')
        names.add(name)
        if (not isinstance(cards, list) or not cards
                or any(type(c) is not int or c not in range(8) for c in cards)
                or len(set(cards)) != len(cards) or assigned.intersection(cards)):
            raise ValueError('distinct disjoint physical card assignments required')
        assigned.update(cards)
        start, end = timestamp(window['first_dispatch']), timestamp(window['last_response'])
        if end <= start:
            raise ValueError('positive completed inference window required')
        evidence = window.get('boundary_evidence_sha256', '')
        if (not isinstance(evidence, str) or len(evidence) != 64
                or any(c not in '0123456789abcdef' for c in evidence)):
            raise ValueError('boundary evidence hash required')
        pairs = [(i, max(start, a), min(end, b))
                 for i, (a, b) in enumerate(zip(times, times[1:]))
                 if a < end and b > start]
        covered = sum(b-a for _, a, b in pairs)
        card_reports = {}
        for card in cards:
            integral = 0.0
            reasons = []
            if not times or times[0] > start or times[-1] < end:
                reasons.append('window_not_bracketed')
            if not math.isclose(covered, end-start, abs_tol=1e-6):
                reasons.append('incomplete_window')
            indices = sorted({j for i, _, _ in pairs for j in (i, i+1)})
            values = {}
            for i in indices:
                observed = samples[i].get('cards', {})
                value = observed.get(str(card), observed.get(card, {})).get('ai')
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or not 0 <= value <= 100):
                    reasons.append('missing_or_invalid_card_observation')
                else:
                    values[i] = value
            for i, left, right in pairs:
                duration = times[i+1]-times[i]
                if duration > gap:
                    reasons.append('sampling_gap_exceeded')
                if i not in values or i+1 not in values:
                    continue
                slope = (values[i+1]-values[i])/duration
                at_left = values[i]+slope*(left-times[i])
                at_right = values[i]+slope*(right-times[i])
                integral += (at_left+at_right)/2*(right-left)
            mean = integral/(end-start) if not reasons else None
            inside = sorted(values[i] for i in values if start <= times[i] <= end)
            card_reports[str(card)] = {
                'time_weighted_mean_ai_percent': mean,
                'observed_inside_window': len(inside),
                'p10_ai_percent': inside[max(0, (len(inside)-1)//10)] if inside else None,
                'zero_observation_fraction': sum(v == 0 for v in inside)/len(inside) if inside else None,
                'strictly_above_threshold_fraction': sum(v > threshold for v in inside)/len(inside) if inside else None,
                'reasons': sorted(set(reasons)),
                'status': 'INSUFFICIENT_EVIDENCE' if reasons else ('PASS' if mean > threshold else 'FAIL'),
            }
        reports.append({'id': name, 'first_dispatch': window['first_dispatch'],
                        'last_response': window['last_response'], 'duration_seconds': end-start,
                        'boundary_evidence_sha256': evidence, 'cards': card_reports})
    if assigned != set(range(8)):
        raise ValueError('all eight physical cards must be assessed')
    statuses = [c['status'] for r in reports for c in r['cards'].values()]
    status = ('INSUFFICIENT_EVIDENCE' if 'INSUFFICIENT_EVIDENCE' in statuses
              else 'FAIL' if 'FAIL' in statuses else 'PASS')
    return {'schema': 'saea-utilization-acceptance-v1', 'status': status,
            'threshold_percent': threshold, 'comparison': 'strictly greater than',
            'max_sample_gap_seconds': gap, 'windows': reports,
            'scope': 'Linear interpolation estimate over supplied complete inference windows; '
                     'boundary provenance and experiment throughput require separate verification.'}


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
        result['time']=iso()  # hardware observation, before HTTP polling
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
    p.add_argument('--queue-dir',type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--assess-samples',type=Path,help='Offline JSONL hardware observations')
    p.add_argument('--windows',type=Path,help='Completed inference-window JSON config')
    p.add_argument('--interval',type=float,default=30)
    p.add_argument('--window',type=int,default=60)
    a=p.parse_args()
    if a.assess_samples or a.windows:
        if not (a.assess_samples and a.windows) or a.queue_dir:
            p.error('assessment requires --assess-samples and --windows, without --queue-dir')
        raw=a.assess_samples.read_bytes(); config_raw=a.windows.read_bytes()
        report=assess_windows([json.loads(line) for line in raw.splitlines() if line.strip()], json.loads(config_raw))
        report['samples_sha256']=hashlib.sha256(raw).hexdigest()
        report['window_config_sha256']=hashlib.sha256(config_raw).hexdigest()
        # Preserve prior reports; evidence revisions use a new output path.
        with a.out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
        raise SystemExit(0 if report['status']=='PASS' else 2)
    if a.queue_dir is None:p.error('--queue-dir required for collection')
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
