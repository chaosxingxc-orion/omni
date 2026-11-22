"""Join device events outside graph replays to closed scheduler steps."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

from io_utils import doc, pin, require, save
from report_batch_stage import report


PREFILL = ('before_attention','attention_kv','attention_gather','post_attention',
           'expert_cat','expert_reduce','weighted_sum')
DECODE = ('decode_before','decode_kv','decode_gather','decode_after','decode_reduce','decode_sum')
ATTENTION_PARTS = ('decode_plan', 'decode_pack_query', 'decode_read_kv', 'decode_fia', 'decode_copy_result')


def summarize(config, state, abi, lane):
    ledger = report(config, state, lane)
    layers = doc(pin(abi))['timed_layers']
    attention_source = abi.parent/'bundle/parents/jev-system-rl/implementation/decode_tiled_attention.py'
    timing_attention = attention_source.is_file() and "'decode_copy_result'" in attention_source.read_text()
    values = defaultdict(list)
    for path in state.glob('stages-*.jsonl'):
        events = [json.loads(line) for line in path.read_bytes().splitlines()]
        frames = [e for e in events if e['kind']=='worker_forward']
        if not frames:
            continue
        timings = defaultdict(dict)
        for e in events:
            if e['kind']!='prefill_interval':
                continue
            key = e['layer'],e['phase']
            require(key not in timings[e['step']], 'duplicate segment')
            require(all(math.isfinite(e[k]) and e[k]>=0 for k in ('device_ms','host_ms')), 'invalid duration')
            timings[e['step']][key] = e
        for step in ledger['steps']:
            frame, = [f for f in frames if step['start_ns']<=f['start']<=f['t']<=step['end_ns']]
            values[step['step']].append(timings[frame['start']])
    out=[]
    for s in ledger['steps']:
        ranks=values[s['step']]
        require(len(ranks)==4, 'missing worker')
        pure=len(s['decode'])==len(s['alloc'])
        phases=DECODE if pure else PREFILL
        expected={(i,p) for i in layers for p in phases}
        if pure and timing_attention:
            expected.update((i,p) for i in layers for p in ATTENTION_PARTS)
        require(all(set(x)==expected for x in ranks), f"incomplete step {s['step']}")
        for i,p in sorted(expected):
            device=[x[i,p]['device_ms'] for x in ranks]
            host=[x[i,p]['host_ms'] for x in ranks]
            out.append(dict(step=s['step'],kind='decode' if pure else 'mixed' if s['decode'] else 'prefill',
                            tokens=s['scheduled_tokens'],layer=i,phase=p,
                            device_median_ms=statistics.median(device),device_min_ms=min(device),
                            device_max_ms=max(device),host_median_ms=statistics.median(host)))
    return out


def aggregate(ledger, rows, layers):
    kinds = {}
    for kind in ('prefill', 'mixed', 'decode'):
        selected = [row for row in rows if row['kind'] == kind]
        indices = {row['step'] for row in selected}
        steps = [step for step in ledger['steps'] if step['step'] in indices]
        if not steps:
            kinds[kind] = dict(steps=0, emitted=0, median_step_ms=None, total_step_s=0, phases_mean_ms={})
            continue
        phases = defaultdict(float)
        for row in selected:
            phases[row['phase']] += row['device_median_ms']
        kinds[kind] = dict(steps=len(steps), emitted=sum(sum(s['out'].values()) for s in steps),
                           median_step_ms=statistics.median(s['ms']['total'] for s in steps),
                           median_core_ms={name:statistics.median(s['ms'][name] for s in steps)
                                           for name in steps[0]['ms']},
                           total_step_s=sum(s['ms']['total'] for s in steps)/1000,
                           phases_mean_ms={name:total/len(steps) for name,total in phases.items()})
    require(sum(k['steps'] for k in kinds.values()) == len(ledger['steps']), 'incomplete step ledger')
    return dict(round=ledger['round_sha256'],sampled_layers=layers,step_kinds=kinds,
                missed_decodes=ledger['missed_already_generating'],totals_ms=ledger['totals_ms'],
                intervals=len(rows))


def capture_tail(ledger,state):
    """Measure first-build host time overlapping each batch's server response tail."""
    requests=defaultdict(list)
    for r in ledger['requests']:
        requests[r['batch']].append(r['sent_ns'])
    spans={batch:(min(ts),max(ts)) for batch,ts in requests.items()}
    result=defaultdict(list)
    rank_count=0
    for path in sorted(state.glob('stages-*.jsonl')):
        events=[json.loads(line) for line in path.read_bytes().splitlines()]
        frames=[e for e in events if e['kind']=='worker_forward']
        if not frames: continue
        rank_count+=1
        builds=defaultdict(list)
        for e in events:
            if e['kind']=='decode_graph_first_build':
                require(e['start']<=e['end']<=e['t'],'invalid build interval')
                builds[e['step']].append(e)
        for step in ledger['steps']:
            frame,=[f for f in frames if step['start_ns']<=f['start']<=f['t']<=step['end_ns']]
            hits=sorted(builds[frame['start']],key=lambda e:e['start'])
            require(all(frame['start']<=e['start']<=e['end']<=frame['t'] for e in hits)
                    and all(a['end']<=b['start'] for a,b in zip(hits,hits[1:])),'unbound build time')
            lo,hi=spans[step['batch']]
            result[step['step']].append(dict(shapes=[(e['layer'],e['rows']) for e in hits],
                build_ms=sum(e['end']-e['start'] for e in hits)/1e6,
                tail_build_ms=sum(max(0,min(hi,e['end'])-max(lo,e['start'])) for e in hits)/1e6))
    require(rank_count==4 and len(result)==len(ledger['steps']),'incomplete workers')
    batches={}
    for batch,(lo,hi) in sorted(spans.items()):
        matched=[]
        overlap_ms=0.; kinds=defaultdict(float); tail_steps=[]
        for step in ledger['steps']:
            if step['batch']!=batch: continue
            ms=max(0,min(hi,step['end_ns'])-max(lo,step['start_ns']))/1e6
            if ms:
                kind='decode' if len(step['decode'])==len(step['alloc']) else 'mixed' if step['decode'] else 'prefill'
                kinds[kind]+=ms
                overlap_ms+=ms
                tail_steps.append(dict(step=step['step'],kind=kind,width=len(step['alloc']),
                                       outputs=sum(step['out'].values()),overlap_ms=ms))
            ranks=result[step['step']]
            require(len(ranks)==4 and all(r['shapes']==ranks[0]['shapes'] for r in ranks),'inconsistent first builds')
            if ranks[0]['shapes']:
                matched.append(dict(step=step['step'],layers=len(ranks[0]['shapes']),
                    shapes=sorted({n for _,n in ranks[0]['shapes']}),
                    build_ms=statistics.median(r['build_ms'] for r in ranks),
                    tail_build_ms=statistics.median(r['tail_build_ms'] for r in ranks)))
        elapsed=(hi-lo)/1e6
        require(overlap_ms<=elapsed+1e-6,'overlapping serving steps')
        batches[batch]=dict(server_first_to_last_ms=elapsed,steps_with_builds=matched,
                            engine_step_overlap_ms=dict(kinds),outside_steps_ms=elapsed-overlap_ms,
                            tail_steps=tail_steps)
    return dict(round=ledger['round_sha256'],batches=batches,ranks=rank_count,
                scope='worker host wall intervals intersected with first-to-last server send; median across TP ranks')


def tail_by_width(ledger, tail, rows, layers):
    require(ledger['round_sha256'] == tail['round'], 'tail belongs to a different run')
    intervals = defaultdict(dict)
    for row in rows:
        if row['kind'] == 'decode':
            key = row['layer'], row['phase']
            step = intervals[row['step']]
            require(key not in step, 'duplicate decode interval')
            step[key] = row['device_median_ms']
    expected = {(i, p) for i in layers for p in DECODE}
    ledger_steps = {s['step']:s for s in ledger['steps']}
    groups = defaultdict(list)
    seen = set()
    for batch in tail['batches'].values():
        for span in batch['tail_steps']:
            if span['kind'] != 'decode':
                continue
            i = span['step']
            require(i not in seen and set(intervals[i]) == expected, 'missing or repeated decode step')
            seen.add(i)
            step = ledger_steps[i]
            require(len(step['alloc']) == len(step['decode']) == span['width'], 'decode width differs')
            groups[span['width']].append((span, step['ms']['total'], intervals[i]))
    return dict(round=tail['round'], layers=layers, ranks=tail['ranks'], widths={
        width: dict(steps=len(items), median_step_ms=statistics.median(ms for _,ms,_ in items),
                    tail_overlap_ms=sum(span['overlap_ms'] for span,_,_ in items),
                    mean_sampled_layer_sum_ms={
                        phase:statistics.mean(sum(device[i, phase] for i in layers) for _,_,device in items)
                        for phase in DECODE}) for width,items in sorted(groups.items())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','state','abi'):
        parser.add_argument(name,type=Path)
    parser.add_argument('lane')
    parser.add_argument('--summary',type=Path)
    parser.add_argument('--capture-tail',type=Path)
    parser.add_argument('--tail-widths',type=Path)
    args=parser.parse_args()
    rows=summarize(args.config,args.state,args.abi,args.lane)
    if args.summary or args.capture_tail or args.tail_widths:
        ledger=report(args.config,args.state,args.lane)
    if args.summary:
        save(args.summary,aggregate(ledger,rows,doc(pin(args.abi))['timed_layers']))
    if args.capture_tail or args.tail_widths:
        require('decode_graph_first_build' in
                (args.abi.parent/'bundle/parents/jev-system-rl/implementation/decode_layer_graph.py').read_text(),
                'cannot count builds without instrumentation')
        tail = capture_tail(ledger,args.state)
        if args.capture_tail:
            save(args.capture_tail,tail)
        if args.tail_widths:
            save(args.tail_widths,tail_by_width(ledger,tail,rows,doc(pin(args.abi))['timed_layers']))
    w=csv.DictWriter(sys.stdout,fieldnames=rows[0],lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
