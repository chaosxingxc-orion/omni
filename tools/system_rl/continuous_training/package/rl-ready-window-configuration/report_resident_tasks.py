"""Join per-host device and engine gauges to a single closed production rollout."""
import argparse
import json
import statistics
from pathlib import Path


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.endswith('\n')]


def devices(samples, cards):
    selected = [[r['devices'][str(i)] for i in cards] for r in samples
                if all(str(i) in r.get('devices', {}) for i in cards)]
    if not selected:
        return {'samples': 0}
    flat = [x for s in selected for x in s]
    return dict(samples=len(selected), unavailable=len(samples)-len(selected),
                avg_aicore_pct=round(statistics.mean(x['aicore'] for x in flat), 2),
                avg_aicore_by_card=[round(statistics.mean(s[i]['aicore'] for s in selected),2)
                                    for i in range(len(cards))],
                peak_aicore_by_card=[max(s[i]['aicore'] for s in selected) for i in range(len(cards))],
                hbm_range_mib=[min(x['hbm_mib'] for x in flat),max(x['hbm_mib'] for x in flat)],
                least_free_mib=min(x['capacity_mib']-x['hbm_mib'] for x in flat))


def summarize(trainer, rollout, cohort, lanes):
    if not (cohort/'GENERATION-CLOSED.json').is_file():
        raise ValueError('cohort is not closed')
    spans = {}
    for p in cohort.glob('groups/*/episode-*/LANE.json'):
        name='lane-'+json.loads(p.read_bytes())['lane']
        a,=(p.parent/'native-journal').glob('*.dispatch.json')
        b,=(p.parent/'native-journal').glob('*.response.json')
        spans.setdefault(name,[]).append((a.stat().st_mtime,b.stat().st_mtime))
    s,e = min(a for lane in spans.values() for a,_ in lane),max(b for lane in spans.values() for _,b in lane)
    out=dict(first_dispatch=s,last_response=e,wall_s=round(e-s,3),lanes={})
    for lane, intervals in spans.items():
        first,last=min(a for a,b in intervals),max(b for a,b in intervals)
        times=[r for r in rollout if first <= r['observed'] <= last]
        eligible=[r for r in rollout if 'num_requests_running' in r.get(lane,{})]
        before=max((r for r in eligible if r['observed']<first),key=lambda r:r['observed'])
        count='request_success_total'
        after=next((r for r in eligible if r['observed']>=last and
                    r[lane][count]-before[lane][count]==len(intervals)),None)
        if after is None: raise ValueError('complete engine counter delta unavailable: '+lane)
        def delta(key):
            return after[lane][key]-before[lane][key]
        def peak(key):
            return max(r[lane][key] for r in times if key in r.get(lane,{}))
        n=len(intervals)
        out['lanes'][lane]=dict(requests=n,devices=devices(times,lanes[lane]),
            running_peak=peak('num_requests_running'), waiting_peak=peak('num_requests_waiting'),
            kv_peak_pct=100*peak('kv_cache_usage_perc'),preemptions=delta('num_preemptions_total'),
            queue_s_per_request=delta('request_queue_time_seconds_sum')/n,
            prefill_s_per_request=delta('request_prefill_time_seconds_sum')/n,
            decode_s_per_request=delta('request_decode_time_seconds_sum')/n,
            prefix_hit_fraction=delta('prefix_cache_hits_total')/delta('prefix_cache_queries_total'),
            prompt_tokens=delta('prompt_tokens_total'), generated_tokens=delta('generation_tokens_total'))
    cards=sorted({int(k) for r in trainer for k in r['devices'] if k.isdigit()})
    out['trainer_during_rollout']=devices([r for r in trainer if s<=r['observed']<=e],cards)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['trainer','rollout','cohort','services']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    lanes={p.parent.name:json.loads(p.read_bytes())['cards'] for p in args.services.glob('*/config.json')}
    print(json.dumps(summarize(rows(args.trainer),rows(args.rollout),args.cohort,lanes),indent=2))
