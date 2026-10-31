"""Read existing scientific COST records and bounded synthetic timing evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def stats(values):
    if not values:
        return {'n': 0, 'mean_s': None, 'p50_s': None, 'p95_s': None}
    ordered = sorted(values)
    return {'n': len(values), 'mean_s': statistics.mean(values),
            'p50_s': statistics.median(values),
            'p95_s': ordered[min(len(ordered)-1, int(.95*(len(ordered)-1)))]}


def analyze(data, recipe, matrix, limit=100):
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    pin = json.loads(recipe.read_bytes())['checkpoint']
    path = (data / pin['relpath']).resolve()
    if not path.is_relative_to(data.resolve()): raise ValueError('checkpoint escapes root')
    checkpoint = read_checkpoint(path, expected_sha256=pin['sha256'])
    timings = defaultdict(list)
    for row in checkpoint.rows:
        if row.keep_defect() is not None: raise ValueError('incomplete source row')
        output, = row.output_rows
        ep = output['native_episode_receipt']
        contacts = {c['agent_contact_id']: c for v in ep['role_visits'] for c in v['execution_receipt']['contacts']}
        trace = {json.loads(s)['record_id']: json.loads(s)['payload'] for s in row.trace_lines}
        total = 0
        for branch in ep['perception_branches']['branches']:
            arm = branch['branch_id']; sampling = branch.get('sampling') or {}; arm_total = 0
            for role, group, field in [('perception','perception_samples','contact_id'), ('solver','solver_samples','solver_contact_id')]:
                samples = (sampling.get(group) or {}).get('samples')
                ids = [v['contact_id'] for v in samples] if samples else [branch[field]]
                seconds = sum(trace[contacts[c]['request_id']+'/cost']['latency_seconds'] for c in ids)
                timings[arm+'/'+role].append(seconds);arm_total += seconds
            timings[arm+'/model_sum'].append(arm_total);total += arm_total
        if abs(total-row.cost['latency_seconds']) > .0001: raise ValueError('row timing reconciliation failed')
        timings['all_branches/model_sum'].append(total)
        sample_events=[v['use_sample_execution_timing'] for v in trace.values() if 'use_sample_execution_timing' in v]
        if len(sample_events)>1: raise ValueError('ambiguous sample timing')
        if sample_events:
            t=sample_events[0]
            timings['all_branches/admission_queue'].append((t['admitted_ns']-t['batch_started_ns'])/1e9)
            timings['all_branches/episode_wall'].append((t['finished_ns']-t['admitted_ns'])/1e9)
    synthetic=[]
    for root in sorted((matrix/'round-0').iterdir()):
        if not (root/'binding.json').exists(): continue
        groups=defaultdict(dict)
        for p in (root/'requests').glob('*.result.json'):
            record=json.loads(p.read_bytes()); r=record['result']
            if r['status']=='ok': groups[r['trajectory']][r['contact']]=(record,p)
        complete=[(max(v[0]['finished_at'] for v in g.values()),k,g) for k,g in groups.items() if set(g)=={0,1,2}]
        selected=sorted(complete)[:limit]
        if len(selected)!=limit: raise ValueError('fewer than requested complete trajectories')
        measures=defaultdict(list); proofs=[]
        for _,key,g in selected:
            rs=[g[i][0]['result'] for i in range(3)]
            elapsed=max(r['started_s']+r['latency_s'] for r in rs)-rs[0]['started_s']
            model=sum(r['latency_s'] for r in rs)
            measures['trajectory_latency'].append(elapsed)
            measures['model_requests_sum'].append(model)
            measures['between_requests_cpu_io_and_scheduling'].append(max(0,elapsed-model))
            for i,r in enumerate(rs):
                measures[f'contact_{i}/request'].append(r['latency_s'])
                if r['ttft_s'] is not None:
                    measures[f'contact_{i}/time_to_first_token'].append(r['ttft_s'])
                    measures[f'contact_{i}/after_first_token'].append(r['latency_s']-r['ttft_s'])
                proofs.append({'file':g[i][1].name,'sha256':hashlib.sha256(g[i][1].read_bytes()).hexdigest()})
        duration=selected[-1][0]-json.loads(next(root.glob('segments/*/metrics.jsonl')).read_text().splitlines()[0])['time']
        binding=json.loads((root/'binding.json').read_bytes())
        synthetic.append({'candidate':binding['candidate']['id'],'concurrency':binding['candidate']['client_concurrency'],
            'rows':limit,'available_complete_rows':len(complete),'selection':'first 100 complete by finish time; completion-order bias; descriptive only',
            'cards':2,'instances':1,'wall_to_100_s':duration,'amortized_wall_s_per_trajectory':duration/limit,
            'timings':{k:stats(v) for k,v in measures.items()},'selected_requests_sha256':hashlib.sha256(json.dumps(proofs,sort_keys=True).encode()).hexdigest()})
    return {'schema':'saea-sample-latency-breakdown-v1','scientific_rows':len(checkpoint.rows),
        'checkpoint_sha256':pin['sha256'],'scientific_request_timings':{k:stats(v) for k,v in timings.items()},
        'scientific_unmeasured':['queue wait','evidence preparation','audio loading/encoding','request TTFT vs decode','between-node CPU','delivery parsing','checkpoint fsync'],
        'synthetic':synthetic,'new_model_calls':0,
        'limitations':['Request latency is client observed and includes server queue/network/prefill/decode.',
          'TTFT cannot separate queue, audio encoder and prefill; after-first-token is not pure device decode.',
          'Scientific per-row model sums are service latency, not inverse aggregate throughput.',
          'Synthetic contact 0/1/2 are not scientific perception/solver roles.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','recipe','matrix','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=analyze(a.data,a.recipe,a.matrix)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'rows':result['scientific_rows'],'synthetic_instances':len(result['synthetic']),'new_model_calls':0}))
