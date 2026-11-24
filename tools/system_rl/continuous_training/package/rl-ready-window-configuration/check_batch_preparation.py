"""Replay real, immutable group inputs and service receipts; no device calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
import subprocess, sys, json, threading


def check_groups(args):
    source, root, draws, rank, world = args
    sys.path.insert(0, str(source))
    from cohort_io import load, read, pin, digest
    cohort = load(Path(__file__).parent/'source/cohort.py','candidate_cohort')
    result = []
    for draw in draws[rank::world]:
        idx = draw['draw_index']
        g = read(pin(root/'groups'/str(idx)/'GROUP.json'))
        assert g['group_id'] == draw['group_id'] and g['query_draw_id'] == draw['query_draw_id']
        before = digest(g)
        group = g['online']['behavior_bundle']['groups'][0]
        calls = [c for e in group['episodes'] for c in e['calls']]
        packed = cohort.group_storage(g)
        table = packed['retained_static_v1']
        audio = [c['training_fields']['retained_static_refs']['audio_features'] for c in calls]
        def size(key):
            return len(json.dumps(table[key],separators=(',',':')).encode())
        sizes = {k:size(k) for k in set(audio)}
        row = dict(rank=rank,draw=idx,calls=len(calls),distinct_audio=len(sizes),
                   audio_before=sum(sizes[k] for k in audio),audio_after=sum(sizes.values()))
        if rank == 0 and not result:
            key, value = next(iter(table.items()))
            table[key] = None
            try:cohort.group_storage(dict(packed),expand=True)
            except ValueError:pass
            else:raise AssertionError('altered static payload accepted')
            table[key] = value
        assert digest(cohort.group_storage(packed,expand=True)) == before
        result.append(row)
    return result


def check_lanes(root, module):
    lanes = json.loads((root/'ACTIVATIONS.json').read_bytes())['lanes']
    barrier = threading.Barrier(len(lanes), timeout=10)
    def fetch(name, ref):
        barrier.wait()
        path = root/name
        expected = json.loads((path/'LANE.json').read_bytes())
        files = {Path(expected[k]['path']).name:expected[k] for k in
                 ('activation_ref','precision_ref','cache_ref','switch_ref')}
        def local_tar(cmd):
            return subprocess.check_output(cmd[3:])
        actual = module.read_lane_receipts(local_tar, 'offline', path, files)
        assert all(sha256(v).hexdigest()==files[k]['sha256'] and len(v)==files[k]['bytes']
                   for k,v in actual.items())
        return len(actual)
    observations = module.apply_lanes(lanes,len(lanes),fetch)
    def fail(name, ref):
        if name==next(iter(lanes)):raise ValueError('one lane unavailable')
        return True
    try:module.apply_lanes(lanes,len(lanes),fail)
    except ValueError as e:assert str(e)=='one lane unavailable'
    else:raise AssertionError('partial activation accepted')
    return observations


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['groups','lanes'])
    parser.add_argument('runtime',type=Path)
    parser.add_argument('root',type=Path)
    parser.add_argument('--world',type=int)
    a=parser.parse_args()
    if a.mode=='lanes':
        sys.path.insert(0,str(a.runtime))
        from dependencies import load
        continuous_policy = load(Path(__file__).parent/'source/continuous_policy.py','batch_activation_candidate')
        print(json.dumps(check_lanes(a.root,continuous_policy)))
    else:
        draws=json.loads((a.root/'SELECTED.json').read_bytes())['draws']
        with ProcessPoolExecutor(a.world) as workers:
            batches=list(workers.map(check_groups,[(a.runtime,a.root,draws,r,a.world) for r in range(a.world)]))
        rows=sum(batches,[])
        assert sorted(r['draw'] for r in rows)==list(range(len(draws)))
        print(json.dumps(dict(rank_counts=list(map(len,batches)),**{k:sum(r[k] for r in rows)
            for k in ('calls','distinct_audio','audio_before','audio_after')},all_groups_digest_equal=True)))
