"""Exercise the release/serving evidence consumer using a sealed real window."""
from pathlib import Path
import argparse,sys,tempfile,copy,json

def main():
    p=argparse.ArgumentParser()
    for k in ('release','lock','published','endpoints','locations','scratch'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    sys.path[:0]=[str(a.release/'tree/parents/jev-system-rl/implementation'),str(Path(__file__).parent/'source')]
    from dependencies import register_locations
    from run_package import pin,read,write
    from continuous_policy import verify
    register_locations(json.loads(a.locations.read_bytes()))
    lock=pin(a.lock);pub=read(a.published);w=read(a.endpoints)
    assert pub['step']==read(lock)['accepted_verdict']['step']
    with tempfile.TemporaryDirectory(dir=a.scratch) as tmp:
        root=Path(tmp);lanes={}
        for k,v in w['lanes'].items():
            lane=dict(read(v),adapter_sha256=pub['assets'][0]['sha256'])
            lanes[k]=write(root/k/'LANE.json',lane)
        rec=dict(schema='continuous-training-activation-v1',precision_lock_ref=lock,checkpoint_ref=pub['checkpoint_ref'],
                 profile_ref=w['profile_ref'],lanes=lanes)
        assert set(verify(pub,write(root/'activation.json',rec)))==set(lanes)
        def reject(candidate,proof):
            try:verify(candidate,proof)
            except ValueError:return
            raise AssertionError('wrong runtime accepted')
        wrong=copy.deepcopy(pub);wrong['scientific_contract']['lora_arithmetic']={}
        reject(wrong,pin(root/'activation.json'))
        name=next(iter(lanes));lane=read(lanes[name]);cache=dict(read(lane['cache_ref']),reset_succeeded=False)
        lane['cache_ref']=write(root/'bad-cache.json',cache)
        bad=dict(rec,lanes={**lanes,name:write(root/'bad-lane.json',lane)})
        reject(pub,write(root/'wrong.json',bad))
        print(json.dumps(dict(verified_sealed_step=pub['step'],real_tp4_workers=4,rejected=['changed_arithmetic','stale_cache'],
            evidence_scope='offline_release_consumer_not_live_hot_switch')))
if __name__=='__main__':main()
