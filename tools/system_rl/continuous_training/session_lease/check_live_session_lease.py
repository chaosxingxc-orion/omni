"""Exercise the session health path against a held TP4 guard without model calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time


def run(cfg, held_ref, output):
    import sys
    sys.path.append(str(Path(__file__).parent/'source'))
    from session_lease import FIELDS,LocalLease,observe
    from formal_rollout import provider_modules
    from run_package import read,pin,write
    from multiprocessing import get_context
    r=read(held_ref)
    with provider_modules() as (m,_,_,_):probe=m.probe
    state=probe(r['remote'])
    assert all(state[k]==r['state'][k] for k in FIELDS)
    assert not output.exists()
    output.mkdir(parents=True)
    opts=read(cfg)['rollout_dispatch']['resource_session']
    owner=dict(epoch=os.urandom(16).hex(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        identity={k:state[k] for k in FIELDS},max_age=opts['max_age_seconds'])
    ref=write(output/'OWNER.json',owner)
    conf=dict(root=str(output),session=str(output),owner_ref=ref)
    mp=get_context('fork');stop=mp.Event();count=mp.Value('i',0)
    def counted(remote):
        count.value+=1
        return probe(remote)
    proc=mp.Process(target=observe,args=(opts,output,owner,ref,r['remote'],counted,stop,os.getpid()))
    proc.start()
    try:
        deadline=time.monotonic()+opts['max_age_seconds']
        while proc.is_alive() and time.monotonic()<deadline and not (output/'HEARTBEAT.json').exists():time.sleep(.02)
        fence=LocalLease(conf)
        def business(_):
            with fence.admission():return 1
        with ThreadPoolExecutor(128) as pool:total=sum(pool.map(business,range(128)))
        assert total==128 and proc.is_alive()
    finally:
        stop.set();proc.join(timeout=35)
    assert proc.exitcode==0
    stamp=read(output/'HEARTBEAT.json')
    time.sleep(max(0,stamp['started']+opts['max_age_seconds']-time.monotonic())+.02)
    try:fence.check_local()
    except ValueError:pass
    else:raise AssertionError('stale heartbeat admitted')
    print(write(output/'CHECK.json',dict(owner_ref=ref,held=pin(held_ref),admissions=total,
              remote_background_probes=count.value,expired_closed=True,model_requests=0)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('cfg','held_ref','output'):p.add_argument(k,type=Path)
    run(**vars(p.parse_args()))
