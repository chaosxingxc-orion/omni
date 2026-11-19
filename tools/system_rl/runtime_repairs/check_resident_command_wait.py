"""Real two-process rendezvous: CPU delay exceeds collective timeout.

Gloo verifies control flow; this is not an NPU/HCCL numerical acceptance.
"""
import argparse
import datetime
import importlib.util
import json
import time
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, source, root):
    root=Path(root)
    dist.init_process_group('gloo',init_method='file://'+str(root/'rendezvous'),
                            rank=rank,world_size=2,timeout=datetime.timedelta(seconds=2))
    spec=importlib.util.spec_from_file_location('wait_candidate',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Isolate the wait protocol from unrelated capacity admission.
    module.check_running=lambda *args:None
    cfg=dict(deadline_epoch=time.time()+30,storage_budget_bytes=0)
    policy=dict(poll_seconds=.1,max_idle_seconds=20)
    begin=time.monotonic()
    try:
        command=module.receive_command(root,1,cfg,policy,root,begin,rank)
        assert command==dict(action='end',run_id='fixture')
        assert time.monotonic()-begin>3
        expired=dict(policy,max_idle_seconds=-1)
        try:module.receive_command(root,2,cfg,expired,root,begin,rank)
        except ValueError as error:assert 'resident idle timeout' in str(error)
        else:raise AssertionError('idle expiry must reach every rank')
        (root/f'rank-{rank}.json').write_text(json.dumps(dict(passed=True)))
    finally:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    (root/'commands').mkdir()
    context=mp.spawn(worker,args=(a.source,str(root)),nprocs=2,join=False)
    # Wait for imports/rendezvous, then leave ranks polling for > timeout.
    time.sleep(8)
    pending=root/'commands/000001.pending'
    pending.write_text(json.dumps(dict(action='end',run_id='fixture')))
    pending.rename(root/'commands/000001.json')
    while not context.join(timeout=1):pass
    assert all(json.loads((root/f'rank-{r}.json').read_text())['passed'] for r in range(2))
    result=dict(passed=True,ranks=2,backend='gloo',collective_timeout_seconds=2,
                delayed_command_seconds=8,expiry_propagated=True,npu_acceptance=False)
    (root/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
