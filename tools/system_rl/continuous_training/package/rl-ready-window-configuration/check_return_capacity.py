"""Replay the original transmitted checkpoint objects against isolated empty inventories."""
import argparse, hashlib, json, os, sys, tarfile, io, time
from pathlib import Path
from unittest.mock import patch

def check(root, run, step, policy, runtime, batch_only=False):
    sys.path.insert(0,str(runtime))
    from dependencies import load
    from storage import configure_scope
    import checkpoint_inventory as control
    from run_package import read, write
    candidate=load(Path(__file__).parent/'source/checkpoint_inventory.py','candidate_inventory')
    width=read(policy)['capacity_interval_bytes']
    source=run/f'step-{step:06d}'/'update'/'returned'
    wire=read(source/'WIRE-MANIFEST.json');control.validate_manifest(wire)
    root.mkdir(parents=True,exist_ok=False)
    prior=read(run/'STORAGE-SCOPE.json')['spec']
    # Charge the actual run tree as read-only auxiliary data in BOTH isolated budgets.
    # All test output is written under scratch, never under the active lease.
    stream=root/'replay.tar'
    with tarfile.open(stream,'w') as out:
        data=json.dumps(wire).encode()
        info=tarfile.TarInfo('MANIFEST.json');info.size=len(data);out.addfile(info,io.BytesIO(data))
        members={r['sha256']:source/k for k,r in wire['files'].items()}
        for digest,desc in wire['objects'].items():
            if desc['member']:out.add(members[digest],arcname=desc['member'],recursive=False)
    results={}
    cases = [('batch',candidate)] if batch_only else [('control',control),('candidate',candidate)]
    for name,module in cases:
        sandbox=root/name;cache=sandbox/'cache';dest=sandbox/'returned'
        configure_scope(sandbox,prior['peak_bytes'],'return-replay-'+name,prior['source_boot'],
            noncheckpoint_limit_bytes=prior['noncheckpoint_limit_bytes'],auxiliary_roots=[run])
        for digest,desc in wire['objects'].items():
            if not desc['member']:
                with members[digest].open('rb') as f:module.install(cache,digest,desc['bytes'],f)
        cost=[0,0.]
        import storage
        full=storage.allocation if batch_only else module.check_running
        def timed(*args):
            start=time.perf_counter()
            try:return full(*args)
            finally:cost[0]+=1;cost[1]+=time.perf_counter()-start
        start=time.perf_counter()
        with patch.object(storage if batch_only else module,'allocation' if batch_only else 'check_running',timed):
            # Bytes input keeps the test envelope intact for both runs.
            module.unpack(stream.read_bytes(),dest,cache,expected_boot=wire['closure']['source_boot'],
                          **({'admit_batch':True} if batch_only else {'capacity_interval':width} if name=='candidate' else {}))
        results[name]=dict(seconds=time.perf_counter()-start,capacity_calls=cost[0],capacity_seconds=cost[1])
        if batch_only and cost[0] != 1:
            raise ValueError('expected a single complete batch admission')
        for doc in ('CLOSED-MANIFEST.json','WIRE-MANIFEST.json','CLOSED-COMMIT.json'):
            assert (dest/doc).read_bytes()==(source/doc).read_bytes()
        for k,ref in wire['files'].items():
            assert control.digest(dest/k)==ref['sha256']
    # The cheap per-write check must refuse a drop in space or free inodes.
    from types import SimpleNamespace as NS
    for s in (NS(f_bavail=1,f_frsize=4096,f_blocks=2**40,f_files=1000,f_favail=1000),
              NS(f_bavail=2**35,f_frsize=4096,f_blocks=2**36,f_files=1000,f_favail=1)):
        with patch.object(os,'fstatvfs',return_value=s):
            try:candidate.check_open_filesystem(-1,0,1)
            except ValueError:pass
            else:raise AssertionError('reserve drop ignored')
    return write(root/'RESULT.json',dict(results=results,step=step,wire_bytes=stream.stat().st_size,
        files=len(wire['files']),same_sealed_graph=True,space_and_inode_rejections=True))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','run','policy','runtime'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--batch-only',action='store_true')
    p.add_argument('--step',type=int,required=True)
    print(json.dumps(check(**vars(p.parse_args()))))
