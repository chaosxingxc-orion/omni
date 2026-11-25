"""Compare full negotiation vs the sealed command's subset, using the same closed source and cache."""
import argparse,hashlib,json,os,shlex,subprocess,tarfile,tempfile,time
from pathlib import Path
from prepare_recording_macros import seal,read
from prepare_transition_config import pin


def check(config,update):
    from checkpoint_inventory import inventory,validate_manifest
    import transport
    spec=read(pin(config));runner=read(spec['runner_ref']);rev=read(runner['controller_revision'])
    if pin(transport.__file__)['sha256']!=rev['changes'][0]['after']['sha256']:raise ValueError('different transport')
    run=read(pin(update/'REQUEST.json'))['run'];fa=read(spec['iteration_ref'])['formal_actor']
    call=read(pin(update/'RESIDENT-CALL.json'));owner=Path(call['target'])
    seq=call['sequence'];target=owner/'hot/resident/results'/f'{seq:06d}'
    argv=['python3','-B',str(owner/'pull_closed.py'),str(target),str(fa['return_max_bytes']),
        '--formal-resident','--owner='+fa['continuous_actor']['run_id']]
    cache=Path(run['checkpoint_cache_root']);timeout=fa['return_timeout_seconds'];t=time.monotonic();full=inventory(cache);baseline=time.monotonic()-t
    t=time.monotonic();subset,count=transport.requested_inventory(update,argv,cache,run,timeout);limited=time.monotonic()-t
    if not set(subset.items())<=set(full.items()) or not count<len(full):raise ValueError('bad subset')
    expected=read(pin(update/'returned/WIRE-MANIFEST.json'))
    if set(subset)!=(full.keys() & expected['objects'].keys()) or count!=len(expected['objects']):raise ValueError('wrong closure object roster')
    # Exclude objects that were missing at the time of the original return.
    # This reproduces the original complete wire tar, including real file data.
    then=read(pin(sorted(update.glob('PULL-INVENTORY-*.json'))[-1]))['verified_objects']
    scope={k:v for k,v in subset.items() if then.get(k)==v}
    def replay(known):
        args,data=transport.closed_reader_input(argv+['--expected-boot='+run['source_boot']],known)
        with tempfile.TemporaryFile(dir=config.parent) as f:
            subprocess.run(['ssh','-oBatchMode=yes',run['host'],shlex.join(args)],input=data,stdout=f,check=True,timeout=timeout)
            size=f.tell();f.seek(0)
            digest=hashlib.file_digest(f,'sha256').hexdigest();f.seek(0)
            with tarfile.open(fileobj=f) as wire:
                if json.load(wire.extractfile('MANIFEST.json'))!=expected:raise ValueError('different historical wire manifest')
            return digest,size
    # The local tar is compacted after verified unpack. Replay both versions
    # against the same immutable source and original inventory; compare all bytes.
    original,size=replay(then)
    actual,replayed_size=replay(scope)
    if actual!=original or size!=replayed_size:raise ValueError('full and scoped replay differ')
    validate_manifest(expected)
    return seal(config.parent/'ACCEPTANCE.json',dict(config_sha256=pin(config)['sha256'],passed=True,
        closure=expected['closure']['closed_sha256'],matched_tar_sha256=actual,wire_bytes=size,
        original_objects=len(full),requested_objects=count,original_verified_bytes=sum(full.values()),
        requested_verified_bytes=sum(subset.values()),original_seconds=baseline,requested_seconds=limited))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('update',type=Path)
    print(json.dumps(check(**vars(p.parse_args()))))
