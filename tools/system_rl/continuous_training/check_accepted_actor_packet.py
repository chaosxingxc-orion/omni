"""CPU replay in the original read-only training mounts. Write only to a fresh scratch."""
import argparse,json,os,sys,time
from pathlib import Path

def equal(a,b,path='packet'):
    import torch,numpy as np
    if type(a) is not type(b):raise ValueError(f'{path}: types {type(a)} {type(b)}')
    if isinstance(a,torch.Tensor):
        ok=a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError(f'{path}: keys')
        for key in a:equal(a[key],b[key],f'{path}.{key}')
        return
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError(f'{path}: length')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,f'{path}[{i}]')
        return
    elif isinstance(a,np.ndarray):ok=np.array_equal(a,b)
    else:ok=a==b
    if not ok:raise ValueError(f'{path}: values differ')

def main(args):
    import torch
    sys.path.insert(0,str(args.spec.parent))
    from bounded_actor_step import generation_sources
    from dependencies import load
    candidate=load(str(Path(__file__).with_name('pack_actor_fields.py')),'test_candidate_packet')
    old=load(str(args.spec.parent/'pack_actor_fields.py'),'original_packet')
    spec=json.loads(args.spec.read_bytes())
    inventory=spec['generation']['generation_inventory_ref']
    stats={}
    def timed(key,fn):
        before=time.monotonic();result=fn();stats[key]=round(time.monotonic()-before,3);return result
    with generation_sources(spec) as runtime:
        from fresh_adapter import scope,consumer as configure
        for name in ('consumer_source','consumer_wrapper'):
            if name in runtime:os.environ['OMNI_CAPTURE_'+name.upper()]=runtime[name]
        with scope():
            from cohort_io import read,pin
            import consumer
            root=args.root
            original=timed('old_rebuild_seconds',lambda:old.build_packet(root))
            receipt=read(pin(root/'COMPLETE.json'))
            fields,report=timed('original_accept_seconds',lambda:configure(consumer).build(read(receipt['files_ref']),read(receipt['consumer_config_ref'])))
            timed('seal_seconds',lambda:candidate.seal_packet(root,fields,report,inventory,packet_directory=args.out))
            received=timed('load_seconds',lambda:candidate.load_packet(root,inventory,packet_directory=args.out))
            timed('compare_seconds',lambda:equal(original,received))
            try:candidate.load_packet(root,{},packet_directory=args.out)
            except ValueError:pass
            else:raise ValueError('accepted wrong inventory')
            import multiprocessing
            ctx=multiprocessing.get_context('fork');barrier=ctx.Barrier(args.ranks,timeout=180)
            def worker(rank):
                torch.set_num_threads(1)
                barrier.wait()
                before=time.monotonic()
                p=candidate.load_packet(root,inventory,packet_directory=args.out)
                load_s=time.monotonic()-before
                equal(original,p)
                (args.out/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,load_seconds=load_s,all_fields_equal=True)))
            jobs=[ctx.Process(target=worker,args=(i,)) for i in range(args.ranks)]
            for j in jobs:j.start()
            for j in jobs:
                j.join(300)
                if j.exitcode!=0:raise ValueError(f'reader failure {j.exitcode}')
            stats.update(dataset=receipt['version'],groups=receipt['selected_groups'],episodes=receipt['complete_episodes'],
                         tensors=sorted(received['tensors']),fields_bytes=(args.out/'actor-packet/fields.pt').stat().st_size,
                         readers=[json.loads((args.out/f'rank-{i}.json').read_text()) for i in range(args.ranks)])
    (args.out/'REPLAY.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('spec','root','out'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--ranks',type=int,required=True)
    main(p.parse_args())
