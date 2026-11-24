"""Actual eight-rank resident command loop; small CPU DDP model, no NPU claims."""
import argparse,json,sys,tempfile,time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def participant(rank,paths,steps,home,release):
    from types import SimpleNamespace as S
    import datetime
    root,run=map(Path,paths);home=Path(home)
    sys.path[:0]=[str(home),str(Path(release)/'tree/parents/jev-system-rl/implementation')]
    from dependencies import load,register_locations
    from run_package import read,write,pin
    import bounded_actor_step as driver
    mod=load(home/'source/resident.py','resident_fixture')
    receipt=load(home/'source/formal_actor_receipt.py','formal_actor_receipt')
    # Historical checkpoints bind real sampled rosters. They are *not* results
    # from this test network. Store new DDP losses only under the temp test root.
    pairs=[]
    for step in steps:
        directory=run/f'step-{step:06d}'/'update'
        for n in (step-1,step):
            prev=run/f'step-{n:06d}'/'update/returned'
            register_locations(read(prev/'LOCATION-MAP.json'))
        prev=read(run/f'step-{step-1:06d}'/'update/returned/RETURN.json')['checkpoint_ref']
        out=read(directory/'returned/RETURN.json')['checkpoint_ref']
        pairs.append((prev,out,read(next(directory.glob('TRAIN-REQUEST*.json')))['generation']))
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(root/'rendezvous'),rank=rank,world_size=8,
                            timeout=datetime.timedelta(seconds=90))
    try:
        torch.manual_seed(301)
        model=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(3,1))
        opt=torch.optim.AdamW(model.parameters(),lr=.01)
        sched=torch.optim.lr_scheduler.StepLR(opt,1,gamma=.99)
        worker=S(actor=S(actor_module=model,actor_optimizer=opt),actor_lr_scheduler=sched)
        c=read(pairs[0][0])['contract'];src=c['sources'];calls=[];restores=[]
        cfg=dict(protocol='formal-resident-v1',run_id='cpu-test',deadline_epoch=read(root/'DEADLINE.json'),
                 max_commands=len(steps)+1,poll_seconds=.05,max_idle_seconds=90)
        initial=dict(name='cpu',lineage=c['lineage'],formal_sampled=True,deterministic_backward=True,
            training=c['training'],devices=c['devices'],image=c['image'],source_pins=src,
            files=dict(src,finalized_generation='fixture',**{'lora_row_precision.py':c['lora_arithmetic']['source_sha256']}),
            lora_row_geometry=c['lora_arithmetic']['geometry'],accepted_fields='finalized_generation',dataset_sha256='fixture',
            generation=dict(pairs[0][2],runtime=dict(reference_batch={'rows_per_rank':4})),
            continuation=dict(prior_step=steps[0]-1),restore_checkpoint=pairs[0][0],training_memory_gib=1280,
            storage_budget_bytes=0,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),continuous_actor=cfg)
        def restore(w,s):
            restores.append(1)
            assert w is worker and len(restores)==1
            return pairs[0][0],read(pairs[0][0])
        driver.restore_continuation=restore
        def execute(w,adapted,s,root_out,rank,world,sealed,*,live):
            i=len(calls);old,new,_=pairs[i];assert live==(old,read(old))
            assert w is worker and adapted is model and world==8
            opt.zero_grad();loss=(model(torch.tensor([[1.,float(rank),1.]])).square()).mean()
            loss.backward();opt.step();sched.step()
            step=float(next(iter(opt.state.values()))['step'])
            assert step==i+1 and sched.last_epoch==i+1
            calls.append(float(loss.detach()))
            write(root_out/f'rank-{rank}/RESULT.json',dict(rank=rank,fixture_loss=calls[-1],adam_step=step))
            return new
        # The transport publisher runs on the same (rank-zero) host in this
        # test. No deployment, numerical GRPO, or NPU receipt is synthesized.
        def close(current,path,prior,produced,refs,owners):
            i=len(calls)-1
            assert len(refs)==8 and all(read(ref)['objects']==owners[j] for j,ref in enumerate(refs))
            closed=write(path/'CLOSED.json',dict(test_only=True,sequence=i))
            base=dict(run_id='cpu-test',sequence=i+1,previous_close_sha256=closed['sha256'])
            if i+1<len(steps):
                gen=pairs[i+1][2]
                cmd=dict(base,action='update',checkpoint_sha256=produced['sha256'],**{k:gen[k] for k in
                      ('generation_inventory_ref','pending_capture_ref')})
            else:cmd=dict(base,action='end')
            write(root/'resident/commands'/f'{i+1:06d}.json',cmd)
        receipt.close_live=close
        mod.serve_formal(worker,model,initial,root,rank,8,None,execute)
        assert len(restores)==1 and len(calls)==len(steps)
        write(root/f'CPU-{rank}.json',dict(restores=len(restores),updates=len(calls),losses=calls,
            weight=model.module.weight.detach().tolist(),scheduler=sched.last_epoch))
    finally:dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser()
    for k in ('run','scratch','release'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--steps',type=int,nargs='+',required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory(dir=a.scratch) as tmp:
        root=Path(tmp);(root/'DEADLINE.json').write_text(json.dumps(time.time()+180))
        mp.start_processes(participant,args=((str(root),str(a.run)),a.steps,str(Path(__file__).parent),str(a.release)),
                           nprocs=8,join=True,start_method='fork')
        ranks=[json.loads((root/f'CPU-{r}.json').read_bytes()) for r in range(8)]
        assert all(x['weight']==ranks[0]['weight'] and x['scheduler']==len(a.steps) for x in ranks)
        print(json.dumps(dict(scope='CPU DDP, stub recovery and publisher; actual resident command loop',
            ranks=8,restores_per_process=1,adam_steps=len(a.steps),same_live_objects=True,agreement=True)))
if __name__=='__main__':main()
