"""Real Clock 8 objective and LoRA bytes across the resident wire (offline)."""
import argparse,io,json,shutil,sys,tempfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for k in ('release','run','scratch'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    a=p.parse_args();step=a.step;base=Path(__file__).parent
    sys.path[:0]=[str(a.release/'tree/parents/jev-system-rl/implementation'),str(base)]
    from dependencies import load,local_path,register_locations
    from run_package import pin,read,write
    from checkpoint_inventory import unpack
    proxy=load(base/'source/pull_closed.py','resident_wire')
    consumer=load(base/'source/formal_actor_transport.py','resident_consumer')
    submit=load(base/'source/formal_command.py','resident_command').submit
    old=a.run/f'step-{step:06d}/update/returned'
    before=a.run/f'step-{step-1:06d}/update/returned'
    for d in (before,old):register_locations(read(d/'LOCATION-MAP.json'))
    orig=read(old/'RETURN.json');actual=read(orig['result_ref']);last=read(before/'RETURN.json')['checkpoint_ref']
    gen=read(next(old.parent.glob('TRAIN-REQUEST*.json')))['generation']
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    with tempfile.TemporaryDirectory(dir=a.scratch) as tmp:
        home=Path(tmp);owner=home/'owner';hot=owner/'hot';ex=hot/'resident';root=ex/'results/000000';run='fixture'
        policy=dict(run_id=run,max_commands=8);cfg=write(owner/'TRAIN.local.json',dict(continuous_actor=policy,boot_id=boot))
        write(hot/'INTENT.json',dict(config_sha256=cfg['sha256'],boot_id=boot))
        ready=write(ex/'READY.json',dict(run_id=run,source_boot=boot))
        members=[]
        def copy(ref,dest):
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(local_path(ref['path']),dest)
            r=pin(dest);assert r['sha256']==ref['sha256'];members.append(r);return r
        checkpoint=copy(orig['checkpoint_ref'],root/f'step-{step}'/'COMMITTED.json')
        # Probe the original immutable-shard custody boundary: the CPU only
        # requires a manifest and inference payload, never an Adam state copy.
        assets=read(orig['policy_ref'])['assets']
        new_assets=[]
        for asset in assets:
            new_assets.append(copy(asset,Path(checkpoint['path']).parent/'lora_adapter'/Path(asset['path']).name))
        old_policy=read(orig['policy_ref'])
        pub=write(root/'publication/POLICY-CANDIDATE.json',dict(read(old_policy['publication_ref']),
              assets=new_assets,checkpoint_ref=checkpoint))
        manifest=write(root/'publication/ASSETS.json',dict(status='CLOSED_IMMUTABLE',assets=new_assets,publication_ref=pub))
        policy_ref=write(root/'publication/POLICY.json',dict(old_policy,assets=new_assets,publication_ref=pub,asset_manifest_ref=manifest))
        members.extend([pub,manifest,policy_ref])
        command=dict(run_id=run,sequence=0,checkpoint_sha256=last['sha256'],pending_capture_ref=gen['pending_capture_ref'],
                     generation_inventory_ref=gen['generation_inventory_ref'],previous_close_sha256=ready['sha256'],action='update')
        intent=write(root/'COMMAND-INTENT.json',dict(command=command,predecessor=last));members.append(intent)
        selected=copy(actual['selected_cohort_ref'],root/'finalized-cohort/SELECTED.json')
        trajectory=copy(orig['trajectory_ref'],root/'finalized-cohort/COMPLETE.json')
        refs=[];live=[]
        for rank,ref in enumerate(actual['rank_refs']):
            path=root/f'rank-{rank}';r=copy(ref,path/'RESULT.json');refs.append(r)
            live.append(write(path/'LIVE.json',dict(rank=rank,intent_ref=intent,result_ref=r)))
        members+=live
        result=write(root/'RESULT.json',dict(actual,checkpoint_ref=checkpoint,policy_ref=policy_ref,resident_intent_ref=intent,
                     selected_cohort_ref=selected,trajectory_ref=trajectory,rank_refs=refs));members.append(result)
        out=write(root/'RETURN.json',dict(orig,checkpoint_ref=checkpoint,policy_ref=policy_ref,result_ref=result,trajectory_ref=trajectory))
        closed=write(root/'CLOSED.json',dict(schema='formal-resident-command-v1',source_boot=boot,immutable=True,
                   run_id=run,sequence=0,result_ref=result,return_ref=out,rank_live_refs=live,projected_members=members))
        wire=io.BytesIO();proxy.emit(root,wire,128*1024**2,expected_boot=boot)
        state=dict(step=step-1,resume_ref=last,sampled_pending_window_ref=actual['sampled_window_ref'])
        loc=home/'returned';unpack(wire.getvalue(),loc,home/'cas',expected_boot=boot)
        # Test receipt references resolve on a separate controller, with no
        # access to the producing command files.
        root.rename(ex/'withheld')
        args=(gen,dict(source_boot=boot),policy,home/'client',loc,owner,0)
        got=consumer.accept_resident_result(state,*args);assert got['checkpoint_ref']==checkpoint
        files=read(loc/'WIRE-MANIFEST.json')['files']
        assert all('jev-master-rank-' not in k for k in files)
        assert len([k for k in files if k.endswith('/LIVE.json')])==8
        assert read(got['result_ref'])['result']==actual['result']
        try:consumer.accept_resident_result(dict(state,sampled_pending_window_ref={}),*args)
        except ValueError:pass
        else:raise AssertionError('changed sampled window accepted')
        (ex/'withheld').rename(root)
        cmd=dict(command,sequence=1,previous_close_sha256=closed['sha256'],checkpoint_sha256=checkpoint['sha256'])
        first=submit(owner,run,boot,cmd);assert first==submit(owner,run,boot,cmd)
        try:submit(owner,run,boot,dict(cmd,checkpoint_sha256='wrong'))
        except ValueError:pass
        else:raise AssertionError('conflicting command accepted')
        print(json.dumps(dict(step=step,ranks=len(live),objective_denominator=actual['result']['objective_observation']['global_episode_denominator'],
            transfer_bytes=len(wire.getvalue()),optimizer_shards_transferred=0,unchanged_retry=True,
            rejected=['wrong sampled window','different command at same sequence'],scope='offline control and custody; no model run')))
if __name__=='__main__':main()
