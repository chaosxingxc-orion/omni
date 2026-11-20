"""Stage a new attempt under the existing physical-device guard; never replace prior outcomes."""
import argparse,hashlib,json,re,shlex,subprocess,tempfile
from pathlib import Path

def stage(c):
    host=c['host']; tag=c['run']; q=shlex.quote
    if not re.fullmatch(r'[A-Za-z0-9_-]+',host) or not re.fullmatch(r'[a-z0-9-]+',tag):raise ValueError('invalid run or alias')
    parent=Path(c['source_bundle']); dest=parent.with_name(tag); d=Path(__file__).resolve().parent
    remote=lambda cmd:subprocess.check_output(['ssh',host,cmd],text=True)
    s=json.loads(remote('cat '+q(str(parent/'HF-ACTOR.local.json'))))
    if json.loads(remote('cat '+q(str(parent/'hot/EXIT.json'))))['exit_code']!=0:raise ValueError('incomplete reference run')
    from compose_actor_arithmetic import MODULES
    files={n:d/n for n in ('bounded_actor_step.py','retained_hf_fsdp_control.py','launch_retained_hf_fsdp.py')}
    files['events.py']=d/'source/events.py'
    files['policy_identity.py']=d/'source/policy_identity.py'
    files['accepted.pt']=Path(c['accepted_fields'])
    s.update(accepted_fields='accepted.pt',dataset_sha256=c['dataset_sha256'],name=tag,hot=str(dest/'hot'),devices=c['devices'])
    if c.get('recoverable'):
        if c['devices']!=list(range(8)):raise ValueError('recoverable first step requires the original world8 checkpoint topology')
        from source.actor_arithmetic import PROFILES
        s['training'].update(world_size=8,actor_arithmetic_profile=PROFILES[8],checkpoint_storage_mode='lora_training_state')
        s.update(storage_budget_bytes=c['storage_budget_bytes'],timeout_seconds=c['timeout_seconds'],
                 lineage=c['lineage'])
        if s['storage_budget_bytes']<2*1024**3:raise ValueError('checkpoint and runtime budget too small')
        files.update({n:d/n for n in MODULES})
        files['actor_arithmetic.py']=d/'source/actor_arithmetic.py'
        if 'restore_run' in c:
            run=Path(c['restore_run'])
            if run.parent!=parent.parent or run==dest:raise ValueError('invalid restore source')
            prior=json.loads(remote('cat '+q(str(run/'hot/EXIT.json'))))
            if prior.get('failure') or prior['exit_code'] or prior['complete_ranks']!=8:raise ValueError('restore source not closed')
            proof=run/'hot/step-1/COMMITTED.json'
            value=remote('sha256sum -- '+q(str(proof))).split()[0]
            s['restore_checkpoint']=dict(directory=str(proof.parent),sha256=value)
    for n,p in files.items():
        (s['source_pins'] if n in s['source_pins'] else s['files'])[n]=hashlib.sha256(p.read_bytes()).hexdigest()
    # Keep installed implementation, worker and model unchanged; overlaid files are never written through hard links.
    link=set(s['files'])-set(files)|{'tree','dispatch_retained_hf_boundary.py'}
    remote('mkdir -m 700 '+q(str(dest))+' && cp -al -- '+' '.join(q(str(parent/n)) for n in sorted(link))+' '+q(str(dest))+'/')
    with tempfile.TemporaryDirectory() as tmp:
        worker=Path('tree/parents/jev-system-rl/implementation/worker_dp8.py')
        old=remote('cat '+q(str(parent/worker)))
        expected=s['source_pins'][worker.name]
        if hashlib.sha256(old.encode()).hexdigest()!=expected:raise ValueError('worker source drift')
        start=old.index('   replaced=[]',old.index('  def export_adapter('))
        end=old.index('   if rank==0:',start)
        new=old[:start]+"   from policy_identity import resident_lora_weights\n   params=resident_lora_weights(self)\n"+old[end:]
        w=Path(tmp)/worker.name;w.write_text(new);remote('unlink '+q(str(dest/worker)))
        subprocess.run(['scp','-q',str(w),host+':'+str(dest/worker)],check=True)
        s['source_pins'][worker.name]=hashlib.sha256(w.read_bytes()).hexdigest()
        if c.get('recoverable'):
            # The reference worker was already overlaid for arithmetic; pin
            # the exact effective worker that will own the saved shards.
            pins=worker.with_name('SOURCE-PINS.json')
            raw=remote('cat '+q(str(parent/pins)))
            manifest=json.loads(raw)
            updated=[r for r in manifest['sources'] if r['path'].endswith('/jev-system-rl/implementation/worker_dp8.py')
                     and '/implementation/hot/' not in r['path']]
            if len(updated)!=1:raise ValueError('expected one current worker source pin')
            updated[0].update(sha256=s['source_pins'][worker.name],bytes=len(w.read_bytes()))
            sealed=Path(tmp)/pins.name;sealed.write_text(json.dumps(manifest,indent=2))
            remote('unlink '+q(str(dest/pins)))
            subprocess.run(['scp','-q',str(sealed),host+':'+str(dest/pins)],check=True)
            s['source_pins'][pins.name]=hashlib.sha256(sealed.read_bytes()).hexdigest()
        p=Path(tmp)/'TRAIN.local.json';p.write_text(json.dumps(s,indent=2))
        for n,f in [*files.items(),(p.name,p)]:
            subprocess.run(['scp','-q',str(f),host+':'+str(dest/n)],check=True)
    remote('cd '+q(str(dest))+' && (nohup python3 -B launch_retained_hf_fsdp.py --config TRAIN.local.json >launch.log 2>&1 </dev/null & )')
    print(dest)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);stage(json.loads(p.parse_args().config.read_bytes()))
