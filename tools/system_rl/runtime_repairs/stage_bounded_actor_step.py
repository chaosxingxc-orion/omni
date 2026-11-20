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
    files={n:d/n for n in ('bounded_actor_step.py','retained_hf_fsdp_control.py','launch_retained_hf_fsdp.py')}
    files['events.py']=d/'source/events.py'
    files['policy_identity.py']=d/'source/policy_identity.py'
    files['accepted.pt']=Path(c['accepted_fields'])
    s.update(accepted_fields='accepted.pt',dataset_sha256=c['dataset_sha256'],name=tag,hot=str(dest/'hot'),devices=c['devices'])
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
        p=Path(tmp)/'TRAIN.local.json';p.write_text(json.dumps(s,indent=2))
        for n,f in [*files.items(),(p.name,p)]:
            subprocess.run(['scp','-q',str(f),host+':'+str(dest/n)],check=True)
    remote('cd '+q(str(dest))+' && (nohup python3 -B launch_retained_hf_fsdp.py --config TRAIN.local.json >launch.log 2>&1 </dev/null & )')
    print(dest)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);stage(json.loads(p.parse_args().config.read_bytes()))
