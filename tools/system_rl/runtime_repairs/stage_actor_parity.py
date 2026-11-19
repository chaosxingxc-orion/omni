"""Reuse a sealed rollout to test the configured actor worker without another generation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from compose_actor_arithmetic import MODULES, worker_bytes

def stage(config):
    c = json.loads(config.read_bytes())
    host, tag = c['host'], c['run']
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9.-]*',host) or not re.fullmatch(r'[a-z0-9-]+',tag):
        raise ValueError('invalid deployment alias or run')
    source = Path(c['source_bundle'])
    dest = source.with_name(tag)
    if source == dest: raise ValueError('new run required')
    d = Path(__file__).resolve().parent
    quote = shlex.quote
    def remote(cmd):
        return subprocess.check_output(['ssh',host,cmd])
    with tempfile.TemporaryDirectory() as td:
        local = Path(td)
        s = json.loads(remote('cat '+quote(str(source/'HF-GROUP.local.json'))))
        if Path(s['hot']).parent != source or s['training']['world_size'] != 4:
            raise ValueError('unexpected parent bundle')
        worker = Path('tree/parents/jev-system-rl/implementation/worker_dp8.py')
        data = worker_bytes(remote('cat '+quote(str(source/worker))))
        (local/'worker_dp8.py').write_bytes(data)
        s['source_pins']['worker_dp8.py'] = hashlib.sha256(data).hexdigest()
        s['training']['actor_arithmetic_profile']='omni-bf16-tp4-fsdp4-v1'
        s.update(hot=str(dest/'hot'),name=tag)
        changed = {n:d/n for n in (*MODULES,'retained_group_parity.py','retained_hf_fsdp_control.py')}
        changed['actor_arithmetic.py']=d/'source/actor_arithmetic.py'
        for n,p in changed.items():
            compile(p.read_bytes(),str(p),'exec')
            ref = s['source_pins'] if n in s['source_pins'] else s['files']
            ref[n] = hashlib.sha256(p.read_bytes()).hexdigest()
        spec = local/'HF-ACTOR.local.json'
        spec.write_text(json.dumps(s,indent=2))
        links = sorted((set(s['files'])-set(changed))|{'launch_retained_hf_fsdp.py','dispatch_retained_hf_boundary.py'})
        cwd = 'cd '+quote(str(dest))+' && '
        remote('mkdir -m 700 '+quote(str(dest))+' && '+cwd+'cp -al -- '+quote(str(source/'tree'))+' . && unlink '+quote(str(worker))+
               ' && cp -l -- '+' '.join(quote(str(source/n)) for n in links)+' .')
        subprocess.run(['scp','-q',str(local/'worker_dp8.py'),host+':'+str(dest/worker)],check=True)
        subprocess.run(['scp','-q',*map(str,changed.values()),str(spec),host+':'+str(dest)+'/'],check=True)
        remote(cwd+'(nohup python3 -B launch_retained_hf_fsdp.py --config HF-ACTOR.local.json >launch.log 2>&1 </dev/null & )')
    print(dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);stage(p.parse_args().config)
