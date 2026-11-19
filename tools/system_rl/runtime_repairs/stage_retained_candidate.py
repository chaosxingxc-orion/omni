"""Stage an immutable derivative of two closed bundles, then use the existing launchers."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def stage(config):
    p=json.loads(config.read_bytes())
    root=config.parent
    host=p['host']
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9.-]*',host):raise ValueError('invalid deployment alias')
    def read(name):return json.loads((root/name).read_bytes())
    def remote(cmd):return subprocess.run(['ssh',host,cmd],check=True)
    q=shlex.quote
    def derive(x):
        if p['from_tag'] not in x:raise ValueError('template run identity missing')
        return x.replace(p['from_tag'],p['tag'])
    def files(files_map,names):
        for name in names:
            if Path(name).name!=name:raise ValueError('local module required')
            data=(Path(__file__).parent/name).read_bytes()
            ast.parse(data); files_map[name]=hashlib.sha256(data).hexdigest()
    assets={}
    for name, ref in p.get('local_assets',{}).items():
        if Path(name).name!=name or not name.endswith('.json'):raise ValueError('expected named JSON input')
        asset=Path(ref['path'])
        if asset.name!=name or hashlib.sha256(asset.read_bytes()).hexdigest()!=ref['sha256']:
            raise ValueError('staged asset identity differs')
        assets[name]=asset
    def bundle(src,dest,names,changes,filename,spec,tree=False):
        path=root/filename
        if path.exists():raise ValueError('existing derived spec')
        with path.open('x') as f:json.dump(spec,f,indent=2)
        base='mkdir -m 700 '+q(dest)
        links=sorted(set(names)-set(changes)-assets.keys())
        if links:base+=' && cp -l -- '+' '.join(q(src+'/'+n) for n in links)+' '+q(dest+'/')
        if tree:base+=' && cp -al -- '+q(src+'/tree')+' '+q(dest+'/tree')
        remote(base);subprocess.run(['scp','-q',*(str(Path(__file__).parent/n) for n in changes),*(str(assets[n]) for n in assets if n in names),str(path),host+':'+dest+'/'],check=True)
    t=read(p['tp_template']);src=t['inputs']
    for k in ('inputs','name','output'):t[k]=derive(t[k])
    for k in p.get('remove',[]):t.pop(k)
    t.update(p.get('tp_options',{}))
    for n in assets:t['input_files'][n]=p['local_assets'][n]['sha256']
    files(t['input_files'],p['tp_changes'])
    tn='tp4-'+p['tag']+'.local.json'
    bundle(src,t['inputs'],t['input_files'],p['tp_changes'],tn,t)
    if p.get('hf_template'):
        h=read(p['hf_template']);old=h['hf_bundle'];dst=derive(old);h.update(tp_output=t['output'],hf_bundle=dst)
        s=h['hf_spec'];s['hot']=derive(s['hot']);s['name']=derive(s['name'])
        s['fresh_group'].update(p.get('hf_options',{}))
        for n in assets:s['files'][n]=p['local_assets'][n]['sha256']
        changes=p['hf_changes'];files(s['source_pins'],[n for n in changes if n in s['source_pins']])
        files(s['files'],[n for n in changes if n not in s['source_pins']])
        hn='handoff-'+p['tag']+'.local.json'
        names=(set(s['files'])|{'follow_retained_group.py','launch_retained_hf_fsdp.py','dispatch_retained_hf_boundary.py','retained_hf_fsdp_control.py'})-{s['fresh_group']['result']}
        bundle(old,dst,names,changes,hn,h,True)
    remote('cd '+q(t['inputs'])+' && (nohup python3 -B dispatch_retained_vllm_boundary.py --idle --config '+q(tn)+' >dispatch.log 2>&1 </dev/null & )')
    if p.get('hf_template'):
        # Wait for this exact run to acquire the pre-existing guard and issue its ticket.
        ticket=q(t['output']+'/CONTAINER.json')
        remote('cd '+q(dst)+' && (nohup sh -c '+q('for i in $(seq 1 90); do test -s '+ticket+' && exec python3 -B follow_retained_group.py --config '+q(hn)+'; sleep 1; done; exit 1')+' >follow.log 2>&1 </dev/null & )')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True,type=Path);stage(a.parse_args().config)
