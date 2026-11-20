"""Collect closed HTTP windows and build one restored-actor scoring packet."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile

def read(p):return json.loads(p.read_bytes())

def collect(host,args,root):
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        staged=Path(tmp)/'tree';staged.mkdir()
        with subprocess.Popen(['ssh',host,shlex.join(args)],stdout=subprocess.PIPE) as proc:
            with tarfile.open(fileobj=proc.stdout,mode='r|') as ar:ar.extractall(staged,filter='data')
            if proc.wait():raise RuntimeError('closed evidence collection failed')
        staged.rename(root)

def main(config):
    from updated_http_parity import prepare
    c=read(config);root=Path(c['output']);root.mkdir(parents=True,exist_ok=True)
    host=c['host'];bundle=Path(c['bundle'])
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',host):raise ValueError('invalid host alias')
    plan=read(bundle/c['plan']);rows=[]
    for file in plan['stages']:
        j=read(bundle/file);phase=Path(j['output']);lane,=j['lanes'];name=lane['name']
        subprocess.run(['ssh',host,shlex.join(['test','-s',str(phase/'DRAINED.json')])],check=True)
    for name in plan['lanes']:
        state=root/('state-'+name)
        if not state.exists():
            serv=read(bundle/f'bundle/SERVER-{name}.json')['service_identity']['container_name']
            collect(host,['docker','exec',serv,'tar','cf','-','-C','/runtime/state/policy',
                        'ACTIVATION.json','CACHE-POLICY.json','PRECISION.json','full-logits'],state)
    for file in plan['stages']:
        j=read(bundle/file);phase=Path(j['output']);lane,=j['lanes'];name=lane['name'];out=root/Path(file).stem
        out.mkdir(exist_ok=True)
        if not (out/'activation-probe01').exists():
            collect(host,['tar','cf','-','-C',str(phase),'.'],out/'activation-probe01')
        if not (out/name).exists():(out/name).symlink_to('../state-'+name)
        row,=prepare(out,bundle/Path(j['wire']['path']).name,Path(c['manifest']),lanes=[name])['rows']
        rows.append(row)
    packet=dict(rows=rows,limits=read(Path(c['criterion_packet']))['limits'],training_use=False)
    path=root/'PAIRED.json'
    if path.exists():
        if read(path)!=packet:raise ValueError('closed replay changed')
    else:
        with path.open('x') as f:json.dump(packet,f,allow_nan=False)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime_repairs'))
    print(main(a.config))
