"""Observe an already deployed replacement using the same provider reader as its parent."""
import argparse,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
from build import save

def observe(old_client, deployment, output, helper):
    old=json.loads(old_client.read_bytes());bundle=json.loads(deployment.read_bytes())
    remote=Path(bundle['remote_root']);oldroot=Path(bundle['predecessor']);host=old['ssh']
    def ssh(args,**kw):return subprocess.check_output(['ssh','-oBatchMode=yes',host,shlex.join(list(map(str,args)))],**kw)
    names=['READY.json','PREPARED.json','bundle/ROWS.json','bundle/PRECISION-SPEC.json','bundle/rank-precision/delta.json']
    for lane in bundle['config']['old_services']:names.extend([lane+'-READY.json','lane-'+lane+'/config.json'])
    data=ssh(['tar','-cf','-','-C',remote,'--',*names])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        if archive.getnames()!=names:raise ValueError('unexpected deployment roster')
        for f in archive:
            if not f.isfile():raise ValueError('unexpected receipt type')
            raw=archive.extractfile(f).read();dest=output/f.name
            from formal_parity import receipt_bytes
            receipt_bytes(dest,raw)
    parent=json.loads(ssh(['cat',old['remote_config']]))
    # A replacement can change the adapter as well as the container.
    # Bind the expected identity to the requested deployment and both lanes.
    version=bundle['config']['version']
    kind,sep,weight=version.partition('-sha256:')
    if (not sep or kind not in ('initial','checkpoint') or len(weight)!=64 or
            any(c not in '0123456789abcdef' for c in weight) or
            any(json.loads((output/(lane+'-READY.json')).read_bytes())['activation']['version']!=version
                for lane in parent['lanes'])):
        raise ValueError('replacement policy identity differs from deployment')
    area=remote/('provider-observation-'+weight)
    parent.update(adapter_sha256=weight,policy_version=version)
    parent.update(output=str(area/'observed'),deployment=str(remote))
    for name in parent['lanes']:
        source=output/('lane-'+name)/'config.json';raw=source.read_bytes()
        parent['lanes'][name]=dict(service=dict(path=str(remote/source.relative_to(output)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)),
                                  container=json.loads(raw)['name']+'-'+bundle['config']['attempt'])
    config=area/'OBSERVE.json'
    prog="""import json,sys,os
from pathlib import Path
p=Path(sys.argv[1]);p.parent.mkdir(exist_ok=True);raw=sys.stdin.buffer.read()
if p.exists():assert p.read_bytes()==raw
else:
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
"""
    ssh(['python3','-B','-c',prog,config],input=json.dumps(parent,sort_keys=True).encode())
    import sys
    sys.path.insert(0,str(helper))
    from observe_handed_off_lanes import client
    local=dict(old,remote_config=str(config),handoff=str(output/'READY.json'),output=str(output/'providers'))
    save(output/'CLIENT.json',local)
    if not (output/'providers/READY.json').exists():client(local)
    print(output/'providers/READY.json')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('old_client','deployment','output','helper'):p.add_argument(k,type=Path)
    observe(**vars(p.parse_args()))
