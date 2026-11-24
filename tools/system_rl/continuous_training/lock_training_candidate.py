"""Identify exact training candidate sources separately from numerical qualification."""
import argparse,hashlib,json
from pathlib import Path

def lock(root,policy,precision,output):
    from run_package import pin,read,write
    spec=read(policy)
    if read(precision).get('purpose')!='precision_alignment':raise ValueError('numerical baseline required')
    roles=spec['source_roles'];assets={}
    for role,files in roles.items():
        if not files or len(files)!=len(set(files)):raise ValueError('empty or repeated source')
        assets[role]={}
        for name in files:
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or str(path)!=name:raise ValueError('invalid source member')
            p=(root/path).resolve(strict=True)
            if not p.is_relative_to(root.resolve()):raise ValueError('source escapes checkout')
            assets[role][name]=pin(p)['sha256']
    identity=dict(schema='continuous-training-source-snapshot-v1',precision_lock_sha256=pin(precision)['sha256'],
                  policy_sha256=pin(policy)['sha256'],roles=assets,version=spec['version'])
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return write(output/digest/'CANDIDATE.json',dict(identity,candidate_sha256=digest,
        purpose='training_candidate',npu_acceptance=False,performance_claim=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','policy','precision','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(lock(a.root,a.policy,a.precision,a.output)))
