"""Exercise the actual switch reader on a closed marker in a separate CPU process."""
import argparse,hashlib,json,sys,types
from pathlib import Path


def check(bundle,marker,sources):
    impl=bundle/'parents/jev-system-rl/implementation'
    peer=bundle/'parents/policy-iteration-successor'
    acl=bundle/'parents/native-recompute-integration/state-repair/acl-runtime-successor'
    sys.path[:0]=list(map(str,(impl,peer,acl)))
    for name in ('policy_identity','publication'):
        row=sources[name]
        raw=bytes.fromhex(row['data'])
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('reader changed')
        m=types.ModuleType(name);m.__file__=str(impl/(name+'.py'));sys.modules[name]=m
        exec(compile(raw,m.__file__,'exec'),m.__dict__)
    from service_bootstrap import local_successor
    change=json.loads(marker.read_bytes())
    accepted=local_successor(marker.parent,marker,change['previous'])
    if accepted!=change['successor']:raise ValueError('different successor')
    return dict(successor=accepted,rank_sources={k:v['sha256'] for k,v in sources.items()},engine_switch=False)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--bundle',type=Path,required=True);a.add_argument('--marker',type=Path,required=True)
    p=a.parse_args();print(json.dumps(check(p.bundle,p.marker,json.load(sys.stdin))))
