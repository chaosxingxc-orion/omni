"""Extend the pinned LoRA-only worker without copying or replacing its training loop."""
import hashlib

WORKER_BASE = '7239accec8ab96981ded5306d9aea2f5ac1d10e38125b980f15581120b6e636a'
MODULES = ('retained_expert_candidate.py', 'retained_attention_candidate.py',
           'retained_router_candidate.py', 'retained_kv_candidate.py')


def worker_bytes(base):
    tail = b' return worker,adapted,names\n'
    hook = (b" if 'actor_arithmetic_profile' in config:\n"
            b"  from actor_arithmetic import install\n"
            b"  worker.close_actor_arithmetic = install(adapted,config['actor_arithmetic_profile'],world)\n")
    if base.endswith(hook + tail):
        base = base[:-len(hook + tail)] + tail
    if hashlib.sha256(base).hexdigest() != WORKER_BASE or not base.endswith(tail):
        raise ValueError('unexpected LoRA worker source')
    return base[:-len(tail)] + hook + tail


def release_manifest(base, directory, manifest):
    """Create inputs to the existing build_delta composer. Never edit a sealed tree."""
    import json
    from pathlib import Path
    from build import pin
    from source.release_completion import require_closed_release
    base = base.resolve()
    require_closed_release(base)
    parent = json.loads((base/'ASSEMBLY.json').read_bytes())['delta_manifest_ref']
    if pin(parent['path']) != parent:
        raise ValueError('parent manifest changed')
    previous = json.loads(Path(parent['path']).read_bytes())
    result = {k: previous[k] for k in ('schema','logical_root','composer_ref','storage_budget_bytes')}
    result.update(base=str(base),base_assembly_ref=pin(base/'ASSEMBLY.json'),changes=[])
    root = Path(__file__).resolve().parent
    rel = Path('parents/jev-system-rl/implementation')
    directory.mkdir(exist_ok=False)
    w = directory/'worker_dp8.py'
    w.write_bytes(worker_bytes((base/'tree'/rel/w.name).read_bytes()))
    # Pass the configured profile to the same worker. Preserve the historical
    # DP8 admission: this release is not authorization to train on four or eight.
    e = directory/'entry.py'
    old = (base/'tree'/rel/e.name).read_bytes()
    original = b"checkpoint_integrity=config.get('checkpoint_integrity')),hot)"
    if hashlib.sha256(old).hexdigest() != '548613d72ec8378a3749a8f5a3eb010e55d9f2f6595a54af1af1849d3b98e6b1' or old.count(original)!=1:
        raise ValueError('unexpected worker construction in native entry')
    e.write_bytes(old.replace(original,b"checkpoint_integrity=config.get('checkpoint_integrity'),**({k:config['training'][k] for k in ('actor_arithmetic_profile',) if k in config['training']})),hot)"))
    for src in [w,e,root/'source/actor_arithmetic.py',*(root/n for n in MODULES)]:
        target = base/'tree'/rel/src.name
        row=dict(target=str(rel/src.name),source_ref=pin(src),result_sha256=pin(src)['sha256'])
        if target.exists(): row['base_ref']=pin(target)
        result['changes'].append(row)
    with manifest.open('x') as f:
        json.dump(result,f,indent=2)

if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser()
    for name in ('base','sources','manifest'):
        p.add_argument('--'+name, type=Path, required=True)
    a=p.parse_args()
    release_manifest(a.base,a.sources,a.manifest)
