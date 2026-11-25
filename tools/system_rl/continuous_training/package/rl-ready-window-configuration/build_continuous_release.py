"""Materialize reviewed training/control changes using the original release composer."""
import argparse,json
from pathlib import Path
from build import build_delta,pin,save
from verify_runtime_sources import training_snapshot

def build(root,policy,locked,precision,base,logical,composer,budget,output):
    snap=training_snapshot(pin(locked),pin(precision)['sha256'])
    cfg=json.loads(policy.read_bytes())
    if snap['policy_sha256']!=pin(policy)['sha256']:raise ValueError('different training config')
    output=output/snap['candidate_sha256'];changes=[]
    table={k:v for role in snap['roles'].values() for k,v in role.items()}
    for target,source in cfg['runtime_members'].items():
        rel=Path(target)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0]!='parents':
            raise ValueError('invalid release member')
        if Path(source).is_absolute() or '..' in Path(source).parts or source not in table:
            raise ValueError('source not named in release snapshot: '+source)
        path=(root/source).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('invalid release file: '+source)
        src=pin(path)
        if table[source]!=src['sha256']:raise ValueError('source outside release snapshot: '+source)
        original=base/'tree'/rel
        changes.append(dict(target=target,source_ref=src,result_sha256=src['sha256'],
                       **({'base_ref':pin(original)} if original.is_file() else {})))
    delta=output/'DELTA-SPEC.json'
    save(delta,dict(schema='omni-pinned-release-delta-v1',base=str(base),base_assembly_ref=pin(base/'ASSEMBLY.json'),
         logical_root=str(logical),composer_ref=pin(composer),storage_budget_bytes=budget,changes=changes))
    release=output/'release'
    if not release.exists():build_delta(delta,release)
    from source.release_completion import require_closed_release
    require_closed_release(release)
    if json.loads((release/'DELTA.json').read_bytes())['manifest_ref']!=pin(delta):raise ValueError('output namespace collision')
    for row in changes:
        if pin(release/'tree'/row['target'])['sha256']!=row['result_sha256']:raise ValueError('installed component drift')
    return save(output/'TRAINING-BUILD.json',dict(source_lock_ref=pin(locked),build_complete_ref=pin(release/'BUILD-COMPLETE.json'),
        precision_lock_ref=pin(precision),installed_members=len(changes),npu_validation=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','policy','locked','precision','base','logical','composer','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--budget',type=int,required=True)
    a=p.parse_args();print(json.dumps(build(**vars(a))))
