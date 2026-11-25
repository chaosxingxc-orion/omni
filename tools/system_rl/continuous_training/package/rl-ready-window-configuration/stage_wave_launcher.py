"""Create a sibling launcher from verified frozen actor files plus the explicitly changed modules."""
import os, shutil, tempfile
from pathlib import Path
from prepare_transition_config import pin
from prepare_recording_macros import seal

HELPERS=('first_wave_forward.py','microbatch_producer.py','microbatch_fields.py','microbatch_roster.py','policy_row_assignment.py')


def build(source,dest,parent,snapshot):
    frozen=Path(parent['path']).parent
    if pin(frozen/'stage_bounded_actor_step.py')!=parent or frozen.name!='rl-ready-window-configuration':
        raise ValueError('unqualified original launcher')
    package=dest/'launcher'/frozen.name
    package.parent.mkdir(exist_ok=True)
    impl=package.parent/'implementation';expected=(frozen.parent/'implementation').resolve(strict=True)
    if not impl.is_symlink():impl.symlink_to(expected,target_is_directory=True)
    if impl.resolve(strict=True)!=expected:raise ValueError('different implementation dependency')
    changes={name:source/name for name in ('bounded_actor_step.py','capture_runtime.py','stage_bounded_actor_step.py')}
    changes.update({'source/'+n:source/'source'/n for n in HELPERS})
    originals={str(p.relative_to(frozen)):p for p in frozen.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    for name in changes:
        role='controller' if name=='stage_bounded_actor_step.py' else 'actor'
        k=frozen.name+'/'+name
        if name in originals and snapshot['roles'][role].get(k)!=pin(originals[name])['sha256']:
            raise ValueError('unqualified original member: '+name)
        snapshot['roles'][role][k]=pin(changes[name])['sha256']
    paths=dict(originals,**changes)
    if not package.exists():
        tmp=Path(tempfile.mkdtemp(dir=package.parent))
        for n,path in paths.items():
            out=tmp/n;out.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(path,out)
        os.rename(tmp,package)
    for n,path in paths.items():
        if pin(package/n)['sha256']!=pin(path)['sha256']:raise ValueError('different packaged bytes: '+n)
    if {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts}!=set(paths):
        raise ValueError('unexpected launcher member')
    manifest=seal(dest/'LAUNCHER-MEMBERS.json',{n:pin(package/n) for n in sorted(paths)})
    return dict(parent_launcher=parent,launcher_ref=pin(package/'stage_bounded_actor_step.py'),launcher_members=manifest)
