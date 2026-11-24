"""Launch the fixed CPU image with the locked dual-host release and local custody mounts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def pin(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def launch(c, args):
    recipe=Path(c['image_recipe']['path'])
    if pin(recipe)!=c['image_recipe']:raise ValueError('changed image recipe')
    source=shlex.split(recipe.read_text().split('exec ',1)[1])
    if source[:2]!=['docker','run'] or source[-3:]!=['python3','-B','$@']:raise ValueError('unexpected image command')
    release=Path(c['release']); complete=release/'BUILD-COMPLETE.json'
    if not complete.is_file():raise ValueError('release build not closed')
    deployment=json.loads(Path(c['deployment']).read_bytes())
    build=json.loads((release.parent/'TRAINING-BUILD.json').read_bytes())
    if build['source_lock_ref']!=deployment['source_lock_ref'] or build['precision_lock_ref']!=deployment['precision_lock_ref'] or build['build_complete_ref']!=pin(complete):
        raise ValueError('controller and build lock differ')
    tree=Path(c['logical_root']); parents=release/'tree/parents'
    members=json.loads(Path(c['policy']).read_bytes())['runtime_members']
    covered={tree/Path(k).relative_to('parents') for k in members}
    revision_mounts = {}
    revision_ref = c.get('controller_revision')
    if revision_ref:
        def checked(ref):
            path = Path(ref['path'])
            if pin(path) != ref:
                raise ValueError('controller revision input changed: ' + str(path))
            return path

        revision = json.loads(checked(revision_ref).read_bytes())
        parent = json.loads(checked(revision['parent_runner']).read_bytes())
        if (parent != {key:value for key,value in c.items() if key != 'controller_revision'}
                or revision['schema'] != 'controller-training-input-revision-v1'
                or checked(revision['parent_build']) != release.parent/'TRAINING-BUILD.json'
                or checked(revision['deployment']) != Path(c['deployment'])
                or json.loads(Path(c['grant']).read_bytes())['iteration_ref'] != revision['iteration']):
            raise ValueError('controller revision and original experiment differ')
        checked(revision['iteration'])
        checked(revision['closed_generation'])
        for change in revision['changes']:
            member = Path(change['member'])
            allowed={'jev-system-rl/implementation/transport.py'}
            if 'timing_schedule_ref' in revision:
                allowed.add('jev-system-rl/implementation/iteration.py')
            if (str(member) not in allowed
                    or checked(change['before']) != parents/member or tree/member in revision_mounts):
                raise ValueError('unexpected controller input transport replacement')
            revision_mounts[tree/member] = checked(change['after'])
        covered.update(revision_mounts)
        if 'timing_schedule_ref' in revision:
            schedule=json.loads(checked(revision['timing_schedule_ref']).read_bytes())
            if (schedule['iteration_ref']!=revision['iteration'] or
                    tree/'jev-system-rl/implementation/iteration.py' not in revision_mounts):
                raise ValueError('timing schedule not bound to controller revision')
    cmd=[];i=0
    while i<len(source)-4:
        arg=source[i];i+=1
        if arg not in ('-e','--mount'):cmd.append(arg);continue
        value=source[i];i+=1
        if arg=='-e' and value.startswith('OMNI_'):continue
        if arg=='--mount':
            fields=dict(item.split('=',1) for item in value.split(',') if '=' in item)
            dest=Path(fields['dst'])
            if dest==tree:value=value.replace('src='+fields['src']+',','src='+str(parents)+',')
            elif dest in covered or ('/control/' in str(dest) and fields['src']==str(dest)):continue
        cmd.extend([arg,value])
    for dest,src in c['mounts'].items():
        cmd+=['--mount',f'type=bind,src={src},dst={dest},readonly']
    for dest,src in revision_mounts.items():
        cmd+=['--mount',f'type=bind,src={src},dst={dest},readonly']
    if revision_ref and 'timing_schedule_ref' in revision:
        cmd+=['-e','OMNI_TIMING_SCHEDULE_REF='+json.dumps(revision['timing_schedule_ref'])]
    if 'grant' in c and Path(c['grant']).is_file():
        cmd+=['-e','OMNI_ROLLOUT_HOST_GRANT_REF='+json.dumps(pin(Path(c['grant'])))]
    cmd+=source[-4:-1]+args
    os.execvp(cmd[0],cmd)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('config',type=Path)
    p.add_argument('--submit',action='store_true')
    p.add_argument('argv',nargs=argparse.REMAINDER)
    a=p.parse_args();c=json.loads(a.config.read_bytes())
    if not a.submit:
        launch(c,a.argv)
    root=a.config.parent;release=Path(c['release']);out=root/'compiler'
    marker=release.parent/'TRAINING-BUILD.json'
    while not marker.exists():
        # A failed build needs operator reconciliation, never launch its partial tree.
        if Path(c['build_log']).exists() and 'Traceback (most recent call last)' in Path(c['build_log']).read_text():
            raise RuntimeError('release build failed')
        time.sleep(5)
    prefix=[sys.executable,'-B',str(Path(__file__).resolve()),str(a.config)]
    subprocess.run(prefix+[c['compiler'],'--manifest',str(root/'MANIFEST.json'),'--output',str(out),'--bind-live'],check=True)
    config=out/'ITERATION.json';spec=json.loads(config.read_bytes())
    if spec['formal_rollout']['ssh']['target']!=c['rollout_target'] or spec['formal_actor']['rollout_lanes']!=list(c['lanes']):
        raise ValueError('deployment outside granted lanes')
    grant=dict(iteration_ref=pin(config),hosts=[c['rollout_target']])
    path=Path(c['grant'])
    if path.exists():
        if json.loads(path.read_bytes())!=grant:raise ValueError('grant binding changed')
    else:
        with path.open('x') as f:
            json.dump(grant,f);f.flush();os.fsync(f.fileno())
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    launch(c,[str(Path(c['logical_root'])/'jev-system-rl/implementation/iteration.py'),'--config',str(config)])
