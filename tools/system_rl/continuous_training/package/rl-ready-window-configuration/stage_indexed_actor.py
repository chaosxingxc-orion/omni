"""Stage verified publication overlays in a sibling capture root; leave the old worker intact."""
import argparse,copy,hashlib,json,shlex,subprocess,tarfile,tempfile
from pathlib import Path
from prepare_transition_config import pin
from prepare_recording_macros import read,seal
from stage_sampled_actor_release import REMOTE


def stage(spec,source,dest):
    it=read(spec['iteration_ref']);actor=it['formal_actor'];parent=actor['stage_template_ref'];lockref=actor['continuous_actor']['source_lock_ref']
    if 'predecessor' in spec:
        from install_indexed_actor import predecessor
        _,sw,lockref=predecessor(spec['predecessor'],spec['iteration_ref'],actor,read=read,pin=pin)
        parent=sw['restore_template']
    template=read(parent)
    mounts=template['capture_mounts'];old=Path(mounts['root']);snap=read(lockref)
    sha=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    sources={};extra={};names=['cohort.py','consumer.py']
    if 'first_wave_forward' in spec:
        if spec['first_wave_forward']!='overlap_publish_v1' or template['generation']['runtime']['group_preparation']!='shard_by_draw_v1':
            raise ValueError('unsupported first-wave deployment')
        from stage_wave_launcher import build
        launcher=sw.get('launcher_ref',actor['launcher_ref']) if 'predecessor' in spec else actor['launcher_ref']
        extra=build(source,dest,launcher,snap)
        names.append('cohort_capture.py')
        template['generation']['runtime']['first_wave_forward']=spec['first_wave_forward']
    for name in names:
        src=source/'source'/name;key='rl-ready-window-configuration/source/'+name;rel='fresh-cohort-successor/'+name
        if mounts['overlays'][rel]!=snap['roles']['actor'][key]:raise ValueError('unqualified base overlay')
        data=pin(src);snap['roles']['actor'][key]=mounts['overlays'][rel]=data['sha256']
        sources['overlays/'+rel]=src
    snap['candidate_sha256']=sha({k:snap[k] for k in ('schema','precision_lock_sha256','policy_sha256','roles','version')})
    root=old.with_name(old.name+'-'+snap['candidate_sha256'][:12]);mounts['root']=str(root)
    # Keep the original cold tree for immutable paths in prior generated data.
    conf=dict(old=str(old),dest=str(root),boot=actor['source_boot'],files={k:pin(v)['sha256'] for k,v in sources.items()},
              bytes=sum(v.stat().st_size for v in sources.values()))
    with tempfile.TemporaryFile(dir=dest) as f:
        with tarfile.open(fileobj=f,mode='w') as tar:
            for k,v in sources.items():tar.add(v,arcname=k,recursive=False)
        f.seek(0);subprocess.run(['ssh','-oBatchMode=yes',template['host'],shlex.join(['python3','-B','-c',REMOTE,json.dumps(conf)])],stdin=f,check=True)
    return seal(dest/'READER-RELEASE.json',dict(parent_template=parent,iteration_ref=spec['iteration_ref'],
                **({'predecessor':spec['predecessor']} if 'predecessor' in spec else {}),
                stage_template=seal(dest/'STAGE.json',template),source_lock=seal(dest/'SOURCE-LOCK.json',snap),**extra))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('spec','source','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(stage(json.loads(a.spec.read_bytes()),a.source,a.out)))
