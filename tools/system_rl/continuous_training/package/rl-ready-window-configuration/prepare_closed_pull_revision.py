"""Qualify a CPU-only closed-return transport against the original acknowledged command."""
import argparse, errno, hashlib, json, os, shlex, subprocess, sys, tarfile, tempfile
from pathlib import Path
from prepare_recording_macros import read, seal
from prepare_transition_config import pin


def prepare(transition,update,transport,out):
    out.mkdir(parents=True,exist_ok=True)
    spec=read(pin(transition));run=read(spec['runner_ref']);it=read(spec['iteration_ref'])
    generation=read(pin(update/'COMMAND.json'))['generation_inventory_ref']
    ack=read(pin(update/'COMMAND-ACK.json'));call=read(pin(update/'RESIDENT-CALL.json'))
    sub=read(pin(update/'SUBMIT-INTENT.json'));wire=read(ack['command_ref'])
    if (sub['command']!=wire or wire['generation_inventory_ref']!=generation or ack['remote']['sequence']!=call['sequence']
            or sub['target']!=call['target'] or ack['remote']['source_boot']!=sub['source_boot']):
        raise ValueError('not the same acknowledged command')
    member='jev-system-rl/implementation/transport.py';release=Path(run['release'])
    after=out/'transport.py'
    if after.exists() and after.read_bytes()!=transport.read_bytes():raise ValueError('changed candidate')
    if not after.exists():after.write_bytes(transport.read_bytes())
    parent=spec['runner_ref'];prior=None
    if 'controller_revision' in run:
        prior=read(run['controller_revision'])
        if len(prior['changes'])!=1 or prior['changes'][0]['member']!=member or 'timing_schedule_ref' in prior:
            raise ValueError('cannot replace unrelated controller changes')
        parent=prior['parent_runner']
        if read(parent)!={k:v for k,v in run.items() if k!='controller_revision'}:raise ValueError('different parent runner')
    revision=dict(schema='controller-training-input-revision-v1',parent_runner=parent,
        parent_build=pin(release.parent/'TRAINING-BUILD.json'), deployment=pin(Path(run['deployment'])),
        iteration=spec['iteration_ref'],closed_generation=generation,
        changes=[dict(member=member,before=pin(release/'tree/parents'/member),after=pin(after))])
    if prior:revision['previous_revision']=run['controller_revision']
    runner=seal(out/'RUNNER.json',dict(run,controller_revision=seal(out/'REVISION.json',revision)))
    candidate=seal(out/'TRANSITION.json',dict(spec,runner_ref=runner))
    return candidate


def qualify(config,update,out):
    from checkpoint_inventory import validate_manifest
    import transport
    spec=read(pin(config));runner=read(spec['runner_ref']);revision=read(runner['controller_revision'])
    if pin(Path(transport.__file__))['sha256']!=revision['changes'][0]['after']['sha256']:
        raise ValueError('wrong mounted transport')
    run=read(pin(update/'REQUEST.json'))['run'];it=read(spec['iteration_ref']);fa=it['formal_actor']
    known=read(pin(sorted(update.glob('PULL-INVENTORY-*.json'))[-1]))['verified_objects']
    call=read(pin(update/'RESIDENT-CALL.json'));owner=Path(call['target']);seq=call['sequence']
    target=owner/'hot/resident/results'/f'{seq:06d}';argv=['python3','-B',str(owner/'pull_closed.py'),str(target),
        str(fa['return_max_bytes']),'--formal-resident','--owner='+fa['continuous_actor']['run_id'],'--expected-boot='+run['source_boot']]
    new,payload=transport.closed_reader_input(argv,known)
    cmd=lambda a:['ssh','-oBatchMode=yes','-oConnectTimeout=15',run['host'],shlex.join(a)]
    try:subprocess.run(['/bin/true',cmd(argv+['--inventory-json='+payload.decode()])[-1]],check=True)
    except OSError as exc:
        if exc.errno!=errno.E2BIG:raise
        too_big=True
    else:too_big=False
    with tempfile.TemporaryFile(dir=out) as f:
        subprocess.run(cmd(new),input=payload,stdout=f,check=True,timeout=fa['return_timeout_seconds'])
        f.seek(0)
        with tarfile.open(fileobj=f) as arc:
            manifest=json.load(arc.extractfile('MANIFEST.json'));validate_manifest(manifest)
            if manifest['closure']['source_boot']!=run['source_boot'] or Path(manifest['closure']['closed_path'])!=target/'CLOSED.json':
                raise ValueError('wrong remote closed result')
            for h,r in manifest['objects'].items():
                if r['member'] is None:
                    if known.get(h)!=r['bytes']:raise ValueError('unavailable reference')
                else:
                    digest=hashlib.sha256()
                    with arc.extractfile(r['member']) as stream:
                        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
                    if digest.hexdigest()!=h:raise ValueError('different wire object')
    # Other callers supply pinned reader source inline.
    probe=['python3','-B','-c',"import json,sys;print(len(json.loads(sys.argv[-1].split('=',1)[1])))"]
    inline,data=transport.closed_reader_input(probe,known)
    if int(subprocess.check_output(inline,input=data))!=len(known):raise ValueError('inline reader lost inventory')
    return seal(out/'ACCEPTANCE.json',dict(passed=True,config_sha256=pin(config)['sha256'],old_e2big=too_big,
        inventory_bytes=len(payload),verified_objects=len(known),sequence=seq,closure_sha256=manifest['closure']['closed_sha256'],
        remote_command_bytes=len(cmd(new)[-1]),inline_reader_verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','qualify'])
    for name in ('config','update','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--transport',type=Path);a=p.parse_args()
    print(json.dumps(prepare(a.config,a.update,a.transport,a.out) if a.mode=='prepare' else qualify(a.config,a.update,a.out)))
