"""Pin the exact CPU transition release, original iteration and static inputs."""
import argparse, hashlib, json
from pathlib import Path
from shutil import copyfile


def pin(p):
    p=Path(p).resolve()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def build(runner,iteration,source,lengths,static,output,threads):
    old=json.loads(runner.read_bytes());impl=Path(old['release'])/'tree/parents/jev-system-rl/implementation'
    compatibility=json.loads((source/'DISPATCH-COMPATIBILITY.json').read_bytes())
    if compatibility['schema']!='dispatch-overlay-bases-v1':raise ValueError('unknown overlay scope')
    for name in ('capture_stage','formal_rollout'):
        actual=pin(impl/(name+'.py'))['sha256']
        replacement=pin(source/'source'/(name+'.py'))['sha256']
        if actual not in (compatibility['bases'][name],replacement):
            raise ValueError('unqualified dispatch source: '+name)
    src={}
    for n in ('transition_controller','transition_static_inputs','capture_stage','formal_rollout'):
        origin=source/('source' if n in ('capture_stage','formal_rollout') else '')/(n+'.py')
        dest=output.parent/(n+'.py')
        if dest.exists() and dest.read_bytes()!=origin.read_bytes():raise ValueError('candidate already holds different source')
        if not dest.exists():copyfile(origin,dest)
        src[n]=pin(dest)
    obj=dict(schema='dispatch-transition-v1',iteration_ref=pin(iteration),runner_ref=pin(runner),sources=src,
             static_supply_ref=pin(static),prompt_length_index_ref=pin(lengths),cpu_threads=threads,
             publication_sha256=pin(impl/'publication.py')['sha256'])
    if output.exists() and json.loads(output.read_bytes())!=obj:raise ValueError('immutable config drift')
    if not output.exists():output.write_text(json.dumps(obj,indent=2)+'\n')
    return obj


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('runner','iteration','source','lengths','static','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--threads',type=int,required=True)
    build(**vars(p.parse_args()))
