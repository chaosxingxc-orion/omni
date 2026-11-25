"""Pin the exact CPU transition release, original iteration and static inputs."""
import argparse, hashlib, json
from pathlib import Path
from shutil import copyfile


def pin(p):
    p=Path(p).resolve()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def build(runner,iteration,source,lengths,static,output,threads,*,producer=None,inbox_wire_format=None):
    old=json.loads(runner.read_bytes());impl=Path(old['release'])/'tree/parents/jev-system-rl/implementation'
    compatibility=json.loads((source/'DISPATCH-COMPATIBILITY.json').read_bytes())
    if compatibility['schema']!='dispatch-overlay-bases-v1':raise ValueError('unknown overlay scope')
    for name in ('capture_stage','formal_rollout'):
        actual=pin(impl/(name+'.py'))['sha256']
        replacement=pin(source/'source'/(name+'.py'))['sha256']
        if actual not in (compatibility['bases'][name],replacement,*compatibility.get('predecessors',{}).get(name,[])):
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
    if inbox_wire_format is not None:
        bind_inbox(obj,source,output.parent,impl,inbox_wire_format)
    if producer is not None:
        from prepare_recording_macros import read,build as prepare_macros
        if set(producer)-{'lane_storage'}!={'recordings_per_macro','training_input_layout'} or producer['training_input_layout']!='group_refs_v1':
            raise ValueError('unknown producer/consumer configuration')
        # A new run uses the published build from its very first actor. Never
        # enable a new encoding against a legacy stage template.
        runner_plan=read(obj['iteration_ref']);stage=read(runner_plan['formal_actor']['stage_template_ref'])
        for name in ('consumer','cohort'):
            expected=pin(source/'source'/(name+'.py'))['sha256']
            rel=stage['generation']['runtime']['sources']['consumer_wrapper' if name=='consumer' else 'cohort_source']
            if stage['capture_mounts']['overlays'][rel]!=expected:
                raise ValueError('staged actor does not implement indexed inputs: '+name)
        obj['recording_macros_ref']=prepare_macros(obj['static_supply_ref'],producer['recordings_per_macro'],output.parent/'recording-pages')
        obj['training_input_layout']=producer['training_input_layout']
        if 'lane_storage' in producer:
            validate_lane_storage(producer['lane_storage'],impl.parents[1])
            obj['lane_storage']=producer['lane_storage']
    if output.exists() and json.loads(output.read_bytes())!=obj:raise ValueError('immutable config drift')
    if not output.exists():output.write_text(json.dumps(obj,indent=2)+'\n')
    return obj


def validate_lane_storage(profile,parents):
    validator=parents/'native-recompute-integration/state-repair/acl-runtime-successor/sync_runtime.py'
    if profile!=dict(mode='live_space_pinned_mount_v1',validator_sha256=pin(validator)['sha256']):
        raise ValueError('unqualified lane storage profile')


def bind_inbox(spec,source,folder,impl,mode):
    approved=json.loads((source/'DISPATCH-COMPATIBILITY.json').read_bytes())['inbox_publisher']
    base=pin(impl/'formal_parity.py')['sha256']
    candidate=source/'source/formal_parity.py'
    if mode!='raw_members_v1' or base not in (approved,pin(candidate)['sha256']):
        raise ValueError('unqualified policy publisher')
    dest=folder/'formal_parity.py'
    if dest.exists() and dest.read_bytes()!=candidate.read_bytes():raise ValueError('publisher drift')
    if not dest.exists():copyfile(candidate,dest)
    spec['sources']['formal_parity']=pin(dest);spec['inbox_wire_format']=mode


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('runner','iteration','source','lengths','static','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--threads',type=int,required=True)
    build(**vars(p.parse_args()))
