"""Apply a sealed CPU-only dispatch configuration to the existing iteration."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


def signature(path):
    p=Path(path).resolve(strict=True)
    s=p.stat()
    return (str(p),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def install_checkpoint_cache(publication):
    # The immutable local return is read by the owner and then its forked lanes.
    # Never share across unrelated processes/boots or use a partial check for full restore.
    original=publication.checkpoint
    cache={}
    def checkpoint(ref,*,verify_shards=False):
        if verify_shards:return original(ref,verify_shards=True)
        from dependencies import local_path
        from run_package import read
        doc=read(ref)
        logical=Path(ref['path'] if isinstance(ref,dict) else ref)
        assets={}
        for rank in doc['ranks']:
            for name,expected in rank['files'].items():
                if name.startswith('lora_adapter/'):
                    if '..' in Path(name).parts or assets.setdefault(name,expected)!=expected:
                        raise ValueError('invalid adapter manifest entry')
        files=[local_path(logical),*(local_path(logical.parent/n) for n in sorted(assets))]
        key=tuple(map(signature,files))
        if key not in cache:
            result=original(ref,verify_shards=False)
            if key!=tuple(map(signature,files)):
                raise ValueError('checkpoint changed during original validator')
            cache.clear();cache[key]=result
        else:
            # Local filesystem timestamps may coincide even across in-place writes.
            # Rehash every export member, reuse only the expensive tensor validation.
            from dependencies import sha
            for name in assets:
                if sha(logical.parent/name)!=assets[name]:
                    raise ValueError('cached adapter content changed: '+name)
        return copy.deepcopy(cache[key])
    publication.checkpoint=checkpoint


def configure(spec):
    from dependencies import require
    from run_package import read,pin
    from transition_static_inputs import for_window
    root=Path(spec['iteration_ref']['path']).resolve()
    cfg=read(spec['iteration_ref'])
    require(pin(root)==spec['iteration_ref'] and Path(cfg['output_root'])==root.parent.parent/'run'
            and cfg.get('formal_rollout',{}).get('continuous_training'),'wrong experiment')
    for ref in spec['sources'].values():read_source(ref)
    for name in ('capture_stage','formal_rollout','transition_static_inputs'):
        require(Path(sys.modules[name].__file__).resolve()==Path(spec['sources'][name]['path']).resolve(),'wrong dispatch source: '+name)
    import publication
    require(pin(publication.__file__)['sha256']==spec['publication_sha256'],'changed checkpoint reader')
    install_checkpoint_cache(publication)
    static=read(spec['static_supply_ref']); lengths=read(spec['prompt_length_index_ref'])
    require(static['length_index_ref']==spec['prompt_length_index_ref'] and
            static['profile_ref']==lengths['profile_ref'] and static['schema']=='p-static-supply-v1','changed static inputs')
    if 'lane_storage' in spec:
        require(spec['lane_storage']==dict(mode='live_space_pinned_mount_v1',validator_sha256=pin(
            __import__('dependencies').ROOT/'native-recompute-integration/state-repair/acl-runtime-successor/sync_runtime.py')['sha256']),
            'lane storage validator differs')
    def placement(config):
        require(pin(config['_path'])==spec['iteration_ref'],'different live controller configuration')
        config['formal_rollout'].update({k:spec[k] for k in ('static_supply_ref','prompt_length_index_ref')})
        if 'recording_macros_ref' in spec:
            config['formal_rollout']['recording_macros_ref']=spec['recording_macros_ref']
        if 'training_input_layout' in spec:
            config['formal_rollout']['training_input_layout']=spec['training_input_layout']
        if 'lane_storage' in spec:
            config['formal_rollout']['lane_storage']=spec['lane_storage']
        if 'idle_recovery_ref' in spec:
            import recover_idle_actor
            recover_idle_actor.install(config,spec)
        elif 'actor_reader_release_ref' in spec:
            import install_indexed_actor
            install_indexed_actor.install(config,spec,Path(spec['sources']['transition_controller']['path']).parent)
        return config
    return root,placement


def read_source(ref):
    data=Path(ref['path']).read_bytes()
    if len(data)!=ref['bytes'] or hashlib.sha256(data).hexdigest()!=ref['sha256']:raise ValueError('source digest differs')
    return data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('config',type=Path);p.add_argument('--check',action='store_true')
    args=p.parse_args()
    spec=json.loads(args.config.read_bytes())
    if spec['schema']!='dispatch-transition-v1' or type(spec['cpu_threads'])!=int or not 1<=spec['cpu_threads']<=32:
        raise ValueError('invalid dispatch transition configuration')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]=str(spec['cpu_threads'])
    from importlib.util import spec_from_file_location,module_from_spec
    for name,ref in spec['sources'].items():
        if name not in ('transition_static_inputs','capture_stage','formal_rollout','transition_controller','install_indexed_actor','recover_idle_actor'):
            raise ValueError('unexpected transition module')
        raw=read_source(ref)
        if name=='transition_controller':
            if Path(ref['path']).resolve()!=Path(__file__).resolve():raise ValueError('unexpected transition entry')
            continue
        origin=spec_from_file_location(name,ref['path'])
        m=module_from_spec(origin);sys.modules[name]=m;exec(compile(raw,origin.origin,'exec'),m.__dict__)
    root,place=configure(spec)
    if args.check:
        print(json.dumps({'ready':True,'iteration':str(root),'config':str(args.config)}))
        return
    import iteration
    iteration.run(root,resume_placement=place)


if __name__=='__main__':main()
