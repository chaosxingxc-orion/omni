"""Compile the selected phase and physical call budget using the original GRPO Clock and scorer."""
import argparse
import json
import sys
from pathlib import Path

from build import save,pin
from paired_experiment import validate
from source.config_profile import resolve,bind_environment
import os

def compile_scope(cfg, dest):
    from source.prompt_capacity import sampled_envelope
    from source.node_training_task import select
    from dependencies import ROOT,upstream
    from scorer_extension import install
    from run_package import read
    task=read(cfg['task'])['runtime_construction']['node_training_task']
    base=read(cfg['learning']); model=read(cfg['experiment']); n=validate(model)
    learning=select(base,task,mode='formal')
    phase,=learning['phases'];scope=phase['rollout_scope']
    counts=cfg['capacity'];node=task.get('node');g=model['episodes_per_recording']
    if g!=learning['kernel']['G']:raise ValueError('learning and experiment G differ')
    cap=dict(schema='sampled-rollout-capacity-v2',scope=scope,nodes=counts)
    calls,tokens,_=sampled_envelope(cap,cfg['budget'],n,g,scope,node)
    pins={n:pin(ROOT/'unified-learning'/n) for n in ('kernel.py','scoring.py','schedule.py','learning.py','dataset.py')}
    pins.update({n:pin(ROOT/'system-objective-cpu'/n) for n in ('objective.py','runtime.py')})
    # The scope plan is not a policy publication or permission to dispatch.
    # In particular, constructing the Clock does not satisfy the DEV gate.
    learning['adapter_routing']['frozen_upstream_adapter_vector']=learning['kernel']['behavior_policy_vector'].copy()
    api=upstream();install(api)
    clock=api.create_clock(learning,consumer_config=dict(source_pins=pins))
    if not phase['required_validations']:
        assert clock.reserve('cpu-scope-check')['trainable_mask']==phase['trainable_mask']
    else:
        try:clock.reserve('cpu-scope-check')
        except ValueError as e:
            if 'validation gate' not in str(e):raise
        else:raise ValueError('group accepted before configured promotion')
    return save(dest/'SCOPE.json',dict(task=task,inputs={k:cfg[k] for k in ('task','learning','experiment')},
        learning_ref=save(dest/'LEARNING.json',learning),window_capacity=cap,scope=scope,
        trainable_mask=phase['trainable_mask'],pending_validations=phase['required_validations'],recordings_per_step=n,
        replicates=g,requests_upper=calls,capture_tokens_upper=tokens,stage_tokens_upper=3*tokens,submitted=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('--profile',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_bytes());profile=json.loads(Path(a.profile).read_bytes())
    if profile['schema']=='omni-local-profile-v2':profile,_=bind_environment(profile,os.environ)
    cfg=resolve(cfg,profile)
    sys.path.insert(0,str(Path(cfg['release'])/'tree/parents/jev-system-rl/implementation'))
    print(json.dumps(compile_scope(cfg,Path(cfg['output']))))
