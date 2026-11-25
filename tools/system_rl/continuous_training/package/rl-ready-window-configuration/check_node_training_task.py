"""Independent node config through original Clock and auxiliary schedule."""
import argparse,copy,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--output',required=True);a=p.parse_args()
impl=Path(a.release)/'tree/parents/jev-system-rl/implementation';sys.path.insert(0,str(impl))
from dependencies import upstream,ROOT
from run_package import read,write,pin
from node_training_task import select,shift_curve,NODES
from scorer_extension import install
task=upstream();install(task)
template=read(impl/'learning.template.json');checks=[];reports={}
pins={n:pin(ROOT/'unified-learning'/n) for n in ('kernel.py','scoring.py','schedule.py','learning.py','dataset.py')}
pins.update({n:pin(ROOT/'system-objective-cpu'/n) for n in ('objective.py','runtime.py')})
consumer=dict(source_pins=pins)
for node in NODES:
    spec=read(Path(__file__).resolve().parent/'config-fragments'/('node-'+node+'.json'))['runtime_construction']['node_training_task']
    config=select(template,spec,mode='formal')
    config['adapter_routing']['frozen_upstream_adapter_vector']=copy.deepcopy(config['kernel']['behavior_policy_vector'])
    config['phases'][0]['frozen_upstream_release']='fixture-pinned-release'
    clock=task.create_clock(config,consumer_config=consumer)
    if config['phases'][0]['required_validations']:
        try:clock.reserve('before-promotion')
        except ValueError as exc:assert 'validation gate' in str(exc)
        else:raise AssertionError('unvalidated node accepted')
        for gate in config['phases'][0]['required_validations']:
            clock.record_validation(gate,dict(scope='node',passed=True,config_sha256=clock.state['config_sha256'],
                evidence_sha256='fixture-only',fixture=True))
        checks.append(node+' rejects reservation before promotion')
    assert clock.phase()['target_nodes']==[node] and clock.state['successful_optimizer_steps']==0
    snapshot=clock.reserve('fixture-'+node)
    assert snapshot['trainable_mask']=={n:n==node for n in NODES}
    origin=next(p['start_step'] for p in template['phases'] if p['target_nodes']==[node])
    phase=config['phases'][0]
    with task.consumer.unified(consumer) as (_,schedule,_):
        for term,old in template['task_loss']['phases'][phase['id']].items():
            new=config['task_loss']['phases'][phase['id']][term]
            for step in (0,1,25,100,200):assert abs(schedule.curve(new,step)-schedule.curve(old,origin+step))<1e-12
    reports[node]=dict(phase=phase['id'],step=0,mask=snapshot['trainable_mask'],independent_version=config['version'])
    checks.append(node+' original Clock and local annealing preserved')
assert len({v['independent_version'] for v in reports.values()})==4
assert select(template,None,mode='finite_engineering')==template
checks.append('default existing workflow unchanged')
for bad,mode in [(dict(spec,node='invalid'),'formal'),(spec,'finite_engineering'),(dict(spec,sequential_only_validations=['not-a-gate']),'formal')]:
    try:select(template,bad,mode=mode)
    except ValueError:pass
    else:raise AssertionError('invalid independent branch accepted')
checks.append('invalid node/init/diagnostic mode rejected')
print(json.dumps(write(Path(a.output)/'RESULT.json',dict(passed=True,checks=checks,reports=reports,
    fixture=True,NPU=False,multi_adapter_backend_accepted=False,model_calls=0))))
