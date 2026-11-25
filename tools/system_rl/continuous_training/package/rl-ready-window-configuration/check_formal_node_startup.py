"""Four real retained-input branches through the original startup bindings.

No requests, inference, optimizer or full-TRAIN acceptance. Diagnostic and
training Clocks must remain separate; retained PCM is not materialized twice.
"""
import argparse, copy, json, sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--release',required=True)
p.add_argument('--runtime-input',required=True);p.add_argument('--config',required=True)
p.add_argument('--materials',required=True);p.add_argument('--output',required=True)
a=p.parse_args();sys.path.insert(0,str(Path(a.release)/'tree/parents/jev-system-rl/implementation'))
from dependencies import ROOT,HERE,upstream
from run_package import read,pin,write
from storage import environment
from scorer_extension import install
from node_training_task import select,NODES
from finite_runtime import prepare_formal_startup,prepare_finite,prepare_diagnostic

root=Path(a.output);environment(root,0);config=read(a.config)
original=read(a.runtime_input);old_learning=read(original['learning_config_ref'])
task=upstream();install(task);template=read(HERE/'learning.template.json')
data=ROOT/'collection-dispatch-integration/production-pipeline-next-version-v1/finite-capture-binding-v1'
config.update(execution_mode='formal',
    finite_data_binding=dict(module=pin(data/'binding.py'),source_closure=write(root/'DATA-SOURCES.json',
        [pin(data/n) for n in ('binding.py','prepare_sources.py','bounded_train_probe.py','CONFIG.json')])))
seal=read(ROOT/'collection-dispatch-integration/production-pipeline-next-version-v1/finite-diagnostic-capture-v1/ASSEMBLED-BINDING.json')
config.update(finite_diagnostic_binding=dict(module=seal['module_ref'],source_closure=seal['source_closure_ref']),
              finite_diagnostic_config_ref=seal['config_ref'])
checks=[];branches={}
def rejected(name,fn):
    try:fn()
    except ValueError:checks.append(name)
    else:raise AssertionError(name)

for node in NODES:
    path=root/node;spec=read(Path(__file__).resolve().parent/'config-fragments'/('node-'+node+'.json'))['runtime_construction']['node_training_task']
    cfg=copy.deepcopy(config);cfg['runtime_construction']['node_training_task']=spec
    learning=select(template,spec,mode='formal')
    learning['system_version']=old_learning['system_version']
    for key in ('behavior_policy_vector','current_policy_vector','reference_policy_vector'):
        learning['kernel'][key]=copy.deepcopy(old_learning['kernel'][key])
    learning['adapter_routing']['frozen_upstream_adapter_vector']=copy.deepcopy(learning['kernel']['current_policy_vector'])
    learning['phases'][0]['frozen_upstream_release']=old_learning['phases'][0]['frozen_upstream_release']
    lr=write(path/'LEARNING.json',learning)
    consumer=read(original['consumer_config_ref']);consumer['learning_config_ref']=lr
    consumer['source_pins']={n:pin(ROOT/'unified-learning'/n) for n in
        ('kernel.py','learning.py','dataset.py','scoring.py','schedule.py','fixtures.py')}
    consumer['source_pins'].update({n:pin(ROOT/'system-objective-cpu'/n) for n in ('objective.py','runtime.py')})
    consumer['sealed_adapter_ref']=pin(ROOT/'training-consumer-adapter/adapter.py')
    clock=task.create_clock(learning,consumer_config=consumer)
    state=write(path/'CLOCK.json',clock.state);consumer['clock_state_ref']=state
    source=copy.deepcopy(original);source.update(learning_config_ref=lr,clock_state_ref=state,
        consumer_config_ref=write(path/'CONSUMER.json',consumer),
        retained_source_material_ref=pin(a.materials))
    ref=write(path/'INPUT.json',source)
    out=prepare_formal_startup(cfg,ref,path/'prepared',source['policy']['current_ref'])
    assert prepare_formal_startup(cfg,ref,path/'prepared',source['policy']['current_ref'])==out
    prepared=read(out['input_ref']);bound=read(prepared['learning_config_ref'])
    assert bound['phases'][0]['target_nodes']==[node]
    assert bound['phases'][0]['trainable_mask']=={n:n==node for n in NODES}
    assert bound['task_loss']==learning['task_loss']
    assert prepared['formal_startup']['target_node']==node and not prepared['formal_startup']['training_ready']
    assert prepared['policy']==source['policy'] and prepared['execution_registry']==source['execution_registry']
    assert prepared['budget']['cumulative_prior_tokens_upper']==source['budget']['cumulative_prior_tokens_upper']
    assert not (path/'prepared/source-material').exists()
    assert read(prepared['consumer_config_ref'])['recording_map_ref']==read(a.materials)['recording_map_ref']
    updated_clock=task.create_clock(bound,read(prepared['clock_state_ref']),consumer_config=read(prepared['consumer_config_ref']))
    assert updated_clock.state['successful_optimizer_steps']==0
    checks.append(node+' original startup preserves node mask/Clock/cost and reuses retained PCM')
    diagnostic=prepare_diagnostic(cfg,out['input_ref'],path/'diagnostic',source['policy']['current_ref'],None,'base_parity')
    evidence=read(diagnostic['input_ref'])
    assert evidence['diagnostic_evidence_only']['trainable'] is False
    assert evidence['diagnostic_evidence_only']['training_input_ref']==out['input_ref']
    assert evidence['formal_startup']['target_node']==node
    assert not evidence['native']['enabled'] and not evidence['authorization']['native_training']
    assert read(out['input_ref'])==prepared
    checks.append(node+' original all-node diagnostic is separate and cannot replace training Clock')
    branches[node]=dict(input_ref=out['input_ref'],diagnostic_ref=diagnostic['input_ref'])
    from capture_stage import _generate
    request=write(path/'UNSCREENED-REQUEST.json',dict(input_ref=out['input_ref'],output_root=str(path/'forbidden-capture')))
    rejected(node+' unscreened formal input cannot enter live capture',lambda:_generate(request,{},None))
    if node!='P':
        rejected(node+' finite training still refuses non-P',lambda:prepare_finite(cfg,ref,path/'invalid-finite',source['policy']['current_ref']))
    bad=copy.deepcopy(cfg);bad['runtime_construction']['node_training_task']['node']='D' if node!='D' else 'P'
    rejected(node+' mismatched configured task rejected',lambda:prepare_formal_startup(bad,ref,path/'invalid-target',source['policy']['current_ref']))

print(json.dumps(write(root/'RESULT.json',dict(passed=True,checks=checks,branches=branches,
    source_records=19,full_TRAIN=False,model_calls=0,optimizer_updates=0,
    actual_startup_and_diagnostic_bindings=True,multi_adapter_backend_accepted=False))))
