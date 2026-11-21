"""Persistent worker lifecycle for the sealed actual93 language arithmetic.

No audio replacement, teacher tensor injection, operator changes or automatic
training admission. The original service/worker owns the lifetime and failures.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


FILES=('boundary_storage.py','post_norm_repair.py','expert_gate_repair.py',
       'attention_contract_compute.py','paged_attention_contract.py','rope_table_compute.py',
       'expert_grouped_combine.py','expert_down_projection.py','expert_arithmetic.py','router_arithmetic.py',
       'attention_family_runtime.py','norm_family_runtime.py','projection_family_runtime.py',
       'qk_family_runtime.py','router_family_runtime.py','router_selection_runtime.py',
       'moe_family_runtime.py','family_runtime.py')
SCHEMA='omni-persistent-family-precision-v1'
ATTR='_omni_persistent_family_precision'


def require(ok,message):
    if not ok:raise ValueError(message)


def checked(ref):
    from dependencies import local_path
    p=local_path(ref['path']);payload=p.read_bytes()
    require(hashlib.sha256(payload).hexdigest()==ref['sha256'],'family source/config drift: '+str(p))
    require(ref.get('bytes',len(payload))==len(payload),'family byte length drift')
    return p


def validate(spec):
    require(set(spec)-{'arithmetic_delta_ref','efficiency_delta_ref','audio_repair_ref','original_operator_control'}=={'schema','source_seal_ref','config_ref','sources','mode','audio_policy'},'explicit family release fields')
    require(spec['schema']==SCHEMA and spec['mode']=='diagnostic_candidate',
            'persistent full-model numerical acceptance still required')
    require(spec['audio_policy'] in ('native_unmodified','conv1_internal_format_disabled_v1'),'explicit audio dispatch policy')
    if spec['audio_policy']=='conv1_internal_format_disabled_v1':
        require('audio_repair_ref' in spec,'pinned audio dispatch implementation required')
        checked(spec['audio_repair_ref'])
    else:
        require('audio_repair_ref' not in spec,'inactive audio repair must not be hidden')
    seal=json.loads(checked(spec['source_seal_ref']).read_text())
    require(seal['schema']=='audio-boundary-candidate-source-v1','original actual93 source seal')
    cfg_path=checked(spec['config_ref']);config=json.loads(cfg_path.read_text())
    require(spec['config_ref']['sha256']==seal['files']['src/FAMILY-RUNTIME.json']['sha256'],
            'family configuration differs from sealed numerical reference')
    require(set(spec['sources'])==set(FILES),'complete explicit arithmetic source closure')
    changes={}
    if 'arithmetic_delta_ref' in spec:
        delta=json.loads(checked(spec['arithmetic_delta_ref']).read_text())
        require(delta['schema']=='omni-family-arithmetic-delta-v1' and
                delta['parent_seal_ref']==spec['source_seal_ref'] and
                delta['training_admission'] is False,'explicit unaccepted arithmetic successor')
        changes=delta['changes']
        require(set(changes) in ({'projection_family_runtime.py'},
                {'projection_family_runtime.py','moe_family_runtime.py'}),
                'only explicit projection layout and expert activity repairs')
        for name,row in changes.items():
            require(row['parent_sha256']==seal['files']['src/'+name]['sha256'] and
                    ('efficiency_delta_ref' in spec or row['result_ref']==spec['sources'][name]),
                    'arithmetic delta custody differs')
    sources={}
    efficiency={}
    if 'efficiency_delta_ref' in spec:
        delta=json.loads(checked(spec['efficiency_delta_ref']).read_text())
        require(set(delta)=={'schema','parent_spec_ref','changes','options','training_admission'} and
                delta['schema']=='omni-empty-expert-efficiency-delta-v1' and
                delta['training_admission'] is False and
                delta['options'].get('skip_empty_expert_chunks') is True and
                set(delta['options']) in ({'skip_empty_expert_chunks'}, {'skip_empty_expert_chunks','packed_active_limit'}) and
                ('packed_active_limit' not in delta['options'] or
                 type(delta['options']['packed_active_limit']) is int and 1 <= delta['options']['packed_active_limit'] <= 32),
                'explicit unaccepted expert transport optimization')
        parent=json.loads(checked(delta['parent_spec_ref']).read_text())
        require('efficiency_delta_ref' not in parent,'one explicit efficiency successor')
        validate(parent)
        for key in ('schema','source_seal_ref','config_ref','mode','arithmetic_delta_ref',
                    'audio_policy','audio_repair_ref','original_operator_control'):
            require(parent.get(key)==spec.get(key),'efficiency change cannot alter numerical parent')
        efficiency=delta['changes']
        require(set(efficiency)=={'family_runtime.py','moe_family_runtime.py',
                'expert_gate_repair.py','expert_down_projection.py'},'exact empty-expert source scope')
        for name,row in efficiency.items():
            require(row['parent_ref']==parent['sources'][name] and
                    row['result_ref']==spec['sources'][name],'empty-expert source custody differs')
        require(all(spec['sources'][name]==parent['sources'][name] for name in FILES if name not in efficiency),
                'unrelated arithmetic changed in efficiency successor')
        config=dict(config,**delta['options'])
    for name in FILES:
        ref=spec['sources'][name];p=checked(ref)
        expected=(efficiency[name]['result_ref']['sha256'] if name in efficiency else
                  changes[name]['result_ref']['sha256'] if name in changes else seal['files']['src/'+name]['sha256'])
        require(p.name==name and ref['sha256']==expected,
                'family arithmetic differs from numerical reference: '+name)
        sources[name]=p
    require(config['layers']==list(range(48)) and config['preserve_original'] is True and
            config['down_weight_layout']=='hf_transposed' and config['router_selection'] is True,
            'exact original all48 candidate configuration')
    if 'original_operator_control' in spec:
        require(spec['original_operator_control'] is False and not efficiency,
                'disable only original operator controls in this candidate')
        config=dict(config,preserve_original=False)
    if config.get('packed_active_limit'):
        runtime_spec=importlib.util.spec_from_file_location('packed_family_check',sources['family_runtime.py'])
        runtime=importlib.util.module_from_spec(runtime_spec)
        runtime_spec.loader.exec_module(runtime)
        runtime.validate_config(config)
    return config,sources


def expected_names(config):
    names={'final_norm','moe'}
    suffixes=('input_layernorm','post_attention_layernorm','attention','qkv','row','qk_rope','router','router_selection')
    for layer in config['layers']:names.update(f'layer.{layer}.{name}' for name in suffixes)
    return names


def imports(sources):
    """Install exact sealed helpers for lazy absolute imports; never shadow peers."""
    created=[]
    def restore():
        for name,module in reversed(created):
            require(sys.modules.get(name) is module,'family helper module ownership changed: '+name)
            del sys.modules[name]
    try:
        for filename in FILES:
            name=Path(filename).stem;p=sources[filename]
            old=sys.modules.get(name)
            if old is not None:
                actual=getattr(old,'__file__',None)
                require(actual and Path(actual).resolve()==p.resolve(),
                        'existing helper namespace has another owner: '+name)
                continue
            spec=importlib.util.spec_from_file_location(name,p)
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;created.append((name,module))
            spec.loader.exec_module(module)
    except BaseException:
        restore();raise
    return sys.modules['family_runtime'],restore


def counters(audit,config):
    modules=audit['modules'];require(set(modules)==expected_names(config),'all48 family attachment coverage')
    result={}
    for name,row in modules.items():
        require(not row['restored'],'family unexpectedly restored while serving: '+name)
        if name=='moe':
            require(set(row['layers'])=={str(i) for i in config['layers']},'all48 MoE attachments')
            result[name]={k:v['calls'] for k,v in row['layers'].items()}
        else:
            require(row['original_calls']==(row['calls'] if config['preserve_original'] else 0),
                    'original operator control count differs from configured mode: '+name)
            result[name]=row['calls']
    return result


def install(worker,spec):
    config,sources=validate(spec)
    runner=worker.model_runner
    require(not getattr(runner,'use_async_scheduling',False),'synchronous family lifecycle required')
    vc=getattr(worker,'vllm_config',None)
    require(vc is not None and getattr(vc.model_config,'enforce_eager',False) is True and
            getattr(vc,'speculative_config',None) is None,'eager nonspeculative worker required')
    require(vc.parallel_config.tensor_parallel_size==4,'TP4 worker required')
    model=runner.get_model();state=getattr(model,ATTR,None)
    if state is not None:
        require(state['spec']==spec,'different family policy already installed')
        counters(state['audit'],config)
        return state['receipt']
    require(getattr(model,'_production_row_precision',None) is None,'legacy language precision already installed')
    module,undo_imports=imports(sources);undo=None;audio_undo=None;audio_audit=None
    try:
        audit,undo=module.attach(model.language_model.model,config)
        if spec['audio_policy']=='conv1_internal_format_disabled_v1':
            from dependencies import load
            repair=load(checked(spec['audio_repair_ref']),'omni_scoped_audio_conv_precision')
            audio_audit,audio_undo=repair.attach(model.audio_tower)
        counts=counters(audit,config)
        require(all(v==0 if isinstance(v,int) else all(x==0 for x in v.values()) for v in counts.values()),
                'install must not execute hidden model calls')
        receipt=dict(schema=SCHEMA,installed=True,attachments=len(audit['modules']),layers=48,
            config_sha256=spec['config_ref']['sha256'],audio_policy=spec['audio_policy'],
            single_prefill_restriction=False,production_acceptance=False,training_admission=False)
        if 'arithmetic_delta_ref' in spec:receipt['arithmetic_delta_ref']=spec['arithmetic_delta_ref']
        for option in ('efficiency_delta_ref','original_operator_control'):
            if option in spec:receipt[option]=spec[option]
        if audio_audit is not None:receipt['audio_repair_ref']=spec['audio_repair_ref']
        setattr(model,ATTR,dict(spec=json.loads(json.dumps(spec)),config=config,audit=audit,
                              restore=undo,restore_imports=undo_imports,receipt=receipt,
                              audio_audit=audio_audit,audio_restore=audio_undo))
        return receipt
    except BaseException:
        if audio_undo is not None:audio_undo()
        if undo is not None:undo()
        undo_imports();raise


def status(worker):
    state=getattr(worker.model_runner.get_model(),ATTR,None)
    require(state is not None,'family precision not installed')
    return dict(state['receipt'],counters=counters(state['audit'],state['config']),audio_audit=state.get('audio_audit'))


def clear(worker):
    model=worker.model_runner.get_model();state=getattr(model,ATTR,None)
    if state is None:return dict(restored=True,installed=False)
    before=counters(state['audit'],state['config'])
    if state.get('audio_restore') is not None:state['audio_restore']()
    state['restore']()
    require(all(r['restored'] for r in state['audit']['modules'].values()),'incomplete family restoration')
    state['restore_imports']();delattr(model,ATTR)
    return dict(restored=True,installed=False,counters=before,training_admission=False)
