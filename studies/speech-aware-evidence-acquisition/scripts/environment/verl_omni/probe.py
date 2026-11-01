"""Offline CPU/meta probe: no checkpoint tensors, no forward/backward, no NPU work."""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import time
import traceback

TARGET = r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'

def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def probe(model_path):
    report={'schema':'omni-verl-environment-probe-v1','time':time.time(),
            'weights_loaded':False,'model_forward_calls':0,'backward_calls':0,'npu_work':False,
            'training_ready':False,'checks':{},'packages':{}}
    for name in ('torch','torch-npu','transformers','peft','accelerate','verl','vllm','vllm-ascend','numpy'):
        try:report['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:report['packages'][name]=None
    def check(name, fn):
        try:
            report['checks'][name]={'ok':True,'result':fn()}
        except Exception as error:
            report['checks'][name]={'ok':False,'error_type':type(error).__name__,'error':str(error)[:1200]}
    for name in ('torch_npu','verl','verl.utils.model','verl.trainer.sft_trainer','verl.trainer.main_ppo'):
        check('import:'+name,lambda name=name: {'module':name,'file':getattr(importlib.import_module(name),'__file__',None)})
    report['model_config_sha256']=fingerprint(model_path/'config.json')
    report['weight_index_sha256']=fingerprint(model_path/'model.safetensors.index.json')
    def structure():
        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
        from peft import LoraConfig, TaskType, get_peft_model
        cfg=AutoConfig.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        if cfg.model_type!='qwen3_omni_moe':raise ValueError('exact Omni base config required')
        tc=cfg.thinker_config
        tc.architectures=['Qwen3OmniMoeThinkerForConditionalGeneration']
        tc._attn_implementation='sdpa';tc.text_config._attn_implementation='sdpa'
        with init_empty_weights(include_buffers=True):
            model=Qwen3OmniMoeThinkerForConditionalGeneration(tc)
            names=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and re.fullmatch(TARGET,n)]
            if len(names)!=4*tc.text_config.num_hidden_layers:raise ValueError('incomplete text-only attention target set')
            candidate=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,
                                                     lora_dropout=0.0,target_modules=names,exclude_modules=r'(audio_tower|visual)\..*',bias='none'))
        trainable={n:p.numel() for n,p in candidate.named_parameters() if p.requires_grad}
        if not trainable or any('lora_' not in n or '.model.layers.' not in n or '.self_attn.' not in n for n in trainable):
            raise ValueError('unexpected trainables: '+repr([n for n in trainable if 'lora_' not in n or '.model.layers.' not in n or '.self_attn.' not in n][:12])+'; total='+str(len(trainable)))
        if any(p.device.type!='meta' for p in candidate.parameters()):raise ValueError('probe must allocate no parameter tensors')
        processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        model_source=Path(inspect.getfile(Qwen3OmniMoeThinkerForConditionalGeneration))
        return {'class':type(model).__name__,'processor':type(processor).__name__,
                'text_layers':tc.text_config.num_hidden_layers,'target_modules':names,
                'trainable_parameters':sum(trainable.values()),'trainable_tensors':len(trainable),
                'target_regex':TARGET,'all_parameters_meta':True,'audio_tower_retained':hasattr(model,'audio_tower'),
                'transformers_model_source_sha256':fingerprint(model_source),
                'forward_parameters':list(inspect.signature(model.forward).parameters),
                'scope':'Meta structure only; no weight loading, audio batch, backward, FSDP, checkpoint or rollout sync proof.'}
    check('thinker_meta_peft',structure)
    def vanilla():
        from transformers import AutoConfig
        from verl.utils.model import get_hf_auto_model_class
        cfg=AutoConfig.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        cls=get_hf_auto_model_class(cfg)
        return {'selected':cls.__name__,'scope':'Selection only, not a successful full-model training load.'}
    check('vanilla_verl_full_config_selection',vanilla)
    report['probe_checks_passed']=all(v['ok'] for v in report['checks'].values())
    report['remaining_gates']=['Thinker loading from original prefixed checkpoint: missing/unexpected keys and base tensor equality.',
        'Original audio processor, audio token expansion, Omni position IDs and response-only loss mask.',
        'Actual verl FSDP nonzero finite LoRA gradient and frozen-base invariance.',
        'Checkpoint/resume equivalence including optimizer/RNG/sampler.',
        'vLLM-Ascend adapter load, versioned rollout and train/rollout log-prob consistency.']
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('fresh output required')
    try:result=probe(a.model)
    except Exception as error:
        result={'training_ready':False,'probe_checks_passed':False,'fatal':type(error).__name__,'error':str(error)[:1200]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'probe_checks_passed':result['probe_checks_passed'],'training_ready':False}))
    raise SystemExit(0 if result['probe_checks_passed'] else 1)
