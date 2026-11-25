"""Create an independent node's step-zero adapter with the training model and PEFT."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from build import pin, save
from prepare_s1_recordings import read, checked, needed, sha
from source.config_profile import resolve, bind_environment


def prepare(cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from multimodal_jev.omni_fsdp_loading import build_meta_thinker
    from multimodal_jev.local_runtime_storage import check_paths
    from paired_experiment import rank_profile
    from rank_profile import validate, load_initial_publication
    from policy_identity import canonical
    from importlib.metadata import version
    import re

    needed(cfg['schema']=='independent-base-adapter-v1' and type(cfg['seed']) is int
           and 0<=cfg['seed']<2**63,'invalid base initialization')
    torch.set_num_threads(1)
    scope=read(checked(cfg['scope']))
    task=scope['task']
    needed(task['schema']=='independent-node-training-v1' and task['initialization']=='frozen_base'
           and scope['scope']=='node' and scope['trainable_mask']=={n:n==task['node'] for n in ('P','S1','S2','D')},
           'independent frozen-base node required')
    exp=read(checked(scope['inputs']['experiment']));rank=validate(rank_profile(exp))
    base=read(checked(cfg['base_manifest']))
    needed(base['status']=='CLOSED_IMMUTABLE' and base['fixture'] is False and base['model']==exp['model'],'wrong base seal')
    root=Path(exp['model_path']); refs={Path(r['path']).name:r for r in base['assets']}
    needed(len(refs)==len(base['assets']) and all(Path(r['path']).parent==root for r in refs.values()),'model roots differ')
    model_config=read(checked(refs['config.json']))
    index=read(checked(refs['model.safetensors.index.json']))['weight_map']
    profile=cfg['projection_names']
    needed(isinstance(profile,list) and len(profile)==len(set(profile)) and set(profile)=={'q_proj','k_proj','v_proj','o_proj'},
           'actor currently supports the four text attention projections')
    out=Path(cfg['output']); budget=cfg['storage_budget_bytes']
    needed(type(budget) is int and budget>0,'invalid storage budget')
    check_paths({'adapter_root':out.parent},min_free_fraction=.10,min_free_bytes=20*1024**3+budget)
    identity=dict(scope=cfg['scope'],base_manifest=cfg['base_manifest'],seed=cfg['seed'],projections=profile,
                  loader=pin(sys.modules[load_initial_publication.__module__].__file__)['sha256'],
                  peft=version('peft'),torch=torch.__version__)
    with (out.parent/(out.name+'.guard')).open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists():
            report=read(out/'ORIGIN.json')
            needed(report['inputs']==identity and all(sha(out/k)==v for k,v in report['files'].items()),'changed adapter or inputs')
            return pin(out/'ORIGIN.json')
        model=build_meta_thinker(model_config); model.config.use_cache=False
        targets=sorted(n for n,_ in model.named_modules() if re.fullmatch(r'model\.layers\.\d+\.self_attn\.\w+',n)
                       and n.rsplit('.',1)[-1] in profile)
        needed(len(targets)==model_config['thinker_config']['text_config']['num_hidden_layers']*len(profile)
               and all(index['thinker.'+n+'.weight'] in refs for n in targets),'actual actor targets differ from sealed index')
        lora=LoraConfig(r=rank['lora_rank'],lora_alpha=rank['lora_alpha'],lora_dropout=rank['lora_dropout'],
            bias='none',target_modules=targets,exclude_modules=r'.*(audio_tower|visual|projector|merger).*',task_type=None)
        adapted=get_peft_model(model,lora)
        gen=torch.Generator(device='cpu').manual_seed(cfg['seed'])
        params={canonical(n):p for n,p in adapted.named_parameters() if p.requires_grad}
        needed(len(params)==2*len(targets) and all(p.is_meta for p in params.values()),'unexpected trainable parameters')
        weights={}
        for target in targets:
            stem='base_model.model.'+target+'.lora_'
            a,b=(params[stem+x+'.weight'] for x in 'AB')
            needed(a.shape[0]==b.shape[1]==rank['lora_rank'] and a.ndim==b.ndim==2,'actor rank differs')
            weights[stem+'A.weight']=torch.nn.init.kaiming_uniform_(torch.empty(tuple(a.shape)),a=5**.5,generator=gen).bfloat16()
            weights[stem+'B.weight']=torch.zeros(tuple(b.shape),dtype=torch.bfloat16)
        needed(sum(v.numel()*v.element_size() for v in weights.values())+(1<<20)<=budget,'adapter exceeds storage budget')
        with tempfile.TemporaryDirectory(prefix='.node-init-',dir=out.parent) as tmp:
            stage=Path(tmp)/'publish';stage.mkdir(mode=0o700)
            lora.base_model_name_or_path=str(root);lora.save_pretrained(stage)
            weightfile=stage/'adapter_model.safetensors';save_file(weights,str(weightfile))
            # Only LoRA parameters receive real storage; use the production actor reader.
            for n,p in list(adapted.named_parameters()):
                if p.requires_grad:
                    parent,name=n.rsplit('.',1)
                    setattr(adapted.get_submodule(parent),name,torch.nn.Parameter(torch.empty(tuple(p.shape),dtype=torch.float32)))
            load_initial_publication(adapted,stage,rank)
            for n,p in adapted.named_parameters():
                if p.requires_grad:
                    value=weights[canonical(n)]
                    needed(torch.equal(p,value.float()) and torch.isfinite(p).all().item(),'actor did not consume exact BF16')
                    needed((torch.count_nonzero(p).item()==0)==('.lora_B.' in n),'expected zero B and nonzero A')
                else:needed(p.is_meta,'unexpected base tensor allocation')
            files={n:sha(stage/n) for n in ('adapter_config.json','adapter_model.safetensors')}
            for n in files:
                with (stage/n).open('rb') as f:os.fsync(f.fileno())
            save(stage/'ORIGIN.json',dict(schema='independent-node-base-adapter-origin-v1',inputs=identity,files=files,
                node=task['node'],branch_id=task['branch_id'],model=base['model'],rank=rank['lora_rank'],
                layers=model_config['thinker_config']['text_config']['num_hidden_layers'],projections=len(targets),
                loaded_tensors=len(weights),actor_max_abs=0,zero_B=len(targets),optimizer_inherited=False,
                formal_updates=0,base_weights_loaded=False,npu_validation=False))
            os.rename(stage,out)
            fd=os.open(out.parent,os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        return pin(out/'ORIGIN.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('--profile',required=True)
    a=p.parse_args();site=read(a.profile)
    if site['schema']=='omni-local-profile-v2':site,_=bind_environment(site,os.environ)
    cfg=resolve(read(a.config),site)
    sys.path.append(str(Path(cfg['implementation'])))
    print(json.dumps(prepare(cfg)))

if __name__=='__main__':main()
