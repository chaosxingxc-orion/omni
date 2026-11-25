"""Measure the actual sampled S1 conditioning with the original actor processor."""
import argparse
import json
import os
from pathlib import Path
import sys
from build import pin,save
from prepare_s1_recordings import checked,read,needed
from source.config_profile import resolve,bind_environment


def measure(cfg):
    import torch
    from transformers import Qwen3OmniMoeProcessor
    from multimodal_jev.local_runtime_storage import check_paths
    from source.prompt_capacity import prompt_bound
    torch.set_num_threads(1)
    root=Path(cfg['release'])/'tree/parents'
    sys.path.insert(0,str(root/'fresh-cohort-successor'))
    from cohort_io import load
    from cohort_capture import prepare_packet
    execution_path=checked(cfg['execution']);sys.path.append(str(execution_path.parent))
    execution=load(execution_path,'panel_length_admission')
    protocol=load(root/'rollout-capture-contract/capture.py','panel_length_protocol')
    data=read(checked(cfg['inputs']));profile=read(checked(cfg['profile']))
    scope=read(checked(cfg['scope']));task=scope['task']
    needed(data['schema']=='s1-selected-inputs-v1' and task.get('node')=='S1','S1 measurements require S1 inputs')
    exp=read(checked(scope['inputs']['experiment']))
    model=exp['model_path'];processor=Qwen3OmniMoeProcessor.from_pretrained(model,local_files_only=True)
    lengths={};rows=[];limit=prompt_bound(cfg['budget'],'S1');cap=scope['window_capacity']['nodes']['S1']['completion_tokens_upper']
    context=cfg['context_limit'];needed(type(context) is int and context>0,'explicit model context required')
    out=Path(cfg['output']);check_paths({'cache':out.parent},min_free_fraction=.10)
    for row in data['rows']:
        p=read(checked(row['packet']))
        needed(p['stage']=='S1' and p['source_binding']['reference_id']==row['reference_id']
               and p['branch']==row['fold'] and p['wire']['model']==exp['model'],'different panel request')
        i=dict(group_id='cpu-measure',episode_id='cpu-measure',attempt_id='cpu-measure',replicate=0,
            seed=0,parents=[],run_id='cpu-measure',boot_id='no-service',query_id=p['sample_id'],
            split='TRAIN',node='S1',fold=p['branch'])
        sampled=protocol.candidate(p,profile,i);key=execution.PromptLengths.key(sampled)
        if key not in lengths:
            lengths[key]=int(prepare_packet(processor,sampled)['encoded']['input_ids'].shape[-1])
        length=lengths[key];max_new=sampled['wire']['max_tokens']
        rows.append(dict(recording=row['reference_id'],fold=p['branch'],key=key,prompt=length,max_new=max_new,
                         fits_prompt=length<=limit,fits_completion=max_new<=cap,fits_context=length+max_new<=context))
    needed(len(rows)==data['rendered'] and len(set(r['recording'] for r in rows))==len(rows),'incomplete measurement')
    ordered=sorted(r['prompt'] for r in rows)
    return save(out,dict(schema='omni-measured-prompt-lengths-v1',processor=execution.prompt_processor_identity(model),
        lengths=lengths,inputs_ref=cfg['inputs'],profile_ref=cfg['profile'],configuration=cfg,rows=rows,
        recordings=len(rows),min_prompt=ordered[0],p50_prompt=ordered[(len(rows)-1)//2],
        p95_prompt=ordered[(95*len(rows)+99)//100-1],max_prompt=ordered[-1],prompt_limit=limit,context_limit=context,
        max_prompt_plus_output=max(r['prompt']+r['max_new'] for r in rows),
        failures={k:sum(not r[k] for r in rows) for k in ('fits_prompt','fits_completion','fits_context')},
        provider_count_verified=False,model_calls=0))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('--profile',required=True)
    a=p.parse_args();site=read(a.profile)
    if site['schema']=='omni-local-profile-v2':site,_=bind_environment(site,os.environ)
    print(json.dumps(measure(resolve(read(a.config),site))))
