"""Fresh fixed-policy actions on sealed TRAIN packets. No retry, optimizer or historical scores."""
import hashlib
import json
import math
from pathlib import Path


def manifest(generated_root, groups, size):
    rows=[]
    for g in groups:
        for n in range(size):
            p=generated_root/'groups'/str(g)/f'episode-{n}'/'GENERATED.json'
            data=p.read_bytes()
            item,=[x for x in json.loads(data)['calls'] if x['packet']['stage']=='P']
            packet=item['packet']; ident=packet['fresh_capture_identity']; wire=packet['wire']
            if (ident['split']!='TRAIN' or ident['replicate']!=n or ident['query_id']!=packet['sample_id']
                    or wire['seed']!=ident['seed'] or wire['temperature']!=1 or wire['top_p']!=1
                    or wire['top_k']!=0 or wire['min_p']!=0 or wire['repetition_penalty']!=1
                    or wire['presence_penalty'] or wire['frequency_penalty'] or wire['n']!=1
                    or wire.get('ignore_eos') or wire.get('min_tokens') or wire.get('stop')
                    or wire.get('stop_token_ids') or wire.get('logit_bias') or wire.get('structured_outputs')):
                raise ValueError('unsupported action distribution or changed group identity')
            rows.append(dict(group=g,replicate=n,packet=packet,prompt_token_ids=item['prompt_token_ids'],
                             source_sha256=hashlib.sha256(data).hexdigest()))
    if len({r['packet']['request_id'] for r in rows})!=len(rows):
        raise ValueError('duplicate source request')
    return dict(rows=rows,group_size=size,groups=groups)


def sample(engine, cfg):
    from vllm import SamplingParams
    from vllm.multimodal.media.audio import AudioMediaIO
    from retained_action_replay import checked
    m=checked(Path(cfg['input']),cfg['fresh_group'])
    trace=cfg.get('fresh_trace')
    selected=m['rows']
    batch_trace=bool(trace and trace.get('batch'))
    traces={}
    if trace is not None:
        i=trace['request_index']
        if not (cfg.get('isolate_fresh_paths') or batch_trace) or type(i) is not int or not 0<=i<len(selected):
            raise ValueError('fresh trace needs one valid isolated request')
        target=selected[i]
        if not batch_trace:selected=[target]
        if trace['source_sha256']!=target['source_sha256']:
            raise ValueError('trace bound to different source')
        n=len(target['prompt_token_ids'])
        if not (trace['layers']==0 if batch_trace else 0<trace['layers']<=8) or trace['prompt_tokens']!=n or not 1<=trace['action_ordinal']<=target['packet']['wire']['max_tokens']:
            raise ValueError('invalid trace extent')
        def begin(mode):
            pos=n+trace['action_ordinal']-2+trace['rope_offset']
            args=(mode,pos,trace.get('attention_layer'),None,False,False,
                  None if batch_trace else dict(rows=n,layers=trace['layers']),2,False,trace.get('moe_sites'))
            if batch_trace:args+=(target['packet']['wire']['seed'],)
            if sorted(engine.collective_rpc('omni_predictor_begin',args=args,timeout=240))!=list(range(4)):
                raise ValueError('incomplete fresh trace launch')
        def finish(mode):
            result=engine.collective_rpc('omni_predictor_finish',timeout=240)
            if sorted(r['rank'] for r in result)!=list(range(4)):
                raise ValueError('incomplete fresh trace result')
            traces[mode]=result
    tok=engine.get_tokenizer()
    pad=tok.convert_tokens_to_ids('<|audio_pad|>')
    if pad is None or pad==tok.unk_token_id:
        raise ValueError('audio token missing')
    requests,params=[],[]
    for row in selected:
        wire=row['packet']['wire'];ids=row['prompt_token_ids']
        positions=[i for i,t in enumerate(ids) if t==pad]
        if not positions or positions!=list(range(positions[0],positions[-1]+1)):
            raise ValueError('expected exactly one audio span')
        au,=[part['input_audio'] for msg in wire['messages'] if isinstance(msg['content'],list)
             for part in msg['content'] if part['type']=='input_audio']
        wave=AudioMediaIO().load_base64('audio/'+au['format'],au['data'])
        if cfg.get('compare_actor_frame_padding'):
            import numpy as np
            samples,rate=wave
            if rate!=16000 or samples.ndim!=1:raise ValueError('expected original mono 16k')
            wave=np.pad(samples,(0,(-len(samples))%160)),rate
        requests.append(dict(prompt_token_ids=ids[:positions[0]+1]+ids[positions[-1]+1:],
                             multi_modal_data={'audio':wave}))
        params.append(SamplingParams(seed=wire['seed'],temperature=wire['temperature'],top_p=wire['top_p'],
                                     top_k=-1,min_p=0.,max_tokens=wire['max_tokens'],logprobs=1))
    isolate=cfg.get('isolate_fresh_paths',False)
    if isolate or cfg.get('independent_cache_salts'):
        for row,req in zip(selected,requests,strict=True):
            req['cache_salt']=row['packet']['request_id']
    if isolate:
        if trace:begin("decode")
        outputs=[engine.generate([req],sampling_params=param,use_tqdm=False)[0]
                 for req,param in zip(requests,params,strict=True)]
        if trace:finish("decode")
    else:
        if trace:begin('decode')
        outputs=engine.generate(requests,sampling_params=params,use_tqdm=False)
        if trace:finish('decode')
    records=[]
    for row,req in zip(selected,outputs,strict=True):
        packet=row['packet']; out,=req.outputs
        if list(req.prompt_token_ids)!=row['prompt_token_ids'] or not out.token_ids or not req.finished:
            raise ValueError('changed multimodal context or incomplete output')
        if cfg.get('independent_cache_salts') and req.num_cached_tokens!=0:
            raise ValueError('independent cache salt had a shared hit')
        scores=[];top=[]
        for t,lp in zip(out.token_ids,out.logprobs,strict=True):
            scores.append(lp[t].logprob);top.append(next(i for i,v in lp.items() if v.rank==1))
        if not all(map(math.isfinite,scores)):
            raise ValueError('nonfinite sampled probability')
        records.append(dict(group=row['group'],replicate=row['replicate'],source_sha256=row['source_sha256'],
                            request_id=packet['request_id'],wire_sha256=packet['wire_sha256'],
                            seed=packet['wire']['seed'],prompt_tokens=len(req.prompt_token_ids),
                            token_ids=list(out.token_ids),behavior_logprobs=scores,raw_top1=top,
                            text=out.text,finish_reason=out.finish_reason,cached_tokens=req.num_cached_tokens))
    if isolate or cfg.get('fresh_prefill_control'):
        for row,req,r in zip(selected,requests,records,strict=True):
            tracing=trace and row['source_sha256']==trace['source_sha256']
            if batch_trace and not tracing:continue
            ids=r['token_ids'];prompt=dict(req,prompt_token_ids=req['prompt_token_ids']+ids)
            if tracing:
                if len(ids)<trace["action_ordinal"]:raise ValueError("requested action absent")
                begin("prefill")
            probe,=engine.generate([prompt],sampling_params=SamplingParams(temperature=1.,top_p=1.,top_k=-1,
                           max_tokens=1,prompt_logprobs=1,seed=r['seed']),use_tqdm=False)
            if list(probe.prompt_token_ids)!=row['prompt_token_ids']+ids or r['cached_tokens']!=0:
                raise ValueError('isolated control not cold or prompt changed')
            if tracing:
                if probe.num_cached_tokens!=0:raise ValueError('scoring prefill read cache')
                finish("prefill")
                r['predictor_trace']=dict(config=trace,**traces)
            per=probe.prompt_logprobs[-len(ids):]
            r['forced_scores']=[lp[t].logprob for lp,t in zip(per,ids,strict=True)]
            r['forced_top1']=[next(k for k,v in lp.items() if v.rank==1) for lp in per]
    return dict(schema='fresh-train-p-predictor-trace-v1' if trace else 'fresh-train-p-parity-group-v1',records=records,
                manifest_sha256=cfg['fresh_group']['sha256'],optimizer_steps=0)


def hf_scores(worker, adapted, manifest, sampled, manifest_hash, rank, world, trace_group=None, origin_indices=None, router_shadow=False):
    """One owned request per rank per FSDP collective; no resampling or weight update."""
    import sys
    import torch
    from dependencies import ROOT
    from diagnostic_logits import observe
    from forward import forward
    from validate_npu import inputs_for
    sys.path.insert(0,str(ROOT/'fresh-cohort-successor'))
    from cohort_capture import prepare_packet
    if sampled['manifest_sha256']!=manifest_hash or len(sampled['records'])!=len(manifest['rows']):
        raise ValueError('fresh action / input manifest mismatch')
    paired=list(zip(manifest['rows'],sampled['records'],strict=True))
    if not paired or len(paired)%world:
        raise ValueError('equal number of real FSDP calls required across ranks')
    cache={};results=[]
    indices=range(rank,len(paired),world) if origin_indices is None else origin_indices
    if origin_indices is not None and (len(indices)!=2 or len(set(indices))!=2 or
                  not all(type(i) is int and 0<=i<len(paired) and paired[i][0]['group']==trace_group for i in indices)):
        raise ValueError('expected two authentic actions of the same recording')
    for index in indices:
        row,r=paired[index]; p=row['packet']
        if (r['request_id']!=p['request_id'] or r['wire_sha256']!=p['wire_sha256']
                or r['source_sha256']!=row['source_sha256'] or r['replicate']!=row['replicate']
                or r['group']!=row['group'] or r['seed']!=p['wire']['seed']):
            raise ValueError('fresh action does not own this input')
        if row['group'] not in cache:cache[row['group']]=prepare_packet(worker.processor,p)
        prep=cache[row['group']]
        if prep['encoded']['input_ids'][0].tolist()!=row['prompt_token_ids']:
            raise ValueError('HF processor IDs differ from expanded vLLM prompt')
        sample=inputs_for(prep,r['token_ids'],adapted,torch.device('npu',rank))
        policy=dict(schema='omni-response-logits-v1',vocabulary=152064,
                    max_bytes=p['wire']['max_tokens']*152064*2)
        hooks=[]; boundary={}
        if row['group']==trace_group:
            from policy_identity import tensor_hash
            base=adapted.get_base_model()
            def record(name,x):
                if name in boundary:raise ValueError('duplicate HF boundary '+name)
                t=x.detach().cpu().contiguous()
                boundary[name]=dict(shape=list(t.shape),sha256=tensor_hash(t))
            def unwrap(m):
                while type(m).__name__ in ('CheckpointWrapper','FullyShardedDataParallel'):
                    m=m._checkpoint_wrapped_module if type(m).__name__=='CheckpointWrapper' else m.module
                return m
            for i,layer in enumerate(base.audio_tower.layers):
                hooks.append(unwrap(layer).register_forward_hook(
                    lambda m,a,v,k=f'audio.{i}':record(k,v[0] if isinstance(v,tuple) else v)))
            hooks.append(unwrap(base.audio_tower).register_forward_hook(
                lambda m,a,v:record('audio_out',v.last_hidden_state)))
            # Full prompt digest excludes sampled action suffix. Also record each
            # action predictor independently so variable responses remain paired.
            n=len(row['prompt_token_ids'])
            def mark(name,t):
                record(name+'.prompt',t[0,:n])
                record(name+'.predictors',t[0,n-1:n-1+len(r['token_ids'])])
            hooks.append(unwrap(base.model.layers[0]).register_forward_pre_hook(lambda m,a,kw:
                mark('decoder.input',a[0] if a else kw['hidden_states']),with_kwargs=True))
            for i,layer in enumerate(base.model.layers):
                for label,mod in [('in',unwrap(layer).input_layernorm),('post',unwrap(layer).post_attention_layernorm)]:
                    hooks.append(mod.register_forward_hook(lambda m,a,v,k=f'{i}.{label}':mark(k,v)))
        if origin_indices is not None:
            layer=unwrap(base.model.layers[1])
            if router_shadow:
                def shadow_router(m,args,v):
                    import torch_npu
                    x=args[0].reshape(-1,2048).contiguous();w=m.weight
                    if (x.dtype!=torch.bfloat16 or tuple(w.shape)!=(128,2048)
                            or w.dtype!=x.dtype or m.training or torch.is_grad_enabled()):
                        raise ValueError('expected frozen BF16 actual router operands')
                    n=len(row['prompt_token_ids'])
                    record('L1.actual_router_input',x[:n])
                    record('L1.actual_router_weight',w)
                    wt=w.T.contiguous().unsqueeze(0)
                    def project(y):
                        return torch_npu.npu_grouped_matmul(x=[y.contiguous()],weight=[wt],
                            split_item=2,group_list_type=1,group_type=0,
                            group_list=torch.tensor([len(y)],device=x.device,dtype=torch.int64))[0]
                    old=torch.nn.functional.linear(x,w)
                    if not torch.equal(old,v[0]):raise ValueError('same-input native router did not reproduce')
                    new=project(x)
                    record('L1.shadow_router_prompt',new[:n])
                    target=2491
                    one=project(x[target:target+1])
                    native_one=torch.nn.functional.linear(x[target:target+1],w)
                    boundary['L1.router_shadow_target']=dict(grouped_full_single_equal=torch.equal(new[target],one[0]),
                        native_full_single_equal=torch.equal(old[target],native_one[0]),
                        native_vs_grouped_unequal=int(torch.count_nonzero(old[target]!=new[target])),
                        grouped=tensor_hash(new[target]),native=tensor_hash(old[target]))
                hooks.append(unwrap(layer.mlp.gate).register_forward_hook(shadow_router))
            for name,mod in [('attention',layer.self_attn),('mlp',layer.mlp),('router',layer.mlp.gate),('experts',layer.mlp.experts)]:
                def observe_op(m,a,v,k=name):
                    values=v if isinstance(v,tuple) else (v,)
                    for j,t in enumerate(values):
                        if not isinstance(t,torch.Tensor):continue
                        if t.shape[:2]==(1,len(row['prompt_token_ids'])+len(r['token_ids'])):
                            prompt=t[0,:len(row['prompt_token_ids'])]
                        elif t.shape[0]==len(row['prompt_token_ids'])+len(r['token_ids']):
                            prompt=t[:len(row['prompt_token_ids'])]
                        else:continue
                        record('L1.'+k+'.'+str(j),prompt)
                        # Per-position digests find first shared row without storing activations.
                        boundary['L1.'+k+'.'+str(j)]['row_sha256']=[
                            hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                            for x in prompt.detach().cpu().contiguous()]
                hooks.append(unwrap(mod).register_forward_hook(observe_op))
        try:
            with torch.no_grad(),observe(worker.actor.actor_module,sample,policy) as cap:
                _, scores=forward(worker.actor,sample,temperature=p['wire']['temperature'],feature_dtype=torch.bfloat16)
        finally:
            for hook in hooks:hook.remove()
        if trace_group is not None and row['group']==trace_group and len([k for k in boundary if not k.startswith('L1.')])!=227:
            raise ValueError('incomplete HF trace '+str(len(boundary)))
        h=scores[0].float().cpu().tolist()
        delta=[a-b for a,b in zip(h,r['behavior_logprobs'],strict=True)]
        top=cap['tensor'].argmax(-1).tolist()
        flips=[j+1 for j,(a,b) in enumerate(zip(top,r['raw_top1'],strict=True)) if a!=b]
        logits=cap['tensor']
        inadmissible=[j+1 for j,t in enumerate(r['raw_top1']) if logits[j,t]!=logits[j,top[j]]]
        results.append(dict(boundaries=boundary,inadmissible_top1=inadmissible,index=index,group=r['group'],replicate=r['replicate'],tokens=len(h),
            request_id=r['request_id'],max_abs=max(map(abs,delta)),sequence_abs=abs(math.fsum(delta)),
            above_limit=[j+1 for j,v in enumerate(delta) if abs(v)>.01],top1_flips=flips,
            scores=h,hf_top1=top,delta=delta))
    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--groups',nargs='+',type=int,required=True);p.add_argument('--size',type=int,required=True)
    a=p.parse_args()
    with a.output.open('x') as f:
        json.dump(manifest(a.source,a.groups,a.size),f)
