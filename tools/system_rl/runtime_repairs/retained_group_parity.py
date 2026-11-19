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
    tok=engine.get_tokenizer()
    pad=tok.convert_tokens_to_ids('<|audio_pad|>')
    if pad is None or pad==tok.unk_token_id:
        raise ValueError('audio token missing')
    requests,params=[],[]
    for row in m['rows']:
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
    if isolate:
        for row,req in zip(m['rows'],requests,strict=True):
            req['cache_salt']=row['packet']['request_id']
        outputs=[engine.generate([req],sampling_params=param,use_tqdm=False)[0]
                 for req,param in zip(requests,params,strict=True)]
    else:
        outputs=engine.generate(requests,sampling_params=params,use_tqdm=False)
    records=[]
    for row,req in zip(m['rows'],outputs,strict=True):
        packet=row['packet']; out,=req.outputs
        if list(req.prompt_token_ids)!=row['prompt_token_ids'] or not out.token_ids or not req.finished:
            raise ValueError('changed multimodal context or incomplete output')
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
    if isolate:
        for row,req,r in zip(m['rows'],requests,records,strict=True):
            ids=r['token_ids'];prompt=dict(req,prompt_token_ids=req['prompt_token_ids']+ids)
            probe,=engine.generate([prompt],sampling_params=SamplingParams(temperature=1.,top_p=1.,top_k=-1,
                           max_tokens=1,prompt_logprobs=1,seed=r['seed']),use_tqdm=False)
            if list(probe.prompt_token_ids)!=row['prompt_token_ids']+ids or r['cached_tokens']!=0:
                raise ValueError('isolated control not cold or prompt changed')
            per=probe.prompt_logprobs[-len(ids):]
            r['forced_scores']=[lp[t].logprob for lp,t in zip(per,ids,strict=True)]
            r['forced_top1']=[next(k for k,v in lp.items() if v.rank==1) for lp in per]
    return dict(schema='fresh-train-p-parity-group-v1',records=records,
                manifest_sha256=cfg['fresh_group']['sha256'],optimizer_steps=0)


def hf_scores(worker, adapted, manifest, sampled, manifest_hash, rank, world):
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
    for index in range(rank,len(paired),world):
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
        with torch.no_grad(),observe(worker.actor.actor_module,sample,policy) as cap:
            _, scores=forward(worker.actor,sample,temperature=p['wire']['temperature'],feature_dtype=torch.bfloat16)
        h=scores[0].float().cpu().tolist()
        delta=[a-b for a,b in zip(h,r['behavior_logprobs'],strict=True)]
        top=cap['tensor'].argmax(-1).tolist()
        flips=[j+1 for j,(a,b) in enumerate(zip(top,r['raw_top1'],strict=True)) if a!=b]
        logits=cap['tensor']
        inadmissible=[j+1 for j,t in enumerate(r['raw_top1']) if logits[j,t]!=logits[j,top[j]]]
        results.append(dict(inadmissible_top1=inadmissible,index=index,group=r['group'],replicate=r['replicate'],tokens=len(h),
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
