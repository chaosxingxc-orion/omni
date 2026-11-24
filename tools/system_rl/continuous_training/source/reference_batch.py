"""Frozen-reference scoring for an already sampled window on resident FSDP.

All ranks receive the same roster of unique keys and lengths. Only the assigned
rank prepares model inputs. Tail ranks participate in FSDP with a read-only
duplicate; padding never enters the score map, loss denominator, or trajectory.
"""
import math


def schedule(lengths, world, rows, tokens):
    if any(type(n) is not int or n <= 0 for n in (world, rows, tokens)):
        raise ValueError('positive batch limits required')
    by_length = []
    for key, (p, a) in lengths.items():
        if type(p) is not int or type(a) is not int or min(p,a) < 1 or p+a > tokens:
            raise ValueError('invalid score lengths or single input exceeds budget')
        by_length.append((p, a, key))
    # Balance length buckets across all ranks. The token limit covers actual
    # padded input (prompt and reply maxima can come from different rows).
    waves = []
    bucket = []
    pmax = amax = 0
    def flush():
        if not bucket:
            return
        waves.append([tuple(bucket[k::world]) for k in range(world)])
    for p,a,key in sorted(by_length):
        np,na=max(pmax,p),max(amax,a)
        if bucket and (len(bucket)==rows*world or
                       (np+na)*math.ceil((len(bucket)+1)/world)>tokens):
            flush()
            bucket=[]; np,na=p,a
        bucket.append(key)
        pmax,amax=np,na
    flush()
    return waves


def collate(singletons, pad_id):
    """Left-pad prompt/right-pad actions; retain real three-axis RoPE and audio."""
    import torch
    if not singletons: raise ValueError('empty score batch')
    ps=[x['input_ids'].shape[-1]-x['responses'].shape[-1] for x in singletons]
    acts=[x['responses'].shape[-1] for x in singletons]
    pmax,amax=max(ps),max(acts)
    if min(ps+acts)<=0: raise ValueError('empty prompt/action')
    n=len(ps); base=singletons[0]['input_ids']
    ids=base.new_full((n,pmax+amax),pad_id)
    attention=base.new_zeros(ids.shape); pos=base.new_zeros((n,3,ids.shape[-1]))
    mm=[]
    for i,(x,p,a) in enumerate(zip(singletons,ps,acts,strict=True)):
        if (x['input_ids'].shape!=(1,p+a) or x['position_ids'].shape!=(1,3,p+a)
                or len(x['multi_modal_inputs'])!=1
                or not torch.equal(x['input_ids'][:,p:],x['responses'])
                or not torch.all(x['attention_mask']==1) or not torch.all(x['response_mask']==1)):
            raise ValueError('single unpadded sampled sequence required')
        sl=slice(pmax-p,pmax+a)
        ids[i,sl]=x['input_ids'][0]; attention[i,sl]=1; pos[i,:,sl]=x['position_ids'][0]
        mm.extend(x['multi_modal_inputs'])
    return dict(input_ids=ids,attention_mask=attention,position_ids=pos,
                responses=ids[:,pmax:],response_mask=attention[:,pmax:],multi_modal_inputs=mm)


def score_missing(worker,adapted,lengths,make_input,*,rows,tokens,pad_id):
    """Return one validated action vector per missing key; never sample a token."""
    import torch
    import torch.distributed as dist
    from forward import forward
    from resident import collectively,agree
    rank,world=dist.get_rank(),dist.get_world_size()
    waves=schedule(lengths,world,rows,tokens)
    agree(dict(roster=sorted(lengths.items()),waves=waves))
    out={}; padding=0
    module=worker.actor.actor_module
    training=module.training
    try:
        module.eval()
        with torch.no_grad(),adapted.disable_adapter():
            for wave in waves:
                assigned=wave[rank]; keys=assigned or wave[0][:1]
                batch=collectively(lambda:collate([make_input(k) for k in keys],pad_id))
                _,scores=forward(worker.actor,batch,temperature=1.,feature_dtype=torch.bfloat16)
                handle=getattr(module,'_handle',None)
                if handle is not None: handle.reshard(True)
                def extract():
                    vals=scores.detach().float().cpu().tolist(); local={}
                    if len(vals)!=len(keys):raise ValueError('score row count changed')
                    for key,v in zip(keys,vals,strict=True):
                        a=lengths[key][1]; v=v[:a]
                        if len(v)!=a or any(not math.isfinite(x) or x>0 for x in v):
                            raise ValueError('invalid reference log probabilities')
                        if assigned: local[key]=v
                    return local
                local=collectively(extract)
                peers=[None]*world
                dist.all_gather_object(peers,local)
                for expected,got in zip(wave,peers,strict=True):
                    if set(expected)!=set(got) or set(out).intersection(got):
                        raise ValueError('score ownership drift')
                    out.update(got)
                padding+=sum(not group for group in wave)
    finally:
        module.train(training)
    if set(out)!=set(lengths):raise ValueError('incomplete frozen reference')
    return out,dict(unique_scored=len(out),collective_forwards=len(waves),tail_participation_rows=padding)


def collect_window(generated, conditioning_key):
    """Scoped to one frozen reference/processor/runtime; no cross-window aliasing."""
    import hashlib,json
    lengths={}; representative={}; requests={}
    for group in generated:
        for episode in group['episodes']:
            for item in episode['calls']:
                packet=item['packet']; rid=packet['request_id']
                if rid in requests:raise ValueError('request occurs twice')
                actions=item['token_ids']; prompt=item['prompt_token_ids']
                if len(actions)!=len(item['behavior_logprobs']) or not prompt:
                    raise ValueError('sampled token evidence incomplete')
                if not actions:
                    requests[rid]=None
                    continue
                payload=[packet['stage'],conditioning_key(packet),prompt,actions]
                key=hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()
                requests[rid]=key
                lengths[key]=(len(prompt),len(actions))
                representative.setdefault(key,item)
    return lengths,representative,requests


def frozen_reference_window(worker, adapted, generated, prepared_cache, inputs_for, limits, *, cache_root, namespace):
    """Real full-window reference operator; same actor weights, disabled LoRA."""
    lengths,items,requests=collect_window(generated,prepared_cache.conditioning_key)
    device=next(worker.actor.actor_module.parameters()).device
    def make_input(key):
        item=items[key]
        prepared=prepared_cache.prepare(item['packet'])
        if prepared['encoded']['input_ids'][0].tolist()!=item['prompt_token_ids']:
            raise ValueError('rollout/processor prompt mismatch')
        return inputs_for(prepared,item['token_ids'],adapted,device)
    # Read/write one immutable table per cohort rather than 81 small files.
    # A resumed partial cohort may present a subset of the original request roster.
    from pathlib import Path
    from run_package import read,write
    from policy_identity import digest
    from resident import collectively
    import torch.distributed as dist
    root=Path(cache_root)/'reference-batches'/digest(namespace)
    table={}
    def read_tables():
        if dist.get_rank()!=0:return None
        values={}
        for path in sorted(root.glob('*.json')):
            obj=read(path)
            if obj['namespace']!=namespace or digest(obj)!=path.stem:raise ValueError('reference table drift')
            for key,value in obj['scores'].items():
                if key in lengths:
                    if (len(value)!=lengths[key][1] or any(type(v) not in (int,float) or not math.isfinite(v) or v>0 for v in value)
                        or key in values and values[key]!=value):raise ValueError('invalid cached reference')
                    values[key]=value
        return values
    event=[collectively(read_tables)]
    dist.broadcast_object_list(event,src=0);table=event[0]
    missing={k:v for k,v in lengths.items() if k not in table}
    fresh,stats=score_missing(worker,adapted,missing,make_input,rows=limits['rows_per_rank'],
        tokens=limits['padded_tokens_per_rank'],pad_id=worker.tokenizer.pad_token_id)
    def store():
        if fresh and dist.get_rank()==0:
            root.mkdir(parents=True,exist_ok=True)
            obj=dict(namespace=namespace,scores=fresh)
            write(root/(digest(obj)+'.json'),obj)
    collectively(store)
    unique=table|fresh
    stats.update(requests=len(requests),persisted_hits=len(table),reused=sum(k is not None for k in requests.values())-len(fresh))
    rows={}
    for rid,key in requests.items():
        source=items[key]['packet']['request_id'] if key is not None else rid
        rows[rid]=(unique[key] if key is not None else [],
                   dict(hit=source!=rid or key in table,source_request_id=source))
    return rows,stats
