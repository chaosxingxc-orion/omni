"""Candidates for the fixed Thinker forward; activated only in measured scopes."""
from collections import OrderedDict
from contextlib import contextmanager
from hashlib import sha256


def audio_key(inputs):
    """Use the actual CPU feature payload, independent of text, seed and actions."""
    h=sha256()
    for row in inputs['multi_modal_inputs']:
        for name in ('input_features','feature_attention_mask'):
            x=row[name].detach().contiguous()
            if x.device.type!='cpu':raise ValueError('prepare audio keys before device transfer')
            h.update(repr((name,tuple(x.shape),x.dtype)).encode())
            h.update(x.reshape(-1).view(__import__('torch').uint8).numpy())
    return h.hexdigest()


class FrozenAudio:
    """Rank-local NPU cache, scoped to one restored worker and one arithmetic profile."""
    def __init__(self, model, max_bytes):
        if model.audio_tower.training and any(
                getattr(m,'p',0) for m in model.audio_tower.modules()
                if isinstance(m,__import__('torch').nn.modules.dropout._DropoutNd)):
            raise ValueError('stochastic tower')
        if any(p.requires_grad for p in model.audio_tower.parameters()):
            raise ValueError('audio tower must be frozen')
        self.model=model
        self.max_bytes=max_bytes
        self.entries=OrderedDict()
        self.bytes=self.hits=self.misses=0

    @contextmanager
    def use(self, key, device):
        import torch
        import torch.distributed as dist
        model=self.model
        original=model.get_audio_features
        if 'get_audio_features' in vars(model):raise ValueError('audio override already installed')
        # Tower contains FSDP collectives. Either ALL ranks call it or all use
        # their own verified entry, even when local cache contents differ.
        has=torch.tensor(int(key in self.entries),dtype=torch.int32,device=device)
        dist.all_reduce(has,op=dist.ReduceOp.MIN)
        cached=bool(has.item())
        calls=0
        def reuse(*args, **kwargs):
            nonlocal calls
            calls+=1
            if cached:
                self.hits+=1
                self.entries.move_to_end(key)
                return self.entries[key][0]
            self.misses+=1
            result=original(*args,**kwargs)
            x=result.last_hidden_state
            if x.requires_grad:raise ValueError('cannot cache an audio result carrying a graph')
            n=x.numel()*x.element_size()
            if n<=self.max_bytes:
                if key in self.entries:self.bytes-=self.entries.pop(key)[1]
                while self.bytes+n>self.max_bytes:self.bytes-=self.entries.popitem(last=False)[1][1]
                self.entries[key]=(type(result)(last_hidden_state=x.detach()),n)
                self.bytes+=n
            return result
        model.get_audio_features=reuse
        try:
            yield
            if calls!=1:raise ValueError('unexpected audio feature call count')
        finally:del model.get_audio_features


@contextmanager
def action_head(model, count):
    """Crop predictor rows before projection; still compute the entire vocabulary."""
    def select(module,args):
        x,=args
        if x.ndim!=3 or not 0<count<x.shape[1]:raise ValueError('bad action predictor suffix')
        return (x[:, -count-1:-1, :].contiguous(),)
    handle=model.lm_head.register_forward_pre_hook(select)
    try:yield
    finally:handle.remove()


def evaluate(worker, adapted, spec, out, rank, world, batch):
    """Same restored worker, real actions, differentiable objective; no optimizer step."""
    import time
    from contextlib import nullcontext
    import torch
    import torch.distributed as dist
    from bounded_actor_step import compact_scores
    from events import partition, slice_fields
    from forward import model_kwargs
    from policy_identity import tensor_hash
    from storage import commit,check_running
    _,fields,plan,_,_=batch
    actor=worker.actor
    actor.actor_module.train()
    model=adapted.get_base_model()
    layout=partition(plan,fields,world)
    schedule=layout['schedules'][rank]
    if len(schedule)!=1 or schedule[0]['kind']!='policy' or not schedule[0]['owned']:
        raise ValueError('paired probe requires one real policy row per rank')
    row=schedule[0]['row']
    source=slice_fields(fields,row)
    cpu={**source['tensors'],**source['non_tensors']}
    params=[(n,p) for n,p in actor.actor_module.named_parameters() if p.requires_grad]
    if not params or any('.lora_' not in n for n,p in params):raise ValueError('unexpected trainable parameter')
    device=next(actor.actor_module.parameters()).device
    key=audio_key(cpu)
    x={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
    x['multi_modal_inputs']=[{k:v.to(device) for k,v in m.items()} for m in cpu['multi_modal_inputs']]
    mask=x['attention_mask'][0].bool(); action_mask=x['response_mask'][0].bool()
    count=int(action_mask.sum()); width=action_mask.numel()
    compact=dict(x)
    for k in ('input_ids','attention_mask','position_ids'):compact[k]=x[k][...,mask]
    for k in ('responses','response_mask'):compact[k]=x[k][...,:count]
    cp=spec['efficiency']
    cache=FrozenAudio(model,cp['audio_cache_bytes'])
    modes=cp['modes']
    if modes[:2]!=['baseline','baseline'] or set(modes)!={'baseline','audio_fill','audio_hit','action_head'} or modes.index('audio_fill')>modes.index('audio_hit'):
        raise ValueError('need warm baseline and independent audio/head comparisons')
    identity=[(n,tensor_hash(p.detach())) for n,p in params]
    observations=[]
    original=None
    for mode in modes:
        check_running(out,spec['storage_budget_bytes'])
        actor.actor_optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        torch.npu.synchronize()
        start=time.monotonic()
        observed=[]
        def inspect_head(_, args, logits):
            sliced=logits if mode=='action_head' else logits[:, -count-1:-1, :]
            observed.append((logits.shape[1],sliced.detach()))
        h=model.lm_head.register_forward_hook(inspect_head)
        try:
            if mode=='action_head':
                kwargs,_=model_kwargs(compact,feature_dtype=torch.bfloat16)
                with action_head(model,count),torch.autocast(device_type=actor.device_name,dtype=actor.param_dtype):
                    logits=actor.actor_module(**kwargs).logits
                scores=logits.float().log_softmax(-1).gather(-1,compact['responses'].unsqueeze(-1)).squeeze(-1)
                scores=torch.nn.functional.pad(scores,(0,width-count))
            else:
                with cache.use(key,device) if mode.startswith('audio_') else nullcontext():
                    scores=compact_scores(actor,x)
            loss,_=plan.base.loss(scores,row,row+1)
            torch.npu.synchronize()
            mid=time.monotonic()
            (loss*world).backward()
            torch.npu.synchronize()
            end=time.monotonic()
        finally:h.remove()
        positions,logits=observed.pop()
        if observed:raise ValueError('unexpected output-head call count')
        logits=logits.float().cpu()
        top=logits.argmax(-1)
        s=scores.detach()[0,:count].cpu()
        grads={n:p.grad.detach().float().cpu() for n,p in params if p.grad is not None}
        if not grads or not all(torch.isfinite(g).all() for g in grads.values()):
            raise ValueError('missing or nonfinite LoRA gradient')
        if original is None:
            original=(s,top,grads,float(loss.detach()))
        standard,original_top,reference_grad,reference_loss=original
        if grads.keys()!=reference_grad.keys():raise ValueError('different gradient support')
        delta=s-standard
        norm=sum(float(g.double().square().sum()) for g in reference_grad.values())
        diff=sum(float((g-reference_grad[n]).double().square().sum()) for n,g in grads.items())
        observations.append(dict(mode=mode,forward_seconds=mid-start,backward_seconds=end-mid,
            predictor_rows=positions,actions=count,score_max=float(delta.abs().max()),
            score_sequence=float(delta.sum().abs()),top1_flips=int((top!=original_top).sum()),
            loss=float(loss.detach()),loss_delta=abs(float(loss.detach())-reference_loss),
            gradient_diff_sq=diff,baseline_gradient_sq=norm,audio_hits=cache.hits,
            audio_misses=cache.misses,audio_cache_bytes=cache.bytes))
        del logits,loss,scores,grads
    actor.actor_optimizer.zero_grad(set_to_none=True)
    if identity!=[(n,tensor_hash(p.detach())) for n,p in params]:raise ValueError('probe changed policy')
    commit(out/f'rank-{rank}/RESULT.json',dict(rank=rank,worker_constructed_once=True,observations=observations,
        optimizer_updates=0,scope='same retained actions and input fields, without optimizer mutation'))
