"""Full-width BF16 projection on the original frozen TP4 audio weight shards."""
from functools import wraps


def attach(tower, sites, rank):
    import torch
    from torch.nn.functional import linear
    from vllm.distributed import tensor_model_parallel_all_gather as gather
    if sites not in (['self_attn.out_proj'],['self_attn.out_proj','fc2']) or rank not in range(4):
        raise ValueError('unsupported frozen audio row selection')
    calls={};undo=[]
    def bind(layer,name,module):
        original=module.forward
        prior=vars(module).get('forward')
        key=f'{layer}.{name}'
        n=1280 if name=='self_attn.out_proj' else 5120
        if not (module.input_is_parallel and module.reduce_results and not module.skip_bias_add
                and module.return_bias and module.tp_rank==rank and module.tp_size==4 and
                module.weight.shape==(1280,n//4) and module.bias.shape==(1280,) and
                module.weight.dtype==module.bias.dtype==torch.bfloat16 and not module.training):
            raise ValueError('unexpected audio projection ABI: '+key)
        calls[key]=0
        # Frozen base weights; installation is bounded to the lifetime of this
        # worker/model. Do not gather weights per request.
        weight=gather(module.weight.detach().contiguous(),dim=1).contiguous()
        if weight.shape!=(1280,n) or not torch.equal(weight[:,rank*n//4:(rank+1)*n//4],module.weight):
            raise ValueError('audio weight shard order: '+key)
        @wraps(original)
        def forward(x):
            if torch.is_grad_enabled() or x.ndim!=2 or x.shape[1]!=n//4 or x.dtype!=torch.bfloat16:
                raise ValueError('frozen BF16 audio projection expected')
            joined=gather(x.contiguous(),dim=1).contiguous()
            result=linear(joined,weight,module.bias)
            calls[key]+=1
            return result,None
        module.forward=forward
        undo.append((module,prior,forward))
    def restore():
        for module,prior,owned in reversed(undo):
            if vars(module).get('forward') is not owned:raise ValueError('projection owner changed')
            if prior is None:del module.forward
            else:module.forward=prior
        undo.clear()
    try:
        with torch.no_grad():
            for i,layer in enumerate(tower.layers):
                for name in sites:bind(i,name,layer.get_submodule(name))
    except BaseException:
        restore();raise
    return calls,restore
