"""Pack one scheduled wave with the registered consumer's actual tensor code."""
from functools import lru_cache
import hashlib


@lru_cache(maxsize=2)
def packer(source,expected_sha256):
    """Extract the packing boundary; do not reimplement tensor axes, masking or dtypes."""
    from dependencies import require
    raw=source.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected_sha256,'changed registered training consumer')
    text=raw.decode().split('def build(files, config, session=None):')
    require(len(text)==2,'unknown consumer build interface')
    starts="    tensors = dict(prompts=torch.full("
    ends="    meta = dict(audio_contract=CONTRACT"
    a=text[1];require(a.count(starts)==1 and a.count(ends)==1,'unknown tensor packing boundary')
    snippet=a[a.index(starts):a.index(ends)]
    require(snippet.count("    for row,(c,m,advantage) in enumerate(selected):")==1 and
            snippet.count("    token_mapping = []")==1,'unknown actor row layout')
    header="""def pack(selected,snap,pad,pmax,rmax):
    import numpy as np
    import torch
    n=len(selected)
"""
    ns={'require':require}
    exec(compile(header+snippet+"    return tensors,non\n",str(source)+':microbatch', 'exec'),ns)
    return ns['pack']


def micro_fields(rows, geometry, *, source, consumer_sha256, snapshot, pad, context_limit):
    from dependencies import require
    pmax,rmax=geometry
    require(rows and pmax>=0 and rmax>=0 and pmax+rmax<=context_limit,'invalid cohort padding range')
    for call,identity,advantage in rows:
        f=call['training_fields']
        require(identity['group_id'] and identity['episode_id'] and
                identity['call_id']==call['call_id'] and identity['request_identity']==call['request_identity']
                and identity['node']==call['node'] and 0<len(f['action_token_ids'])<=rmax
                and len(f['prompt_token_ids'])<=pmax,'microbatch row identity or padding mismatch')
    return packer(source,consumer_sha256)(rows,snapshot,pad,pmax,rmax)


def overlap_microbatches(waves, prepare):
    """Bounded CPU lookahead. Only the caller runs device collectives/Adam.

    Yield in the original collective order, never completion order. The caller
    exhausts the iterator before optimizer_step, including any producer error.
    """
    from concurrent.futures import ThreadPoolExecutor
    iterator=iter(waves); sentinel=object()
    first=next(iterator,sentinel)
    if first is sentinel:return
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='actor-input') as pool:
        pending=pool.submit(prepare,first)
        try:
            while True:
                fields=pending.result()
                following=next(iterator,sentinel)
                pending=None if following is sentinel else pool.submit(prepare,following)
                yield fields
                if pending is None:break
        finally:
            if pending is not None:pending.cancel()


def scheduled_transport(waves,prepare,meta,keys,non_keys,*,micro,agree_ready):
    """A lazy, single-minibatch input for verl's original update_policy loop.

    CPU production cannot call distributed collectives. All ranks agree on a
    ready microbatch (including local preparation errors) in the caller thread
    before any rank enters the corresponding forward.
    """
    from concurrent.futures import ThreadPoolExecutor
    from verl import DataProto
    from dependencies import require
    n=len(waves)
    require(n and all(len(w)==micro for w in waves),'unequal collective wave widths')
    class Batch:
        meta_info=dict(meta,temperature=1.,use_dynamic_bsz=False)
        batch=dict.fromkeys(keys)
        non_tensor_batch=dict.fromkeys(non_keys)
        def select(self,batch_keys=None,non_tensor_batch_keys=None):
            self.tensor_keys=batch_keys
            self.non_keys=list(dict.fromkeys(list(non_tensor_batch_keys or [])+['record_identity','adapter_route']))
            return self
        def split(self,size):
            require(size==n*micro,'changed optimizer batch size')
            return [Mini(self)]
    class Mini:
        def __init__(self,outer):self.outer=outer
        def split(self,size):
            require(size==micro,'changed micro width')
            with ThreadPoolExecutor(max_workers=1,thread_name_prefix='actor-input') as pool:
                def pack(wave):
                    tensors,non=prepare(wave)
                    return DataProto.from_dict(tensors={k:tensors[k] for k in self.outer.tensor_keys},
                        non_tensors={k:non[k] for k in self.outer.non_keys},meta_info=self.outer.meta_info)
                pending=pool.submit(pack,waves[0])
                for i in range(n):
                    fields=agree_ready(pending.result)
                    if i+1<n:pending=pool.submit(pack,waves[i+1])
                    yield fields
    return Batch()


def run_scheduled_update(actor,data,*,reconcile,agree_ready):
    """Only commit after every rank has reconciled the stream with admission."""
    from dependencies import require
    original=actor._optimizer_step
    owned='_optimizer_step' in vars(actor)
    committed=False
    def step():
        nonlocal committed
        require(not committed,'multiple optimizer steps for one generation')
        agree_ready(reconcile)
        committed=True
        return original()
    actor._optimizer_step=step
    try:
        result=actor.update_policy(data)
        require(committed,'no complete optimizer attempt')
        return result
    except BaseException:
        actor.actor_optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        if owned:actor._optimizer_step=original
        else:del actor._optimizer_step


class CommitGate:
    """Bind consumed fields to the authoritative admission before optimizer.step.

    Keep digests only. The producer and final reader use the same exact tensor
    bytes/dtypes/shapes and non-tensor values in original collective order.
    """
    def __init__(self,waves,expected_meta):
        self.waves=waves;self.meta=expected_meta;self.observed=[];self.accepted=False

    @staticmethod
    def fingerprint(tensors,non):
        import torch,numpy as np,json
        def encode(value):
            if isinstance(value,torch.Tensor):
                if value.device.type!='cpu':raise ValueError('expected CPU transport')
                raw=value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
                return ['tensor',str(value.dtype),list(value.shape),hashlib.sha256(raw).hexdigest()]
            if isinstance(value,np.ndarray):return encode(value.tolist())
            if isinstance(value,dict):
                if any(type(k)!=str for k in value):raise ValueError('unexpected non-string transport key')
                return ['dict',[(k,encode(value[k])) for k in sorted(value)]]
            if isinstance(value,(tuple,list)):return [type(value).__name__,[encode(x) for x in value]]
            if value is not None and type(value) not in (str,float,int,bool):raise ValueError('unsupported transport type')
            return value
        blob=json.dumps(encode([tensors,non]),separators=(',',':'),allow_nan=False).encode()
        return hashlib.sha256(blob).hexdigest()

    def observe(self,slots,tensors,non):
        from dependencies import require
        i=len(self.observed)
        require(not self.accepted and i<len(self.waves) and self.waves[i]==slots,'out-of-order wave')
        self.observed.append(self.fingerprint(tensors,non))

    def reconcile(self,packet,actual_waves):
        from dependencies import require
        require(not self.accepted and not packet['meta']['synthetic'] and packet['report']['required_fields_validated']
                and actual_waves==self.waves and len(self.observed)==len(self.waves)>0,'incomplete or different cohort')
        require(all(packet['meta'][k]==v for k,v in self.meta.items()),'different full-batch loss metadata')
        for wave,observed in zip(self.waves,self.observed,strict=True):
            order=[i for i,owned in wave]
            require(self.fingerprint({k:v[order] for k,v in packet['tensors'].items()},
                        {k:[v[i] for i in order] for k,v in packet['non_tensors'].items()})==observed,
                    'provisional fields differ from accepted actor fields')
        self.accepted=True
