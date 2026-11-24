"""Compare the isolated scoring projection with the sealed consumer at full batch size."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path


def consume(cfg,root):
    from bounded_actor_step import generation_sources
    with generation_sources(cfg) as rt:
        from fresh_adapter import scope
        with scope():
            import consumer
            from cohort_io import read,pin,digest
            closed=read(pin(root/'COMPLETE.json'))
            start=time.monotonic()
            f,r,b=consumer.build_with_bundle(read(closed['files_ref']),read(closed['consumer_config_ref']))
            elapsed=time.monotonic()-start
            if digest(r)!=digest(read(pin(root/'CONSUMER-REPORT.json'))):
                raise ValueError('changed consumer report')
            tensors={n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(
                v.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()) for n,v in f['tensors'].items()}
            def normalize(value):
                import torch
                if isinstance(value,torch.Tensor):
                    return dict(tensor_dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(
                        value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
                if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
                if isinstance(value,(list,tuple)):return [normalize(v) for v in value]
                return value
            return dict(seconds=elapsed,report=digest(r),bundle=digest(b),tensors=tensors,
                        non_tensors=digest(normalize({n:list(v) for n,v in f['non_tensors'].items()})),meta=digest(normalize(f['meta_info'])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('spec','cohort','out'):p.add_argument(k,type=Path)
    a=p.parse_args()
    sys.path.insert(0,str(a.spec.parent))
    result=consume(json.loads(a.spec.read_bytes()),a.cohort)
    with a.out.open('x') as f:
        json.dump(result,f);f.flush();os.fsync(f.fileno())
    print(result['seconds'],flush=True)
