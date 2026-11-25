"""Paired original/candidate scorer on the same accepted training groups in a CPU replica."""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path

def replay(root,original,candidate):
    from check_accepted_actor_packet import equal
    import torch
    torch.set_num_threads(1)
    from dependencies import load
    from bounded_actor_step import generation_sources
    import fresh_adapter
    with generation_sources(json.loads(Path('/sealed/TRAIN.local.json').read_bytes())) as rt:
        os.environ['OMNI_CAPTURE_CONSUMER_SOURCE']=rt['consumer_source']
        os.environ['OMNI_CAPTURE_CONSUMER_WRAPPER']=rt['consumer_wrapper']
        def once(path):
            load(str(path),'scorer_extension')
            with fresh_adapter.scope():
                from cohort_io import read,pin
                load(rt['cohort_source'],'cohort')
                mod=fresh_adapter.consumer(load(rt['consumer_wrapper'],'replay_consumer'))
                cfg=read(pin(root/'LOCAL-CONSUMER.json')); files=read(pin(root/'FILES.json'))
                start=time.monotonic(); fields,report,_=mod.build_with_bundle(files,cfg)
                return time.monotonic()-start,dict(tensors=fields['tensors'],non={k:list(v) for k,v in fields['non_tensors'].items()},meta=fields['meta_info'],report=report)
        first,ref=once(original)
        timings=[]
        for src in (candidate,original):
            gc.collect();elapsed,out=once(src);equal(out,ref);timings.append(elapsed);del out
        source=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        return dict(original_sha256=source(original),candidate_sha256=source(candidate),original_first=first,
            candidate_seconds=timings[0],original_repeat_seconds=timings[1],equal=True,rows=ref['report']['actor_rows'])
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','original','candidate'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,'/sealed')
    print(json.dumps(replay(a.root,a.original,a.candidate)))
