"""CPU-only paired publish replay using closed real groups and the live consumer."""
import argparse, json, os, sys, time
from pathlib import Path


def replay(root,scratch,spec_path,code):
    import torch
    from check_accepted_actor_packet import equal
    sys.path.insert(0,str(spec_path.parent))
    from bounded_actor_step import generation_sources
    import shutil
    from dependencies import load
    stats={}
    def run(mode,cohort_src,wrapper):
        dst=scratch/mode;dst.mkdir()
        for item in root.iterdir():
            if item.name in ('groups','SELECTED.json','LOCAL-CONSUMER.json','RESERVED-CLOCK.json','INPUT.json'):
                if item.is_dir():shutil.copytree(item,dst/item.name,copy_function=os.link)
                else:os.link(item,dst/item.name)
        with generation_sources(json.loads(spec_path.read_bytes())) as rt:
            os.environ['OMNI_CAPTURE_CONSUMER_SOURCE']=rt['consumer_source']
            os.environ['OMNI_CAPTURE_CONSUMER_WRAPPER']=str(wrapper or rt['consumer_wrapper'])
            from fresh_adapter import scope
            with scope():
                from cohort_io import read,pin
                module=load(str(cohort_src or rt['cohort_source']),'cohort')
                cfg=read(pin(root/'INPUT.json'))
                cfg['capture']['official_source_config_ref']=pin(root/'LOCAL-OFFICIAL-SOURCE.json')
                if mode=='indexed':cfg['capture']['training_input_layout']='group_refs_v1'
                def sink(folder,fields,report): saved.extend([fields,report])
                saved=[];t=time.monotonic()
                module.publish(cfg,dst,accepted_sink=sink,consumer_ref=pin(root/'LOCAL-CONSUMER.json'),row_encoding='compact_json_v1')
                stats[mode]=dict(seconds=time.monotonic()-t,online_bytes=(dst/'online.json').stat().st_size)
                print(json.dumps({mode:stats[mode]}),flush=True)
                if mode=='indexed':
                    import consumer
                    files=read(pin(dst/'FILES.json'));settings=read(pin(root/'LOCAL-CONSUMER.json'))
                    fields,report=consumer.build(files,settings)
                    compare(fields,report,*saved)
                    index=read(files['refs']['online']);index['groups'][0],index['groups'][1]=index['groups'][1],index['groups'][0]
                    path=dst/'bad-index.json';path.write_text(json.dumps(index));ref=pin(path)
                    bad=dict(files,online=str(path),refs=dict(files['refs'],online=ref))
                    try:consumer.build(bad,settings)
                    except ValueError:pass
                    else:raise ValueError('reordered index accepted')
                return saved
    def compare(f,r,g,s):
        equal(f['tensors'],g['tensors']);equal(f['meta_info'],g['meta_info'])
        equal({k:list(v) for k,v in f['non_tensors'].items()},{k:list(v) for k,v in g['non_tensors'].items()})
        def norm(report):
            return dict(report,asset_hash_counts={str(k).replace(str(scratch/'indexed'),'<replay>').replace(str(scratch/'original'),'<replay>'):v for k,v in report['asset_hash_counts'].items()})
        equal(norm(r),norm(s))
    before=run('original',None,None);after=run('indexed',code/'cohort.py',code/'consumer.py')
    compare(*before,*after)
    original=json.loads((root/'CONSUMER-REPORT.json').read_bytes())
    equal({k:v for k,v in original.items() if k!='asset_hash_counts'},json.loads(json.dumps({k:v for k,v in after[1].items() if k!='asset_hash_counts'})))
    return dict(**stats,rows=after[1]['actor_rows'],recordings=after[1]['loaded_count'],recovery_equal=True,wrong_order_rejected=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','scratch','spec','code'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.scratch.mkdir(parents=True,exist_ok=False)
    print(json.dumps(replay(a.root,a.scratch,a.spec,a.code)),flush=True)
