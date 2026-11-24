"""Compare historical build + restore with the candidate in original immutable mounts."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from check_accepted_actor_packet import equal

def run(a):
    sys.path.insert(0,str(a.spec.parent))
    from bounded_actor_step import generation_sources
    from run_package import write
    times={}
    def timed(key,work):
        start=time.monotonic(); result=work();times[key]=time.monotonic()-start;print(key,times[key],flush=True);return result
    spec=json.loads(a.spec.read_bytes())
    with generation_sources(spec) as runtime:
        from fresh_adapter import scope
        for key in ('consumer_source','consumer_wrapper'):os.environ['OMNI_CAPTURE_'+key.upper()]=runtime[key]
        with scope():
            from cohort_io import read,pin,digest
            import consumer
            receipt=read(pin(a.root/'COMPLETE.json'))
            files=read(receipt['files_ref']);cfg=read(receipt['consumer_config_ref'])
            old,report=timed('old_build_s',lambda:consumer.build(files,cfg))
            bundle,_=timed('old_restore_again_s',lambda:consumer.restore(files,cfg))
        src=Path(__file__).with_name('source')/'consumer.py'
        os.environ['OMNI_CAPTURE_CONSUMER_WRAPPER']=str(src)
        with scope():
            import consumer
            from cohort_io import read,pin,digest
            fields,r,b=timed('new_build_and_restore_s',lambda:consumer.build_with_bundle(files,cfg))
            if bundle!=b:equal(bundle,b,'restored_bundle')
            if r!=report:equal(report,r,'in_memory_report')
            disk=read(pin(a.root/'CONSUMER-REPORT.json'))
            if digest(r)!=digest(disk):raise ValueError('different sealed report keys: '+str([k for k in r if digest(r[k])!=digest(disk[k])]))
            equal(old['tensors'],fields['tensors'])
            equal(old['meta_info'],fields['meta_info'])
            equal({k:list(v) for k,v in old['non_tensors'].items()},{k:list(v) for k,v in fields['non_tensors'].items()})
            # A validly rehashed but wrong draw must still be rejected BEFORE training.
            import copy
            bad=copy.deepcopy(files)
            rows=read(files['refs']['provenance']);rows[0]['sample_id']+=':drift'
            ref=write(a.out/'mismatch.json',rows)
            bad['refs']['provenance']=ref;bad['provenance']=ref['path']
            try:consumer.build_with_bundle(bad,cfg)
            except ValueError as error:
                if 'draw probabilities/identity/outcome changed' not in str(error):raise
            else:raise ValueError('wrong selected draw accepted')
    result=dict(times=times,episodes=receipt['complete_episodes'],groups=receipt['selected_groups'],
         inputs_equal=True,bundles_equal=True,reports_equal=True,wrong_draw_rejected=True,
         wrapper_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),npu=False)
    write(a.out/'CONSUMER-COMPARISON.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('spec','root','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--ranks',type=int,required=True)
    run(p.parse_args())
