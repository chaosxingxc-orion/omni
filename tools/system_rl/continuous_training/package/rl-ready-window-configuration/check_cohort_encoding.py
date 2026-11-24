"""CPU A/B of the actual publisher on the same sealed batch; no GPU or synthetic records."""
import argparse,json,tempfile,time,sys
from pathlib import Path

def check(source, candidate, output, runtime):
    sys.path.insert(0,str(runtime))
    from cohort_io import load,pin,read,require,save
    module=load(candidate,'candidate_cohort')
    receipts=read(pin(source/'FILES.json'))
    out={}
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        for name,ref in receipts['refs'].items():
            original=source/(name+'.json')
            identity=pin(original);require(all(identity[k]==ref[k] for k in ('sha256','bytes')),'wrong source file')
            rows=read(identity)
            row={}
            for mode in ('original','compact_json_v1'):
                path=Path(tmp)/(name+'-'+mode+'.json')
                start=time.perf_counter();published=module.save_training_rows(path,rows,mode); elapsed=time.perf_counter()-start
                check=read(published);require(check==rows,'round trip changed content')
                if mode=='original':require(published['sha256']==ref['sha256'],'control differs from sealed baseline')
                row[mode]=dict(seconds=elapsed,bytes=published['bytes'],rows=len(check));del check
            out[name]=row
    return save(output/'ENCODING-COMPARISON.json',dict(files=out,cohort=pin(source/'COMPLETE.json'),npu=False))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source','candidate','output','runtime'):p.add_argument(k,type=Path)
    print(json.dumps(check(**vars(p.parse_args()))))
