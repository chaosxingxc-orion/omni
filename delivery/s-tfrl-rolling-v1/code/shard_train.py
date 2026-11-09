"""Four contiguous shards of the unchanged frozen remainder; no new dataset."""
import copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'delivery/s-tfrl-rolling-v1'
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import read,lines,save,sha,require

def main():
    parent=OUT/'train-supply/train-eligible_remainder-single-v1'
    samples=lines(parent/'samples.private.jsonl');manifest=lines(parent/'single_intent_no_examples/manifest.private.jsonl')
    gold=lines(parent/'offline/gold.private.jsonl');proof=lines(parent/'source-proof.private.jsonl')
    require(len(samples)==3734 and len(manifest)==11202,'whole original remainder')
    cfg=read(parent/'single_intent_no_examples/native-UNBOUND.json');tasks=[];seen=set()
    for i in range(4):
        begin,end=len(samples)*i//4,len(samples)*(i+1)//4;block=samples[begin:end];ids={r['sample_id'] for r in block}
        require(not seen&ids,'disjoint contiguous blocks');seen|=ids
        name='train-remainder-part'+str(i+1)+'-v1';folder=OUT/'train-supply'/name
        require(not folder.exists(),'immutable shard')
        rel=lambda p:str(p.relative_to(ROOT))
        chosen=[copy.deepcopy(r) for r in manifest if r['source_sample_id'] in ids]
        for row in chosen:
            src=ROOT/row['payload'];dst=folder/'single_intent_no_examples/packets'/src.name
            dst.parent.mkdir(exist_ok=True,parents=True);os.link(src,dst);row['payload']=rel(dst)
        selected=[]
        for row in proof:
            if row['sample_id'] not in ids:continue
            row=copy.deepcopy(row);src=ROOT/row['original_packet']
            dst=folder/'source-packets'/src.name;dst.parent.mkdir(exist_ok=True,parents=True);os.link(src,dst)
            row['original_packet']=rel(dst);selected.append(row)
        for file in ['snapshot.projected.private.json','budget-parent.json']:
            save(folder/file,read(parent/file))
        save(folder/'samples.private.jsonl',block,jsonl=True)
        save(folder/'source-proof.private.jsonl',selected,jsonl=True)
        save(folder/'offline/gold.private.jsonl',[r for r in gold if r['sample_id'] in ids],jsonl=True)
        path=folder/'single_intent_no_examples/manifest.private.jsonl';save(path,chosen,jsonl=True)
        native=cfg|dict(run_id='S-ROLLING-'+name.upper(),manifest=rel(path),manifest_sha256=sha(path.read_bytes()),
            expected_rows=len(chosen),full_denominator=len(block),prepared=rel(folder),samples=rel(folder/'samples.private.jsonl'),
            snapshot=rel(folder/'snapshot.projected.private.json'),budget_parent=dict(path=rel(folder/'budget-parent.json'),sha256=sha((folder/'budget-parent.json').read_bytes())),
            reference_CPU_only=rel(folder/'offline/gold.private.jsonl'),
            partition_custody=dict(parent=str(parent.relative_to(ROOT)),parent_denominator=3734,range_begin=begin,range_end_exclusive=end,stable_contiguous=True))
        save(folder/'single_intent_no_examples/native-UNBOUND.json',native)
        save(folder/'CPU-PREPARED.json',dict(status='PREPARED_NEEDS_FIXED_IMAGE_BUDGET',samples=len(block),branches=len(chosen),
            original_remainder_denominator=3734,stable_range=[begin,end],audio_encodes=0,P_recollection=0,model_calls=0))
        tasks.append(dict(task_id=name,prepared=rel(folder),config=rel(folder/'single_intent_no_examples/native-UNBOUND.json'),
            runner='delivery/s-tfrl-rolling-v1/code/run_n.py',bind_script='delivery/s-tfrl-rolling-v1/code/bind_n.py',
            parent_version='prepared-v5-dynamic/single_intent_no_examples',preferred_lane='ABCD'[i],method='original_remainder_contiguous_shard',
            purpose='new real S trajectories; original3734 union; hold excluded from updates'))
    require(seen=={r['sample_id'] for r in samples},'whole3734 union')
    save(OUT/'config/train-remainder-shards.json',dict(tasks=tasks,at=time.time(),full_denominator=3734,branches=11202,
        disjoint=True,order='original stable contiguous quarters',sample_ids_sha256=sha(('\n'.join(r['sample_id'] for r in samples)).encode())))
    print('STABLE_ORIGINAL3734_FOUR_SHARDS_PREPARED')

if __name__=='__main__':main()
