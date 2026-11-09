"""Pair original frozen TRAIN control with current D score-top2 policy, CPU only."""
import collections,copy,json,os,sys,time,tarfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'delivery/s-tfrl-rolling-v1'
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import read,save,sha,lines,canonical,require,validate_packet
from retrieval import CachedRetrieval,select,example
from speech_aware_evidence_acquisition.rails import candidate_handoff as hb
from speech_aware_evidence_acquisition.core.model.vllm_transport import vllm_request_body
import numpy as np

def durable_copy(src,dst):
    import shutil
    dst.parent.mkdir(parents=True,exist_ok=True)
    with src.open('rb') as source,dst.open('xb') as target:
        shutil.copyfileobj(source,target);target.flush();os.fsync(target.fileno())

class TrainRetrieval:
    def __init__(self):
        roster=lines(ROOT/'delivery/s-single-intent-fewshot-v1/prepared-v5-dynamic/samples.private.jsonl')
        self.base=CachedRetrieval(ROOT/'delivery/s-single-intent-fewshot-v1/config/retrieval-v5.json',roster)
        c=read(ROOT/'delivery/s-train-p-collection-v1/config/prepare.json')
        pins=read(ROOT/'delivery/s-train-p-collection-v1/pins/sources.json')
        source=read(ROOT/c['train']/'inputs/source-pins.private.json')
        qp=ROOT/c['query_vectors'];kp=ROOT/c['key_vectors'];enc=ROOT/c['train']/'encoding-config.private.json'
        for p in (qp,kp,enc):require(sha(p.read_bytes())==pins[str(p.relative_to(ROOT))]['sha256'],'frozen P query/key provenance '+str(p))
        self.q=np.load(qp,mmap_mode='r',allow_pickle=False);self.k=np.load(kp,mmap_mode='r',allow_pickle=False).astype(np.float64)
        require(self.q.shape==(10000,2048) and self.k.shape==(8554,2048),'original vectors dimensions')
        require(sha(kp.read_bytes())==source['keys']['sha256'],'original key vector provenance')
        self.k/=np.linalg.norm(self.k,axis=1)[:,None]
        jobs=read(enc)['jobs'];self.rows={j['id']:i for i,j in enumerate(jobs)}
        require(len(self.rows)==10000,'unique original TRAIN query identity')
        self.masks={r['sample_id']:set(r['excluded_source_groups']) for r in read(ROOT/c['train']/'inputs/source-masks.private.json')['rows']}
        self.policy=self.base.policy|dict(query_matrix_shape=list(self.q.shape),
            original_TRAIN_vectors=True,new_embeddings=0,query_transcript_used=False,query_gold_used=False,
            score_mode='original saved normalized TRAIN audio query vectors cast float64 dot normalized same8554 keys',
            query_specific_source_group_exclusion=True,pins=self.base.policy['pins']|{
            c['query_vectors']:sha(qp.read_bytes()),c['key_vectors']:sha(kp.read_bytes()),
            str(enc.relative_to(ROOT)):sha(enc.read_bytes()),
            str((ROOT/c['train']/'inputs/source-masks.private.json').relative_to(ROOT)):sha((ROOT/c['train']/'inputs/source-masks.private.json').read_bytes())})

    def choose(self,sid,cid,allowed):
        row=self.rows[sid]
        if getattr(self,'cached_sid',None)!=sid:
            self.cached_scores=np.asarray(self.q[row],dtype=np.float64)@self.k.T;self.cached_sid=sid
        scores=self.cached_scores
        require(np.isfinite(scores).all(),'finite audio similarity')
        excluded=self.masks[sid];group=self.base.groupdoc['groups'][str(int(sid.split(':')[2]))]
        require(group in excluded,'query source group excluded')
        pool=[j for j in self.base.pools[cid] if self.base.groups[j] not in excluded
            and {e['type'] for e in self.base.records[j]['entities']}<=set(allowed)]
        chosen,covered,bounds=select(pool,scores,self.base.records,self.base.groups)
        require(len(chosen)<=2 and all(self.base.groups[j]!=group for j in chosen),'source disjoint max2')
        return [example(self.base.records[j],cid) for j in chosen],dict(sample_id=sid,candidate_id=cid,
            score_row=row,selected_columns=chosen,selected_scores=[float(scores[j]) for j in chosen],
            selected_source_groups=[self.base.groups[j] for j in chosen],legal_pool_size=len(pool),
            legal_pool_sha256=sha(canonical(pool)),query_gold_used=False,query_transcript_used=False)

def prepare(task,retrieval):
    start,cpu=time.monotonic(),time.process_time();name=task['task_id'];logical=ROOT/task['prepared']
    if (logical/'CPU-PREPARED.json').exists():return
    hot=Path('/var/tmp/omni-s-tfrl-rolling-v1/train-paired-fewshot')/name
    require(not logical.exists() and not hot.exists(),'immutable paired TRAIN')
    parent=ROOT/task['control_prepared'];cfg=read(parent/'single_intent_no_examples/native-UNBOUND.json')
    arm='single_intent_fewshot';rel=lambda p:str((logical/p.relative_to(hot)).relative_to(ROOT))
    for path in ['samples.private.jsonl','snapshot.projected.private.json','budget-parent.json','offline/gold.private.jsonl']:
        src=parent/path;dst=hot/path;durable_copy(src,dst)
    proofs=lines(parent/'source-proof.private.jsonl')
    for r in proofs:
        src=ROOT/r['original_packet'];dst=hot/'source-packets'/src.name;durable_copy(src,dst);r['original_packet']=rel(dst)
    save(hot/'source-proof.private.jsonl',proofs,jsonl=True)
    snapshot=read(parent/'snapshot.projected.private.json');values={v['value_id']:v['content'] for v in snapshot['values']}
    reports=[];manifest=[];cache={}
    for prior in lines(ROOT/cfg['manifest']):
        raw=(ROOT/prior['payload']).read_bytes();require(sha(raw)==prior['packet_sha256'],'control packet custody')
        packet=json.loads(raw);cid=prior['candidate_ids'][0];sid=prior['source_sample_id']
        ex,report=retrieval.choose(sid,cid,packet['allowed_types'])
        original=copy.deepcopy(packet);packet['arm']=arm;packet['examples']=ex
        if ex:
            demo=[dict(candidate_id=cid,input=e['input'],output=e['output'],annotated_mentions=e['annotated_mentions']) for e in ex]
            packet['wire']['messages'][1]['content'].insert(-1,dict(type='text',text='TRAIN demonstrations (not current-audio evidence):\n'+canonical(demo).decode()))
        inverse=copy.deepcopy(packet);inverse['arm']=original['arm'];inverse['examples']=[]
        if ex:inverse['wire']['messages'][1]['content'].pop(-2)
        require(inverse==original,'arms differ only real example block and arm/examples metadata')
        validate_packet(packet,selected=[cid],expected_audio=original['wire']['messages'][1]['content'][0],
            snapshot=cfg['snapshot_sha256'],values=values,handoff_module=hb)
        path=hot/arm/'packets'/Path(prior['payload']).name;save(path,packet)
        row=copy.deepcopy(prior);row.update(arm=arm,payload=rel(path),packet_sha256=sha(path.read_bytes()),
            abstract_wire_sha256=sha(canonical(packet['wire'])),wire_sha256=sha(canonical(vllm_request_body(packet['wire'],packet['wire']['model']))))
        manifest.append(row);reports.append(report)
    save(hot/'source-policy.json',retrieval.policy)
    save(hot/'retrieval-audit.private.jsonl',reports,jsonl=True)
    path=hot/arm/'manifest.private.jsonl';save(path,manifest,jsonl=True)
    native=cfg|dict(arm=arm,run_id='S-ROLLING-'+name.upper(),manifest=rel(path),manifest_sha256=sha(path.read_bytes()),
        snapshot=rel(hot/'snapshot.projected.private.json'),prepared=str(logical.relative_to(ROOT)),samples=rel(hot/'samples.private.jsonl'),
        budget_parent=dict(path=rel(hot/'budget-parent.json'),sha256=sha((hot/'budget-parent.json').read_bytes())),
        reference_CPU_only=rel(hot/'offline/gold.private.jsonl'),paired_control_config=str((parent/'single_intent_no_examples/native-UNBOUND.json').relative_to(ROOT)))
    save(hot/arm/'native-UNBOUND.json',native)
    save(hot/'CPU-PREPARED.json',dict(status='PREPARED_NEEDS_FIXED_IMAGE_BUDGET',samples=cfg['full_denominator'],branches=len(manifest),
        model_calls=0,new_embeddings=0,audio_encodes=0,P_recollection=0,query_gold_used=False,query_transcript_used=False,
        real_example_branches=sum(len(r['selected_columns'])>0 for r in reports),wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu))
    pending=logical.with_name(name+'.pending');pending.mkdir(parents=True)
    archive=Path('/var/tmp/omni-s-tfrl-rolling-v1')/(name+'.tar')
    with tarfile.open(archive,'w') as tf:
        for path in hot.rglob('*'):
            if path.is_file():tf.add(path,arcname=str(path.relative_to(hot)),recursive=False)
    with archive.open('rb') as f:subprocess.run(['tar','xf','-','-C',str(pending)],stdin=f,check=True)
    subprocess.run(['sync','-f',str(pending)],check=True);os.rename(pending,logical)
    print(json.dumps(read(logical/'CPU-PREPARED.json')),flush=True)

def main():
    import producer
    os.environ['OPENBLAS_NUM_THREADS']='2'
    r=TrainRetrieval();tasks=read(OUT/'config/train-paired-fewshot-tasks.json')['tasks']
    save(OUT/'TRAIN-PAIRED-PRODUCER-START.json',dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),agent=False,tasks=tasks))
    passed=[];failed=[]
    for task in tasks:
        try:prepare(task,r);producer.verify(task);producer.publish(task)
        except Exception as e:
            import traceback
            save(OUT/'evidence'/task['task_id']/'PAIRED-PRODUCER-FAILED.json',dict(at=time.time(),error=str(e),traceback=traceback.format_exc(),model_calls=0))
            failed.append(task['task_id'])
        else:passed.append(task['task_id'])
    save(OUT/'TRAIN-PAIRED-BATCH-END.json',dict(at=time.time(),status='CPU_BATCH_COMPLETE' if not failed else 'CPU_BATCH_PARTIAL_FAILURE',
        passed=passed,failed=failed,planned=len(tasks),P_recollection=0,new_embeddings=0))

if __name__=='__main__':main()
