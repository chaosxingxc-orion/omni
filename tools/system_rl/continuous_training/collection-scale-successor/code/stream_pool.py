"""Bounded sealing; unchanged v1 sampler contract exposed by lazy streams."""
import json,hashlib,os,tempfile
from pathlib import Path
from collections import Counter
from common import canonical,digest,save,require,fingerprint,mkdir,syncdir

def iter_jsonl(ref):
    p=Path(ref['path']);before=fingerprint(p);h=hashlib.sha256();n=0
    with p.open('rb') as f:
        for line in f:
            h.update(line);n+=len(line)
            if line.strip():yield json.loads(line)
    require(h.hexdigest()==ref['sha256'] and n==ref['bytes'] and fingerprint(p)==before,'STREAM_PIN_DRIFT')

def projection(engine,row):
    sid=row['sample_id'];online,offline=engine.rawdb.execute('SELECT online,offline FROM samples WHERE id=?',(sid,)).fetchone()
    sample=engine.content.loads(online);labels=json.loads(offline)
    roles={n.get('role'):n['id'] for n in engine.nodes};st=row['stages']
    def output(role):
        v=st.get(roles.get(role),{}).get('output')
        return engine.content.expand(v) if v is not None else None
    gate=output('gate');score=output('score');export=output('export')
    domain={'train_real':'real','train_synth':'synthetic','real':'real','synthetic':'synthetic'}.get(sample['domain'],sample['domain'])
    src=dict(recording_ref_id=sample.get('reference_id',sid),split=sample['split'],domain=domain,source=sample.get('dataset','SLURP'),source_id=sample['source_id'])
    eligible=bool(gate and gate.get('eligible'))
    outcome='ineligible' if row['outcome']=='excluded' else score.get('outcome','unknown') if score else 'unknown'
    if outcome not in ('success','error','ineligible'):outcome='unknown'
    if outcome in ('success','error') and not eligible:outcome='unknown'
    result=dict(**src,status=outcome,gold_in_actual_pre_P_dynamic_retrieval=eligible)
    record=None
    if outcome in ('success','error'):
        online_stages={s:dict(status=v['status'],error=v['error']) for s,v in st.items() if not engine.byid[s].get('offline')}
        record=dict(**src,sample_id=sid,system_version=engine.system_version,rollout_scope=engine.cfg.get('rollout_scope','system'),
            outcome=outcome,online_gold_injected=False,
            online=dict(trajectory_export=export.get('online_trajectories') if export else None,stages=online_stages,
                        evidence=engine.content.put({k:v for k,v in st.items() if not engine.byid[k].get('offline')})),
            offline_supervision=dict(labels=labels,score=st.get(roles.get('score'),{}).get('output'),export=export.get('offline_supervision') if export else None),
            synthetic_fixture=engine.mode in ('synthetic','fixture'),purpose='screen',content_root=str(engine.content.root))
    return src,result,record

def publish(engine):
    prefix=engine.root/'pool';mkdir(prefix)
    counts=Counter();domains=Counter();population_hash=hashlib.sha256();population_hash.update(b'[');first=True
    paths=[prefix/(n+'.jsonl') for n in ('population','outcomes','records')]
    handles=[];temps=[];hashes=[hashlib.sha256() for _ in paths];sizes=[0]*3;ns=[0]*3
    try:
        for p in paths:
            fd,tmp=tempfile.mkstemp(dir=prefix,prefix='.pool-');temps.append(tmp);handles.append(os.fdopen(fd,'wb'))
        for i,row in enumerate(engine.rows()):
            if i%128==0:engine.storage.check()
            src,outcome,record=projection(engine,row);domains[src['domain']]+=1;counts[outcome['status']]+=1
            if not first:population_hash.update(b',')
            population_hash.update(canonical(src));first=False
            for j,value in enumerate((src,outcome,record)):
                if value is None:continue
                raw=canonical(value)+b'\n';handles[j].write(raw);hashes[j].update(raw);sizes[j]+=len(raw);ns[j]+=1
        population_hash.update(b']')
        for f in handles:f.flush();os.fsync(f.fileno());f.close()
        for tmp,p in zip(temps,paths):os.replace(tmp,p)
        syncdir(prefix)
    finally:
        for f in handles:
            if not f.closed:f.close()
        for tmp in temps:
            if Path(tmp).exists():Path(tmp).unlink()
    refs={p.stem:dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=size,count=n,format='jsonl') for p,h,size,n in zip(paths,hashes,sizes,ns)}
    if engine.purpose=='evaluation':
        return save(engine.root/'EVALUATION-OUTCOMES.json',dict(schema='full-population-evaluation-outcomes-v2',purpose='evaluation',split=engine.dataset['split'],
            population=engine.dataset['population'],population_accounted=ns[0]==engine.dataset['population'],outcomes_ref=refs['outcomes'],
            gold_admission_filter=False,training_ready=False,official_corpus_metrics_computed=False,system_version=engine.system_version))
    full=ns[0]==engine.dataset['population'] and not engine.dataset.get('bounded_selection')
    if full and not engine.dataset.get('synthetic'):
        from train_population import official_domains
        full=dict(domains)==official_domains(engine.dataset)
    complete=full and counts['unknown']==0
    dataset_path=str(engine.paths[0])
    # Reuse the exact hash recorded by Engine.pin_inputs at startup.
    (dataset_sha,)=engine.rawdb.execute('SELECT sha FROM inputs WHERE path=?',(dataset_path,)).fetchone()
    lineage=dict(raw_dataset=dict(path=dataset_path,sha256=dataset_sha,metadata=engine.dataset.get('inputs',[])),system_release=engine.system_version,
        knowledge=engine.release.get('knowledge_pin','pinned business release'),retrieval=engine.release.get('retrieval_pin','pinned business release'),
        prompt=engine.release.get('pins'),rails=engine.runtime_pins,scorer=engine.release.get('scorer_pin','pinned business release'))
    return save(engine.root/'TRAINING-POOL.json',dict(schema='sealed-training-pool-v1',storage_format='content-ref-jsonl-v1',sealed=True,
        version=engine.version,system_version=engine.system_version,lineage=lineage,official_train_scope='entire_official_TRAIN_real_and_synthetic',
        population_complete=complete,official_train_count=ns[0],synthetic_fixture=engine.mode in ('synthetic','fixture'),
        population_sha256=population_hash.hexdigest(),streams=refs,content_root=str(engine.content.root),
        cross_project_exclusions=False,unknown_count=counts['unknown'],training_ready=False))

class Sequence:
    def __init__(self,ref):self.ref=ref
    def __len__(self):return self.ref['count']
    def __iter__(self):return iter_jsonl(self.ref)

class Pool(dict):
    def __init__(self,manifest):
        super().__init__(manifest)
        for name,ref in manifest['streams'].items():self[name]=Sequence(ref)

def sampler_module(base):
    from common import load_module
    base=Path(base);p=base/'kernel.py'
    load_module(dict(path=str(p),sha256=digest(p.read_bytes())),'kernel')
    p=base/'dataset.py';m=load_module(dict(path=str(p),sha256=digest(p.read_bytes())),'collection_existing_sampler')
    original=m.digest
    def streamed_digest(value):
        if not isinstance(value,Sequence):return original(value)
        h=hashlib.sha256();h.update(b'[')
        for i,row in enumerate(value):
            if i:h.update(b',')
            h.update(canonical(row))
        h.update(b']');return h.hexdigest()
    m.digest=streamed_digest
    return m
