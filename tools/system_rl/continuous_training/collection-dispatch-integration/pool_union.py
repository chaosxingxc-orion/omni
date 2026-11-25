"""Streaming sealed-run union, verified science, original record identities."""
import argparse, collections, hashlib, json, sqlite3
from pathlib import Path
from base import checked, read, save, require, canonical, digest, Storage, mkdir, HERE, official_domains, fingerprint
from science import extract
from stream_pool import iter_jsonl, Pool, sampler_module
from content import jsonl
from native import inventory


def rows(pool, name):
    if 'streams' in pool: yield from iter_jsonl(pool['streams'][name])
    else: yield from pool[name]


def union(config):
    c=read(config); root=Path(c['output_root']);Storage(root,c['storage']);mkdir(root)
    require(not (root/'COMPLETE.json').exists(), 'UNION_ALREADY_IMMUTABLE')
    require(not (root/'union.sqlite').exists(), 'PARTIAL_UNION_USE_NEW_OUTPUT')
    dataset=checked(c['dataset'])
    expected=official_domains(dataset); total=dataset['population']
    db=sqlite3.connect(root/'union.sqlite');db.execute('PRAGMA synchronous=FULL')
    db.executescript('''CREATE TABLE roster(id TEXT PRIMARY KEY,src TEXT NOT NULL);
       CREATE TABLE observed(id TEXT PRIMARY KEY,sample TEXT UNIQUE,source TEXT,outcome TEXT,record TEXT,run TEXT);
       CREATE TABLE population(id TEXT PRIMARY KEY,src TEXT);
       CREATE TABLE outcomes(id TEXT PRIMARY KEY,value TEXT);
       CREATE TABLE records(id TEXT PRIMARY KEY,sample TEXT UNIQUE,value TEXT);''')
    try:
        # inventory() reads raw metadata; bind those bytes to the configured roster.
        snapshots={}
        for domain in dataset['domains']:
            meta=domain['metadata']; p=Path(meta['path']); snapshots[p]=fingerprint(p)
            with p.open('rb') as f:
                h=hashlib.sha256()
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            require(h.hexdigest()==meta['sha256'],'OFFICIAL_METADATA_PIN')
        observed_domains=collections.Counter()
        for x in inventory(dataset):
            s=x['online']; src=dict(recording_ref_id=s['reference_id'],split='TRAIN',domain={'train_real':'real','train_synth':'synthetic'}[s['domain']],source='SLURP',source_id=s['source_id'])
            observed_domains[src['domain']]+=1
            db.execute('INSERT INTO roster VALUES (?,?)',(s['reference_id'],canonical(src).decode()))
        db.commit()
        require(all(fingerprint(p)==fp for p,fp in snapshots.items()) and observed_domains==expected, 'OFFICIAL_ROSTER_COUNT')
        runs=[];science=None;systems={};fixture=False
        for entry in c['runs']:
            marker=checked(entry['complete']); require(marker['immutable'],'UNSEALED_RUN')
            version=checked(marker['version'])
            require(version['status'].startswith('SEALED') and version['unresolved_requests']==0 and version['pending_tasks']==0, 'UNKNOWN_OR_NONTERMINAL_RUN')
            pool=checked(version['training_pool'])
            require(pool['sealed'] and pool['unknown_count']==0,'UNKNOWN_POOL')
            is_fixture=pool.get('synthetic_fixture') is True or version.get('synthetic') is True
            require(not is_fixture or c.get('allow_cpu_fixture') is True,'FIXTURE_NOT_PRODUCTION')
            fixture=fixture or is_fixture
            runtime=pool['system_version']
            require(runtime==entry['system']['sha256']==version['system_version'],'ACTUAL_SYSTEM_IDENTITY')
            evidence=extract(entry['system'],entry['business'],version,entry.get('config'))
            if science is None:science=evidence['identity']
            require(science==evidence['identity'],'INCOMPATIBLE_SCIENTIFIC_SYSTEM')
            systems[runtime]=evidence
            for table in ('population','outcomes','records'): db.execute('DELETE FROM '+table)
            for name in ('population','outcomes','records'):
                for row in rows(pool,name):
                    rid=row['recording_ref_id']
                    try:
                        if name=='records':db.execute('INSERT INTO records VALUES (?,?,?)',(rid,row['sample_id'],canonical(row).decode()))
                        else:db.execute('INSERT INTO '+name+' VALUES (?,?)',(rid,canonical(row).decode()))
                    except sqlite3.IntegrityError:raise ValueError('DUPLICATE_ID_WITHIN_RUN:'+name+':'+rid)
            n=db.execute('SELECT COUNT(*) FROM population').fetchone()[0]
            require(n==pool['official_train_count'],'RUN_POPULATION_COUNT')
            require(not db.execute('SELECT id FROM population EXCEPT SELECT id FROM outcomes').fetchone() and not db.execute('SELECT id FROM outcomes EXCEPT SELECT id FROM population').fetchone(),'RUN_OUTCOME_ACCOUNTING')
            require(not db.execute('SELECT id FROM records EXCEPT SELECT id FROM population').fetchone(),'ORPHAN_RUN_RECORD')
            for rid,srcraw,outraw,recraw,sid in db.execute('SELECT p.id,p.src,o.value,r.value,r.sample FROM population p JOIN outcomes o ON o.id=p.id LEFT JOIN records r ON r.id=p.id ORDER BY p.id'):
                src=json.loads(srcraw);out=json.loads(outraw);rec=json.loads(recraw) if recraw else None
                official=db.execute('SELECT src FROM roster WHERE id=?',(rid,)).fetchone()
                require(official and official[0]==srcraw,'NONOFFICIAL_SOURCE')
                require(all(out.get(k)==v for k,v in src.items()),'OUTCOME_SOURCE_DRIFT')
                require(out['status'] in ('success','error','ineligible') and type(out['gold_in_actual_pre_P_dynamic_retrieval']) is bool,'UNKNOWN_OUTCOME')
                admitted=out['status'] in ('success','error')
                require((rec is not None)==admitted,'ADMITTED_RECORD_ACCOUNTING')
                if rec:
                    require(all(rec.get(k)==v for k,v in src.items()) and rec['outcome']==out['status'],'RECORD_SOURCE_OR_OUTCOME_DRIFT')
                    require(out['gold_in_actual_pre_P_dynamic_retrieval'] and rec['system_version']==runtime and rec['online_gold_injected'] is False,'RECORD_RUNTIME_OR_ADMISSION_DRIFT')
                    # Exact original content_root wins; never overwrite per-record CAS.
                    if not rec.get('content_root') and pool.get('content_root'):rec=dict(rec,content_root=pool['content_root'])
                try:db.execute('INSERT INTO observed VALUES (?,?,?,?,?,?)',(rid,sid,srcraw,outraw,canonical(rec).decode() if rec else None,entry['complete']['path'],))
                except sqlite3.IntegrityError:raise ValueError('DUPLICATE_ID_ACROSS_RUNS:'+rid)
            db.commit();runs.append(dict(complete=entry['complete'],version=marker['version'],runtime_system=runtime,scientific_identity=science,population=n,evidence=evidence))
        def stream(column):
            for (raw,) in db.execute('SELECT '+column+' FROM observed WHERE '+column+' IS NOT NULL ORDER BY id'):yield json.loads(raw)
        refs={name:jsonl(root/(name+'.jsonl'),stream(column)) for name,column in [('population','source'),('outcomes','outcome'),('records','record')]}
        n=refs['population']['count'];h=hashlib.sha256();h.update(b'[')
        for i,row in enumerate(stream('source')):
            if i:h.update(b',')
            h.update(canonical(row))
        h.update(b']')
        domains=collections.Counter(r['domain'] for r in stream('source'))
        complete=n==total and domains==expected and not fixture
        manifest=dict(schema='sealed-training-pool-v1',storage_format='content-ref-jsonl-v1',sealed=True,
          version=digest(canonical(runs)),system_version=science,scientific_identity=science,compatible_runtime_systems=systems,
          lineage=dict(raw_dataset=c['dataset'],system_release=science,knowledge=runs,retrieval=runs,prompt=runs,rails=runs,scorer=runs),
          official_train_scope='entire_official_TRAIN_real_and_synthetic',population_complete=complete,official_train_count=n,
          official_expected_count=total,missing_population=total-n,population_sha256=h.hexdigest(),streams=refs,
          synthetic_fixture=fixture,cross_project_exclusions=False,unknown_count=0,training_ready=False,runs=runs)
        pool_ref=save(root/'TRAINING-POOL.json',manifest)
        sampling=None
        if c.get('sampling'):
            from sampling import publish
            if not complete:
                sampling=dict(status='WAITING_FULL_TRAIN',missing_population=total-n,training_ready=False)
            else:
                sample_config=read(c['sampling']) if isinstance(c['sampling'],str) else dict(c['sampling'])
                if sample_config.get('pool_version')=='AUTO_SEALED_UNION':sample_config['pool_version']=manifest['version']
                if sample_config.get('system_version')=='AUTO_SEALED_UNION':sample_config['system_version']=science
                sampling=publish(manifest,sample_config,root/'sample',c['readonly_roots'])
        save(root/'COMPLETE.json',dict(immutable=True,pool=pool_ref,production_population_complete=complete,training_ready=False,sampling=sampling))
        return manifest
    finally: db.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args();print(json.dumps(union(a.config)))
