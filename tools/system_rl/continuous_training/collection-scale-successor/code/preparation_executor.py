"""Configured original-producer executor; CPU planning cannot dispatch a model.

One immutable projected shard/config per run, three original encoder processes
per shard, durable Docker names/IDs. Unknown launch is reconciled, never copied
into an inference exactly-once claim. No business rendering/parsing lives here.
"""
import copy,fcntl,json,os,subprocess,threading,time
from pathlib import Path
from common import canonical,digest,load_module,mkdir,require,save
from transport import Unknown

class Executor:
    def __init__(self,session):
        self.session=session;self.spec=session.spec['encoder'];self.root=session.root/'preparation'
        mkdir(self.root);self.lock=threading.RLock();self.shards={};self.byref={};self.consumer=None
        self.contract=session.read(self.spec['handoff_contract']);self.index=session.read(self.spec['projection_index'])
        self.stats=dict(index_loads=1,shard_loads=0,model_launches=0,audio_materializations=0)

    def shard(self,sample):
        proof=self.session.collection.resolver.proof(sample['reference_id'])
        n=int(digest(canonical(proof['source_component']))[:16],16)%64
        key=sample['domain']+'/part-'+str(n).zfill(3)+'.private.jsonl'
        with self.lock:
            if key in self.shards:return self.shards[key]
            matches=[r for r in self.index['shards'] if r['key']==key];require(len(matches)==1,'PREP_PROJECTED_SHARD')
            row=matches[0]
            def lines(ref):
                raw=Path(ref['path']).read_bytes();require(digest(raw)==ref['sha256'],'PREP_SHARD_PIN');return [json.loads(x) for x in raw.splitlines()]
            tasks=lines(row['tasks']);bindings={r['task_id']:r for r in lines(row['bindings'])}
            require(len(bindings)==len(tasks),'PREP_TASK_BINDING_ROSTER')
            value=dict(row=row,tasks=tasks,bindings=bindings,root=self.root/digest(key.encode()))
            self.shards[key]=value;self.stats['shard_loads']+=1
            for task in tasks:
                for rid in task['reference_ids']:self.byref[rid]=task
            return value

    def original_consume(self):
        if hasattr(self,'consume'):return self.consume
        here=Path(self.spec['handoff_entry']['path']).parent
        for name in ('prepare','consume'):
            p=here/(name+'.py');load_module(dict(path=str(p),sha256=digest(p.read_bytes())),name)
        import sys
        self.consume=sys.modules['consume'];return self.consume

    def plan(self,sample):
        with self.lock:return self._plan(sample)

    def _plan(self,sample):
        shard=self.shard(sample);out=shard['root'];mkdir(out)
        if 'plan' in shard:return shard
        consume=self.original_consume();hist=self.session.read(self.contract['historical_encoding'])
        cfg=consume.export_encoder_config(shard['tasks'],shard['bindings'],self.contract,hist,
            model_path=self.spec['model_path'],probe_base=self.spec['probe_base'])
        config=save(out/'execution/encoding-config.private.json',cfg)
        runtime=self.session.read(self.spec['runtime_template'])
        for m in runtime['mounts']:
            if m['target']=='/job':m['source']=str(out/'execution')
            elif m['target']=='/job/encoding-config.private.json':m['source']=config['path']
            elif m['target']=='/root/ascend/log':m['source']=str(out/'execution/home-ascend-log')
        runtime.update(model_config_sha256=config['sha256'],model_execution_enabled=False,dispatch_allowed=False,
            shard_mapping=[dict(shard=i,rows=list(range(i,len(cfg['jobs']),3))) for i in range(3)])
        commands=[]
        for i in range(3):
            name='omni-collection-'+digest(canonical(dict(run=str(out),config=config['sha256'],shard=i)))[:24]
            command=['docker','create','--label','omni.config-sha256='+config['sha256'],'--label','omni.plan='+digest(str(out).encode()),'--entrypoint',runtime['image_python'],'--name',name,'--network','none','--read-only','--log-driver=json-file','--log-opt','max-size=100m','--log-opt','max-file=3']
            for m in runtime['mounts']:command+=['--mount','type=bind,src='+m['source']+',dst='+m['target']+(',readonly' if m['readonly'] else '')]
            for k,v in runtime['environment'].items():command+=['--env',k+'='+v]
            for driver in ['/usr/local/Ascend/driver','/usr/local/dcmi','/etc/ascend_install.info']:
                command+=['--mount','type=bind,src='+driver+',dst='+driver+',readonly']
            command+=['--env','ASCEND_RT_VISIBLE_DEVICES=0','--security-opt','label=disable','--cpus','16','--memory','48g','--shm-size','2g']
            command+=[runtime['image']]+runtime['original_entries'][i][1:]
            commands.append(dict(shard=i,name=name,create=command,start=['docker','start',name],wait=['docker','wait',name],inspect=['docker','inspect',name]))
        plan=dict(schema='automatic-original-preparation-plan-v2',tasks=shard['row']['tasks'],bindings=shard['row']['bindings'],
            encoding_config=config,runtime=runtime,commands=commands,
            dependency_chain=['original_encode','original_consume.validate_vector','original_audio_materializer','original_P_supply','original_S1_Consumer.cache_row','existing_StageRails'],
            dispatch_enabled=False,new_audio_reads=0,new_model_calls=0)
        shard.update(cfg=cfg,plan=save(out/'PLAN.json',plan));return shard

    def enabled(self):
        s=self.session
        return s.mode=='live' and s.authorization.get('embedding_calls') is True and s.execution.get('enabled') is True

    def execute(self,shard):
        s=self.session;require(self.enabled(),'PREPARATION_EXECUTION_NOT_AUTHORIZED')
        require(s.authorization.get('run_id')==s.spec['journal_identity']['run_id'],'PREPARATION_RUN_AUTHORITY')
        import datetime as dt
        expiry=dt.datetime.fromisoformat(s.authorization['expires_at']);require(expiry>dt.datetime.now(dt.timezone.utc),'PREPARATION_AUTHORITY_EXPIRED')
        devices=s.execution.get('shard_devices');require(isinstance(devices,list) and len(devices)==3 and all(isinstance(x,list) and x for x in devices),'ENCODER_THREE_SHARD_DEVICE_MAPPING')
        limit=s.execution.get('max_encoder_jobs',0);require(0<len(shard['cfg']['jobs'])<=limit,'PREPARATION_JOB_BUDGET')
        plan=s.read(shard['plan']);out=shard['root'];runtime=plan['runtime'];s.store.check()
        for m in runtime['mounts']:
            if not m['readonly']:mkdir(m['source'])
        # A run-local preparation lease does not replace/steal the resource guard.
        guard=Path(s.execution['guard_path']);require(guard.exists(),'PREPARATION_EXISTING_RESOURCE_GUARD')
        resource=guard.open('r+b');fcntl.flock(resource,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock=(out/'EXECUTION.guard').open('ab');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owned=[]
        try:
            ledger_path=self.root/'ADMISSION.json'
            ledger=json.loads(ledger_path.read_bytes()) if ledger_path.exists() else dict(shards={},total_jobs=0)
            token=plan['encoding_config']['sha256']
            if token not in ledger['shards']:
                require(ledger['total_jobs']+len(shard['cfg']['jobs'])<=limit,'PREPARATION_CUMULATIVE_JOB_BUDGET')
                ledger['shards'][token]=len(shard['cfg']['jobs']);ledger['total_jobs']+=len(shard['cfg']['jobs']);save(ledger_path,ledger)
            for item in plan['commands']:
                state=subprocess.run(item['inspect'],capture_output=True,text=True)
                if state.returncode:
                    require('No such object' in state.stderr or 'No such container' in state.stderr,'DOCKER_INSPECT_UNAVAILABLE_UNKNOWN')
                    intent=out/('container-'+str(item['shard'])+'.intent.json')
                    if intent.exists():raise Unknown('PREPARATION_CREATE_UNKNOWN: '+str(intent))
                    save(intent,dict(name=item['name'],config=plan['encoding_config'],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
                    command=item['create'];at=command.index(runtime['image']);extra=[]
                    for device in devices[item['shard']]:extra+=['--device',device]
                    subprocess.run(command[:at]+extra+command[at:],check=True,capture_output=True,text=True)
                    state=subprocess.run(item['inspect'],check=True,capture_output=True,text=True)
                obj=json.loads(state.stdout)[0];self.verify_container(obj,item,plan,devices[item['shard']]);owned.append(item)
                save(out/('container-'+str(item['shard'])+'.json'),dict(id=obj['Id'],name=item['name'],config=plan['encoding_config']))
                if obj['State']['Status']=='created':subprocess.run(item['start'],check=True,capture_output=True);self.stats['model_launches']+=1
                elif obj['State']['Status']=='exited':require(obj['State']['ExitCode']==0,'ENCODER_FAILED_KEEP_RECEIPTS')
                else:require(obj['State']['Running'],'ENCODER_CONTAINER_UNKNOWN_STATE')
            # Original processes resume their own valid completed rows. No new
            # container per query; wait checks capacity and configured deadline.
            deadline=time.monotonic()+s.execution['max_wall_seconds']
            while True:
                s.store.check();pending=False
                for item in plan['commands']:
                    obj=json.loads(subprocess.run(item['inspect'],check=True,capture_output=True,text=True).stdout)[0]
                    if obj['State']['Running']:pending=True
                    else:require(obj['State']['ExitCode']==0,'ENCODER_FAILED')
                if not pending:break
                if time.monotonic()>deadline or dt.datetime.now(dt.timezone.utc)>=expiry:
                    raise Unknown('ENCODER_BUDGET_OR_AUTHORITY_EXHAUSTED_STOP_OWNED')
                time.sleep(min(1,s.execution.get('poll_seconds',1)))
        finally:
            # Stop only authenticated containers owned by this exact plan before
            # releasing the resource lease, including timeout/expiry/interrupt.
            for item in owned:
                obj=json.loads(subprocess.run(item['inspect'],check=True,capture_output=True,text=True).stdout)[0]
                self.verify_container(obj,item,plan,devices[item['shard']])
                if obj['State']['Running']:subprocess.run(['docker','stop','--time','30',item['name']],check=True,capture_output=True)
                obj=json.loads(subprocess.run(item['inspect'],check=True,capture_output=True,text=True).stdout)[0]
                require(not obj['State']['Running'],'OWNED_ENCODER_STOP_NOT_CONFIRMED_KEEP_LEASE')
            lock.close();resource.close()

    def verify_container(self,obj,item,plan,devices):
        runtime=plan['runtime'];cfg=obj['Config'];labels=cfg.get('Labels') or {}
        require(cfg['Image']==runtime['image'] and labels.get('omni.config-sha256')==plan['encoding_config']['sha256'] and
            labels.get('omni.plan')==digest(str(Path(plan['encoding_config']['path']).parents[1]).encode()),'ENCODER_CONTAINER_IDENTITY')
        require(cfg['Entrypoint']==[runtime['image_python']] and cfg['Cmd']==runtime['original_entries'][item['shard']][1:],'ENCODER_CONTAINER_COMMAND')
        env=dict(x.split('=',1) for x in cfg['Env'] if '=' in x)
        require(all(env.get(k)==v for k,v in runtime['environment'].items()) and env.get('ASCEND_RT_VISIBLE_DEVICES')=='0','ENCODER_CONTAINER_ENV')
        mounts={m['Destination']:(m['Source'],not m['RW']) for m in obj['Mounts']}
        require(all(mounts.get(m['target'])==(m['source'],m['readonly']) for m in runtime['mounts']),'ENCODER_CONTAINER_MOUNTS')
        actual={x['PathOnHost'] for x in obj['HostConfig'].get('Devices',[])}
        require(actual==set(devices),'ENCODER_CONTAINER_DEVICES')

    def encoder(self,sample):
        shard=self.plan(sample);task=self.byref[sample['reference_id']];bound=shard['bindings'][task['task_id']]
        jobs=shard['cfg']['jobs'];ix=next((i for i,j in enumerate(jobs) if j['id']==task['task_id']),None)
        consume=self.original_consume()
        if ix is None:
            entries=consume.reuse_alias(task,bound)
        else:
            receipt=shard['root']/f'execution/encoding/shard{ix%3}/rows/{ix:05}.json'
            if not receipt.exists():
                if self.enabled():self.execute(shard)
                else:raise Unknown('PREPARATION_RUNTIME_NOT_AUTHORIZED: '+shard['plan']['path'])
            cfg=consume.read_encoding_config(shard['root']/'execution/encoding-config.private.json')
            vector,entries=consume.validate_vector(task,bound,json.loads(receipt.read_bytes()),cfg)
            import numpy as np
            dest=shard['root']/('vector-'+task['task_id']+'.npy');tmp=dest.with_suffix('.tmp')
            with tmp.open('wb') as f:np.save(f,np.asarray([vector]),allow_pickle=False);f.flush();os.fsync(f.fileno())
            os.replace(tmp,dest)
            from common import syncdir
            syncdir(dest.parent);ref=dict(path=str(dest),sha256=digest(dest.read_bytes()))
            for e in entries:e.update(row=0,vector=ref)
        self.session.vector_overlays.update({e['reference_id']:e for e in entries})
        save(shard['root']/('INGEST-'+digest(sample['reference_id'].encode())+'.json'),dict(entries=entries,task=task['task_id'],plan=shard['plan'],model_called_by_ingest=False))
        return dict(status='original_encoder_ingested',entry=self.session.vector_overlays[sample['reference_id']],plan=shard['plan'])

    def audio(self,sample,bound):
        """Actual original PCM materializer; never executes in this CPU-only run."""
        s=self.session;sid=sample['sample_id'];path=self.root/'audio'/(digest(sid.encode())+'.json')
        if path.exists():return json.loads(path.read_bytes())
        # Audio preparation is CPU-only; authorization gates only original encoder execution.
        # Bind raw bytes/stat before calling original soundfile->PCM16 function.
        from common import fingerprint
        require(fingerprint(bound['audio_path'])==bound['audio_stat'],'MATERIALIZER_BOUND_STAT')
        require(digest(Path(bound['audio_path']).read_bytes())==bound['audio_sha256'],'MATERIALIZER_BOUND_BYTES')
        import ast,io,numpy as np,base64
        from business import ROOT
        source=ROOT/'delivery/s1-round03-bound-layout-combination-v1/package/src/speech_aware_evidence_acquisition/analysis/slurp_mf7b_vectors.py'
        fn=next(n for n in ast.parse(source.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='read_audio_wav_bytes')
        env=dict(np=np,io=io,Path=Path,SAMPLE_RATE_HZ=16000)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
        raw=env['read_audio_wav_bytes'](bound['audio_path'])
        audio=dict(type='input_audio',input_audio=dict(data=base64.b64encode(raw).decode(),format='wav'))
        s.collection.adapter.audio_reader.audio_bytes(audio)
        ar=save(self.root/'audio'/(digest(sid.encode())+'.object.json'),dict(source_sample_id=sid,audio_sha256=bound['audio_sha256'],audio=audio))
        ar.update(content_path=['audio'],audio_sha256=bound['audio_sha256'],audio_object_sha256=digest(canonical(audio)))
        custody=dict(sample_id=sid,audio_ref=ar,audio_sha256=bound['audio_sha256'],audio_object_sha256=digest(canonical(audio)),transport_audio_sha256=digest(raw),
          source_materializer=dict(path=str(source),sha256=digest(source.read_bytes())),bound=bound,custody='ORIGINAL_MATERIALIZER_BOUND_RAW_STAT_SHA_PCM16')
        save(path,custody);self.stats['audio_materializations']+=1;return custody

    def supply(self,sample,event,entry):
        s=self.session
        if not hasattr(self,'repair'):
            from business import BASE,module
            self.repair=module(BASE/'p-card-field-repair/repair.py','repair').Repair(self.spec['p_repair_config']['path'])
            self.pmaterializer=module(BASE/'p-supply-materializer/materialize.py','integration_preparation_materializer').Materializer(self.spec['p_materializer_config']['path'])
        import numpy as np
        m=self.pmaterializer;r=self.repair
        require(r.permanent==m.frozen['permanent_labels']==s.renderer.permanent and not set(event['actual_returned_ranked_intent_ids']) & set(r.permanent),
                'P_PERMANENT_KNOWLEDGE_POLICY_MISMATCH')
        # This is input assembly only. Original legal-source status, field-view,
        # rank kernel and renderer remain the sole business implementations.
        proof=s.collection.resolver.proof(sample['reference_id']);ex=proof['legacy_components'];component=proof['source_component']
        q=s.matrices.get(entry['vector'])[entry['row']];manifest=s.read(s.spec['retrieval_manifest']);keys=s.rank.normalize(s.matrices.get(manifest['keys']))
        labels=event['actual_returned_ranked_intent_ids'];fewshots={};audit={}
        historical=m.bindings.get(sample['sample_id'],{})
        base=dict(reference_id=sample['reference_id'],identity=dict(component_id=component),excluded_source_groups=ex)
        for label in labels:
            old=historical.get(label,[]);selection=None;mode='original_kernel_source_distinct_same_label'
            if old:
                choices={digest(canonical(v['selected'])) for v in old};require(len(choices)==1,'P_HISTORICAL_SELECTION_CONFLICT')
                selected=old[0]['selected']
                if selected is None:mode='preserve_historical_None'
                elif not m.source_status(str(selected['utterance_id']),set(ex),component):selection=selected;mode='preserve_legal_historical_selection'
            if mode not in ('preserve_historical_None','preserve_legal_historical_selection'):
                ix=[i for i,v in enumerate(m.members) if v['intent']==label and not m.source_status(str(v['utterance_id']),set(ex),component)]
                if ix:
                    scores=np.zeros(len(m.members),dtype=np.float64);scores[ix]=np.asarray(q,dtype=np.float64)@keys[ix].T
                    kernel=s.rank.load_kernel(manifest['ranking_identity']['kernel'])
                    order=kernel.source_distinct_order(scores,ix,m.bank['sourcegroups'],set(ex))
                    j=int(order[0]);v=m.members[j];uid=str(v['utterance_id'])
                    selection=dict(member_index=j,utterance_id=uid,component=m.registry['train_real:'+uid]['component_id'],text_sha256=v['sentence_sha256'])
            audit[label]=dict(selected=selection,status='complete' if selection else 'explicit_no_legal_bank_example',mode=mode)
            if selection:
                uid=str(selection['utterance_id']);text=m.support[uid];require(digest(text.encode())==selection['text_sha256'],'P_ORIGINAL_SUPPORT_HASH')
                fewshots[label]=dict(text=text,provenance=dict(selection=audit[label],support_text_ref=m.support_refs[uid],source_record=m.registry['train_real:'+uid]))
            else:fewshots[label]=None
        cards={l:r.view(l,base) for l in labels+r.permanent}
        gaps=[l for l,v in cards.items() if v['provenance'].get('gaps')];require(not gaps,'P_FIELD_SUPPORT_UNAVAILABLE: '+','.join(gaps))
        value=dict(reference_id=sample['reference_id'],status='complete_offline_text_supply',gaps=[],supply=dict(dynamic_labels=labels,cards=cards,fewshots=fewshots),fewshot_audit=audit,
            identity=dict(source_id=sample['source_id'],domain=sample['domain'],component_id=component),current_qualification=False,dispatch_allowed=False)
        ref=save(self.root/'supplies'/(digest(sample['reference_id'].encode())+'.json'),value)
        s.supply_overlays[sample['reference_id']]=value;return ref

    def s1_overlay(self,sample,custody,entry):
        s=self.session
        if self.consumer is None:
            from business import module
            directory=Path(self.spec['s1_supplement']['path']).parent
            module(directory/'runtime_io.py','runtime_io')
            consumer=module(directory/'s1_supplement.py','integration_preparation_s1')
            self.consumer=consumer.Consumer.load(self.spec['s1_supplement']['path'])
        shard=self.shard(sample);task=self.byref[sample['reference_id']];bound=shard['bindings'][task['task_id']]
        audio=s.read(custody['audio_ref'])
        for k in custody['audio_ref']['content_path']:audio=audio[k]
        proof=s.collection.resolver.proof(sample['reference_id']);vector=s.matrices.get(entry['vector'])[entry['row']]
        row=self.consumer.cache_row(task,bound,vector,audio,{proof['source_component']})
        row=dict(sample_id=sample['sample_id'],source_component=task['component_id'],
            audio_sha256=row['audio_sha256'],audio_object_sha256=row['audio_object_sha256'],
            excluded_source_components=row['excluded_source_components'],neighbors=row['neighbors'])
        # A new query is not smuggled under an old cache ID. Version a single
        # sample cache/rails and route this recording consistently through it.
        cache=copy.deepcopy(s.collection.cache);cache['queries']=[row];cache.pop('cache_id');cache['cache_id']=digest(canonical(cache))
        policies=copy.deepcopy(s.collection.policies)
        for p in policies:p['s1_consumption']['cache_id']=cache['cache_id']
        from business import module,BASE
        policy=module(BASE/'s1-source-policy-successor/policy.py','integration_policy')
        c=copy.copy(s.collection);a=copy.copy(c.adapter);c.adapter=a;c.cache=cache;c.policies=policies
        a.s1=policy.policy_rails(c.resolver,c.knowledge,cache,policies[0]);a.s2=policy.policy_rails(c.resolver,c.knowledge,cache,policies[1])
        contract=a.scoped(a.cfg['s2_contract'],'integration_s2_contract',a.scoped(a.cfg['s2_common'],'integration_s2_common'));a.s2=contract.apply(a.s2)
        c.ids=dict(c.ids,cache_id=cache['cache_id']);s.collections[sample['sample_id']]=c
        ref=save(self.root/'s1'/(digest(sample['reference_id'].encode())+'.json'),dict(cache=cache,policies=policies,parent_cache_id=s.collection.cache['cache_id']))
        return ref
