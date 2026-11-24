"""Concrete original rails + DirectHTTP/ContactJournal + processor/actor capture.

No caller callback. Disabled until explicit new capture configuration. Uncertain
requests are reconciled by the existing journal; no generation retries/refills.
"""
import base64
import copy
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import subprocess
import sys
from cohort_io import BASE,ACL,CODE,HERE,read,raw,pin,save,require,digest,imports,load,storage


def request_once(transport,packet):
    state=transport.inspect(packet)
    if state['state']!='absent':
        result=transport.reconcile(packet['request_id'],packet)
        require(result['status']=='known','UNKNOWN: original journal reconciliation required; never resend')
        return result['result']
    return transport.submit(packet['request_id'],packet)


def sample_packet(packet,profile,identity,protocol):
    p=protocol.candidate(packet,profile,identity)
    # Original business parsers consume the same packet schema and actual wire.
    return dict(packet,wire=p['wire'],wire_sha256=p['wire_sha256'],request_id=p['request_id'],
                fresh_capture_identity=identity,sampling_config_hash=p['sampling_config_hash'],original_packet=packet)


def labels_for(node,online,annotation):
    """Reward labels retain entity occurrence ownership and unknown-value masks."""
    slots={};owners={};unknown=set()
    ts=annotation.get('tokens');es=annotation.get('entities')
    complete=isinstance(ts,list) and isinstance(es,list)
    tokens={t['id']:t.get('surface') for t in ts or []}
    require(len(tokens)==len(ts or []),'duplicate official token ID')
    for index,entity in enumerate(es or []):
        typ=entity['type'];span=entity.get('span');slots.setdefault(typ,[]);owners.setdefault(typ,[])
        if not span or not all(i in tokens and isinstance(tokens[i],str) for i in span):
            unknown.add(typ);continue
        value=' '.join(tokens[i] for i in span)
        if not value.strip():unknown.add(typ);continue
        slots[typ].append(value)
        owners[typ].append(digest(dict(recording=annotation['recording_ref_id'],source=annotation['source_record_sha256'],entity_index=index,type=typ,span=span)))
    if node=='P':return dict(intent=annotation['gold_intent_id'])
    if node=='S1':
        return dict(present_types=[t for t in online['type_universe'] if t in slots],annotated_types=online['type_universe']) if complete else None
    if node=='S2':return dict(slots={t:slots.get(t,[]) for t in online['supplied_types']},
        value_annotation_mask={t:complete and t not in unknown for t in online['supplied_types']},
        value_occurrence_ids={t:owners.get(t,[]) for t in online['supplied_types']})
    return dict(intent=annotation['gold_intent_id'],slots=slots,joint_labels_observable=True) if complete and not unknown else None


def normalized(packet,parsed,reply,assets):
    node=packet['stage'];terminal='valid_final'
    if reply['finish_reason']=='length':terminal='model_length'
    elif parsed.get('status') not in ('valid','valid_empty','valid_nonempty') and not ('answer' in parsed and parsed['answer'] is not None):terminal='protocol_error'
    online=dict(messages=packet['wire']['messages'],audio_assets=assets)
    if node=='P':online['retrieved_candidates']=[x['label'] for x in packet['ranking']];out=dict(selected=[x['candidate_id'] for x in parsed.get('candidates',[])])
    elif node=='S1':online['type_universe']=packet['allowed_types'];out=dict(present_types=parsed.get('selected_types',[]))
    elif node=='S2':
        online['supplied_types']=packet['allowed_types'];out=dict(slots={t:[] for t in packet['allowed_types']})
        for item in parsed.get('entities',[]):
            typ=item.get('type',packet['branch']);out['slots'][typ].append(item.get('filler',item.get('value')))
    else:
        candidates=packet.get('actual_candidates',packet.get('packet',{}).get('actual_candidates'))
        values=packet.get('actual_slot_values',packet.get('packet',{}).get('actual_slot_values'))
        if candidates is None and packet.get('packet',{}).get('P') is not None:
            candidates=[x['candidate_id'] for x in packet['packet']['P']['candidates']]
        if candidates is None:
            schema=packet.get('original_packet',packet)['wire']['response_format']['json_schema']['schema']
            candidates=schema['properties']['candidate_id']['enum']
        if values is None:
            evidence=packet.get('packet',{}).get('evidence',{})
            values={}
            for branch in evidence.get('s2',[]):
                for entity in branch.get('entities',[]):values.setdefault(entity['type'],[]).append(entity.get('filler',entity.get('value')))
            require('s2' in evidence,'D actual supplied slot evidence missing')
        online.update(actual_candidates=candidates,actual_slot_values=values)
        answer=parsed.get('answer') or {};slots={}
        for e in answer.get('entities',[]):slots.setdefault(e['type'],[]).append(e.get('filler',e.get('value')))
        intent=answer.get('candidate_id',answer.get('intent'))
        if intent is None and 'scenario' in answer and 'action' in answer:intent=answer['scenario']+'_'+answer['action']
        out=dict(intent=intent,slots=slots)
    out['terminal']=terminal
    return online,out


def selected_material(draw):
    from material import selected
    return selected(draw)[0]


def business_scope():
    return imports([BASE/'collection-dispatch-integration/automatic-successor',CODE],
        ('common','transport','direct_transport','business','content','asset_index',
         'source_policy','storage','integration','source_bridge','base','session','official_source'))


def diagnostic_targets(config,packets):
    """Bound frozen-input diagnostics before HTTP; never subsample training.

    The original order is retained, matching the existing full-logits first
    input quota. Selection cannot see current outputs, rewards or labels.
    Each retained input still receives every replicate in its original group.
    """
    spec=config.get('diagnostic_packet_selection')
    if spec is None:return packets
    from prompt_capacity import diagnostic_quotas
    quotas=diagnostic_quotas(spec)
    evidence=config.get('diagnostic_evidence_only',{})
    require(config.get('prepare_mode')=='finite_diagnostic_capture' and
            evidence.get('trainable') is False and evidence.get('input_mode')=='frozen_node_inputs' and
            config['native']['enabled'] is False and config['authorization']['native_training'] is False,
            'packet quota is only for nontraining frozen-input diagnostics')
    nodes=('P','S1','S2','D')
    seen={n:0 for n in nodes};selected=[]
    for packet in packets:
        node=packet['stage'];require(node in seen,'unknown diagnostic node')
        if seen[node]<quotas[node]:selected.append(packet);seen[node]+=1
    require(all(seen[n]==quotas[n] for n in nodes),'diagnostic selected draw lacks configured inputs')
    return selected


def lane_plan(plan, capture, routes):
    """Bind selected groups to the configured and authenticated TP4 instances."""
    lanes = capture['lanes']
    width = (1, 2) if plan.get('sampled_window_ref') else (2,)
    require(len(lanes) in width and len(lanes) == len(set(lanes)) and set(lanes) == set(routes),
            'configured physical lanes must match the authenticated routes')
    require(set(capture['lane_transports']) == set(lanes) and
            all(routes[name]['endpoint'] == capture['lane_transports'][name]['remote_endpoint']
                for name in lanes), 'physical lane destination differs from activation')
    indices = [draw['draw_index'] for draw in plan['draws']]
    require(indices and indices == list(range(len(indices))), 'capture draw indices are not contiguous')
    if not plan.get('sampled_window_ref'):
        require(len(indices) == len(lanes), 'bounded capture requires one group per lane')
    require(len({capture['lane_transports'][name]['local_endpoint'] for name in lanes}) == len(lanes),
            'TP4 lanes must have distinct local endpoints')
    return {index: lanes[index % len(lanes)] for index in indices}


def episode_concurrency(config, plan, lane, replicates):
    capture = config['capture']
    settings = capture.get('episode_concurrency')
    if settings is None:
        return 1
    require(plan.get('sampled_window_ref') and lane is not None and
            set(settings) == set(capture['lanes']),
            'episode concurrency requires an explicit formal TP4 lane profile')
    mode=capture.get('dispatch_unit','recording')
    require(mode in ('recording','training_batch'),'unknown rollout dispatch unit')
    limit=replicates if mode=='recording' else len(plan['draws'])*replicates
    require(all(type(n) is int and n>=1 and (mode=='training_batch' or n<=limit) for n in settings.values()),
            'invalid per-lane in-flight capacity')
    return settings[lane]


def complete_wave(episodes, concurrency):
    """Finish CPU admission, run only HTTP concurrently, parse in original order."""
    if concurrency == 1:
        return [finish(request_once(transport, packet)) for transport, packet, finish in episodes]
    replies = {}
    new = {}
    for i, (transport, packet, _) in enumerate(episodes):
        if transport.inspect(packet)['state'] == 'absent':
            new[i] = (transport, packet)
        else:
            # Resolve every old intent before submitting any fresh work in
            # this wave. A crash after admission is not a request outcome.
            replies[i] = request_once(transport, packet)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        requests = {i:pool.submit(request_once, transport, packet) for i,(transport,packet) in new.items()}
        for i, request in requests.items():
            replies[i] = request.result()
    return [finish(replies[i]) for i, (_, _, finish) in enumerate(episodes)]


def dispatch_draws(draws,prepare,concurrency,unit):
    """Schedule a whole policy window; keep admission and result writes serial."""
    if unit=='recording':
        for draw in draws:
            pending=[]
            for item in prepare(draw):
                pending.append(item)
                if len(pending)==concurrency:
                    for (_,store),result in zip(pending,complete_wave([p for p,_ in pending],concurrency),strict=True):store(result)
                    pending.clear()
            if pending:
                for (_,store),result in zip(pending,complete_wave([p for p,_ in pending],concurrency),strict=True):store(result)
        return
    require(unit=='training_batch' and concurrency>=1,'wrong training dispatch mode')
    # Prepare all lane-owned requests, reconciling every old intent before
    # any new contact. This also handles partial groups and the last short batch.
    pending=[];seen=set()
    for draw in draws:
        for (transport,packet,finish),store in prepare(draw):
            rid=packet['request_id']
            require(rid not in seen,'duplicate request in training batch');seen.add(rid)
            if transport.inspect(packet)['state']=='absent':pending.append((transport,packet,finish,store))
            else:store(finish(request_once(transport,packet)))
    remaining=iter(pending)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        active={}
        def fill():
            while len(active)<concurrency:
                job=next(remaining,None)
                if job is None:break
                transport,packet,finish,store=job
                active[pool.submit(request_once,transport,packet)]=finish,store
        fill()
        while active:
            ready,_=wait(active,return_when=FIRST_COMPLETED)
            results=[]
            for future in ready:
                finish,store=active.pop(future);results.append((finish,store,future.result()))
            for finish,store,reply in results:store(finish(reply))
            fill()


def admitted_routes(current, endpoint, profile, sampled):
    from bounded_tp4 import validate_routes
    from publication import validate_publication
    if current['kind']=='branch_initial':
        pub=read(current['admission_ref'])
        require(sampled and pub['status']=='published_for_fresh_sampling' and
                pub['initial_publication_ref']==current['publication_ref'] and
                pub['assets']==current['assets'] and pub['artifact_version']==current['version'],
                'initial adapter has no matching training admission')
    else:
        pub=validate_publication(current['publication_ref'])
    return validate_routes(pub,endpoint,profile,sampled=sampled)


def capture(config,root,*,execution_enabled=False,lease=None,draw_index=None,lane=None):
    """Real generation via the existing DirectHTTP protocol; actor capture later."""
    root=Path(root);require(execution_enabled is True,'capture disabled')
    require((root/'SELECTED.json').exists(),'prepare single input first')
    require(read(pin(root/'INPUT.json'))==config,'input changed after prepare')
    c=config['capture'];storage(root,c['runtime_storage_budget_bytes'])
    plan=read(pin(root/'SELECTED.json'));learning=read(config['learning_config_ref']);clock=read(pin(root/'RESERVED-CLOCK.json'))
    from policy import validate
    current,reference=validate(config['policy'],learning,clock)
    require(c['unknown']=='journal_reconcile_never_resend' and c['retry'] is False,'no replay/refill')
    protocol=load(BASE/'rollout-capture-contract/capture.py','fresh_original_capture_protocol')
    profile=read(c['profile_ref']);protocol.validate_profile(profile)
    require(c['endpoint_policy_ref']==config['policy']['runtime_evidence_ref'],'endpoint must be the witnessed published policy')
    runtime_evidence=read(c['endpoint_policy_ref'])
    base_transport=read(c['transport_config_ref'])
    routes=None
    if runtime_evidence.get('schema') in ('omni-bounded-tp4-endpoints-v1','continuous-training-endpoints-v1'):
        routes=admitted_routes(current,c['endpoint_policy_ref'],profile,bool(plan.get('sampled_window_ref')))
        assignments=lane_plan(plan,c,routes)
    else:
        require(runtime_evidence['target_http_sampling_supported'] is True and runtime_evidence['observed_token_ids_and_logprobs'] is True,'actual target HTTP token/probability protocol not witnessed')
        require(runtime_evidence['endpoint']==base_transport['endpoint'] and runtime_evidence['engine_required']==profile['engine_required'] and runtime_evidence['sampling']==profile['sampling'],'effective behavior engine/distribution not authenticated')
    require(draw_index is None or (lane is None and routes is not None and
                type(draw_index) is int and draw_index in assignments),
            'partial capture requires a bound TP4 draw')
    require(lane is None or (routes is not None and plan.get('sampled_window_ref') and lane in routes),
            'lane capture requires a frozen sampled TRAIN window')
    require(read(pin(root/'CONSUMER.json'))['sampling_distribution']==profile['sampling'],'consumer behavior distribution differs from actual sampler')
    mapping=read(read(pin(root/'CONSUMER.json'))['recording_map_ref'])
    origins={r['recording_ref_id']:r for r in mapping['records']}
    replicates=learning['kernel']['G']
    concurrency=episode_concurrency(config,plan,lane,replicates)
    batch_dispatch=c.get('dispatch_unit','recording')=='training_batch'
    if batch_dispatch:require(plan.get('sampled_window_ref') and lane is not None,'batch dispatch requires a sampled lane')
    from official import sources
    with sources(config) as (official_sources,_), business_scope():
        from direct_transport import DirectHTTP
        from business import Session
        screened_p_parser = None
        # Shared asset index remains ORIGINAL. Each episode owns isolated mutable
        # business state; duplicate query draws cannot share request identities.
        boot=read(save(root/'RUN-BOOT.json',dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())))['boot_id'] if not (root/'RUN-BOOT.json').exists() else read(pin(root/'RUN-BOOT.json'))['boot_id']
        def prepare_draw(d):
            nonlocal screened_p_parser
            index=d['draw_index']
            if draw_index is not None and index!=draw_index:return
            if lane is not None and assignments[index]!=lane:return
            directory=root/'groups'/str(d['draw_index']);directory.mkdir(parents=True,exist_ok=True)
            if (directory/'GENERATED.json').exists():return
            origin=origins[d['record']['recording_ref_id']]
            assets=[origin['audio_ref']]+origin.get('context_audio_refs',[])
            annotation=selected_annotation(d,c,root,official_sources)
            snapshot=clock['groups'][d['group_id']]['snapshot']
            require((concurrency==1 and not batch_dispatch) or (snapshot['rollout_scope']=='node' and snapshot['target_nodes']==['P']),
                    'concurrent capture requires a single independent P request per episode')
            episodes={}
            def remember(k,episode):
                require(k not in episodes and episode['replicate']==k and
                        episode['episode_id']==d['group_id']+':e'+str(k),'episode belongs to another draw')
                episodes[k]=episode
                if len(episodes)==replicates:
                    save(directory/'GENERATED.json',dict(group_id=d['group_id'],query_draw_id=d['query_draw_id'],
                        episodes=[episodes[i] for i in range(replicates)],policy=config['policy']))
            def prepare_episode(replicate):
                nonlocal screened_p_parser
                started=time.monotonic()
                ep=directory/('episode-'+str(replicate));ep.mkdir(exist_ok=True)
                eid=d['group_id']+':e'+str(replicate)
                identity=dict(run_id=plan['version'],attempt_id=eid,arm='fresh',protocol_hash=digest(profile),split_identity_hash=digest(config['screen_ref']))
                require(lease is not None,'original registry/fence required')
                cfg=base_transport
                if routes is not None:
                    name=assignments[index]
                    route=c['lane_transports'][name]
                    cfg=dict(base_transport,endpoint=route['local_endpoint'])
                    save(ep/'LANE.json',dict(lane=name,route=route,activation=runtime_evidence['lanes'][name]))
                transport=lease.transport(DirectHTTP(ep,dict(cfg,journal_identity=identity)))
                pending=[]
                def transform(packet,parents):
                    seed=int(digest(dict(seed=c['seed'],draw=d['draw_index'],replicate=replicate,node=packet['stage'],branch=packet.get('branch')))[:15],16)
                    ident=dict(group_id=d['group_id'],episode_id=eid,node=packet['stage'],fold=packet['branch'] if packet['stage']=='S1' else None,
                        attempt_id=eid,replicate=replicate,seed=seed,parents=parents,run_id=plan['version'],boot_id=boot,query_id=d['record']['sample_id'],split='TRAIN')
                    if packet['stage']=='S2':ident['slot_type']=packet['branch']
                    return sample_packet(packet,profile,ident,protocol)
                def admit(packet):
                    storage(root,c['runtime_storage_budget_bytes'])
                    if transport.inspect(packet)['state']=='absent':
                        require(lease is not None,'existing capture lease required');lease.check(packet,transport)
                def consume(packet,parse,reply):
                    reply.update(request_id=packet['request_id'],wire_sha256=packet['wire_sha256'],response_sha256=reply['native_response_sha256'])
                    reply['receipt']=dict(request_sha256=packet['wire_sha256'],response_sha256=reply['response_sha256'])
                    value=parse(packet,reply)
                    online,output=normalized(packet,value,reply,wire_assets(packet,origin,ep))
                    state=transport.inspect(packet);raw_response=state['response_bytes']
                    obj=json.loads(raw_response);choice=obj['choices'][0]
                    tokens=choice.get('token_ids');items=(choice.get('logprobs') or {}).get('content')
                    require(isinstance(tokens,list) and isinstance(items,list) and len(tokens)==len(items),'observed sampled token/logprob coverage required')
                    require(all(x['token']=='token_id:'+str(t) and x['logprob'] is not None and x['logprob']!=-9999 for t,x in zip(tokens,items)),'clamped/missing sampled probability: retain evidence, do not refill')
                    require(obj.get('model')=='qwen3-omni-30b-a3b-instruct','actual model mismatch')
                    if not tokens:output['terminal']='model_length' if reply['finish_reason']=='length' else 'protocol_error'
                    reply['receipt']=dict(request_sha256=packet['wire_sha256'],response_sha256=reply['response_sha256'])
                    envelope=reply['native_envelope'];save(ep/(packet['request_id']+'.packet.json'),packet)
                    item=dict(packet=packet,reply=reply,online=online,output=output,annotation=annotation,
                        parse_diagnostic={key:copy.deepcopy(value[key]) for key in
                            ('status','reason','error','model_failure','provenance_status','provenance_audit_errors') if key in value},
                        token_ids=tokens,behavior_logprobs=[x['logprob'] for x in items],prompt_token_ids=obj.get('prompt_token_ids'),
                        raw_output=choice['message'].get('content') or '',usage=obj.get('usage'),envelope=envelope)
                    item['annotation_ref']=annotation.get('source')
                    pending.append(item);return value,reply
                def close_episode():
                    require(all(transport.inspect(x['packet'])['state']=='observed' for x in pending),
                            'unresolved physical request cannot close episode')
                    result=dict(episode_id=eid,replicate=replicate,calls=pending,synthetic=False)
                    save(ep/'GENERATED.json',result)
                    return result
                def call(packet,parse):
                    admit(packet)
                    return consume(packet,parse,request_once(transport,packet))
                if clock['groups'][d['group_id']]['snapshot']['rollout_scope']=='node':
                    snap=clock['groups'][d['group_id']]['snapshot']
                    from rails import session_for,bind_branches,bind_selected_prepared
                    screened = 'node_input_ref' in d['record']['online']
                    if screened:
                        from material import node_screen_packet
                        candidates=[node_screen_packet(d,snap,annotation,origin)]
                        if screened_p_parser is None:
                            from rails import node_p_parser
                            screened_p_parser=node_p_parser(c)
                    else:
                        session=session_for(Session,c,ep,identity)
                        candidates=selected_material(d)
                        bind_selected_prepared(session,d)
                        sample=selected_sample(d,session)
                        session.prepare(sample)
                    from material import hydrate_packets
                    # Node training consumes only its frozen target packets.
                    # Hydrating every historical S2/D packet first introduced
                    # unrelated parsers, source reconstruction and failure gates.
                    targets=[p for p in candidates if p['stage'] in snap['target_nodes']]
                    targets=diagnostic_targets(config,targets)
                    if config.get('diagnostic_packet_selection'):
                        save(ep/'DIAGNOSTIC-TARGETS.json',dict(
                            selection=config['diagnostic_packet_selection'],
                            available=len(candidates),selected=[dict(node=p['stage'],
                            request_id=p['request_id'],wire_sha256=p['wire_sha256']) for p in targets],
                            outcome_used=False,trainable=False))
                    d_runtimes={}
                    if 'node_input_ref' not in d['record']['online']:
                        targets=hydrate_packets(session,d,targets,d_runtimes=d_runtimes)
                    require(targets and set(p['stage'] for p in targets)==set(snap['target_nodes']),'selected frozen target packets missing')
                    for packet in targets:
                        p=transform(packet,[])
                        if screened:
                            require(p['stage']=='P','screened node parser supports only P')
                            parser=screened_p_parser
                        elif p['stage']=='P':
                            from rails import bind_p
                            bind_p(session.collector(p['sample_id']).adapter)
                            parser=lambda q,r:session.collector(q['sample_id']).adapter.parse_p(q,r)
                        elif p['stage'] in ('S1','S2'):parser=lambda q,r:session.collector(q['sample_id']).adapter.parse_branches(q['stage'],[q],[r])[0]
                        else:
                            # Retained D packet contains exact original runtime
                            # source/config; factory consumes its current source.
                            # Hydration already verified and reconstructed this
                            # exact runtime. Consume it once within this episode;
                            # do not repeat knowledge/source restoration.
                            runtime,packet=d_runtimes.pop(packet['request_id'])
                            p=transform(packet,[])
                            original_request=runtime.request
                            runtime.request=lambda inner:transform(original_request(inner),[])
                            session.runtime[p['sample_id']]=runtime
                            parser=lambda q,r:session.parse_d([q],[r])
                        if concurrency>1 or batch_dispatch:
                            require(len(targets)==1 and p['stage']=='P','expected one P wire')
                            admit(p)
                            supply=ep/'SUPPLY.json'
                            if not supply.exists():
                                save(supply,dict(prepare_and_admit_seconds=time.monotonic()-started))
                            def finish(reply):
                                consume(p,parser,reply)
                                return close_episode()
                            return transport,p,finish
                        call(p,parser)
                else:
                    from rails import session_for,bind_branches,bind_selected_prepared
                    session=session_for(Session,c,ep,identity)
                    from rails import bind_selected_features
                    bind_selected_features(session,d)
                    sample=selected_sample(d,session)
                    prepared=session.prepare(sample)['prepared'];event=session.retrieve(sample)
                    # Full screen owns eligibility. If actual retrieval identity
                    # changed this is an error, never a new ineligible exclusion.
                    if config.get('prepare_mode')!='domain_boundary_acquisition':
                        require(annotation['gold_intent_id'] in event['actual_returned_ranked_intent_ids'],'retrieval changed from admitted screen; no ineligible/refill')
                    stages=dict(prepare=dict(output=dict(prepared=prepared)))
                    # Current per-episode journal identity is consumed by the
                    # original closure verifier for all node receipts.
                    session.collection.cfg=copy.deepcopy(session.collection.cfg)
                    for stage in ('P','S1','S2','D'):session.collection.cfg['stages'][stage]['journal_identity']=identity
                    actual_adapter=session.collector(sample['sample_id']).adapter
                    from rails import bind_p
                    bind_p(actual_adapter)
                    bind_branches(actual_adapter,transform)
                    ps=[transform(p,[]) for p in session.p(prepared,event)]
                    execute_system(session,sample,prepared,stages,ps,call,transform)
                return close_episode()
            for replicate in range(replicates):
                saved=directory/f'episode-{replicate}'/'GENERATED.json'
                if saved.exists():
                    remember(replicate,read(pin(saved)))
                elif concurrency==1 and not batch_dispatch:
                    remember(replicate,prepare_episode(replicate))
                else:
                    yield prepare_episode(replicate),lambda episode,k=replicate:remember(k,episode)
        dispatch_draws(plan['draws'],prepare_draw,concurrency,c.get('dispatch_unit','recording'))
    if draw_index is None and lane is None:
        close_generation(root,plan)


def close_generation(root,plan):
    for d in plan['draws']:
        group=read(pin(root/'groups'/str(d['draw_index'])/'GENERATED.json'))
        require(group['group_id']==d['group_id'],'wrong group completion')
    save(root/'GENERATION-CLOSED.json',dict(selected_cohort_ref=pin(root/'SELECTED.json'),groups=plan['group_ids'],training_ready=False,
        next='original NativeOmniWorker current/reference same-action capture -> consumer -> configured_update',automatic_next_cohort=False))


def execute_system(session,sample,prepared,stages,ps,call,transform):
    """Original independent P/S1 roots, S1->S2, original terminal D route."""
    pr=[call(p,lambda q,r:session.parse_p([q],[r]))[1] for p in ps]
    s1=session.s1(prepared)
    sr=[call(p,lambda q,r:session.parse_s('S1',[q],[r])[0])[1] for p in s1]
    # Original prepare_s2 computes supplied types from all parsed S1 branches,
    # including retained partial failures. P failure never suppresses this root.
    s2=session.s2(prepared,sr,s1)
    r2=[call(p,lambda q,r:session.parse_s('S2',[q],[r])[0])[1] for p in s2]
    for n,pp,rr in (('P',ps,pr),('S1',s1,sr),('S2',s2,r2)):
        stages[n]=dict(status='done',transport_packets=pp,transport_replies=rr)
    # Session.p cached its plain packet before sampling. Original advance must
    # replay the actual sampled root, just as close_source receives in stages.
    session.prepared[sample['sample_id']]['P']=ps[0]
    session.prepared[sample['sample_id']]['S1']=s1
    observed={n:{p['request_id']:r['native_envelope'] for p,r in zip(stages[n]['transport_packets'],stages[n]['transport_replies'])} for n in ('P','S1','S2')}
    # SAME original terminal contract decides whether the closed Source can D.
    state=session.collector(sample['sample_id']).advance(session.prepared[sample['sample_id']],observed)['terminal_contract']
    if not state['upstream_ready']:return state
    source=session.close_source(sample,stages)
    dp=session.d(sample,source)
    runtime=session.runtime[sample['sample_id']];old_request=runtime.request
    parent_ids=[p['request_id'] for p in ps+s1+s2]
    runtime.request=lambda packet:transform(old_request(packet),parent_ids)
    for p in dp:call(transform(p,parent_ids),lambda x,r:session.parse_d([x],[r]))
    return state


def finalize_on_worker(worker,adapted,native_config,root,rank,*,directory=None,
                       resident=False,cost_receipt=None,accepted_sink=None):
    """Current/reference fixed-action forwards on the ORIGINAL resident worker.

    Uses original inputs_for and forward. No worker, optimizer or trainer copy.
    Reference is explicitly the independent frozen base, never current alias.
    """
    import torch
    import torch.distributed as dist
    if not resident:
        from native_update import configure_imports
        configure_imports()
    from validate_npu import inputs_for
    from forward import forward
    from cohort import publish
    pending=read(native_config['pending_capture_ref']);config=read(pending['input_ref'])
    if native_config.get('local_official_source_ref'):
        config['capture']['official_source_config_ref']=native_config['local_official_source_ref']
    directory=Path(directory if directory is not None else pending['cohort_root'])
    plan=read(pin(directory/'SELECTED.json'))
    require(native_config['enabled'] is True,'explicit real capture/training enabled required')
    require((directory/'GENERATION-CLOSED.json').is_file(),'full selected generation not closed')
    learning=read(config['learning_config_ref']);clock=read(pin(directory/'RESERVED-CLOCK.json'))
    from policy import validate
    current,reference=validate(config['policy'],learning,clock)
    if current['kind']=='checkpoint' and not resident:
        # Restore native optimizer/current weights once BEFORE any current-policy
        # score. Original configured_update restores again for audited update.
        from training_state import restore_checkpoint
        from task_runtime import task_module
        prior_learning=read(native_config['restore_learning_config_ref'])
        prior_consumer=read(native_config['restore_consumer_config_ref'])
        restore_checkpoint(worker,native_config['restore_checkpoint_ref'],lambda s:task_module().create_clock(prior_learning,s,consumer_config=prior_consumer))
    consumer_ref=native_config.get('local_consumer_ref') or pin(directory/'CONSUMER.json')
    consumer_config=read(consumer_ref)
    mapping=read(consumer_config['recording_map_ref']);origins={r['recording_ref_id']:r for r in mapping['records']}
    processor=worker.processor;tokenizer=worker.tokenizer
    model_path=Path(native_config['model_path']);assets=[pin(model_path/n) for n in ('tokenizer.json','tokenizer_config.json') if (model_path/n).exists()]
    if rank==0:
        save(directory/'TOKENIZER.json',dict(schema='retained-tokenizer-v1',assets=assets,vocab_size=len(tokenizer),pad_token_id=tokenizer.pad_token_id))
        save(directory/'PROCESSOR.json',dict(schema='retained-audio-processor-v1',assets=assets+[pin(model_path/'preprocessor_config.json')],feature_layout='audio_feature_time',feature_bins=128,
            position_layout='retained_multi_axis',position_axes=['rope_axis','tokens'],position_non_token_shape=[3],audio_pad_token_id=tokenizer.convert_tokens_to_ids('<|audio_pad|>')))
    dist.barrier()
    version=learning['kernel']['behavior_policy_vector'];device=next(worker.actor.actor_module.parameters()).device
    from reference_scores import cached_reference,commit_reference
    def broadcast_cache(value):
        objects=[value];dist.broadcast_object_list(objects,src=0);return objects[0]
    parity_first=None
    prepared_cache=PreparedPacketCache(processor,native_config.get('prepared_cache_bytes',
        config['capture'].get('prepared_cache_bytes',64*1024*1024)))
    batched_reference=None
    if native_config.get('reference_batch'):
        from reference_batch import frozen_reference_window
        # A resumed window retains committed GROUPs. Only score uncommitted groups.
        def pending_generated():
            for draw in plan['draws']:
                folder=directory/'groups'/str(draw['draw_index'])
                if not (folder/'GROUP.json').exists():
                    generated=read(pin(folder/'GENERATED.json'))
                    require(generated['policy']==config['policy'],'reference batch mixed policy')
                    yield generated
        batched_reference,batch_stats=frozen_reference_window(worker,adapted,pending_generated(),
            prepared_cache,inputs_for,native_config['reference_batch'],cache_root=directory,
            namespace=dict(reference=config['policy']['reference_ref'],processor=pin(directory/'PROCESSOR.json'),
                native_config=digest(native_config),input_ref=pending['input_ref']))
        if rank==0:
            save(directory/('REFERENCE-BATCH-'+digest([sorted(batched_reference),batch_stats])+'.json'),batch_stats)
    sharded = native_config.get('group_preparation') == 'shard_by_draw_v1'
    if sharded:
        require(resident and accepted_sink is not None and batched_reference is not None,
                'group sharding requires completed collective reference scores')
    processor_ref = pin(directory/'PROCESSOR.json')
    tokenizer_ref = pin(directory/'TOKENIZER.json')

    def assemble_group(d):
        nonlocal parity_first
        storage(directory,config['capture']['runtime_storage_budget_bytes'])
        gd=directory/'groups'/str(d['draw_index'])
        if (gd/'GROUP.json').exists():return
        generated=read(pin(gd/'GENERATED.json'))
        static_fields = {}
        origin=origins[d['record']['recording_ref_id']];episodes=[];supervision={};costs=[];frozen={}
        snapshot=clock['groups'][d['group_id']]['snapshot']
        # One restored actor in eval mode, one recording group, no optimizer step.
        group_scores={} if native_config.get('reuse_group_current_scores',False) else None
        for ge in generated['episodes']:
            calls=[];off={};idmap={x['packet']['request_id']:ge['episode_id']+':c'+str(i) for i,x in enumerate(ge['calls'])}
            for ci,item in enumerate(ge['calls']):
                packet=item['packet'];node=packet['stage'];tokens=item['token_ids'];online=item['online'];output=item['output']
                cache_hits_before=prepared_cache.hits
                prepared=prepared_cache.prepare(packet)
                prepared_cache_hit=prepared_cache.hits>cache_hits_before
                require(item['prompt_token_ids']==prepared['encoded']['input_ids'][0].tolist(),'actual HTTP/HF prompt tokens differ; preserve capture, no refill')
                inputs=inputs_for(prepared,tokens,adapted,device)
                if batched_reference is not None:
                    # The legacy scalar CPU reducer requires the pre-update policy
                    # probabilities. Same-version rollout observed these directly.
                    # Native GRPO's differentiable current scores come from actor.forward.
                    require(config['policy'].get('formal_sampled') and generated['policy']==config['policy']
                            and current['step']==clock['successful_optimizer_steps'],
                            'sampled pre-update policy does not match this window')
                    cur=item['behavior_logprobs']
                    refp,reference_entry=batched_reference[packet['request_id']]
                    reused_current=True
                else:
                    was_training=worker.actor.actor_module.training;worker.actor.actor_module.eval()
                    try:
                        score_key=None
                        if group_scores is not None:
                            score_key=(prepared_cache.conditioning_key(packet),node,
                                       tuple(item['prompt_token_ids']),tuple(tokens))
                        reused_current=group_scores is not None and score_key in group_scores
                        with torch.no_grad():
                            if reused_current:
                                cur=group_scores[score_key]
                            elif tokens:
                                _,cur=forward(worker.actor,inputs,temperature=1.,feature_dtype=torch.bfloat16)
                                cur=cur[0].detach().float().cpu().tolist()
                            else:cur=[]
                            handle=getattr(worker.actor.actor_module,'_handle',None)
                            if handle is not None:handle.reshard(True)
                        def compute_reference():
                            with torch.no_grad(),adapted.disable_adapter():
                                if tokens:
                                    _,value=forward(worker.actor,inputs,temperature=1.,feature_dtype=torch.bfloat16)
                                    value=value[0].detach().float().cpu().tolist()
                                else:value=[]
                                handle=getattr(worker.actor.actor_module,'_handle',None)
                                if handle is not None:handle.reshard(True)
                                return value
                        refp,reference_entry=cached_reference(directory,dict(
                            input_ref=pending['input_ref'],reference_ref=config['policy']['reference_ref'],
                            native_config=digest(native_config),request_id=packet['request_id'],
                            conditioning_sha256=digest(packet['wire']['messages']),
                            prompt_token_ids=item['prompt_token_ids'],action_token_ids=tokens,
                            processor_ref=pin(directory/'PROCESSOR.json')),len(tokens),compute_reference,
                            rank=rank,broadcast=broadcast_cache)
                    finally:worker.actor.actor_module.train(was_training)
                    peers=[None]*dist.get_world_size();dist.all_gather_object(peers,dict(current=cur,reference=refp))
                    require(all(p==peers[0] for p in peers),'same-action rank capture differs')
                    if group_scores is not None and tokens and not reused_current:
                        group_scores[score_key]=cur
                    commit_reference(reference_entry,refp,rank=rank,broadcast=broadcast_cache)
                    old=torch.tensor(item['behavior_logprobs']);now=torch.tensor(cur)
                    tolerance=native_config['parity']
                    compare_behavior(directory,packet,item,old,now,tolerance,rank)
                    if current['kind']=='base':require(cur==refp,'initial base vs zero-init adapter identity failed')
                enc=prepared['encoded'];prompt=enc['input_ids'][0].tolist();pos=inputs['position_ids'][0].cpu().tolist()
                # Only inputs_for's action-dependent positions are rebuilt per reply.
                # Scope to this draw, keyed by the full processor input including audio.
                key_static = prepared_cache.conditioning_key(packet) if sharded else None
                if not sharded or key_static not in static_fields:
                    audio = dict(audio_features=enc['input_features'].tolist(),
                                 feature_mask=enc['feature_attention_mask'].long().tolist(),
                                 effective_audio_lengths=enc['feature_attention_mask'].sum(-1).tolist())
                    if sharded:static_fields[key_static] = audio
                else:
                    audio = static_fields[key_static]
                f=dict(prompt_token_ids=prompt,action_token_ids=tokens,behavior_logprobs=item['behavior_logprobs'],action_mask=[1]*len(tokens),
                    behavior_policy_version=version[node],adapter_vector=version,probability_semantics='same_effective_behavior_distribution_and_action_history',
                    sampling_distribution=consumer_config['sampling_distribution'],processor_manifest=processor_ref,tokenizer_manifest=tokenizer_ref,**audio,
                    attention_mask=[1]*(len(prompt)+len(tokens)),position_ids=pos,modality_assets=online['audio_assets'],recording_id=origin['capture_recording_id'],
                    audio_token_lengths=audio_spans(prompt,tokenizer.convert_tokens_to_ids('<|audio_pad|>')),prompt_messages=packet['wire']['messages'],prompt_text=prepared['prompt'],
                    action_text=item['raw_output'],decoded_action_text=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                    input_boundary='online_only_no_reference',capture_origin='runtime_capture')
                # Retain exact decode: any EOS/text policy mismatch stops admission.
                require(f['action_text']==f['decoded_action_text'],'decoded sampled token text differs')
                f['captured_tensor_schema']=dict(processor_manifest_sha256=digest(f['processor_manifest']),consumer_contract_version='unified-omni-retained-padded-v1',
                    tensors={k:dict(shape=list(torch.tensor(f[k]).shape),axes=a) for k,a in dict(audio_features=['audio','feature','time'],feature_mask=['audio','time'],position_ids=['rope_axis','tokens'],effective_audio_lengths=['audio']).items()})
                f['capture_binding']=dict(online_input_sha256=digest(online),output_sha256=digest(output),fields_sha256=digest(f))
                identity=packet['fresh_capture_identity'];request_identity={k:identity[k] for k in ('boot_id','run_id','attempt_id')};request_identity['request_id']=packet['request_id']
                cid=idmap[packet['request_id']];key=node+(':'+str(packet['branch']) if node in ('S1','S2') else '')
                call=dict(call_id=cid,node=node,input_key=key,parents=[idmap[p] for p in identity['parents']],request_identity=request_identity,
                    observed_terminal=True,online_gold_injected=False,online_input=online,output=output,behavior_policy_version=version[node],adapter_vector=version,
                    probability_semantics=f['probability_semantics'],token_ids=tokens,token_kinds=['generated_eos' if t==tokenizer.eos_token_id else 'generated_action' for t in tokens],
                    action_mask=[1]*len(tokens),old_logprobs=item['behavior_logprobs'],current_logprobs=cur,reference_logprobs=refp,training_fields=f,raw_output=item['raw_output'])
                if batched_reference is not None:
                    call['preupdate_logprob_source']='observed_actor_rollout_same_policy'
                if not tokens:call['zero_action_terminal_observed']=True
                # Offline source facts stay in the separated supervision file.
                call['receipt']=dict(request_identity=request_identity,episode_id=ge['episode_id'],call_id=cid,payload_sha256=digest(call))
                calls.append(call)
                if key in frozen:require(frozen[key]==online or snapshot['rollout_scope']=='system','node inputs changed inside group')
                else:frozen[key]=online
                part=dict(online=online,output=output,labels=labels_for(node,online,item['annotation']),call_ids=[cid])
                component='system' if node=='D' and snapshot['rollout_scope']=='system' else node
                off.setdefault(component,dict(calls=[]))['calls'].append(part)
                if component=='system':off.setdefault('D',dict(calls=[]))['calls'].append(copy.deepcopy(part))
                costs.append(dict(request_id=packet['request_id'],usage=item['usage'],journal=item['envelope'],current_forward_tokens=len(prompt)+len(tokens) if tokens and not reused_current else 0,
                    reference_forward_tokens=len(prompt)+len(tokens) if tokens and not reference_entry['hit'] else 0,
                    reference_cache_hit=reference_entry['hit'],
                    reference_score_source_request_id=reference_entry['source_request_id'],
                    prepared_input_cache_hit=prepared_cache_hit,
                    currency_cost=None,npu_seconds=None))
                if group_scores is not None:
                    costs[-1]['current_score_cache_hit']=reused_current
                if batched_reference is None and parity_first is None and snapshot['trainable_mask'][node] and tokens:
                    parity_first=(prepared,item,pos,dict(group_id=d['group_id'],episode_id=ge['episode_id'],call_id=cid,node=node))
            terminal='valid_final' if all(c['output']['terminal']=='valid_final' for c in calls) else next(c['output']['terminal'] for c in calls if c['output']['terminal']!='valid_final')
            ep=dict(episode_id=ge['episode_id'],replicate=ge['replicate'],query_draw_id=d['query_draw_id'],query_id=d['record']['sample_id'],recording_id=origin['capture_recording_id'],sampling_round=plan['version'],
                terminal=terminal,physical_requests_drained=True,rollout_scope=snapshot['rollout_scope'],request_roster=[c['request_identity'] for c in calls],expected_call_keys=[c['input_key'] for c in calls],calls=calls)
            if snapshot['rollout_scope']=='system':ep.update(system_episode_closed=True,s2_requested_input_keys=[c['input_key'] for c in calls if c['node']=='S2'],early_termination_accounted=True,joint_labels_observable=labels_for('D',{},ge['calls'][0]['annotation']) is not None)
            episodes.append(ep);supervision[ge['episode_id']]=off
        group=dict(group_id=d['group_id'],query_draw_id=d['query_draw_id'],query_id=d['record']['sample_id'],recording_id=origin['capture_recording_id'],sampling_round=plan['version'],split='TRAIN',
            gold_in_actual_pre_P_dynamic_retrieval=True,fresh_on_policy=True,closed=True,system_version=learning['system_version'],baseline=learning['kernel']['baseline'],behavior_policy_vector=version,
            group_size=learning['kernel']['G'],snapshot=snapshot,frozen_upstream_release=snapshot['frozen_upstream_release'],frozen_inputs=frozen if snapshot['rollout_scope']=='node' else {},episodes=episodes,
            closure=dict(group_id=d['group_id'],episode_ids=[e['episode_id'] for e in episodes],snapshot_sha256=snapshot['snapshot_sha256']))
        result=dict(group_id=d['group_id'],query_draw_id=d['query_draw_id'],policy=config['policy'],synthetic=False,costs=costs,
            task_records=[dict(group_id=d['group_id'],episode_id=ge['episode_id'],call_id=ge['episode_id']+':c'+str(i),node=item['packet']['stage'],annotation=item['annotation'],packet=item['packet']['original_packet']) for ge in generated['episodes'] for i,item in enumerate(ge['calls'])],
            online=dict(online=dict(audio_assets=[origin['audio_ref']]),behavior_bundle=dict(schema='unified-behavior-bundle-v1',synthetic=False,online_gold_injected=False,global_query_count=1,groups=[group])),
            offline=dict(supervision=d['record']['offline_supervision'],group_supervision=supervision))
        if sharded:
            from cohort import group_storage
            save(gd/'GROUP.json', group_storage(result))
        elif rank == 0:
            save(gd/'GROUP.json',result)

    if sharded:
        from resident import collectively
        def assemble_owned():
            for draw in plan['draws'][rank::dist.get_world_size()]:
                assemble_group(draw)
        collectively(assemble_owned)
    else:
        for draw in plan['draws']:
            assemble_group(draw)
            dist.barrier()
    parity_first=None
    if batched_reference is None and parity_first is None:
        # Recover first trainable row from durable captured groups; do not sample
        # or repeat actor/reference forwards when GROUP is already committed.
        for d in plan['draws']:
            data=read(pin(directory/'groups'/str(d['draw_index'])/'GENERATED.json'))
            for ge in data['episodes']:
                for ci,item in enumerate(ge['calls']):
                    node=item['packet']['stage']
                    if clock['groups'][d['group_id']]['snapshot']['trainable_mask'][node] and item['token_ids']:
                        prepared=prepared_cache.prepare(item['packet'])
                        retained=read(pin(directory/'groups'/str(d['draw_index'])/'GROUP.json'))
                        call=retained['online']['behavior_bundle']['groups'][0]['episodes'][ge['replicate']]['calls'][ci]
                        parity_first=(prepared,item,call['training_fields']['position_ids'],{})
                        break
                if parity_first:break
            if parity_first:break
    def close_on_publisher():
        nonlocal config
        if rank!=0:return
        # Reuse actual first actor row for original checkpoint->rollout parity.
        # Generated from real capture, never user-supplied future hashes.
        if parity_first:
            prepared,item,positions,ident=parity_first
            target=directory/'PARITY';target.mkdir(exist_ok=True)
            from native_artifacts import save_torch
            if not (target/'prepared.pt').exists():save_torch(target/'prepared.pt',prepared)
            save(target/'capture.json',dict(records=[dict(prompt_token_ids=item['prompt_token_ids'],token_ids=item['token_ids'])]))
            save(target/'positions.json',dict(positions=positions))
            config=copy.deepcopy(config);config['native'].update(parity_input_refs={k:pin(target/name) for k,name in dict(prepared='prepared.pt',capture='capture.json',positions='positions.json').items()},parity_record_index=0)
        release=publish(config,directory,**({'row_encoding':native_config['row_encoding']} if 'row_encoding' in native_config else {}),**({'accepted_sink':accepted_sink} if accepted_sink else {}),**({'consumer_ref':consumer_ref} if native_config.get('local_consumer_ref') else {}))
        generated_native=read(release['native_config_ref']);generated_native['enabled']=True
        from execution import cost_plan
        generated_native.update(cost_receipt if cost_receipt is not None else cost_plan(config,directory,worker.tokenizer))
        generated_native.pop('pending_capture_ref',None)
        save(directory/'NATIVE.accepted.json',generated_native)
    if accepted_sink is not None:
        from resident import collectively
        collectively(close_on_publisher)
    else:
        close_on_publisher()
        dist.barrier()
    return read(pin(directory/'NATIVE.accepted.json'))


def compare_behavior(directory,packet,item,old,now,tolerance,rank):
    require(old.shape==now.shape,'behavior/current action count differs')
    delta=old-now
    passed=bool((delta.abs()<=tolerance['token_atol']+
        tolerance['token_rtol']*old.abs()).all()) and abs(float(delta.sum()))<=tolerance['sequence_atol']
    if not passed and rank==0:
        absolute=delta.abs()
        save(directory/'BEHAVIOR-PARITY-FAILED.json',dict(request_id=packet['request_id'],
            node=packet['stage'],prompt_tokens=len(item['prompt_token_ids']),
            action_tokens=len(item['token_ids']),thresholds=tolerance,
            max_token_abs=float(absolute.max()) if absolute.numel() else 0.0,
            max_token_index=int(absolute.argmax()) if absolute.numel() else None,
            sequence_abs=abs(float(delta.sum()))))
    require(passed,'behavior/current parity failed; no training')


def audio_spans(tokens,audio_id):
    lengths=[];previous=-2
    for i,t in enumerate(tokens):
        if t==audio_id:
            if i!=previous+1:lengths.append(0)
            lengths[-1]+=1;previous=i
    return lengths


class PreparedPacketCache:
    """Bounded worker-local CPU cache; no model outputs or action state.

    The processor is fixed for this instance. Conditioning includes audio,
    message order and every option read by prepare_packet/render_prompt. Decode
    seed and limits do not affect preprocessing and vary between replicas.
    Copies protect against consumers that
    mutate tensors while constructing action-dependent training inputs.
    """
    def __init__(self,processor,max_bytes=64*1024*1024):
        from collections import OrderedDict
        require(type(max_bytes) is int and max_bytes>=0,'prepared cache byte budget')
        self.processor=processor;self.max_bytes=max_bytes;self.entries=OrderedDict()
        self.bytes=0;self.hits=0;self.misses=0

    @staticmethod
    def size(value):
        import sys
        if hasattr(value,'numel') and hasattr(value,'element_size'):
            return value.numel()*value.element_size()+sys.getsizeof(value)
        if hasattr(value,'nbytes'):return int(value.nbytes)+sys.getsizeof(value)
        if isinstance(value,dict):return sys.getsizeof(value)+sum(PreparedPacketCache.size(k)+PreparedPacketCache.size(v) for k,v in value.items())
        if isinstance(value,(list,tuple)):return sys.getsizeof(value)+sum(PreparedPacketCache.size(v) for v in value)
        return sys.getsizeof(value)

    @staticmethod
    def conditioning_key(packet):
        wire=packet['wire']
        return digest(dict(messages=wire['messages'],
            chat_template_kwargs=wire.get('chat_template_kwargs',{}),
            add_generation_prompt=wire.get('add_generation_prompt',True),
            continue_final_message=wire.get('continue_final_message',False)))

    def prepare(self,packet):
        key=self.conditioning_key(packet)
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key)
            return copy.deepcopy(self.entries[key][0])
        self.misses+=1
        value=prepare_packet(self.processor,packet);size=self.size(value)+self.size(key)+128
        if size<=self.max_bytes:
            while self.entries and self.bytes+size>self.max_bytes:
                _,(_,old_size)=self.entries.popitem(last=False);self.bytes-=old_size
            self.entries[key]=(copy.deepcopy(value),size);self.bytes+=size
        return value


def prepare_packet(processor,packet):
    """Same processor/waveform boundary as original prepare_cpu, no model load."""
    import io
    import numpy as np
    import soundfile as sf
    messages=copy.deepcopy(packet['wire']['messages']);waves=[]
    for m in messages:
        if isinstance(m['content'],list):
            for i,item in enumerate(m['content']):
                if item['type']=='input_audio':
                    wave,rate=sf.read(io.BytesIO(base64.b64decode(item['input_audio']['data'])),dtype='float32')
                    require(rate==16000 and wave.ndim==1,'original mono16k wire required')
                    wave=np.pad(wave,(0,(-len(wave))%160));waves.append(wave);m['content'][i]=dict(type='audio',audio=wave)
    require(waves,'original audio required')
    protocol=load(BASE/'rollout-capture-contract/capture.py','actor_prompt_contract')
    prompt=protocol.render_prompt(processor,messages,packet['wire'])
    enc=processor(text=[prompt],audio=waves,sampling_rate=16000,return_tensors='pt',padding=True,truncation=False)
    return dict(encoded=dict(enc),waves=waves,prompt=prompt)


def selected_annotation(draw,capture_config,root,official_sources=None):
    from official import annotation,sources
    if official_sources is None:
        with sources(dict(capture=capture_config)) as (bound,_):return selected_annotation(draw,capture_config,root,bound)
    result=annotation(draw,official_sources)
    save(Path(root)/'annotations'/(digest(draw['record']['recording_ref_id'])+'.json'),result)
    return result


def selected_sample(draw,session):
    proof=session.proof(draw['record']['recording_ref_id'])
    require(proof['sample_id']==draw['record']['sample_id'] and proof['source_id']==draw['record']['source_id'],'selected source proof mismatch')
    return dict(sample_id=proof['sample_id'],reference_id=proof['reference_id'],source_id=proof['source_id'],domain=proof['domain'],split='TRAIN',recording_file=proof['recording_file'],source_record_sha256=proof['source_record_sha256'])


def wire_assets(packet,origin,ep):
    import hashlib
    candidates=[origin['audio_ref']]+origin.get('context_audio_refs',[])
    ordered=[]
    for m in packet['wire']['messages']:
        if isinstance(m['content'],list):
            for item in m['content']:
                if item.get('type')=='input_audio':
                    data=base64.b64decode(item['input_audio']['data']);sha=hashlib.sha256(data).hexdigest()
                    matches=[a for a in candidates if a['sha256']==sha and a['bytes']==len(data)]
                    # Derive wire custody from original pinned audio using the
                    # exact existing prepare_cpu decoder. Store both identities.
                    if not matches:
                        from material import materialize_audio
                        for asset in candidates:
                            raw(asset)
                            if materialize_audio(asset['path'])==data:matches.append(asset)
                    require(len(matches)==1,'actual wire waveform differs from pinned original/context asset')
                    save(Path(ep)/'audio-derivations'/(sha+'.json'),dict(original_ref=matches[0],wire_sha256=sha,wire_bytes=len(data),
                        equality='exact original bytes or original PCM16 materializer bytes',wire_retained_in_packet=True))
                    ordered.append(matches[0])
    require(ordered and any(a['sha256']==origin['audio_ref']['sha256'] for a in ordered),'primary original audio absent from wire mapping')
    return [dict(asset,media_type=asset.get('media_type','audio')) for asset in ordered]
