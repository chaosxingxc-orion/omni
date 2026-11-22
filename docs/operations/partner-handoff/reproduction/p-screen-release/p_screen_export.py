"""Project confirmed P screen observations into immutable inputs for fresh node rollout.

Check the regenerated original business wire against the actual submitted wire.
Offline labels, screen receipt and transport audio remain independently addressable.
"""
import argparse,base64,hashlib,json,sqlite3,sys,random,contextlib
from collections import Counter,defaultdict
from pathlib import Path
from dataset_encoder import put,save,doc
from p_screen_inputs import Inputs,lines
from p_screen_run import transport_payload
def digest(raw):return hashlib.sha256(raw).hexdigest()
def ref(p):
    b=p.read_bytes()
    return dict(path=str(p.resolve()),bytes=len(b),sha256=digest(b))
def transport_proof(packet,row,response,binding,policy):
    if binding is None:
        if digest(transport_payload(packet))!=row['wire'] or response.get('model')!=packet['wire']['model']:
            raise ValueError('unbound transport or response differs from source')
        return {}
    alias=binding['model']
    if (binding['source_wire']!=packet['wire_sha256'] or binding['rid']!=row['rid']
            or alias!=policy['version']):
        raise ValueError('screen policy, source, or response mismatch')
    engine=binding['engine_id']
    if not isinstance(engine,str) or len(engine)!=64 or any(c not in '0123456789abcdef' for c in engine):
        raise ValueError('invalid engine identity')
    overlay=row['wire']==binding['source_wire']
    model=packet['wire']['model'] if overlay else alias
    if response.get('model')!=model:raise ValueError('response differs from bound wire model')
    payload=transport_payload(packet,dict(base=packet['wire']['model'],alias=alias,policy_overlay=overlay))
    if digest(payload)!=row['wire']:raise ValueError('screen transport payload differs')
    return dict(transport=dict(source_sha256=packet['wire_sha256'],wire_sha256=row['wire'],
                               model=model,engine_id=engine,**({'policy_version':alias} if overlay else {})))

def materialize(x,bound,row,dest,policy,binding=None):
    rid=row['rid'];g=x.gate[rid];assert row['status']=='valid'
    meta,official=bound.row(rid)
    labels=bound.offline(rid)
    assert labels['gold_intent_id']==x.gold[rid] and official['recording_file']==g['recording'] and official['utterance_id']==g['source_id']
    sid='slurp:train:'+official['utterance_id'].zfill(6)+':'+Path(official['recording_file']).stem
    p=x.packet(rid);order=[r['label'] for r in p['ranking']]
    assert order==json.loads(row['ranking'])
    raw=bytes(row['raw']);response=json.loads(raw);choice,=response['choices']
    assert choice['finish_reason']=='stop'
    transport=transport_proof(p,row,response,binding,policy)
    pred=[r['label'] for r in x.parse(choice['message']['content'],p['ranking'],3)]
    gold=x.gold[rid];rank=pred.index(gold)+1 if gold in pred else 0
    labels_p = g['labels'] if 'witnesses' in g else x.meta[rid][2]
    assert rank==row['rank'] and labels_p[g['rank']-1]==gold
    key=digest(rid.encode());folder=dest/'records'/key[:2]/key
    # No actions/probabilities from the screening reply enter a new behavior group.
    # A generated request ID is reserved for fresh capture by the original transform.
    packet=dict(reference_id=rid,sample_id=sid,stage='P',branch='shared',version=policy['version'],
        request_id='screen-wire:'+row['wire'],wire=p['wire'],wire_sha256=p['wire_sha256'],ranking=p['ranking'],
        actual_returned_ranked_intent_ids=labels_p)
    save(folder/'PACKET.json',packet)
    put(folder/'RESPONSE.json',raw)
    part,=[v for m in p['wire']['messages'] if isinstance(m['content'],list) for v in m['content'] if v.get('type')=='input_audio']
    wav=base64.b64decode(part['input_audio']['data'],validate=True)
    audio=dest/'pcm'/(digest(wav)+'.wav');put(audio,wav)
    origin=dict(recording_ref_id=rid,source_id=official['source_id'],source='SLURP',domain=row['domain'],
        split='TRAIN',capture_recording_id=rid,recording_filename=audio.name,audio_ref=ref(audio),context_audio_refs=[])
    save(folder/'ORIGIN.json',origin)
    proof=dict(schema='node-screen-observation-v1',node='P',reference_id=rid,sample_id=sid,split='TRAIN',
        source_id=official['source_id'],source_record_sha256=official['source_record_sha256'],audio_sha256=g['audio_sha256'],
        packet_ref=ref(folder/'PACKET.json'),response_ref=ref(folder/'RESPONSE.json'),policy_version=policy['version'],
        domain=row['domain'],lane=row['lane'],issued=row['issued'],finished=row['finished'],dynamic_gold_rank=g['rank'],
        response_gold_rank=rank,valid=True,**transport)
    save(folder/'OBSERVATION.json',proof)
    record=dict(recording_ref_id=rid,sample_id=sid,source_id=official['source_id'],source='SLURP',domain=row['domain'],split='TRAIN',
        synthetic_fixture=False,rollout_scope='node',outcome='success' if rank else 'error',
        online_gold_injected=False,online=dict(node_input_ref=ref(folder/'PACKET.json')),
        offline_supervision=dict(labels=labels),screen_observation_ref=ref(folder/'OBSERVATION.json'))
    save(folder/'RECORD.json',record)
    return dict(record_ref=ref(folder/'RECORD.json'),recording_origin_ref=ref(folder/'ORIGIN.json'))
def usage_receipt(db):
    counts={};cells={};prompt=completion=0
    for status,raw,finished,domain,rank in db.execute('select status,raw,finished,domain,rank from calls'):
        if status=='valid':
            cell=domain+'/'+('success' if rank else 'error');cells[cell]=cells.get(cell,0)+1
        if finished is None or status not in ('valid','invalid'):raise ValueError('unreconciled P screen request: '+status)
        usage=json.loads(raw)['usage'];p=usage['prompt_tokens'];c=usage['completion_tokens']
        if not(type(p) is type(c) is int and min(p,c)>=0 and p+c==usage['total_tokens']):
            raise ValueError('invalid P screen token accounting')
        prompt+=p;completion+=c;counts[status]=counts.get(status,0)+1
    result=dict(requests=sum(counts.values()),statuses=counts,valid_cells=cells,prompt_tokens=prompt,completion_tokens=completion,total_tokens=prompt+completion)
    if db.execute("select 1 from sqlite_master where type='table' and name='screen_pre_generation_refusals'").fetchone():
        rejected=db.execute('select count(*) from screen_pre_generation_refusals').fetchone()[0]
        result.update(pre_generation_refusals=rejected,http_attempts=result['requests']+rejected)
    return result


def project_v2(config_paths, dest, *, selection_policy, preview=False):
    """Seal a sample from disjoint, fully screened admissions of the same bank."""
    from p_screen_delta import DeltaInputs
    from p_screen_run import apport
    from recall_cpu_prepare import checked
    if selection_policy!='cap_each_cell_at_available':
        raise ValueError('unsupported full screen selection policy')
    configs = [doc(p) for p in config_paths]
    c = configs[0]
    d = c['recall_delta']
    base = Path(c['source_root'])/'task-loss-integration/official-source-successor'
    sys.path.insert(0,str(base))
    from sources import OfficialTrainSources
    program=doc(c['program']); policy=doc(checked(program['policy_iteration']['policy_ref']))
    request=doc(checked(program['request_ref']));capture=doc(checked(request['input_ref']))['capture']
    version_file=Path(d['bank'])/'READBACK.json'
    expected=doc(version_file)
    s=dict(c['sampling']);s.pop('completion',None)
    maximum=s['sample_count']
    quota={f'{dom}/{out}':n for out,total in apport(maximum,s['error_success']).items()
           for dom,n in apport(total,s['domain_mix']).items()}
    if not dest.is_dir():
        dest.mkdir(mode=0o700,parents=True)
    inventory={}
    scored={}
    groups=defaultdict(list)
    sources=[]
    usage=Counter()
    status=Counter()
    valid_cells=Counter()
    with contextlib.ExitStack() as stack:
        for path,config in zip(config_paths,configs,strict=True):
            if (any(config[k]!=c[k] for k in ('sampling','source_root','program','business','model_bindings_by_port'))
                    or any(config['recall_delta'][k]!=d[k] for k in ('trace','bank'))):
                raise ValueError('mixed screening identities')
            root=Path(config['output'])
            x=DeltaInputs(config)
            if inventory.keys() & x.gate.keys():
                raise ValueError('overlap between screening queues')
            inventory.update(x.gate)
            if (x.receipt['knowledge_manifest_sha256']!=expected['manifest_sha256'] or
                    x.receipt['recall_results_sha256']!=expected['results_sha256']):
                raise ValueError('mixed retrieval evidence')
            db=stack.enter_context(contextlib.closing(sqlite3.connect((root/'p-screen.sqlite').as_uri()+'?mode=ro',uri=True)))
            db.row_factory=sqlite3.Row
            if not preview:
                seal=doc(root/'COMPLETE.json')
                evidence=checked(dict(path=str(root/'P-DELTA-SCORED.private.jsonl'),sha256=seal['scored_sha256']))
                q=usage_receipt(db)
                if (seal['candidate_sha256']!=x.receipt['queue_sha256'] or
                        seal['screened']!=q['requests'] or q['requests']!=len(x.gate) or seal['statuses']!=q['statuses']):
                    raise ValueError('incomplete screening queue')
                source_rows={v['reference_id']:v for v in lines(evidence)}
                if len(source_rows)!=seal['valid'] or len(source_rows)!=q['statuses'].get('valid',0):
                    raise ValueError('score seal differs')
                status.update(q['statuses']);valid_cells.update(q['valid_cells'])
                usage.update({k:q[k] for k in ('requests','prompt_tokens','completion_tokens','total_tokens')})
                sources.append(dict(config_ref=ref(path),queue_ref=ref(root/'DELTA.json'),
                                    complete_ref=ref(root/'COMPLETE.json')))
            else:
                source_rows={}
            for rid,dom,rank in db.execute("select rid,domain,rank from calls where status='valid' order by rid"):
                if rid not in x.gate or rank not in (0,1,2,3) or dom!={
                        'train_real':'real','train_synth':'synthetic'}[rid.split(':')[0]]:
                    raise ValueError('record identity differs from queue')
                cell=dom+'/'+('success' if rank else 'error')
                groups[cell].append((rid,db,x))
                if not preview:
                    z=source_rows.pop(rid)
                    if (any(z[k]!=v for k,v in x.gate[rid].items())
                            or (z['p_rank'],z['domain'])!=(rank,dom)):
                        raise ValueError('score does not bind the current v2 roster')
                    scored[rid]=z
            if source_rows:
                raise ValueError('unused scored records')
        if not preview and (len(inventory)!=expected['hits'] or sum(valid_cells.values())!=len(scored)):
            raise ValueError('v2 population incomplete')
        rng=random.Random(s['seed'])
        picked=[]
        for cell,target in quota.items():
            available=sorted(groups[cell],key=lambda row:row[0])
            limit=min(target,len(available))
            picked.extend(available[:1] if preview else rng.sample(available,limit))
        if not picked:
            raise ValueError('no valid samples to train')
        cfg=doc(base/'config.official-train.json');cfg['index_cache_root']=str(dest/'official-index')
        save(dest/'official-config.json',cfg)
        bound=OfficialTrainSources(str(dest/'official-config.json')); stack.callback(bound.close)
        draws=[];cells=Counter()
        for rid,db,x in picked:
            row=db.execute('select * from calls where rid=?',(rid,)).fetchone()
            binding=db.execute('select * from screen_bindings where rid=?',(rid,)).fetchone()
            if binding is None:
                raise ValueError('screen observation has no engine binding')
            z=scored.get(rid)
            if z and (row['wire'],digest(row['raw']))!=(z['wire_sha256'],z['response_sha256']):
                raise ValueError('score and actual transport differ')
            draws.append(materialize(x,bound,row,dest,policy,binding))
            cells[row['domain']+'/'+('success' if row['rank'] else 'error')]+=1
    if preview:
        save(dest/'PREVIEW.json',dict(schema='closed-node-screen-inputs-v1',preview=True,
                                    rows=len(draws),draws=draws,formal_sample_locked=False))
        return dict(rows=len(draws),cells=dict(cells),preview=True)

    official=doc(Path(c['recall'])/'RECALL-COMPLETE.json')['population']
    s['sample_count']=len(draws)
    save(dest/'SCREEN-CONFIG.json',dict(sampling=s,program=c['program'],sources=sources,
        knowledge_bank_ref=ref(version_file),requested_cells=quota,selection_policy=selection_policy))
    roster=dest/'P-SAMPLE.private.jsonl'
    put(roster,b''.join((json.dumps(scored[rid],sort_keys=True)+'\n').encode() for rid,_,_ in picked))
    save(dest/'COMPLETE.json',dict(rows=len(draws),roster_sha256=ref(roster)['sha256'],counts=dict(cells),
        requested=quota,deficits={k:v-cells[k] for k,v in quota.items() if cells[k]<v},screened=len(inventory)))
    save(dest/'RECALL-COMPLETE.json',dict(population=official,p_gate_hits=len(inventory),
        retrieval_denominator=expected['ranked'],knowledge_bank_ref=ref(version_file),sources=sources))
    save(dest/'USAGE.json',dict(usage,statuses=dict(status),valid_cells={k:valid_cells[k] for k in quota}))
    save(dest/'INPUTS.json',dict(usage_ref=ref(dest/'USAGE.json'),recall_ref=ref(dest/'RECALL-COMPLETE.json'),
        screen_config_ref=ref(dest/'SCREEN-CONFIG.json'),policy_ref=program['policy_iteration']['policy_ref'],
        official_train_count=official,admitted_population=len(inventory),population_scope='all v2 retrieval admissions screened',
        complete_ref=ref(dest/'COMPLETE.json'),schema='closed-node-screen-inputs-v1',preview=False,rows=len(draws),
        cells=dict(cells),policy=policy['version'],screen_scope='P',draws=draws,
        historical_actions_are_behavior=False,selection_ref=ref(roster),formal_sample_locked=True))
    version=program['inputs']['system_version']['sha256']
    inputs=ref(dest/'INPUTS.json')
    save(dest/'SAMPLING.json',dict(s,pool_version=inputs['sha256'],system_version=version))
    save(dest/'NODE-POOL.json',dict(schema='node-screen-selection-v1',sealed=True,node='P',version=inputs['sha256'],
        inputs_ref=inputs,policy_ref=program['policy_iteration']['policy_ref'],official_train_count=official,
        retrieval_population_complete=True,synthetic_fixture=False,system_version=version))
    save(dest/'SCREEN.json',dict(schema='node-screen-data-result-v1',status='complete',fixture=False,
        request_ref=program['request_ref'],program_config_ref=ref(Path(c['program'])),usage_ref=ref(dest/'USAGE.json'),
        transport_config_ref=capture['transport_config_ref'],complete=True,population=official,screen_scope='node',node='P',
        screen_ref=ref(dest/'NODE-POOL.json'),sampling_config_ref=ref(dest/'SAMPLING.json'),system_version=version))
    return dict(rows=len(draws),cells=dict(cells))

def project(c,dest,preview):
    from recall_cpu_prepare import checked
    base=Path(c['source_root']);owner=base/'task-loss-integration/official-source-successor'
    sys.path.insert(0,str(owner))
    from sources import OfficialTrainSources
    root=Path(c['output']);dest.mkdir(parents=True,exist_ok=True,mode=0o700)
    s=c['sampling']
    program=doc(c['program']);policy=doc(checked(program['policy_iteration']['policy_ref']))
    original=owner/'config.official-train.json'
    cfg=doc(original);cfg['index_cache_root']=str(dest/'official-index')
    save(dest/'official-config.json',cfg)
    db=sqlite3.connect((root/'p-screen.sqlite').as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    has_bindings=db.execute("select 1 from sqlite_master where type='table' and name='screen_bindings'").fetchone()
    if preview:
        ids=[db.execute("select rid from calls where domain=? and status='valid' and (rank>0)=? order by issued limit 1",(d,hit)).fetchone()[0] for d in s['domain_mix'] for hit in (False,True)]
        roster=None
    else:
        seal=doc(root/'COMPLETE.json');roster={r['reference_id']:r for r in lines(checked(dict(path=str(root/'P-SAMPLE.private.jsonl'),sha256=seal['roster_sha256'])))}
        assert len(roster)==seal['rows']==s['sample_count']
        ids=list(roster)
    x=Inputs(c);bound=OfficialTrainSources(str(dest/'official-config.json'));draws=[];cells={}
    try:
        for rid in ids:
            r=db.execute('select * from calls where rid=?',(rid,)).fetchone()
            if roster is not None:
                z=roster[rid];assert all(z[k]==v for k,v in x.gate[rid].items())
                assert (r['wire'],r['rank'],digest(r['raw']),r['domain'])==(z['wire_sha256'],z['p_rank'],z['response_sha256'],z['domain'])
            binding=db.execute('select * from screen_bindings where rid=?',(rid,)).fetchone() if has_bindings else None
            draw=materialize(x,bound,r,dest,policy,binding);draws.append(draw)
            label=r['domain']+'/'+('success' if r['rank'] else 'error');cells[label]=cells.get(label,0)+1
        if not preview:usage=usage_receipt(db)
    finally:bound.close();db.close()
    if not preview:
        from p_screen_run import apport
        expected={d+'/'+o:q for o,t in apport(s['sample_count'],s['error_success']).items() for d,q in apport(t,s['domain_mix']).items()}
        assert expected==cells
    recall_ref=ref(Path(c['recall'])/'RECALL-COMPLETE.json')
    admission=doc(recall_ref['path'])
    assert admission['population']==admission['p_gate_hits']+admission['p_gate_misses'] and len(x.gate)==admission['p_gate_hits']
    if not preview:save(dest/'USAGE.json',usage)
    save(dest/('PREVIEW.json' if preview else 'INPUTS.json'),dict(usage_ref=None if preview else ref(dest/'USAGE.json'),recall_ref=recall_ref,
        screen_config_ref=ref(root/'CONFIG.local.json'),policy_ref=program['policy_iteration']['policy_ref'],
        official_train_count=admission['population'],admitted_population=admission['p_gate_hits'],
        population_scope='all official recording references evaluated for pre-P retrieval; only selected members require valid P responses',
        complete_ref=None if preview else ref(root/'COMPLETE.json'),schema='closed-node-screen-inputs-v1',preview=preview,rows=len(ids),cells=cells,
        policy=policy['version'],screen_scope='P',draws=draws,historical_actions_are_behavior=False,
        selection_ref=None if preview else ref(root/'P-SAMPLE.private.jsonl'),formal_sample_locked=not preview))
    if not preview:
        # The controller consumes an authenticated selected-input release. Whole-population
        # coverage is only asserted for retrieval, never for P or downstream outcomes.
        inputs=ref(dest/'INPUTS.json')
        sampling=dict(s,pool_version=inputs['sha256'],system_version=program['inputs']['system_version']['sha256'])
        save(dest/'SAMPLING.json',sampling)
        pool=dict(schema='node-screen-selection-v1',sealed=True,node='P',version=inputs['sha256'],
            inputs_ref=inputs,policy_ref=program['policy_iteration']['policy_ref'],official_train_count=admission['population'],
            retrieval_population_complete=True,synthetic_fixture=False,system_version=sampling['system_version'])
        save(dest/'NODE-POOL.json',pool)
        request=doc(checked(program['request_ref']));capture=doc(checked(request['input_ref']))['capture']
        save(dest/'SCREEN.json',dict(schema='node-screen-data-result-v1',status='complete',fixture=False,
            request_ref=program['request_ref'],program_config_ref=ref(Path(c['program'])),usage_ref=ref(dest/'USAGE.json'),
            transport_config_ref=capture['transport_config_ref'],complete=True,population=admission['population'],
            screen_scope='node',node='P',screen_ref=ref(dest/'NODE-POOL.json'),
            sampling_config_ref=ref(dest/'SAMPLING.json'),system_version=sampling['system_version']))
    print(json.dumps(dict(rows=len(draws),cells=cells,preview=preview)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('config',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--preview',action='store_true')
    ap.add_argument('--continue-from',type=Path,action='append',default=[])
    ap.add_argument('--selection-policy',choices=['cap_each_cell_at_available'])
    a=ap.parse_args()
    if a.continue_from:
        if not a.selection_policy:ap.error('--continue-from requires --selection-policy')
        print(json.dumps(project_v2([a.config,*a.continue_from],a.output,preview=a.preview,
                                    selection_policy=a.selection_policy)))
    else:project(doc(a.config),a.output,a.preview)
