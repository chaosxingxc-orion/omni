"""Project confirmed P screen observations into immutable inputs for fresh node rollout.

Check the regenerated original business wire against the actual submitted wire.
Offline labels, screen receipt and transport audio remain independently addressable.
"""
import argparse,base64,hashlib,json,sqlite3,sys
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
    assert rank==row['rank'] and x.meta[rid][2][g['rank']-1]==gold
    key=digest(rid.encode());folder=dest/'records'/key[:2]/key
    # No actions/probabilities from the screening reply enter a new behavior group.
    # A generated request ID is reserved for fresh capture by the original transform.
    packet=dict(reference_id=rid,sample_id=sid,stage='P',branch='shared',version=policy['version'],
        request_id='screen-wire:'+row['wire'],wire=p['wire'],wire_sha256=p['wire_sha256'],ranking=p['ranking'],
        actual_returned_ranked_intent_ids=x.meta[rid][2])
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
    a=ap.parse_args();project(doc(a.config),a.output,a.preview)
