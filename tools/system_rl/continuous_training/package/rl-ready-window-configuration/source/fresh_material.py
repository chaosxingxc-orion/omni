"""Expand only selected original content; preserve actual frozen upstream packets."""
import copy
from pathlib import Path
from cohort_io import read,raw,pin,require,BASE


def materialize_audio(path):
    """Invoke the exact function used by original preparation_executor.audio."""
    import ast,io,numpy as np
    source=BASE.parents[1]/'delivery/s1-round03-bound-layout-combination-v1/package/src/speech_aware_evidence_acquisition/analysis/slurp_mf7b_vectors.py'
    fn=next(n for n in ast.parse(source.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='read_audio_wav_bytes')
    env=dict(np=np,io=io,Path=Path,SAMPLE_RATE_HZ=16000)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
    return env['read_audio_wav_bytes'](path)


def expand(value):
    if isinstance(value,dict):
        if set(value)=={'$collection_content_v1'}:return expand(read(value['$collection_content_v1']))
        return {k:expand(v) for k,v in value.items()}
    if isinstance(value,list):return [expand(v) for v in value]
    return value


def node_screen_packet(draw, snapshot, annotation, origin):
    """Validate the independently retained node request against its actual screening receipt."""
    from hashlib import sha256
    import base64,json
    record=draw['record']
    observation=read(record['screen_observation_ref'])
    require(snapshot['rollout_scope']=='node' and snapshot['target_nodes']==[observation['node']]
        and observation['schema']=='node-screen-observation-v1' and observation['node'] in ('P','S1')
        and record['split']==observation['split']=='TRAIN','wrong screen/rollout scope')
    packet=read(record['online']['node_input_ref'])
    node=observation['node'];binding=packet if node=='P' else packet['source_binding']
    require(record['online']['node_input_ref']==observation['packet_ref'] and
        binding['reference_id']==observation['reference_id']==origin['recording_ref_id']==record['recording_ref_id']==annotation['recording_ref_id'] and
        record['source_id']==origin['source_id']==observation['source_id'] and record['domain']==observation['domain']==origin['domain'] and
        observation['source_record_sha256']==annotation['source_record_sha256'] and
        packet['sample_id']==record['sample_id']==observation['sample_id'] and packet['stage']==observation['node'],
        'node screen recording or source differs')
    digest=sha256(json.dumps(packet['wire'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    require(digest==packet['wire_sha256'],'node screen wire differs')
    if node=='P':
        require(packet['version']==observation['policy_version'],'P policy differs')
        labels=packet['actual_returned_ranked_intent_ids'];k=observation['dynamic_gold_rank']
        require(type(k) is int and 0<k<=len(labels) and labels[k-1]==annotation['gold_intent_id'], 'actual P admission differs')
    else:
        require(binding['domain']==record['domain'] and binding['source_id']==record['source_id']
                and binding['split']=='TRAIN' and binding['source_component']==observation['source_component']
                and packet['branch']==observation['fold'] and packet['wire_sha256']==observation['source_wire']
                and packet['version']==observation['packet_version']
                and packet['s1_candidate_presentation']=='binary_marking_v1'
                and packet['allowed_types']==observation['supervision']['annotated_types']
                and isinstance(annotation.get('entities'),list) and isinstance(annotation.get('tokens'),list)
                and [t for t in packet['allowed_types'] if t in {e['type'] for e in annotation['entities']}]
                    ==observation['supervision']['present_types'], 'S1 fold, source, or supervision differs')
    parts=[v for m in packet['wire']['messages'] if isinstance(m['content'],list) for v in m['content'] if v.get('type')=='input_audio']
    require(len(parts)==1 and base64.b64decode(parts[0]['input_audio']['data'],validate=True)==raw(origin['audio_ref']),
            'node wire and actor recording differ')
    return packet


def selected(draw):
    evidence=expand(draw['record']['online']['evidence'])
    packets=[]
    ref=draw['record']['online']['trajectory_export'];blob=raw(ref)
    trajectories=[__import__('json').loads(line) for line in blob.splitlines() if line.strip()] if ref['path'].endswith('.jsonl') else __import__('json').loads(blob)
    for row in trajectories:
        for span in row['spans']:
            if span.get('packet_ref'):
                ref=span['packet_ref'];packet=expand(read(ref))
                for key in ref.get('pointer',[]):packet=packet[key]
            else:
                wire=expand(read(span['online_request']))
                for key in span['online_request'].get('pointer',[]):wire=wire[key]
                packet=dict(sample_id=draw['record']['sample_id'],stage=span['node'],branch=span['branch'],
                    request_id=span['request_id'],wire=wire,wire_sha256=span['wire_sha256'],version=span['collector_code_version'])
                if span['node']=='S2':packet['allowed_types']=[span['branch']]
            packets.append(packet)
    require(packets and all('wire' in p for p in packets),'original selected transport packets missing')
    return packets,evidence


def frozen_d_runtime(session,draw,packet):
    """Restore ORIGINAL selected Source/config, never manufacture D upstream."""
    packets,evidence=selected(draw)
    source=next(v['output'] for v in evidence.values() if isinstance(v.get('output'),dict) and 'closure' in v['output'] and 'prepared' in v['output'])
    prepared=read(source['prepared']);c=copy.copy(session.collector(packet['sample_id']))
    c.ids=prepared['binding'];c.cfg=copy.deepcopy(c.cfg)
    closure=read(source['closure'])
    for stage,ident in closure['journal_identities'].items():c.cfg['stages'][stage]['journal_identity']=ident
    # Original Session.d published this deterministic config beside prepared/.
    directory=Path(source['prepared']['path']).parent.parent
    dref=pin(directory/'d-config'/(source['closure']['sha256']+'.json'))
    from source_policy import source as original_source
    src=original_source(c,dref['path'],prepared,source['observed'],source['validation'],BASE)
    runtime=session.dmodule.factory(c,dref['path'],source=src)
    original=runtime.request(runtime.prepare_recording(packet['sample_id']))
    require(all(original[k]==packet[k] for k in ('request_id','wire_sha256','wire','version')),'selected frozen D source/request changed')
    return runtime,original


def hydrate_packets(session,draw,packets,*,d_runtimes=None):
    """Rebuild missing exported S2/D packet metadata through original rails.

    Exported standalone wires are insufficient for task rendering. Reconstruct
    from the actual closed prepared/observed source and require exact identity;
    never infer a schema or a supplied label from the reference annotation.
    """
    _,evidence=selected(draw)
    source=next((v['output'] for v in evidence.values() if isinstance(v.get('output'),dict) and 'closure' in v['output'] and 'prepared' in v['output']),None)
    if source is None:
        require(all(p['stage'] not in ('S2','D') or ('allowed_types' in p if p['stage']=='S2' else 'packet' in p) for p in packets),'frozen target source unavailable; retain selected draw')
        return packets
    prepared=read(source['prepared']);c=copy.copy(session.collector(prepared['sample_id']))
    c.cfg=copy.deepcopy(c.cfg)
    closure=read(source['closure'])
    for stage,identity in closure['journal_identities'].items():c.cfg['stages'][stage]['journal_identity']=identity
    s2=[]
    if any(p['stage']=='S2' for p in packets):
        replies=[session.original.verify_observed(c,'S1',p,source['observed']['S1'][p['request_id']]) for p in prepared['S1']]
        _,s2=c.adapter.prepare_s2(prepared['recording'],prepared['S1'],replies)
    originals={p['request_id']:p for p in [prepared['P'],*prepared['S1'],*s2]}
    result=[]
    for packet in packets:
        if packet['stage']=='D':
            runtime,actual=frozen_d_runtime(session,draw,packet)
            if d_runtimes is not None:
                require(packet['request_id'] not in d_runtimes,'duplicate frozen D runtime')
                d_runtimes[packet['request_id']]=(runtime,actual)
        else:
            require(packet['request_id'] in originals,'exported target missing original rails packet')
            actual=originals[packet['request_id']]
        require(all(actual[k]==packet[k] for k in ('request_id','wire_sha256','wire','version')),'frozen packet cannot be regenerated exactly')
        result.append(actual)
    return result
