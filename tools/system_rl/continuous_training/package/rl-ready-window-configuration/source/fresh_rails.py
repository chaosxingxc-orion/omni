"""Per-episode binding of ORIGINAL semantic rails to observed sampled wires."""
import copy
import sqlite3
import types
from pathlib import Path
from cohort_io import read,pin,save,require,raw


def bind_p(adapter):
    parent=type(adapter).parse_p
    import inspect,textwrap
    source=textwrap.dedent(inspect.getsource(parent))
    needle="packet['wire']['response_format']['json_schema']['schema']['maxItems']"
    require(source.count(needle)==1,'original P parser schema seam changed')
    source=source.replace(needle,"packet.get('original_packet',packet)['wire']['response_format']['json_schema']['schema']['maxItems']")
    scope=dict(parent.__globals__);exec(compile(source,parent.__code__.co_filename+':sampled','exec'),scope)
    adapter.parse_p=types.MethodType(scope['parse_p'],adapter)


def node_p_parser(capture):
    """Load the same ranked reply parser without opening a system Session."""
    import json
    import os
    from cohort_io import BASE, load

    member=capture.get('screened_p_parser_member')
    if member is not None:
        rel=Path(member['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and
                rel.name=='ranked_candidate_indices.py','invalid packaged P parser')
        source=(BASE/rel).resolve(strict=True)
        require(source.is_relative_to(BASE.resolve()),'P parser outside release')
        ref=dict(member,path=str(source))
    else:
        config_ref = capture.get('screened_p_parser_config_ref')
        if config_ref is None:
            config_ref = json.loads(os.environ['OMNI_SCREENED_P_PARSER_CONFIG_REF'])
        config = read(config_ref)
        require(config['schema'] == 'full-system-cpu-collection-v1' and config['thinking'] is False,
                'bound system parser configuration required')
        root = BASE.parents[1]
        relative = Path(config['package_src'])/'speech_aware_evidence_acquisition/rails/ranked_candidate_indices.py'
        ref, = [item for item in config['code_pins'] if Path(item['path']) == relative]
        source=root/relative
        ref=dict(ref,path=str(source))
    parser = types.ModuleType('fresh_ranked_candidate_indices')
    exec(compile(raw(ref), str(source), 'exec'), parser.__dict__)
    cls = load(BASE/'full-system-collection-adapter/adapter.py', 'fresh_original_p_adapter').Adapter
    adapter = cls.__new__(cls)
    adapter.parse_indices = parser.parse
    bind_p(adapter)
    return adapter.parse_p


def bind_branches(adapter,transform):
    """Replay generation AND original Writer/Source closure against same roster."""
    original_s1=adapter.prepare_s1;original_s2=adapter.prepare_s2
    adapter.prepare_s1=lambda recording:[transform(p,[]) for p in original_s1(recording)]
    def s2(recording,packets,replies):
        frozen,plain=original_s2(recording,packets,replies)
        return frozen,[transform(p,[q['request_id'] for q in packets]) for p in plain]
    adapter.prepare_s2=s2


def session_for(Session,config,ep,identity):
    """Derive exact journal config; reuse existing index without population rebuild."""
    spec=read(config['business_config_ref']);spec['journal_identity']=identity
    reuse=copy.deepcopy(spec['index_reuse'])
    if 'schema' not in reuse:reuse=read(reuse)
    reuse['root']=str(ep.parent.parent/'index-reuse')
    spec['index_reuse']=reuse
    spec_ref=save(ep/'BUSINESS.json',spec)
    # Full official TRAIN is the configured population. Reuse the existing
    # successor; legacy business.Session still enforces a historical subset.
    from session import Session as OfficialSession
    require(issubclass(OfficialSession,Session),'official Session parent differs')
    session=OfficialSession(spec_ref['path'],ep,mode='live')
    # DATA owns the immutable index router. Reuse its explicit closed seed;
    # each episode still owns mutable custody and original rails state.
    from asset_index import IndexRouter
    session.index=IndexRouter(ep/'assets.sqlite',spec['asset_files'],reuse)
    session.index_factory=lambda *_:session.index
    # Selected full-screen records must already have vectors. Missing preparation
    # is an input error; fresh capture must never launch an implicit encoder.
    def encoder_disabled(*args,**kwargs):raise ValueError('fresh capture requires existing selected encoder assets')
    original_executor=session.executor
    def executor():
        worker=original_executor();worker.encoder=encoder_disabled;return worker
    session.executor=executor
    bind_p(session.collection.adapter)
    return session


def parse_d(session,packet,reply):
    # Session owns binding checks and observed model failure mapping.
    return session.parse_d([packet],[reply])


def bind_selected_features(session,draw):
    """Reuse authenticated encoder assets, never old policy responses."""
    from material import expand
    from cohort_io import digest
    record=draw['record'];evidence=expand(record['online']['evidence'])
    encoded=evidence.get('encoder',{}).get('output',{})
    material=encoded.get('materialized')
    if material is None:
        # Cached recordings retain the original immutable index lookup.
        return
    entry=copy.deepcopy(material['entry']);binding=material['binding'];task=material['task']
    proof=session.proof(record['recording_ref_id'])
    for key in ('reference_id','source_id','source_record_sha256','recording_file'):
        require(entry[key]==proof[key],'selected feature source mismatch: '+key)
    require(entry['component_id']==proof['source_component'],'selected feature component mismatch')
    require(binding['task_sha256']==digest(task),'selected feature task mismatch')
    require(binding['audio_sha256']==entry['audio_sha256'] and
            binding['encoder_input_id']==entry['encoder_input_id'],'selected feature audio identity')
    require(binding['encoder_recipe_sha256']==digest(session.spec['encoder']['recipe']),
            'selected feature encoder recipe changed')
    matches=[r for r in read(material['consumer_result'])['entries'] if r['reference_id']==entry['reference_id']]
    require(len(matches)==1,'selected feature missing original consumer receipt')
    original=matches[0]
    require({k:v for k,v in original.items() if k!='vector'}=={k:v for k,v in entry.items() if k!='vector'} and
            original['vector']['sha256']==entry['vector']['sha256'],'selected feature receipt drift')
    raw(entry['vector'])
    matrix=session.matrices.get(entry['vector'])
    import numpy as np
    require(matrix.ndim==2 and matrix.shape[1]==session.spec['encoder']['recipe']['dimension'] and
            0<=entry['row']<matrix.shape[0] and np.isfinite(matrix[entry['row']]).all(),
            'selected feature vector layout')
    old=session.vector_overlays.get(entry['reference_id'])
    require(old is None or old==entry,'selected feature overlay conflict')
    session.vector_overlays[entry['reference_id']]=entry
    # The immutable projected index predates some authenticated incremental
    # encoder returns. Carry their verified task/audio binding with the vector,
    # instead of looking up a missing task in that older shard.
    require(entry['reference_id'] in task['reference_ids'],'selected feature task reference')
    worker=session.executor()
    with worker.lock:
        if not hasattr(worker,'selected_task_bindings'):worker.selected_task_bindings={}
        selected=dict(task=copy.deepcopy(task),binding=copy.deepcopy(binding))
        previous=worker.selected_task_bindings.get(entry['reference_id'])
        require(previous is None or previous==selected,'selected task binding conflict')
        worker.selected_task_bindings[entry['reference_id']]=selected


def bind_selected_prepared(session,draw):
    """Node replay uses actual closed upstream preparation for this exact draw."""
    from material import selected
    _,evidence=selected(draw)
    source=next(v['output'] for v in evidence.values() if isinstance(v.get('output'),dict) and 'closure' in v['output'] and 'prepared' in v['output'])
    prepared=read(source['prepared']);proof=read(prepared['source_proof_ref'])
    require(prepared['sample_id']==draw['record']['sample_id'] and proof['reference_id']==draw['record']['recording_ref_id'],'selected prepared source identity differs')
    selected_proof=session.proof(draw['record']['recording_ref_id'])
    require(proof['source_record_sha256']==selected_proof['source_record_sha256'] and proof['source_id']==draw['record']['source_id'],'selected prepared record hash differs')
    raw(prepared['original_audio_ref'])
    from hashlib import sha256
    from cohort_io import canonical
    root=Path(source['prepared']['path']).parent.parent
    overlay_ref=pin(root/'preparation/s1'/(sha256(prepared['reference_id'].encode()).hexdigest()+'.json'))
    overlay=read(overlay_ref)
    require(overlay['cache']['cache_id']==prepared['binding']['cache_id'],'retained cache differs from frozen prepared binding')
    require(overlay['cache']['cache_id']==sha256(canonical({k:v for k,v in overlay['cache'].items() if k!='cache_id'})).hexdigest(),'retained cache content identity')
    from business import module,BASE
    policy=module(BASE/'s1-source-policy-successor/policy.py','integration_policy')
    # build_collection.recording closes over these ORIGINAL objects. Each node
    # episode owns one selected query; restoring a shallow copy would leave the
    # original recording closure looking at the template's unrelated cache.
    c=session.collection;a=c.adapter
    require(overlay['parent_cache_id']==c.cache['cache_id'],'retained overlay parent cache identity')
    require(prepared['binding']['knowledge_id']==c.knowledge['knowledge_id'] and
            prepared['binding']['source_policy_id']==c.knowledge['source_policy']['policy_id'],
            'retained upstream knowledge/source policy identity')
    require(len(overlay['policies'])==len(c.policies) and all(
        dict(p,s1_consumption=dict(p['s1_consumption'],cache_id=overlay['cache']['cache_id']))==q
        for p,q in zip(c.policies,overlay['policies'])), 'retained policies differ beyond selected cache')
    c.cache=overlay['cache'];c.policies=overlay['policies'];c.ids=prepared['binding']
    # D version includes the original sealed runtime-policy REF, not just its
    # contents. Restore the actual per-recording policy seal/config written by
    # recording_overlay; substituting the parent seal changes request identity
    # even when its wire remains byte-identical.
    bound_cfg=read(pin(root/'bindings'/sha256(prepared['sample_id'].encode()).hexdigest()/'CONFIG.json'))
    require(sha256(canonical(bound_cfg)).hexdigest()==prepared['binding']['config_version'],'retained collection config version')
    bound_seal=read(bound_cfg['source_policy']['complete'])
    members={Path(r['path']).name:r for r in bound_seal['artifacts']}
    require(read(members['retrieval.private.json'])==c.cache and all(
        read(members[n])==p for n,p in zip(('s1.runtime.json','s2-v2.runtime.json'),c.policies)), 'retained policy seal assets')
    c.cfg=bound_cfg;c.policy_seal=bound_seal;c.version=prepared['binding']['config_version']
    # Replayed dynamic caches retain the same domain-aware consumption path as
    # their producer. Legacy caches preserve the original factory and version.
    if c.cache.get('query_source_bindings'):
        from domain_rails import policy_rails
        factory=lambda runtime:policy_rails(policy,c.resolver,c.knowledge,c.cache,runtime)
    else:
        factory=lambda runtime:policy.policy_rails(c.resolver,c.knowledge,c.cache,runtime)
    a.s1=factory(c.policies[0]);a.s2=factory(c.policies[1])
    contract=a.scoped(a.cfg['s2_contract'],'integration_s2_contract',a.scoped(a.cfg['s2_common'],'integration_s2_common'));a.s2=contract.apply(a.s2)
    query=a.s1.s1_consumer.queries[prepared['sample_id']]
    require(query['source_component']==proof['source_component'],'retained cache source component identity')
    require(query['audio_sha256']==prepared['audio_sha256'] and query['audio_object_sha256']==prepared['audio_object_sha256'],'retained cache audio identity')
    session.collections[prepared['sample_id']]=c
    session.frozen_ready=prepared
