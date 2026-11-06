import copy
import json
from pathlib import Path
import sys
import pytest
from test_llama_server_transport import _spawn_live_matching_process

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
from psd_construction_campaign import compile_campaign, check_compiled, queue_manifest


def config():
    ids=[r['sample_id'] for r in json.loads((ROOT/'docs/receipts/rosters/slurp-stage3-construction-r01c.json').read_text())['items']]
    return {'schema':'saea-psd-construction-campaign-v1','name':'construction-fixture','run_id':'CONSTRUCTION-FIXTURE',
            'workload':{'schema':'saea-intent-slot-workload-v1','name':'template','run_id':'TEMPLATE','sample_ids':ids,
                        'runtime_binding':{'model_fragment':'model','service_session':'session','profile':'profile','runtime_receipt':'receipt'},
                        'execution':{'admitted_level':1},'material':{},'node_arm':'enhanced-p-slot-decision',
                        'population':{'role':'construction','dataset':'slurp-speech-native-stage3-construction-r01c',
                                      'dataset_sha256':'a'*64,'roster':{},'source_baseline':{}},
                        'pipeline':{'protocol_version':'v2','node_rails':'sdk-handoff-v1','answer_only_fallback':'budgeted-v1'}},
            'collection':{'temperature':0.7,'seeds':[10,11]},'deployment_policy':{'temperature':0}}


def test_compile_registers_independent_full_graph_and_budget():
    doc=compile_campaign(config());check_compiled(doc)
    assert [m['fit_run_id'] for m in doc['members']]==['CONSTRUCTION-FIXTURE-K001-FIT','CONSTRUCTION-FIXTURE-K002-FIT']
    assert doc['budget']['normal_generation_calls']==3750
    assert doc['budget']['generation_cap']==7500
    assert doc['budget']['fit_cap']==3750
    assert doc['budget']['gpu_hours_all_generation_and_fit']==64
    assert [m['job_config']['collection_decoding'] for m in doc['members']]==[{'temperature':0.7,'seed':10},{'temperature':0.7,'seed':11}]
    altered=copy.deepcopy(doc);altered['members'][0]['job_config']['collection_decoding']['temperature']=0
    with pytest.raises(ValueError):check_compiled(altered)


def test_queue_uses_existing_schema_and_member_dependencies():
    doc=compile_campaign(config())
    pins=[{'launch':{'path':f'/tmp/launch{i}','sha256':'a'*64},'preflight':{'path':f'/tmp/preflight{i}','sha256':'b'*64}} for i in range(2)]
    q=queue_manifest(doc,'test','/tmp/owned/state',pins)
    assert q['schema']=='saea-configured-job-queue-v1' and len(q['jobs'])==2
    assert all(j['depends_on']==[] and j['after'] is None for j in q['jobs'])


@pytest.mark.parametrize('field', ['duplicate_seed','temperature','fold','pipeline'])
def test_compiler_rejects_mixed_or_invalid_inputs(field):
    c=config()
    if field=='duplicate_seed':c['collection']['seeds']=[1,1]
    if field=='temperature':c['collection']['temperature']=float('nan')
    if field=='fold':c['workload']['population']['role']='evaluation'
    if field=='pipeline':c['workload']['pipeline']['node_rails']='off'
    with pytest.raises(ValueError):compile_campaign(c)


def test_two_real_sdk_episodes_supply_group_reader(tmp_path, monkeypatch):
    """Real P/Slot/D SDK execution and scoring; sealed-flight boundary is a fixture."""
    import hashlib
    from dataclasses import replace
    from types import SimpleNamespace
    import psd_construction_group_read as reader
    from test_intent_slot_runner import harness
    from test_template_variants_delivery import _runtime, _raw_native
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import canonical
    from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
    from speech_aware_evidence_acquisition.reproduction.slurp_scoring import score_slu, parse_slu_response, SlotEntity
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience, DENSE_REWARD

    c=config(); sid='slurp:train:000001:audio-fixture';c['workload']['sample_ids']=[sid]
    doc=compile_campaign(c); sources={}; scores={}; fits={}; job_by_name={}
    for member in doc['members']:
        job=member['job_config']; idx=member['index']; job_by_name[job['name']]=job
        h=harness(tmp_path/f'run{idx}',sample_ids=(sid,))
        policy=json.loads(h.config.intent_slot_pipeline_json)
        policy.update(schema='saea-intent-slot-pipeline-v2',duplicate_candidate_policy='group_v2',node_rails='sdk-handoff-v1',answer_only_fallback='budgeted-v1')
        h.program=default_phase_program('phase-native-intent-slot-batched-v2')
        h.config=replace(h.config,intent_slot_pipeline_json=canonical(policy))
        original=h._output
        def output(key):
            return canonical({'candidate_id':'play_radio'}) if key[1]=='solver' else original(key)
        h._output=output
        try:
            summary=h.run(decoding_params={'temperature':0.7,'seed':member['seed'],'max_tokens':64})
            raw=_raw_native(h,solver_role={'max_iterations':1});raw['intent_slot_pipeline']=policy
            rows=_runtime(h,summary,raw)._canonical_output_rows(h.client)
        finally: cost=h.close()['cost']
        records={rid:payload for _,rid,payload in h.trace.records}
        _verify_outputs_trace_binding(tuple(rows),records)
        visits=summary.results[0].native_episode_receipt['role_visits']
        contacts=[x for visit in visits for x in visit['execution_receipt']['contacts']]
        assert len(contacts)==3 and cost['calls_used']==3 and len(h.fit_calls)==3
        for contact in contacts:
            decoding=records[contact['request_id']+'/request']['payload']['decoding_params']
            assert decoding['temperature']==0.7 and decoding['seed']==member['seed']
        parsed=parse_slu_response(rows[0]['text'])
        score=score_slu(gold_scenario='play',gold_action='radio',gold_entities=[SlotEntity(type='person',filler='Bob')],predicted_scenario=parsed[0],predicted_action=parsed[1],predicted_entities=parsed[2]).as_metrics_dict()
        score['sample_id']=sid
        scores[idx]={'slurp-slu-official-train':{'per_sample':[score]}}
        fits[idx]={'stages':{'run_phases':{'dispatch':{'native_contact_fit_receipts':[{'agent_contact_id':x['agent_contact_id']} for x in contacts]}}}}
        values={'request_decoding_overrides':job['collection_decoding'],'decoding_params':{'temperature':0,'max_tokens':64},
                'native_episode':{'intent_slot_pipeline':{'node_rails':'sdk-handoff-v1'}},
                'native_contact_fit':{'run_id':member['fit_run_id'],'attempt_id':'fit-v1'},'knowledge_stores':{'same':'pinned'}}
        manifest={'status':'COMPLETED','cost':cost,'config':{'config_hash':'c'*64,'fragments':{'model':'model','dataset':job['population']['dataset'],'baseline':job['name'],'experiment':job['name']}}}
        source=SimpleNamespace(plan=SimpleNamespace(run_id=job['run_id'],attempt_id='generation-v1'),sample_ids=(sid,),
             composed=SimpleNamespace(values=values),bundle=SimpleNamespace(manifest=manifest,attempt_state='FINALIZED',trace_records=records),
             phase_results={sid:{'native_episode_receipt':summary.results[0].native_episode_receipt}},output_rows={sid:rows[0]})
        sources[idx]=source
        p=tmp_path/'docs/readiness'/job['name'];p.mkdir(parents=True)
        (p/'candidate.json').write_text(json.dumps({'job_config':job,'composed_config_hash':'c'*64,'dataset':{}}))
        p=tmp_path/'configs/experiment';p.mkdir(parents=True,exist_ok=True);(p/(job['name']+'.json')).write_text('{}')
    def pinned(data,pin,label):
        idx=pin['index']
        if label=='system feedback binding':return ({'index':idx,'manifest':{'relpath':'x','sha256':'0'*64},'phases':{'relpath':'x','sha256':'0'*64},'flight_receipts':{'index':idx},'scorer_id':'slurp-slu-official-train'},'x')
        if label=='official scores':return scores[idx],'x'
        if label=='flight receipts':return fits[idx],'x'
        raise AssertionError(label)
    monkeypatch.setattr(reader,'_pinned_json',pinned)
    monkeypatch.setattr(reader,'_confined_file',lambda *args:tmp_path/'x')
    monkeypatch.setattr(reader,'load_system_feedback',lambda **kw:{'run_id':sources[kw['binding']['index']].plan.run_id,'resolved_auxiliary_sources':{'scores':{'index':kw['binding']['index']}}})
    monkeypatch.setattr(reader,'verify_native_episode_source',lambda **kw:next(iter(sources.values())))
    # The production verifier receives a unique manifest path for each source.
    monkeypatch.setattr(reader,'_confined_file',lambda root,path:tmp_path/path)
    for i in sources:
        (tmp_path/f'm{i}').touch();(tmp_path/f'p{i}').touch()
    original_pin=pinned
    def binding_pin(data,pin,label):
        value,digest=original_pin(data,pin,label)
        if label=='system feedback binding':
            value['manifest']['relpath']=f"m{pin['index']}";value['phases']['relpath']=f"p{pin['index']}"
        return value,digest
    monkeypatch.setattr(reader,'_pinned_json',binding_pin)
    monkeypatch.setattr(reader,'verify_native_episode_source',lambda **kw:sources[int(kw['manifest_path'].name[1:])])
    monkeypatch.setattr(reader,'pinned',lambda *args:{})
    monkeypatch.setattr(reader.NodeWorkload,'validate',lambda *args:None)
    monkeypatch.setattr(reader,'verify_fit_receipts',lambda data,native,fit,member:len(fit))
    monkeypatch.setattr(reader,'knowledge_index',lambda data,values:{})
    # Harness request identity includes temporary media roots; normalize only its
    # common-input hash seam after the native binding checks have run.
    monkeypatch.setattr(reader,'initial_speech_input_hash',lambda p:'a'*64)
    bindings=[{'index':1},{'index':2}]
    gold={sid:{'scenario':'play','action':'radio','entities':[{'type':'person','filler':'Bob'}]}}
    kwargs=dict(repo=tmp_path,data=tmp_path,gold=gold,gold_pin={},parent=empty_experience('model'),common={'reward_mode':DENSE_REWARD},learning_config_sha256='b'*64,bounds={})
    result=reader.read_groups(doc,bindings,**kwargs)
    rolls=result['batch']['groups'][0]['rollouts']
    assert len(rolls)==2 and rolls[0]['reward']==rolls[1]['reward']==1
    assert result['provenance']['generation_calls_used']==6 and result['provenance']['fit_tokenizer_calls_used']==6
    from speech_aware_evidence_acquisition.rails.tfgrpo_external_proposals import relative_evidence
    from speech_aware_evidence_acquisition.rails.external_exploration import expand_evidence
    groups=copy.deepcopy(result['tfgrpo_evidence']['groups'])
    for group in groups:
        packed=group['evidence']; table=packed.get('development_evidence_dictionary',{})
        group['evidence']=expand_evidence({k:v for k,v in packed.items() if not k.startswith('development_evidence_')},table)
    relative=relative_evidence(groups)
    assert [x['sample_weight'] for x in relative['groups'][0]['rollouts']]==[0.5,0.5]
    sources[2].composed.values['knowledge_stores']={'changed':'yes'}
    with pytest.raises(ValueError,match='mixed model'):reader.read_groups(doc,bindings,**kwargs)
    sources[2].composed.values['knowledge_stores']={'same':'pinned'}
    sources[2].bundle.attempt_state='UNKNOWN'
    with pytest.raises(ValueError,match='not final'):reader.read_groups(doc,bindings,**kwargs)
    sources[2].bundle.attempt_state='FINALIZED'
    fits[2]['stages']['run_phases']['dispatch']['native_contact_fit_receipts'].pop()
    with pytest.raises(ValueError,match='FIT contact custody'):reader.read_groups(doc,bindings,**kwargs)


def test_parent_cannot_be_invented_or_hidden(tmp_path):
    from psd_construction_group_read import verify_actor_parent
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience, sha
    job=compile_campaign(config())['members'][0]['job_config']
    parent=empty_experience('model')
    assert verify_actor_parent(job,parent,repo=tmp_path,data=tmp_path,binding=None)==[]
    altered=copy.deepcopy(parent);altered['items']={'x':'unreviewed'};altered['sha256']=sha({k:v for k,v in altered.items() if k!='sha256'})
    with pytest.raises(ValueError,match='empty unexposed'):
        verify_actor_parent(job,altered,repo=tmp_path,data=tmp_path,binding=None)
    job['tfgrpo_experience']={'target':'slot'}
    with pytest.raises(ValueError,match='experienced actor cannot'):
        verify_actor_parent(job,parent,repo=tmp_path,data=tmp_path,binding=None)
    del job['tfgrpo_experience']
    p=tmp_path/'configs/probes/slot.txt';p.parent.mkdir(parents=True);p.write_text('Extract observed slots.\n')
    import hashlib
    job['slot_policy']={'instruction_template':{'relpath':'configs/probes/slot.txt','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}
    assert verify_actor_parent(job,parent,repo=tmp_path,data=tmp_path,binding=None)==[]
    p.write_text('Use these conditional experiences only for Slot extraction.\n')
    job['slot_policy']['instruction_template']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='TF-GRPO template'):
        verify_actor_parent(job,parent,repo=tmp_path,data=tmp_path,binding=None)


def test_knowledge_diagnostics_resolve_only_actual_snapshot_values(tmp_path):
    from test_knowledge_stores import _config
    from psd_construction_group_read import knowledge_index
    cfg, snapshots=_config(tmp_path)
    index=knowledge_index(tmp_path,cfg)
    for store, snapshot in snapshots.items():
        value=snapshot.values[0]
        assert index[store+'::'+value.value_id][0]['source_utterances']==sorted({':'.join(s.split(':')[:3]) for s in value.source_sample_ids})
    assert 'not-supplied' not in index
    pin=cfg['knowledge_stores']['intent-boundaries']['snapshot'];path=tmp_path/pin['relpath']
    path.write_text('{}')
    with pytest.raises(Exception):knowledge_index(tmp_path,cfg)


def test_locked_real_three_store_content_hashes():
    import os
    data=os.environ.get('SAEA_PSD_LOCKED_KNOWLEDGE_TEST_ROOT')
    if data is None:
        pytest.skip('requires explicitly mounted locked knowledge data root')
    from psd_construction_group_read import knowledge_index
    baseline=json.loads((ROOT/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_text())
    index=knowledge_index(Path(data),baseline)
    for store in baseline['knowledge_stores']:
        assert any(key.startswith(store+'::') for key in index)
    slot=baseline['knowledge_stores']['slot-contracts']['snapshot']
    import hashlib
    assert slot['sha256']=='e035c1b827b10cb711c7c31d345deb5e668489679b0d3979acfc95427f6e2a5e'
    assert hashlib.sha256((Path(data)/slot['relpath']).read_bytes()).hexdigest() != slot['sha256']
