"""P-only grouping uses test combinations, never claims four real draws."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT/'scripts/analysis'))
import psd_construction_group_read as reader
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience, canonical


def test_four_native_node_groups_and_dynamic_teacher(tmp_path, monkeypatch):
    from speech_aware_evidence_acquisition.core import bundle as bundle_mod, config as config_mod
    from speech_aware_evidence_acquisition.analysis import perception_node_score
    import registered_workload
    ids = [f'slurp:train:{i:06}:audio' for i in range(80)]
    parent = empty_experience('model')
    members=[]; bundles={}; candidates={}; bindings=[]; fit_docs={}
    for i in range(4):
        job={'name':f'job{i}', 'run_id':f'RUN{i}', 'sample_ids':ids, 'runtime_binding':{'model_fragment':'model'},
             'population':{'dataset':'dataset'}, 'collection_decoding':{'temperature':0.7,'seed':i+1}}
        member={'job_config':job,'seed':i+1,'generation_attempt_id':'generation-v1','fit_run_id':f'FIT{i}',
                'fit_attempt_id':'fit-v1','fit_cap':480,'generation_cap':480}
        members.append(member); fragments={'model':'model','dataset':'dataset','baseline':job['name'],'experiment':job['name']}
        manifest={'status':'COMPLETED','result_kind':'node','plan':{'run_id':job['run_id'],'attempt_id':'generation-v1','split_identity_hash':'a'*64},
                  'config':{'fragments':fragments,'config_hash':f'cfg{i}'},'cost':{'calls_used':80}}
        rows=[];trace={};fits=[]
        for n,sid in enumerate(ids):
            rid=f'{i}-{n}'; cid=f'contact-{rid}'
            payload={'request_id':rid,'sample_id':sid,'carrier_lock_key':'slurp','media_relpath':'a','speech_ref':'a',
                     'speech_packet_sha256':'b'*64,'raw_speech_query':'audio','sample_rate_hz':16000,'audio_seconds':1,
                     'task_instruction':'prompt','history':[],'supplied_evidence':[],'evidence_provenance':[],
                     'decoding_params':{'temperature':0.7,'seed':i+1,'max_tokens':4096}}
            contact={'request_id':rid,'agent_contact_id':cid,'request_sha256':'c'*64,'response_sha256':'d'*64,'response_text_sha256':'e'*64}
            response='win' if (n == 0 and i == 0) or n == 1 else 'fail'
            node={'status':'ok','response':response,'node_receipt':{'contacts':[contact],'final_contact':contact}}
            rows.append({'sample_id':sid,'node_outcome':node})
            trace[rid+'/request']={'payload':payload};trace[rid+'/response']={'text':response,'usage':{'prompt_tokens':1},'finish_reason':'stop'}
            trace[rid+'/cost']={'calls':1,'prompt_tokens':1,'completion_tokens':2,'latency_seconds':0.1}
            fits.append({'agent_contact_id':cid})
        if i == 0:
            primary = rows[2]['node_outcome']['node_receipt']['contacts'][0]
            rid = 'repair-0-2'; cid = primary['agent_contact_id']+':perception-length-repair'
            repair = {**primary, 'request_id':rid, 'agent_contact_id':cid}
            trace[rid+'/request']={'payload':copy.deepcopy(trace['0-2/request']['payload'])}
            trace[rid+'/response']={'text':'fail','finish_reason':'stop','usage':{'prompt_tokens':2}}
            trace[rid+'/cost']={'calls':1,'prompt_tokens':2,'completion_tokens':3,'latency_seconds':0.2}
            trace['0-2/cost']['finish_reason']='length'
            rows[2]['node_outcome']['node_receipt']['contacts'].append(repair)
            rows[2]['node_outcome']['node_receipt']['final_contact']=repair
            fits.append({'agent_contact_id':cid}); manifest['cost']['calls_used'] += 1
        bundles[f'm{i}']=SimpleNamespace(manifest=manifest,attempt_state='FINALIZED',outputs_rows=rows,trace_records=trace)
        candidates[job['name']]={'job_config':job,'composed_config_hash':f'cfg{i}'}
        fit_docs[f'f{i}']={'stages':{'run_phases':{'dispatch':{'native_contact_fit_receipts':fits}}}}
        bindings.append({'relpath':f'b{i}','sha256':'1'*64})
    compiled={'members':members,'policy_sha256':'2'*64,'content_sha256':'3'*64}
    monkeypatch.setattr(reader,'check_compiled',lambda x: x)
    def pinned(data,pin,label):
        name=pin['relpath']
        if name.startswith('b'): return ({'manifest':{'relpath':'m'+name[1:],'sha256':'4'*64},'flight_receipts':{'relpath':'f'+name[1:],'sha256':'5'*64}},None)
        if name in fit_docs: return fit_docs[name],None
        if name.startswith('m'): return bundles[name].manifest,None
        return json.loads((data/name).read_bytes()),None
    monkeypatch.setattr(reader,'_pinned_json',pinned)
    monkeypatch.setattr(reader,'_confined_file',lambda data,rel: Path(rel))
    monkeypatch.setattr(bundle_mod,'verify_bundle',lambda p: bundles[str(p)])
    def asset(repo,rel):
        name=rel.split('/')[-2]; p=tmp_path/f'{name}.json';p.write_text(json.dumps(candidates[name]));return p
    monkeypatch.setattr(reader,'resolve_asset',asset)
    monkeypatch.setattr(registered_workload,'validate_workload',lambda *a:None)
    monkeypatch.setattr(config_mod,'compose',lambda repo,*frags:SimpleNamespace(config_hash='cfg'+frags[-1][-1],values={'request_decoding_overrides':members[int(frags[-1][-1])]['job_config']['collection_decoding']}))
    monkeypatch.setattr(reader,'effective_request_decoding_params',lambda values: values['request_decoding_overrides'])
    monkeypatch.setattr(reader,'verify_fit_receipts',lambda data,native,fit,member:len(fit))
    monkeypatch.setattr(perception_node_score,'score',lambda roster,rows,**kw:{'rows':[{'status':'valid' if r['node_outcome']['response']=='win' else 'invalid', 'candidate_ids':['a'] if r['node_outcome']['response']=='win' else [],'gold_recalled':r['node_outcome']['response']=='win'} for r in rows]})
    common={'agent_objective':'P intent','learning_objective':'improve evidence use','num_experiences':2}
    kwargs=dict(repo=tmp_path,data=tmp_path,gold={sid:'a' for sid in ids},gold_pin={'sha256':'6'*64},legal_ids=['a','b'],parent=parent,common=common,learning_config_sha256='7'*64,bounds={'max_calls':481,'concurrency':2,'max_operations':128,'max_items':128,'max_characters':16000})
    result=reader.read_p_only_groups(compiled,bindings,**kwargs)
    assert result['metrics']['mixed_groups']==1 and result['metrics']['all_correct']==1
    assert result['metrics']['all_wrong']==78 and result['metrics']['ties']==79
    assert len(result['batch']['groups'])==80 and result['provenance']['fit_tokenizer_calls_used']==321
    assert result['metrics']['repaired_draws']==1
    repaired=json.loads(result['batch']['groups'][2]['rollouts'][0]['trajectory'])
    assert len(repaired['contacts'])==2 and repaired['contacts'][0]['finish_reason']=='length'
    feedback=tmp_path/'feedback.json';feedback.write_bytes(canonical(result))
    from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
    config={'plan':{'run_id':'LEARN','attempt_id':'learn-v1','execution_profile':'bounded-discovery-probe','carrier_lock_key':'slurp','split_role':'discovery','split_identity_hash':split_identity_hash(prefixed('slurp',ids)),'planned_gpu_hours':1.0,'planned_speech_audio_seconds':1},'decoding_params':{'temperature':0,'max_tokens':4096},'max_context_bytes':2000000,'runtime_identity_sha256':'8'*64}
    prepared=reader.prepare_p_teacher(result,data=tmp_path,feedback_pin={'relpath':'feedback.json','sha256':__import__('hashlib').sha256(feedback.read_bytes()).hexdigest()},roster_sha256='a'*64,config=config)
    assert len(prepared['registration']['contacts'])==7 and len(prepared['source']['batch']['groups'])==80
    from speech_aware_evidence_acquisition.rails.tfgrpo_source import load_dynamic_source
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    import hashlib
    source=tmp_path/'source.json';source.write_bytes(canonical(prepared['source']))
    values={'prepared_training_feedback':{'schema':'saea-prepared-tfgrpo-dynamic-v1','source':{'root':'data','relpath':'source.json','sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}, 'tfgrpo_learning_registration':prepared['registration']}
    loaded, receipt=load_dynamic_source(values,plan=ExecutionPlan(**prepared['plan']),repo_root=tmp_path,data_root=tmp_path,sample_ids=ids,roster_sha256='a'*64)
    assert receipt['contacts']==7 and len(loaded['batch']['groups'])==80
    from test_tfgrpo_external_reference import material, write
    from speech_aware_evidence_acquisition.rails import tfgrpo_external_reference as ref
    (tmp_path/'external').mkdir()
    path, external=material(tmp_path/'external')
    external.update(p_only_binary=True, source=write(tmp_path/'p-evidence.json',{'document':result['tfgrpo_evidence']}),
        source_batch=write(tmp_path/'p-batch.json',result['batch']), parent=write(tmp_path/'p-parent.json',parent),
        query_ids=ids,actor_model_id='model')
    external['bounds'].update(max_calls=481,max_completion_tokens_total=481*131072)
    write(path,external)
    dry=ref.prepare(path)
    assert dry['plan']['contrast_groups']==1 and dry['plan']['planned_calls']==7
    assert dry['actor_model_id']=='model' and dry['learner_model_id']=='external-model' and dry['new_model_calls']==0
    bundles['m1'].trace_records['1-0/request']['payload']['task_instruction']='changed'
    with pytest.raises(ValueError,match='actual prompt'):
        reader.read_p_only_groups(compiled,bindings,**kwargs)


def test_frozen_real_p_response_is_scored_as_test_combination():
    import os
    path=os.environ.get('P_ONLY_FROZEN_RECORDS')
    if not path:
        pytest.skip('private frozen P-only response fixture not mounted')
    from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import document
    records=[json.loads(line) for line in Path(path).read_text().splitlines()]
    valid=next(r for r in records if r['native_P_valid'] and r['P_contacts'] and r['P_contacts'][-1]['response']['text'])
    text=valid['P_contacts'][-1]['response']['text']
    parsed=document(text)
    assert isinstance(parsed['candidates'],list) and parsed['candidates']
    # Reused real text and a deliberately truncated test variant are not independent draws.
    assert len([text,text,text[:max(1,len(text)//2)],]) == 3


def test_frozen_real_p_request_response_and_scoring():
    import os, hashlib, base64
    path=os.environ.get('P_ONLY_FROZEN_RECORDS')
    if not path: pytest.skip('private frozen P-only response fixture not mounted')
    from speech_aware_evidence_acquisition.rails.solver_protocol_audit import canonical_request
    from speech_aware_evidence_acquisition.analysis.perception_node_score import score
    from speech_aware_evidence_acquisition.core.preparation_fit import boundary_identity
    records=[json.loads(line) for line in Path(path).read_text().splitlines()]
    r=next(x for x in records if x['native_P_valid'] and len(x['P_contacts'])==1)
    c=r['P_contacts'][0]; req=c['request']; payload=req['payload']; response=c['response']
    raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    assert hashlib.sha256(raw).hexdigest()==req['request_sha256']==c['wrapper']['request_sha256']
    wire=base64.b64decode(response['wire_response_b64']);assert hashlib.sha256(wire).hexdigest()==response['wire_response_sha256']
    choice=json.loads(wire)['choices'][0];assert choice['message']['content']==response['text']
    normalized,fmt=canonical_request(payload);assert fmt is not None and normalized['sample_id']==r['sample_id']
    assert len(boundary_identity((payload,)))==64
    inv={'sample_id':r['sample_id'],'candidate_ids':[], 'sources':[{'source_id':e['evidence_id'],'kind':'knowledge','source_sha256':hashlib.sha256(e['content'].encode()).hexdigest()} for e in payload['supplied_evidence']]}
    node={'result_type':'node_outcome','stage':'perception','sample_id':r['sample_id'],'status':'ok',
          'response':response['text'],'finish_reason':response['finish_reason'],'pipeline_config':r['native_P']['reference_config'],
          'evidence_inventory':inv,'response_format':payload['decoding_params']['response_format']}
    output={'sample_id':r['sample_id'],'result_kind':'node','text':node['response'],'node_outcome':node}
    result=score([{'sample_id':r['sample_id'],'gold_intent':r['gold_feedback_private']['intent']}], [output],maximum=3,
                 legal_ids=set(r['candidate_ids'])|{r['gold_feedback_private']['intent']})
    assert result['rows'][0]['status']=='valid'
    assert response['usage']['prompt_tokens']>=0 and response['usage']['completion_tokens']>=0


def test_native_fit_receipt_cost_boundary(tmp_path):
    import hashlib
    from speech_aware_evidence_acquisition.core.preparation import digest
    from speech_aware_evidence_acquisition.core.preparation_fit import boundary_identity
    sid='slurp:train:fixture:audio'; payload={'request_id':'req','sample_id':sid,'decoding_params':{'max_tokens':4096}}
    inputs={'native_contact':{'agent_contact_id':'contact'},'fit_plan':{'run_id':'FIT','attempt_id':'fit-v1'},
            'sources':{'boundary_sha256':boundary_identity((payload,))},'split_identity_hash':'a'*64}
    output={'every_request_fits':True,'rows':[{}],'agent_contact_id':'contact','sample_id':sid,'contacts':2}
    document={'schema':'saea-prepared-artifact-v1','kind':'request-fit','inputs':inputs,'contact_class':'tokenizer',
              'output':output,'output_sha256':digest(output)}
    path=tmp_path/'fit.json';path.write_bytes(canonical(document));h=hashlib.sha256(path.read_bytes()).hexdigest()
    native=SimpleNamespace(sample_ids=[sid],plan=SimpleNamespace(split_identity_hash='a'*64),
        phase_results={sid:{'native_episode_receipt':{'role_visits':[{'execution_receipt':{'contacts':[{'agent_contact_id':'contact','request_id':'req'}]}}]}}},
        bundle=SimpleNamespace(trace_records={'req/request':{'payload':payload}}))
    receipts=[{'agent_contact_id':'contact','path':str(path),'sha256':h}]
    member={'fit_run_id':'FIT','fit_attempt_id':'fit-v1','fit_cap':2}
    assert reader.verify_fit_receipts(tmp_path,native,receipts,member)==2
    native.bundle.trace_records['req/request']['payload']['decoding_params']['max_tokens']=2048
    with pytest.raises(ValueError,match='FIT request identity'):
        reader.verify_fit_receipts(tmp_path,native,receipts,member)
