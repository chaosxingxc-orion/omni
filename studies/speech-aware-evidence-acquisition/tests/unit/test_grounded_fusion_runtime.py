"""Shared final fusion dispatch; source and endpoint are explicit boundary doubles."""
import copy
from dataclasses import replace
import json
from types import SimpleNamespace
import pytest
from speech_aware_evidence_acquisition.rails import grounded_fusion_runtime as m
from speech_aware_evidence_acquisition.rails import tfgrpo_runtime
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical,sha,empty_experience
from test_tfgrpo_dispatch import make_runtime


def setup(tmp_path,monkeypatch):
    runtime,transport,captured=make_runtime(tmp_path,monkeypatch)
    runtime._plan=replace(runtime.plan,planned_model_calls=1)
    old=runtime.frozen_core_client_kwargs
    runtime.frozen_core_client_kwargs=lambda:{**old(),'plan':runtime.plan}
    sid=runtime._samples[0].sample_id
    source={'model_id':'frozen-model','unknown_contacts':0,
        'original_total_usage':{'calls':312,'prompt_tokens':1000,'completion_tokens':200},
        'batch':{'parent':empty_experience('frozen-model'),'groups':[{'query_id':sid}],
                 'rollout_kind':'shared-perception-adaptation','common':{'reward_mode':'intent-exact-slot-f1-v1'}}}
    document={'input_sha256':'d'*64}
    measured={'messages':[{'role':'system','content':'learn'},{'role':'user','content':'evidence'}],
        'messages_sha256':'e'*64,'input_tokens':10,'completion_cap':100,
        'reserve_tokens':10,'resident_context':1000,'max_items':2,'max_characters':100,
        'native_fit_completed':False,'generation_authorized':False}
    measured['messages_sha256']=sha(measured['messages'])
    monkeypatch.setattr(m,'load_token_counter',lambda p:(lambda x:10,{'fixture':True}))
    monkeypatch.setattr(m,'load_finalized_shards',lambda *a,**k:copy.deepcopy(source))
    monkeypatch.setattr(m,'build_fusion_input',lambda source:copy.deepcopy(document))
    monkeypatch.setattr(m,'fusion_messages',lambda *a,**k:copy.deepcopy(measured))
    monkeypatch.setattr(m,'validate_fusion_output',lambda text,*a,**k:{
        'operations':[{'operation':'ADD','id':None,'content':'Only apply when evidence supports it.'}],
        'semantic_support_verified':False})
    reg,_=m.build_registration(runtime.plan,source=source,document=document,measured=measured,
        model_id='frozen-model',decoding_params={'max_tokens':100,'temperature':0},
        max_context_bytes=10000,runtime_identity_sha256=sha(runtime.session_required_identity()),
        roster_sha256='c'*64)
    doc={'schema':'saea-grounded-fusion-runtime-source-v1','source':{},
        'tokenizer_relpath':'tokenizer','measurement':measured,'input_sha256':document['input_sha256']}
    (tmp_path/'fusion.json').write_bytes(canonical(doc))
    runtime._composed.values.update(tfgrpo_learning_registration=reg,
        prepared_training_feedback={'schema':m.SOURCE_SCHEMA,'source':{
            'root':'data','relpath':'fusion.json','sha256':sha(doc)}})
    return runtime,transport,captured,doc


def test_fusion_dispatch_seals_candidate_and_preserves_actual_cost(tmp_path,monkeypatch):
    runtime,transport,captured,_=setup(tmp_path,monkeypatch)
    receipt=runtime.run_phases()
    assert receipt['contact_count']==len(transport.calls)==1
    candidate=json.loads((runtime._run_dir()/'tfgrpo/learner/candidate.json').read_bytes())
    assert candidate['historical_fusion_custody']['original_total_usage']['calls']==312
    assert candidate['semantic_review_required'] and not candidate['deployment_accepted']
    assert receipt['learning']['candidate']['experience_sha256']==candidate['sha256']
    seal=runtime.seal()
    assert seal['cost']['calls_used']==1 and captured['finalized']


@pytest.mark.parametrize('fault',['messages','stage','schema','cap','source','roster'])
def test_fusion_binding_drift_refused_before_contact(tmp_path,monkeypatch,fault):
    runtime,transport,_,doc=setup(tmp_path,monkeypatch)
    reg=runtime._composed.values['tfgrpo_learning_registration']
    if fault=='messages':reg['contacts'][0]['messages_sha256']='0'*64
    elif fault=='stage':reg['contacts'][0]['stage']='grounded_shard_update'
    elif fault=='schema':reg['schema']=tfgrpo_runtime.GROUNDED_RUNTIME_SCHEMA
    elif fault=='cap':reg['completion_cap']=99
    elif fault=='source':(tmp_path/'fusion.json').write_text('{}')
    else:reg['contacts'][0]['sample_id']='slurp:train:foreign'
    with pytest.raises(ValueError):runtime.run_phases()
    assert not transport.calls


def test_observed_handoff_reused_in_live_client(tmp_path,monkeypatch):
    runtime,transport,_,_=setup(tmp_path,monkeypatch)
    original=tfgrpo_runtime.MeteredExperienceCaller.__call__
    failed=[]
    def crash(self,request):
        response=original(self,request)
        if not failed:
            failed.append(True)
            raise OSError('handoff')
        return response
    monkeypatch.setattr(tfgrpo_runtime.MeteredExperienceCaller,'__call__',crash)
    with pytest.raises(OSError):runtime.run_phases()
    assert len(transport.calls)==1
    runtime.run_phases()
    assert len(transport.calls)==1


def test_native_fit_failure_has_no_generation_intent(tmp_path,monkeypatch):
    runtime,transport,_,_=setup(tmp_path,monkeypatch)
    from speech_aware_evidence_acquisition.core import tfgrpo_fit
    def reject(request):raise ValueError('wrapped input too large')
    monkeypatch.setattr(tfgrpo_fit,'build_tfgrpo_fit_gate',lambda *a,**k:reject)
    with pytest.raises(ValueError,match='wrapped'):runtime.run_phases()
    assert not transport.calls
    assert not list((runtime._run_dir()/'tfgrpo/learner/contacts').glob('*.intent.json'))
    runtime._client.close()
