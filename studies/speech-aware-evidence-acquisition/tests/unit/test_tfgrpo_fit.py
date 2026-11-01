"""Real vLLM token counting path with a local fake endpoint; no model contact."""
import json
from types import SimpleNamespace
from dataclasses import asdict
import pytest
from test_tfgrpo_runtime import setup,request
from speech_aware_evidence_acquisition.core.tfgrpo_fit import build_tfgrpo_fit_gate, PreparationError


def fixture(tmp_path, *, cap=None):
    caller,client,_,_,_,_=setup(tmp_path)
    class Runtime:
        plan=caller.plan
        _data_root=tmp_path
        _session=SimpleNamespace(host='127.0.0.1',port=8200)
        _gate=object()
        _adapter=SimpleNamespace(serving_geometry=lambda:{'slot_tokens':1000})
        _composed=SimpleNamespace(values={'transport':'vllm','arm':'prepared-context-text',
            'tfgrpo_contact_fit':{'run_id':'fit-run','attempt_id':'fit-attempt','contact_cap':cap or 1}})
        _transport=SimpleNamespace(render_request=lambda raw:{'model':'frozen-model','messages':[
            {'role':'user','content':json.loads(raw)['history'][0]['content']}], 'max_tokens':100})
        def session_required_identity(self):return {'model_fragment':'frozen-model'}
        def assert_generation_lease(self):pass
        def _gate_touch(self,g,p):self.checked=p
        def _require_exposure_row_unconsumed(self,r):self.exposure=r
    calls=[]
    class Tokenizer:
        def __init__(self,**kwargs):pass
        def request_json(self,method,route,payload):
            calls.append((method,route,payload));return {'count':10,'tokens':list(range(10))}
    runtime=Runtime()
    gate=build_tfgrpo_fit_gate(runtime,registration=caller.registration,client_factory=Tokenizer)
    return caller,client,runtime,gate,calls


def test_exact_vllm_render_count_and_durable_reuse(tmp_path):
    caller,client,runtime,gate,calls=fixture(tmp_path)
    _,prepared=caller.prepare(request())
    receipt=gate(prepared)
    assert receipt['measured_prompt_tokens']==10 and receipt['every_request_fits']
    assert calls[0][:2]==('POST','/tokenize') and len(calls)==1
    assert gate(prepared)==receipt and len(calls)==1
    assert runtime.checked.planned_model_calls==1 and runtime.checked.planned_speech_audio_seconds==runtime.plan.planned_speech_audio_seconds
    client.close()


def test_unknown_tokenizer_outcome_never_resends(tmp_path):
    caller,client,runtime,gate,calls=fixture(tmp_path)
    _,prepared=caller.prepare(request())
    from speech_aware_evidence_acquisition.rails.tfgrpo_runtime import contact_key
    key=contact_key('summary','r1')
    root=tmp_path/'prepared'/'fit-run'/'fit-attempt'
    (root/(key+'.endpoint.json')).write_text('{}')
    with pytest.raises(PreparationError,match='unknown tokenizer'):gate(prepared)
    assert not calls
    client.close()


def test_context_change_cannot_reuse_fit(tmp_path):
    caller,client,_,gate,calls=fixture(tmp_path)
    _,prepared=caller.prepare(request());gate(prepared)
    changed=request();changed['messages'][1]['content']='changed'
    _,other=caller.prepare(changed)
    with pytest.raises(PreparationError,match='saved FIT identity'):gate(other)
    assert len(calls)==1
    client.close()


def test_registration_cap_is_finite(tmp_path):
    with pytest.raises(PreparationError,match='one endpoint'):fixture(tmp_path,cap=2)


def test_changed_renderer_cannot_reuse_fit(tmp_path):
    caller,client,runtime,gate,calls=fixture(tmp_path)
    _,prepared=caller.prepare(request());gate(prepared)
    original=runtime._transport.render_request
    runtime._transport.render_request=lambda raw:{**original(raw),'chat_template_kwargs':{'enable_thinking':False}}
    with pytest.raises(PreparationError,match='saved FIT identity'):gate(prepared)
    assert len(calls)==1
    client.close()
