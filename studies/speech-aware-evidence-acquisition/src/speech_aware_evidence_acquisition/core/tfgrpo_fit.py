"""Registered vLLM FIT for realized TF-GRPO prepared-text contacts."""
import hashlib
import json
import threading
from dataclasses import asdict, replace

from .atomic_file import publish_new
from .preparation import PreparationError, canonical_bytes, digest, prepare_artifact, load_artifact
from .preparation_fit import GuardedTokenizer, fit_requests


def validate_tfgrpo_fit(values, plan, registration):
    spec = values.get('tfgrpo_contact_fit')
    if (values.get('transport') != 'vllm' or values.get('arm') != 'prepared-context-text'
            or values.get('live_phase_fit') or values.get('whole_roster_wave_barrier')
            or values.get('native_contact_fit') or values.get('prepared_request_fit')
            or not isinstance(spec, dict) or set(spec) != {'run_id','attempt_id','contact_cap'}):
        raise PreparationError('dynamic TF-GRPO requires its exclusive vLLM text FIT')
    import re
    for key in ('run_id','attempt_id'):
        if not isinstance(spec[key],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',spec[key]):
            raise PreparationError('plain FIT run and attempt identities required')
    if (spec['run_id'] == plan.run_id or spec['attempt_id'] == plan.attempt_id
            or type(spec['contact_cap']) is not int
            or spec['contact_cap'] != len(registration['contacts'])):
        raise PreparationError('FIT requires distinct identities and one endpoint per registered contact')
    return replace(plan,run_id=spec['run_id'],attempt_id=spec['attempt_id'],
                   planned_model_calls=spec['contact_cap']).validate()


def build_tfgrpo_fit_gate(runtime, *, registration, client_factory=None):
    from ..rails.core_client import PreparedCoreRequest
    from ..rails.tfgrpo_runtime import validate_registration, learning_source_plan
    from .driver.speech_native import _LoopbackPromptAuditClient
    from .preparation import measure_vllm_multimodal_request
    plan = learning_source_plan(runtime.plan,
        replay=getattr(runtime, '_native_contact_replay', None),data_root=runtime._data_root)
    model_id = runtime.session_required_identity()['model_fragment']
    contacts = validate_registration(registration,plan=plan,model_id=model_id)
    if registration['runtime_identity_sha256'] != digest(runtime.session_required_identity()):
        raise PreparationError('FIT resident identity differs from generation configuration')
    fit_plan = validate_tfgrpo_fit(runtime._composed.values,plan,registration)
    if runtime._session is None or runtime._transport is None:
        raise PreparationError('FIT requires the attached resident')
    runtime.assert_generation_lease()
    runtime._gate_touch(runtime._gate,fit_plan)
    runtime._require_exposure_row_unconsumed(fit_plan.run_id)
    root=runtime._data_root/'prepared'/fit_plan.run_id/fit_plan.attempt_id
    root.mkdir(parents=True,exist_ok=True)
    lock=threading.Lock(); active=set(); receipts=[]
    identity=runtime.session_required_identity(); session=runtime._session
    registration_sha=digest(registration)
    def gate(prepared):
        if not isinstance(prepared,PreparedCoreRequest):
            raise PreparationError('immutable prepared request required')
        payload=prepared.as_payload(); history=payload.get('history',[])
        if (payload.get('raw_speech_query') or payload.get('speech_feature_parts')
                or payload.get('supplied_evidence') or not history):
            raise PreparationError('TF-GRPO FIT accepts text-only prepared feedback')
        cid=history[-1].get('contact_id','')
        key=cid.removeprefix('tfgrpo-')
        if cid!='tfgrpo-'+key or key not in contacts or payload['sample_id']!=contacts[key]['sample_id']:
            raise PreparationError('FIT contact outside registered stage roster')
        runtime.assert_generation_lease()
        body=runtime._transport.render_request(canonical_bytes(payload))
        # The same operational slot routes FIT rendering and generation.
        inputs={'schema':'saea-tfgrpo-contact-fit-input-v1','payload_sha256':digest(payload),
                'registration_sha256':registration_sha,'fit_plan':asdict(fit_plan),
                'rendered_body_sha256':digest(body),
                'session_identity':identity,'slot_tokens':int(runtime._adapter.serving_geometry()['slot_tokens']),
                'fit_multiple':float(runtime._composed.values.get('prompt_fit_multiple',1.0))}
        path=root/(key+'.request-fit.json'); journal=root/(key+'.endpoint.json')
        with lock:
            if key in active:raise PreparationError('FIT contact already in progress')
            active.add(key)
        try:
            if path.exists():
                document=json.loads(path.read_bytes())
                if document.get('inputs')!=inputs or document.get('output',{}).get('every_request_fits') is not True:
                    raise PreparationError('saved FIT identity or outcome differs')
                load_artifact(path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kind='request-fit',inputs=inputs)
                endpoint=json.loads(journal.read_bytes())
                if endpoint != {'method':'POST','route':'/tokenize','fit_plan':asdict(fit_plan),
                    'payload_sha256':digest(payload),'session_identity_sha256':digest(identity)}:
                    raise PreparationError('saved FIT endpoint custody differs')
                runtime.assert_generation_lease()
            else:
                if journal.exists():raise PreparationError('unknown tokenizer outcome; never resend')
                runtime.assert_generation_lease()
                def produce():
                    def contact(method,route):
                        publish_new(journal,canonical_bytes({'method':method,'route':route,
                            'fit_plan':asdict(fit_plan),'payload_sha256':digest(payload),
                            'session_identity_sha256':digest(identity)}))
                    client=GuardedTokenizer((client_factory or _LoopbackPromptAuditClient)(
                        host=session.host,port=session.port,timeout_seconds=600.),
                        guard=runtime.assert_generation_lease,contact_cap=1,contact_sink=contact)
                    result=fit_requests((payload,),expected_sample_ids=(payload['sample_id'],),
                        transport=runtime._transport,client=client,speech_counts=lambda _:(),media_marker=None,
                        slot_tokens=inputs['slot_tokens'],fit_multiple=inputs['fit_multiple'],
                        rendered_bodies=(body,),request_measure=measure_vllm_multimodal_request)
                    return {**result,'contacts':client.counts.total(),'generation_contacts':0}
                prepare_artifact(path,kind='request-fit',inputs=inputs,producer=produce,contact_class='tokenizer')
                document=json.loads(path.read_bytes())
                if document['output']['every_request_fits'] is not True:
                    raise PreparationError('prepared learning request exceeds context capacity')
            receipt={'payload_sha256':digest(payload),'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'artifact_relpath':str(path.relative_to(runtime._data_root)),
                     'every_request_fits':True,'measured_prompt_tokens':document['output']['rows'][0]['measured_prompt_tokens']}
            with lock:
                if receipt not in receipts:receipts.append(receipt)
            return receipt
        finally:
            with lock:active.remove(key)
    gate.execution_plan=fit_plan;gate.receipts=receipts
    return gate
