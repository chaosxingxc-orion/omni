"""Bind TF-GRPO learner contacts to the shared gated, metered text client.

This bridge supplies no exposure authority. Its runtime factory requires a
preflighted/served TRAIN artifact runtime and a registered batch binding. The
request-fit callable must be that runtime's registered tokenizer gate. Dynamic
stage orchestration and accepting a candidate remain the caller's duties.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import re
import threading

from ..core.atomic_file import publish_new
from ..core.tracing import TraceBuffer, get_current_trace_buffer, set_current_trace_buffer
from .core_client import FrozenCoreClient
from .hashing import content_hash
from .prepared_feedback import states_from_contexts
from .prepared_text import PREPARED_TEXT_ARM, PREPARED_TEXT_INSTRUCTION
from .tfgrpo_learning import LearningError, canonical, sha

RENDERING_VERSION = 'saea-tfgrpo-prepared-text-v1'

def learning_configuration(binding):
    return {'schema':'saea-tfgrpo-generation-config-v1', 'model_id':binding['model_id'],
            'decoding_params':binding['decoding_params'], 'max_context_bytes':binding['max_context_bytes'],
            'rendering_version':RENDERING_VERSION, 'runtime_identity_sha256':binding['runtime_identity_sha256']}


STAGES = frozenset({'summary', 'advantage', 'group_update', 'batch_update'})


def contact_key(stage, identity):
    return sha({'stage': stage, 'identity': identity})


def validate_registration(binding, *, plan, model_id):
    """A finite contact roster binds stage identities to original TRAIN anchors.

    Exclude protocol_hash from the embedded plan to avoid a self-referential
    config hash. The runtime protocol binds this registration externally."""
    required = {'schema', 'plan', 'model_id', 'batch_sha256', 'construction_roster_sha256',
                'contacts', 'decoding_params', 'max_context_bytes', 'learning_config_sha256', 'runtime_identity_sha256'}
    if not isinstance(binding, dict) or set(binding) != required:
        raise LearningError('exact TF-GRPO runtime registration required')
    if (binding['schema'] != 'saea-tfgrpo-runtime-binding-v1'
            or binding['plan'] != {k: v for k, v in asdict(plan).items() if k != 'protocol_hash'} or binding['model_id'] != model_id
            or plan.split_role != 'discovery'
            or plan.execution_profile != 'bounded-discovery-probe'):
        raise LearningError('learning registration requires its exact TRAIN discovery plan and model')
    for name in ('batch_sha256', 'construction_roster_sha256', 'runtime_identity_sha256'):
        if not isinstance(binding[name], str) or re.fullmatch('[a-f0-9]{64}', binding[name]) is None:
            raise LearningError('learning registration needs pinned batch and roster')
    contacts = binding['contacts']
    if not isinstance(contacts, list) or not contacts or len(contacts) > plan.planned_model_calls:
        raise LearningError('finite contact roster exceeds plan budget')
    indexed = {}
    for row in contacts:
        if (not isinstance(row, dict) or set(row) != {'stage', 'identity', 'sample_id'}
                or row['stage'] not in STAGES
                or any(not isinstance(row[k], str) or not row[k] for k in row)):
            raise LearningError('invalid learning stage anchor')
        key = contact_key(row['stage'], row['identity'])
        if key in indexed:
            raise LearningError('duplicate learning stage identity')
        indexed[key] = row
    decoding = binding['decoding_params']
    if (not isinstance(decoding, dict) or 'id_slot' in decoding
            or type(decoding.get('max_tokens')) is not int or decoding['max_tokens'] < 1):
        raise LearningError('registered output bound required; routing slot is operational')
    if type(binding['max_context_bytes']) is not int or binding['max_context_bytes'] < 1:
        raise LearningError('positive prepared context bound required')
    if binding['learning_config_sha256'] != sha(learning_configuration(binding)):
        raise LearningError('learning generation configuration digest differs')
    canonical(binding)
    return indexed


class MeteredExperienceCaller:
    """An ExperienceLearner callable using actual adapter COST/RESPONSE records.

    The bridge owns neither the client lifetime nor experiment acceptance.
    Captured trace records and response receipts preserve each contact's usage
    under parallel summaries without slicing global counters or inferring cost.
    """
    def __init__(self, *, client, plan, registration, model_id, receipt_root,
                 request_fit, assert_lease):
        if (not isinstance(client, FrozenCoreClient) or client.arm != PREPARED_TEXT_ARM
                or client._plan != plan):
            raise LearningError('shared prepared-text FrozenCoreClient and exact plan required')
        if not callable(request_fit) or not callable(assert_lease):
            raise LearningError('registered request FIT and generation lease checks required')
        self.registration = copy.deepcopy(registration)
        self.contacts = validate_registration(self.registration, plan=plan, model_id=model_id)
        if not {row['sample_id'] for row in self.contacts.values()} <= client._binding.allowed_sample_ids:
            raise LearningError('learning anchor outside runtime binding')
        self.client, self.plan, self.model_id = client, plan, model_id
        self.root = Path(receipt_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.request_fit, self.assert_lease = request_fit, assert_lease
        self._lock = threading.Lock()
        self._prepared = {}
        self._slots = {key: i for i, key in enumerate(sorted(self.contacts))}
        pin = self.root/'registration.json'
        if pin.exists():
            if pin.read_bytes() != canonical(self.registration):
                raise LearningError('durable learning registration changed')
        else:
            publish_new(pin, canonical(self.registration))

    def prepare(self, request):
        request = copy.deepcopy(request)
        if (not isinstance(request, dict) or set(request) != {'schema', 'batch_sha256', 'model_id',
                'stage', 'identity', 'messages', 'learning_config_sha256'}
                or request['schema'] != 'saea-tfgrpo-learning-request-v1'
                or request['batch_sha256'] != self.registration['batch_sha256']
                or request['model_id'] != self.model_id
                or request['learning_config_sha256'] != self.registration['learning_config_sha256']):
            raise LearningError('learning request differs from admitted batch')
        key = contact_key(request['stage'], request['identity'])
        row = self.contacts.get(key)
        if row is None:
            raise LearningError('learning contact is not in the registered finite roster')
        messages = request['messages']
        if (not isinstance(messages, list) or len(messages) != 2
                or any(not isinstance(m, dict) or set(m) != {'role', 'content'}
                       or not isinstance(m['content'], str) or not m['content'].strip() for m in messages)
                or [m['role'] for m in messages] != ['system', 'user']):
            raise LearningError('learner requires exact system/user text messages')
        context = {'schema': 'saea-tfgrpo-model-context-v1', 'stage': request['stage'],
                   'source_messages': messages}
        sample = row['sample_id']
        states, _ = states_from_contexts(
            {sample: {'model_context': context, 'model_context_sha256': content_hash(context),
                      'source_feedback_sha256': request['batch_sha256']}}, active_ids=(sample,),
            prepared_sha256=sha(request), source_bindings_sha256=sha(self.registration),
            roster_sha=self.registration['construction_roster_sha256'],
            maximum=self.registration['max_context_bytes'])
        prepared = self.client.prepare_request(
            [{'role': 'system', 'content': PREPARED_TEXT_INSTRUCTION},
             {'role': 'user', 'content': []}], text_only_sample_id=sample,
            agent_state=[states[sample].as_json()], agent_contact_id='tfgrpo-'+key,
            contact_instruction=('Execute the registered TF-GRPO stage. Follow the system '
                'instruction in source_messages[0] and process source_messages[1] as its '
                'input. Return only the requested stage output.'),
            decoding_params=self.registration['decoding_params'])
        return key, prepared

    def preflight(self, request):
        """Validate and fit before the learner writes a generation intent."""
        key, prepared = self.prepare(request)
        self.assert_lease()
        fit_receipt = self.request_fit(prepared)
        if (not isinstance(fit_receipt, dict)
                or fit_receipt.get('payload_sha256') != sha(prepared.as_payload())
                or fit_receipt.get('every_request_fits') is not True
                or not isinstance(fit_receipt.get('artifact_sha256'), str)
                or re.fullmatch('[a-f0-9]{64}', fit_receipt['artifact_sha256']) is None):
            raise LearningError('request FIT must return its bound artifact receipt')
        with self._lock:
            current = self._prepared.get(key)
            if current is not None and current[0] != request:
                raise LearningError('prepared contact changed')
            self._prepared[key] = (copy.deepcopy(request), prepared, fit_receipt)
        return fit_receipt

    def __call__(self, request):
        key = contact_key(request['stage'], request['identity'])
        intent = self.root/(key+'.intent.json')
        with self._lock:
            if intent.exists():
                raise LearningError('learning contact already attempted; use durable observed receipt, never resend')
            cached = self._prepared.get(key)
        if cached is None:
            self.preflight(request)
            cached = self._prepared[key]
        if cached[0] != request:
            raise LearningError('prepared contact changed')
        prepared, fit_receipt = cached[1:]
        with self._lock:
            if intent.exists():
                raise LearningError('learning contact already attempted; never resend')
            publish_new(intent, canonical({'request': request, 'payload': prepared.as_payload()}))
            publish_new(self.root/(key+'.fit.json'), canonical(fit_receipt))
        self.assert_lease()
        previous = get_current_trace_buffer()
        buffer = TraceBuffer(self.plan.run_id)
        set_current_trace_buffer(buffer)
        try:
            completion = self.client.complete_prepared_with_metadata(prepared, id_slot=self._slots[key])
            records = list(buffer.iter_records())
            publish_new(self.root/(key+'.trace.json'), canonical(records))
            responses = [p for channel, rid, p in records
                         if channel == 'MODEL_RESPONSE' and rid == completion.request_id+'/response']
            costs = [p for channel, rid, p in records
                     if channel == 'COST' and rid == completion.request_id+'/cost']
            if len(responses) != 1 or len(costs) != 1:
                raise LearningError('exact adapter response and cost required')
            response, cost = responses[0], costs[0]
            if (response['text'] != completion.text or response['response_sha256'] != completion.response_sha256
                    or response['response_text_sha256'] != completion.response_text_sha256
                    or type(cost.get('calls')) is not int or cost['calls'] != 1):
                raise LearningError('adapter response/cost custody differs')
            usage = {'calls': cost['calls']}
            for field in ('prompt_tokens', 'completion_tokens'):
                value = response['usage'].get(field)
                if type(value) is not int or value < 0 or cost.get(field) != value:
                    raise LearningError('actual metered token usage missing or inconsistent')
                usage[field] = value
            receipt = {'text': completion.text, 'model_id': self.model_id,
                       'receipt_id': f'{self.plan.run_id}/{self.plan.attempt_id}/{completion.request_id}',
                       'usage': usage, 'request_sha256': completion.request_sha256,
                       'response_sha256': completion.response_sha256,
                       'trace_sha256': sha(records), 'fit_receipt_sha256': sha(fit_receipt)}
            publish_new(self.root/(key+'.response.json'), canonical({'request_sha256': sha(request), 'response': receipt}))
            return receipt
        finally:
            set_current_trace_buffer(previous)
            self.client.flush_trace_buffer(buffer)

    def reconcile(self, request):
        key, _ = self.prepare(request)
        if not (self.root/(key+'.response.json')).exists():
            return None
        response = self.observed(request)
        return {'request_sha256':sha(request), 'response':response, 'response_sha256':sha(response)}

    def observed(self, request):
        """Read exact observed receipt for explicit reconciliation, with no contact."""
        key = contact_key(request['stage'], request['identity'])
        intent = json.loads((self.root/(key+'.intent.json')).read_bytes())
        record = json.loads((self.root/(key+'.response.json')).read_bytes())
        if intent.get('request') != request or record['request_sha256'] != sha(request):
            raise LearningError('observed learning request changed')
        trace = json.loads((self.root/(key+'.trace.json')).read_bytes())
        fit = json.loads((self.root/(key+'.fit.json')).read_bytes())
        if record['response']['trace_sha256'] != sha(trace) or record['response']['fit_receipt_sha256'] != sha(fit):
            raise LearningError('observed learning receipt custody differs')
        response = record['response']
        matching = [(rid, p) for channel, rid, p in trace if channel == 'MODEL_RESPONSE']
        costs = [(rid, p) for channel, rid, p in trace if channel == 'COST']
        if len(matching) != 1 or len(costs) != 1:
            raise LearningError('observed receipt needs one adapter response and cost')
        rid, payload = matching[0]
        request_id = rid.removesuffix('/response')
        expected_usage = {'calls': costs[0][1].get('calls'),
                          **{k: payload['usage'].get(k) for k in ('prompt_tokens', 'completion_tokens')}}
        if (response['model_id'] != self.model_id
                or response['text'] != payload['text']
                or response['response_sha256'] != payload['response_sha256']
                or response['usage'] != expected_usage
                or response['receipt_id'] != f'{self.plan.run_id}/{self.plan.attempt_id}/{request_id}'
                or costs[0][0] != request_id+'/cost'
                or any(costs[0][1].get(k) != v for k, v in expected_usage.items())):
            raise LearningError('observed response differs from adapter records')
        return copy.deepcopy(response)


def bind_runtime(runtime, *, receipt_root, request_fit=None):
    """Attach only after runtime preflight/serve; do not mint gates or authority.

    The registration must already be part of composed configuration, hence its
    protocol binding. Caller retains and closes the returned client's lifetime.
    """
    receipt_root = Path(receipt_root).resolve()
    if not receipt_root.is_relative_to(Path(runtime._data_root).resolve()):
        raise LearningError('learning receipts must remain under the runtime data root')
    values = runtime._composed.values
    if (values.get('arm') != PREPARED_TEXT_ARM or values.get('result_kind') != 'artifact'
            or values.get('scorers') != []):
        raise LearningError('TF-GRPO runtime must be a scoreless prepared-text TRAIN stage')
    model_id = runtime.session_required_identity()['model_fragment']
    registration = values.get('tfgrpo_learning_registration')
    validate_registration(registration, plan=runtime.plan, model_id=model_id)
    if registration['runtime_identity_sha256'] != sha(runtime.session_required_identity()):
        raise LearningError('registered generation runtime identity differs')
    runtime.assert_generation_lease()
    if request_fit is None:
        from ..core.tfgrpo_fit import build_tfgrpo_fit_gate
        request_fit = build_tfgrpo_fit_gate(runtime, registration=registration)
    kwargs = runtime.frozen_core_client_kwargs()
    kwargs['contact_journal'] = True
    client = FrozenCoreClient(**kwargs)
    return MeteredExperienceCaller(client=client, plan=runtime.plan, registration=registration,
        model_id=model_id, receipt_root=receipt_root, request_fit=request_fit,
        assert_lease=runtime.assert_generation_lease)


def run_registered_learning(runtime, *, receipt_root, learner_root):
    """Execute the pinned batch after ordinary runtime preflight/serve/lease.

    The coordinator retains runtime finalization and candidate acceptance. This
    entry uses no alternate transport or fabricated token fit.
    """
    from .tfgrpo_source import load_dynamic_source
    from .tfgrpo_learning import ExperienceLearner
    values=runtime._composed.values
    registration=values['tfgrpo_learning_registration']
    source,_=load_dynamic_source(values,plan=runtime.plan,repo_root=runtime._repo_root,
        data_root=runtime._data_root,sample_ids=tuple(s.sample_id for s in runtime._samples),
        roster_sha256=registration['construction_roster_sha256'])
    batch=source['batch']
    caller=bind_runtime(runtime,receipt_root=receipt_root)
    try:
        learner=ExperienceLearner(learner_root,model_id=batch['model_id'],metered_call=caller,
            learning_config_sha256=batch['learning_config_sha256'],preflight=caller.preflight,
            reconcile=caller.reconcile,**batch['bounds'])
        return learner.learn(batch['groups'],batch['parent'],policy_sha256=batch['policy_sha256'],
            rollout_kind=batch['rollout_kind'],**batch['common'])
    finally:
        caller.client.close()
