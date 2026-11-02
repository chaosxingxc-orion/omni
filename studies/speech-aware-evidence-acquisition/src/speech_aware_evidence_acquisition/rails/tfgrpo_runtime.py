"""Bind TF-GRPO learner contacts to the shared gated, metered text client.

This bridge supplies no exposure authority. Its runtime factory requires a
preflighted/served TRAIN artifact runtime and a registered batch binding. The
request-fit callable must be that runtime's registered tokenizer gate. Shared flight dispatch runs the stages; accepting a candidate remains separate.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import threading

from ..core.atomic_file import publish_new
from ..core.tracing import TraceBuffer, get_current_trace_buffer, set_current_trace_buffer
from ..core.stage_timing import measure
from .core_client import FrozenCoreClient
from .hashing import content_hash
from .prepared_feedback import states_from_contexts
from .prepared_text import PREPARED_TEXT_ARM, PREPARED_TEXT_INSTRUCTION
from .tfgrpo_learning import LearningError, canonical, sha

RENDERING_VERSION = 'saea-tfgrpo-prepared-text-summary-dedup-v2'

def learning_configuration(binding):
    return {'schema':'saea-tfgrpo-generation-config-v1', 'model_id':binding['model_id'],
            'decoding_params':binding['decoding_params'], 'max_context_bytes':binding['max_context_bytes'],
            'rendering_version':RENDERING_VERSION, 'runtime_identity_sha256':binding['runtime_identity_sha256']}


STAGES = frozenset({'summary', 'advantage', 'group_update', 'batch_update'})
GROUNDED_RUNTIME_SCHEMA = 'saea-grounded-shards-runtime-binding-v1'
FUSION_RUNTIME_SCHEMA = 'saea-grounded-fusion-runtime-binding-v1'
ACE_RUNTIME_SCHEMA = 'saea-ace-runtime-binding-v1'
REFLEXION_RUNTIME_SCHEMA = 'saea-reflexion-runtime-binding-v1'


def _timing_identity(caller, request):
    return {'learning_stage':request.get('stage'), 'learning_contact':request.get('identity')}


def learning_source_plan(plan, *, replay=None, data_root=None):
    """Keep scientific registration immutable across verified fresh attempts."""
    if replay is None:
        return plan
    from ..core.contact_journal import NativeContactReplay
    if not isinstance(replay, NativeContactReplay) or data_root is None:
        raise LearningError('verified native replay required for learning recovery')
    replay.assert_target(data_root, plan, PREPARED_TEXT_ARM)
    return replay.scientific_source_plan


def contact_key(stage, identity):
    return sha({'stage': stage, 'identity': identity})


def validate_registration(binding, *, plan, model_id):
    """A finite contact roster binds stage identities to original TRAIN anchors.

    Exclude protocol_hash from the embedded plan to avoid a self-referential
    config hash. The runtime protocol binds this registration externally."""
    if isinstance(binding, dict) and binding.get('schema') == REFLEXION_RUNTIME_SCHEMA:
        from .reflexion_runtime import validate_registration as validate_reflexion
        return validate_reflexion(binding, plan=plan, model_id=model_id)
    if isinstance(binding, dict) and binding.get('schema') == ACE_RUNTIME_SCHEMA:
        from .ace_runtime import validate_registration as validate_ace
        return validate_ace(binding, plan=plan, model_id=model_id)
    required = {'schema', 'plan', 'model_id', 'batch_sha256', 'construction_roster_sha256',
                'contacts', 'decoding_params', 'max_context_bytes', 'learning_config_sha256', 'runtime_identity_sha256'}
    grounded = isinstance(binding, dict) and binding.get('schema') in (GROUNDED_RUNTIME_SCHEMA, FUSION_RUNTIME_SCHEMA)
    if grounded:
        required |= {'execution_sha256', 'completion_cap'}
    if not isinstance(binding, dict) or set(binding) != required:
        raise LearningError('exact TF-GRPO runtime registration required')
    if (binding['schema'] not in ('saea-tfgrpo-runtime-binding-v1', GROUNDED_RUNTIME_SCHEMA, FUSION_RUNTIME_SCHEMA)
            or binding['plan'] != {k: v for k, v in asdict(plan).items() if k != 'protocol_hash'} or binding['model_id'] != model_id
            or plan.split_role != 'discovery'
            or plan.execution_profile != 'bounded-discovery-probe'):
        raise LearningError('learning registration requires its exact TRAIN discovery plan and model')
    for name in ('batch_sha256', 'construction_roster_sha256', 'runtime_identity_sha256') + (('execution_sha256',) if grounded else ()):
        if not isinstance(binding[name], str) or re.fullmatch('[a-f0-9]{64}', binding[name]) is None:
            raise LearningError('learning registration needs pinned batch and roster')
    contacts = binding['contacts']
    if not isinstance(contacts, list) or not contacts or len(contacts) > plan.planned_model_calls:
        raise LearningError('finite contact roster exceeds plan budget')
    indexed = {}
    for row in contacts:
        if (not isinstance(row, dict) or set(row) != ({'stage', 'identity', 'sample_id', 'messages_sha256'} if grounded else {'stage', 'identity', 'sample_id'})
                or row['stage'] not in ({'grounded_final_update'} if binding['schema'] == FUSION_RUNTIME_SCHEMA
                    else {'grounded_shard_update'} if grounded else STAGES)
                or any(not isinstance(row[k], str) or not row[k] for k in row)):
            raise LearningError('invalid learning stage anchor')
        key = contact_key(row['stage'], row['identity'])
        if key in indexed:
            raise LearningError('duplicate learning stage identity')
        indexed[key] = row
        if grounded and (not row['sample_id'].startswith('slurp:train:')
                or re.fullmatch('[a-f0-9]{64}', row['messages_sha256']) is None):
            raise LearningError('grounded contact requires TRAIN anchor and exact messages')
    if binding['schema'] == FUSION_RUNTIME_SCHEMA and len(contacts) != 1:
        raise LearningError('fusion requires exactly one final contact')
    decoding = binding['decoding_params']
    if (not isinstance(decoding, dict) or 'id_slot' in decoding
            or type(decoding.get('max_tokens')) is not int or decoding['max_tokens'] < 1):
        raise LearningError('registered output bound required; routing slot is operational')
    if type(binding['max_context_bytes']) is not int or binding['max_context_bytes'] < 1:
        raise LearningError('positive prepared context bound required')
    if grounded and (type(binding['completion_cap']) is not int
            or binding['completion_cap'] != decoding['max_tokens']
            or len(contacts) != plan.planned_model_calls):
        raise LearningError('grounded contact budget or completion cap differs')
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
                 request_fit, assert_lease, request_validator=None):
        if (not isinstance(client, FrozenCoreClient) or client.arm != PREPARED_TEXT_ARM
                or client._plan != plan):
            raise LearningError('shared prepared-text FrozenCoreClient and exact plan required')
        if not callable(request_fit) or not callable(assert_lease):
            raise LearningError('registered request FIT and generation lease checks required')
        self.registration = copy.deepcopy(registration)
        if registration.get('schema') in (ACE_RUNTIME_SCHEMA, REFLEXION_RUNTIME_SCHEMA) and not callable(request_validator):
            raise LearningError('ACE requires observed-predecessor request derivation')
        self.request_validator = request_validator
        source_plan = learning_source_plan(plan,replay=client._contact_replay,data_root=client._binding.data_root)
        self.contacts = validate_registration(self.registration, plan=source_plan, model_id=model_id)
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

    @measure('tfgrpo_request_prepare', _timing_identity)
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
        if self.request_validator is not None:
            self.request_validator(request, self)
        if self.registration['schema'] in (GROUNDED_RUNTIME_SCHEMA, FUSION_RUNTIME_SCHEMA) and sha(request['messages']) != row['messages_sha256']:
            raise LearningError('grounded messages differ from registered shard')
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
            decoding_params={**self.registration['decoding_params'], 'id_slot':self._slots[key]})
        return key, prepared

    @measure('tfgrpo_request_preflight', _timing_identity)
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

    @measure('tfgrpo_generation', _timing_identity)
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
            timings = [asdict(t) for t in self.client.timings
                       if t.request_id == completion.request_id]
            if len(timings) != 1:
                raise LearningError('exact original adapter timing required')
            timing = timings[0]
            if (timing['sample_id'] != completion.sample_id
                    or timing['response_sha256'] != completion.response_sha256
                    or timing['request_sha256'] != completion.request_sha256
                    or timing['latency_seconds'] != cost['latency_seconds']):
                raise LearningError('original timing differs from adapter response')
            publish_new(self.root/(key+'.timing.json'), canonical(timing))
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
                       'trace_sha256': sha(records), 'fit_receipt_sha256': sha(fit_receipt),
                       'timing_sha256': sha(timing)}
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
        timing = json.loads((self.root/(key+'.timing.json')).read_bytes())
        if response.get('timing_sha256') != sha(timing):
            raise LearningError('observed timing custody differs')
        matching = [(rid, p) for channel, rid, p in trace if channel == 'MODEL_RESPONSE']
        costs = [(rid, p) for channel, rid, p in trace if channel == 'COST']
        if len(matching) != 1 or len(costs) != 1:
            raise LearningError('observed receipt needs one adapter response and cost')
        rid, payload = matching[0]
        request_id = rid.removesuffix('/response')
        if (timing.get('request_id') != request_id
                or timing.get('sample_id') != self.contacts[key]['sample_id']
                or timing.get('request_sha256') != response['request_sha256']
                or timing.get('response_sha256') != response['response_sha256']
                or timing.get('response_text_sha256') != payload['response_text_sha256']
                or timing.get('latency_seconds') != costs[0][1]['latency_seconds']):
            raise LearningError('observed timing differs from actual adapter trace')
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


def bind_runtime(runtime, *, receipt_root, request_fit=None, request_validator=None):
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
    replay = getattr(runtime, '_native_contact_replay', None)
    source_plan = learning_source_plan(runtime.plan,replay=replay,data_root=runtime._data_root)
    validate_registration(registration, plan=source_plan, model_id=model_id)
    from ..core.reasoning_knobs import fragment_wire_params
    for key, value in fragment_wire_params(values).items():
        if registration['decoding_params'].get(key) != value:
            raise LearningError('registered learning decoding omits or changes fragment reasoning controls')
    if registration['runtime_identity_sha256'] != sha(runtime.session_required_identity()):
        raise LearningError('registered generation runtime identity differs')
    runtime.assert_generation_lease()
    if request_fit is None:
        from ..core.tfgrpo_fit import build_tfgrpo_fit_gate
        request_fit = build_tfgrpo_fit_gate(runtime, registration=registration)
    kwargs = runtime.frozen_core_client_kwargs()
    kwargs['contact_journal'] = True
    if replay is not None:
        kwargs['contact_replay'] = replay
    client = FrozenCoreClient(**kwargs)
    return MeteredExperienceCaller(client=client, plan=runtime.plan, registration=registration,
        model_id=model_id, receipt_root=receipt_root, request_fit=request_fit,
        assert_lease=runtime.assert_generation_lease, request_validator=request_validator)


def run_registered_learning(runtime, *, receipt_root, learner_root, retain_client=False):
    """Execute the pinned batch after ordinary runtime preflight/serve/lease.

    The shared dispatcher retains the original client for ordinary sealing.
    Standalone callers close it here. This entry uses no alternate transport
    or fabricated token fit; a process restart never silently resets costs.
    """
    from .tfgrpo_source import load_dynamic_source
    from .tfgrpo_learning import ExperienceLearner
    from .tfgrpo_verified import VerifiedImportContext
    values=runtime._composed.values
    registration=values['tfgrpo_learning_registration']
    verified_context = VerifiedImportContext(repo_root=runtime._repo_root,
        data_root=runtime._data_root)
    source,_=load_dynamic_source(values,plan=runtime.plan,repo_root=runtime._repo_root,
        data_root=runtime._data_root,sample_ids=tuple(s.sample_id for s in runtime._samples),
        roster_sha256=registration['construction_roster_sha256'],
        replay=getattr(runtime, '_native_contact_replay', None),
        verified_context=verified_context)
    batch=source['batch']
    historical={}
    if 'summary_import' in batch:
        from .tfgrpo_summary_import import load_import
        summaries,custody=load_import(batch['summary_import'],data_root=runtime._data_root,
            groups=batch['groups'],common=batch['common'],model_id=batch['model_id'],
            verified_context=verified_context)
        historical={'summary_import':batch['summary_import'],'imported_summaries':summaries,'import_custody':custody}
    if 'final_import' in batch:
        from .tfgrpo_final_import import load_import
        operations,custody=load_import(batch['final_import'],data_root=runtime._data_root,
            groups=batch['groups'],common=batch['common'],model_id=batch['model_id'],
            parent=batch['parent'],policy_sha256=batch['policy_sha256'],rollout_kind=batch['rollout_kind'],
            verified_context=verified_context)
        historical={'final_import':batch['final_import'],'imported_operations':operations,'stage_custody':custody}
    sample_ids = {s.sample_id for s in runtime._samples}
    if {row['sample_id'] for row in registration['contacts']} != sample_ids:
        raise LearningError('every artifact roster row requires a registered learning contact')
    learner_root = Path(learner_root).resolve()
    if not learner_root.is_relative_to(Path(runtime._data_root).resolve()):
        raise LearningError('learner artifacts must remain under the runtime data root')
    # A fresh client cannot inherit a prior attempt's adapter cost or trace.
    # Such a restart requires the registered recovery coordinator, never a
    # silent replay into a newly opened shared trace sink.
    caller = getattr(runtime, '_tfgrpo_caller', None) if retain_client else None
    if caller is not None:
        if (caller.root.resolve() != Path(receipt_root).resolve()
                or caller.registration != registration or caller.plan != runtime.plan
                or caller.client is not getattr(runtime, '_client', None)
                or caller.client._closed):
            raise LearningError('live learning recovery client or registration changed')
    else:
        if any(Path(receipt_root).glob('*.intent.json')):
            raise LearningError('prior learning contacts require registered recovery; refusing a fresh client')
        caller=bind_runtime(runtime,receipt_root=receipt_root)
    if retain_client:
        runtime._client = caller.client
        runtime._tfgrpo_caller = caller
    try:
        learner=ExperienceLearner(learner_root,model_id=batch['model_id'],metered_call=caller,
            learning_config_sha256=batch['learning_config_sha256'],preflight=caller.preflight,
            reconcile=caller.reconcile,**batch['bounds'])
        return learner.learn(batch['groups'],batch['parent'],policy_sha256=batch['policy_sha256'],
            rollout_kind=batch['rollout_kind'],**batch['common'],**historical)
    finally:
        if not retain_client:
            caller.client.close()


@dataclass(frozen=True)
class LearningStageSummary:
    """Actual learning contacts; these are not task predictions or scores."""
    document: dict
    results: tuple = ()

    def as_json(self):
        return copy.deepcopy(self.document)


def dispatch_registered_learning(runtime):
    """Shared run_phases branch; ordinary FlightRuntime.seal owns finalization.

    Each output is the anchor's LAST actual model response. The deterministic
    candidate is separately pinned into the trace and phase summaries, never
    misrepresented as an independently generated response for every anchor.
    """
    from .tfgrpo_learning import validate_experience

    run_dir = runtime._run_dir()
    root = run_dir / 'tfgrpo'
    started = getattr(runtime, '_tfgrpo_started', None)
    if started is None:
        started = runtime._clock()
        runtime._tfgrpo_started = started
    candidate = run_registered_learning(runtime, receipt_root=root/'contacts',
        learner_root=root/'learner', retain_client=True)
    ended = runtime._clock()
    return seal_registered_candidate(runtime, candidate, started=started, ended=ended)


def seal_registered_candidate(runtime, candidate, *, started, ended):
    """Record a checked candidate and exact native contacts for ordinary seal."""
    from .tfgrpo_learning import validate_experience
    root = runtime._run_dir()/'tfgrpo'
    runtime._contact_window = (started, ended)
    caller = runtime._tfgrpo_caller
    validate_experience(candidate, caller.model_id)
    candidate_path = root/'learner/candidate.json'
    if json.loads(candidate_path.read_bytes()) != candidate:
        raise LearningError('candidate file differs from learned proposal')
    candidate_pin = {'root':'data',
        'relpath':str(candidate_path.resolve().relative_to(Path(runtime._data_root).resolve())),
        'sha256':hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        'experience_sha256':candidate['sha256'], 'deployment_accepted':False}
    observed = {}
    stage_contacts = {row['stage']:[] for row in caller.registration['contacts']}
    for row in caller.registration['contacts']:
        key = contact_key(row['stage'], row['identity'])
        request = json.loads((caller.root/(key+'.intent.json')).read_bytes())['request']
        response = caller.observed(request)
        rid = response['receipt_id'].removeprefix(f'{caller.plan.run_id}/{caller.plan.attempt_id}/')
        if rid in observed:
            raise LearningError('duplicate metered learning request identity')
        observed[rid] = (row, response)
        stage_contacts[row['stage']].append({'sample_id':row['sample_id'],
            'identity':row['identity'], 'request_id':rid, **response})
    timings = caller.client.timings
    if {t.request_id for t in timings} != set(observed):
        raise LearningError('learning outputs require all original adapter timing custody')
    last = {}
    for timing in timings:
        row, response = observed[timing.request_id]
        if timing.sample_id != row['sample_id'] or timing.response_sha256 != response['response_sha256']:
            raise LearningError('learning contact timing differs from response receipt')
        last[timing.sample_id] = {'sample_id':timing.sample_id, 'response_text':response['text']}
    if set(last) != {sample.sample_id for sample in runtime._samples}:
        raise LearningError('learning artifact does not cover the exact TRAIN roster')
    usage = {key:sum(response['usage'][key] for _,response in observed.values())
             for key in ('calls','prompt_tokens','completion_tokens')}
    if candidate['learning_usage'] != usage:
        raise LearningError('candidate usage differs from actual adapter contact receipts')
    runtime._output_rows = [last[s.sample_id] for s in runtime._samples]
    runtime._phase_summaries = tuple(LearningStageSummary({
        'schema':'saea-tfgrpo-stage-summary-v1', 'phase':stage, 'result_kind':'artifact',
        'contacts':len(contacts), 'rows':len({r['sample_id'] for r in contacts}),
        'contact_receipts':contacts,
        **({'candidate':candidate_pin} if stage in ('batch_update', 'grounded_final_update', 'ace_curate', 'reflexion_reflect') else {}),
    }) for stage,contacts in stage_contacts.items())
    binding = {'schema':'saea-tfgrpo-flight-proposal-v1', 'candidate':candidate_pin,
        'batch_sha256':caller.registration['batch_sha256'],
        'learning_config_sha256':caller.registration['learning_config_sha256'],
        'contact_response_sha256':candidate['contact_response_sha256'],
        'usage':usage, 'result_kind':'artifact'}
    buffer = TraceBuffer(caller.plan.run_id)
    buffer.emit('ORG', caller.plan.run_id+'/tfgrpo-candidate', {'org_learning_proposal':binding})
    caller.client.flush_trace_buffer(buffer)
    wall = max(ended-started, 1e-9)
    payload = {
        'dispatch':'registered-tfgrpo-learning', 'policy':runtime._composed.values['policy'],
        'result_kind':'artifact', 'artifact_result':dict(runtime._artifact_result_binding),
        'contact_count':len(observed), 'outputs_row_count':len(last),
        'wall_seconds':wall, 'contacts_per_second':len(observed)/wall,
        'tokens_per_second':usage['completion_tokens']/wall,
        'tokens_per_second_basis':'actual_metered_completion_tokens',
        'average_latency_seconds':sum(t.latency_seconds for t in timings)/len(timings),
        'average_seconds_per_dataset_row':wall/len(last),
        'processed_dataset_rows':len(last), 'total_dataset_rows':len(runtime._samples),
        'remaining_dataset_rows':0, 'estimated_remaining_seconds':0,
        'input_mode':'text-only-zero-audio',
        'contact_window':{'from':started,'to':ended}, 'learning':binding,
        'phases':[{'phase':stage,'contacts':len(rows)} for stage,rows in stage_contacts.items()],
    }
    from ..core.flight.receipts import require_mandatory_throughput_fields
    require_mandatory_throughput_fields('run_phases', payload, ('contacts_per_second','tokens_per_second'))
    return runtime.receipts.record('run_phases', payload)
