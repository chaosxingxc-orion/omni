"""Durable FIT custody for native contact recovery, without tokenizer replay."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .atomic_file import publish_new
from .contact_journal import ContactJournal, canonical, comparable_request, digest, logical_contact
from .preparation import PreparationError, load_artifact
from .stage_timing import measured_lock, span


def runtime_only_fit_repair(current, original):
    """Raw request, boundary, service and tokenizer implementations stay exact.

    The caller independently verifies all original FIT output/journal bytes and
    the immutable current request. A runtime file edit alone cannot change the
    token length of those identical bytes on that same serving deployment.
    """
    from copy import deepcopy
    comparable = deepcopy(current)
    try:
        before = original['sources']['implementation']['runtime']
        after = current['sources']['implementation']['runtime']
        if before == after or any(not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in (before,after)):
            raise ValueError('distinct runtime hashes required')
        if not current['sources']['boundary_sha256']:
            raise ValueError('actual request boundary required')
        comparable['sources']['implementation']['runtime'] = before
    except (KeyError,TypeError,ValueError) as error:
        raise PreparationError('predecessor FIT configuration or source changed') from error
    if canonical(comparable) != canonical(original):
        raise PreparationError('predecessor FIT non-runtime identity changed')
    return {'schema':'saea-native-fit-runtime-repair-v1',
        'previous_runtime_sha256':before,'current_runtime_sha256':after,
        'previous_inputs_sha256':digest(canonical(original)),
        'current_inputs_sha256':digest(canonical(current)),
        'exact_request_boundary_sha256':current['sources']['boundary_sha256']}


class NativeFitJournal:
    def __init__(self, data_root, plan, arm):
        self.contacts = ContactJournal(data_root, plan, arm)
        self.root = self.contacts.root

    def path(self, payload):
        return self.root / (digest(canonical(comparable_request(payload))) + '.fit.json')

    def record(self, payload, inputs, receipt, measured, session_identity):
        path = Path(receipt['path'])
        journal = path.with_name(path.name + '.contacts.jsonl')
        if not journal.is_file():
            raise PreparationError('successful native FIT lacks durable endpoint journal')
        if type(measured) is not int or measured < 0:
            raise PreparationError('native FIT recovery requires exact measured prompt tokens')
        self.contacts._ensure_directory()
        document = {'schema': 'saea-native-fit-custody-v1',
            'identity': self.contacts.identity, 'payload': payload, 'inputs': inputs,
            'receipt': receipt, 'measured_prompt_tokens': measured,
            'session_identity': session_identity,
            'endpoint_journal_sha256': digest(journal.read_bytes())}
        publish_new(self.path(payload), canonical(document))
        return document


class RecoverableNativeFitGate:
    """Lazily construct the normal gate only for genuinely new FIT contacts.

    Reuse validates original bytes, current implementation/config/media inputs,
    current service identity, and the original endpoint journal. Unknown source
    FIT outcomes are refused by the original gate's durable contact journal.
    """
    def __init__(self, runtime, plan, gate_factory, *, replay=None):
        self.runtime, self.plan, self.factory, self.replay = runtime, plan, gate_factory, replay
        self.journal = NativeFitJournal(runtime._data_root, plan, runtime._arm())
        self._gate = None
        self._lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._init_failed = False
        self.receipts = []
        self._used = set()
        self._source = None
        self._pins = {}
        if replay is not None:
            replay.assert_target(runtime._data_root, plan, runtime._arm())
            self._source = NativeFitJournal(runtime._data_root, replay.source_plan, runtime._arm())
            self._pins = {p: digest(p.read_bytes()) for p in self._source.root.glob('*.fit.json')}

    def __call__(self, request, *, prompt_limit_tokens=None):
        from ..rails.core_client import PreparedCoreRequest
        if not isinstance(request, PreparedCoreRequest):
            raise PreparationError('native FIT requires an immutable prepared request')
        payload = request.as_payload()
        with measured_lock(self._lock, "native_fit_recovery_lock_wait", lambda: {
                "sample_id": payload["sample_id"], "request_id": payload["request_id"]}):
            logical = logical_contact(payload)
            key = (logical['sample_id'], logical['agent_contact_id'])
            if key in self._used:
                raise PreparationError('native FIT repeated agent_contact_id')
            # Claim before any I/O; failures and unknown outcomes retain this
            # identity. Other contacts may proceed without replaying this one.
            self._used.add(key)
        if self.replay is not None:
            observed_payload = self.replay.match(payload)
            source_path = self._source.path(payload)
            if source_path in self._pins:
                return self._reuse(source_path, payload, prompt_limit_tokens)
            if source_path.exists() or observed_payload is not None:
                raise PreparationError('predecessor FIT missing or changed; no automatic tokenizer repeat')
        # Initialization may attach a service and perform slow identity I/O.
        # It must not block claims or completed/reused receipt publication.
        with measured_lock(self._init_lock, "native_fit_initialization_wait", lambda: {
                "sample_id": payload["sample_id"], "request_id": payload["request_id"]}):
            if self._init_failed:
                raise PreparationError('native FIT initialization previously failed; new attempt required')
            if self._gate is None:
                try:
                    with span("native_fit_initialization"):
                        gate = self.factory()
                        if not callable(gate) or not callable(getattr(gate, 'receipt_for', None)):
                            raise PreparationError('concurrent native FIT requires contact-bound receipt access')
                    self._gate = gate
                except BaseException:
                    # Partial initialization can already have durable effects.
                    # Other waiting contacts must not implicitly retry it.
                    self._init_failed = True
                    raise
            gate = self._gate
        if not callable(getattr(gate, 'receipt_for', None)):
            raise PreparationError('concurrent native FIT requires contact-bound receipt access')
        measured = gate(request, **(
            {'prompt_limit_tokens': prompt_limit_tokens} if prompt_limit_tokens is not None else {}))
        receipt = gate.receipt_for(request)
        if (receipt.get('sample_id'), receipt.get('agent_contact_id')) != key:
            raise PreparationError('native FIT receipt contact differs')
        self.journal.record(payload, self.runtime.request_fit_inputs((payload,)), receipt,
            measured, self.runtime.session_required_identity())
        with self._lock:
            self.receipts.append(dict(receipt))
        return measured

    def _reuse(self, path, payload, prompt_limit_tokens):
        try:
            self.runtime.assert_generation_lease()
            blob = path.read_bytes()
            if digest(blob) != self._pins[path]:
                raise PreparationError('predecessor FIT custody changed')
            document = json.loads(blob)
            original = document['payload']
            if (document['schema'] != 'saea-native-fit-custody-v1'
                    or document['identity'] != self._source.contacts.identity
                    or comparable_request(original) != comparable_request(payload)
                    or self._source.path(original) != path):
                raise PreparationError('predecessor FIT request/identity differs')
            if document['session_identity'] != self.runtime.session_required_identity():
                raise PreparationError('predecessor FIT service identity changed')
            current = self.runtime.request_fit_inputs((original,))
            runtime_repair = None
            if canonical(current) != canonical(document['inputs']):
                runtime_repair = runtime_only_fit_repair(current, document['inputs'])
            receipt = document['receipt']
            artifact_path = Path(receipt['path'])
            artifact = json.loads(artifact_path.read_bytes())
            fit_plan = artifact['inputs']['fit_plan']
            expected_root = (self.runtime._data_root / 'prepared' / fit_plan['run_id'] / fit_plan['attempt_id']).resolve()
            if artifact_path.resolve().parent != expected_root or not expected_root.is_relative_to(self.runtime._data_root.resolve()):
                raise PreparationError('predecessor FIT artifact escaped preparation root')
            expected_inputs = {**document['inputs'], 'native_contact': artifact['inputs']['native_contact'], 'fit_plan': fit_plan}
            output = load_artifact(artifact_path, expected_sha256=receipt['sha256'], kind='request-fit', inputs=expected_inputs)
            from .preparation_fit import boundary_identity
            measured = document['measured_prompt_tokens']
            if (output.get('every_request_fits') is not True
                    or output.get('boundary_sha256') != boundary_identity((original,))
                    or output.get('sample_id') != original['sample_id']
                    or output.get('agent_contact_id') != logical_contact(original)['agent_contact_id']
                    or type(measured) is not int or measured < 0
                    or output['rows'][0].get('measured_prompt_tokens') != measured):
                raise PreparationError('predecessor FIT output does not certify this raw request')
            journal = artifact_path.with_name(artifact_path.name + '.contacts.jsonl')
            if digest(journal.read_bytes()) != document['endpoint_journal_sha256']:
                raise PreparationError('predecessor FIT endpoint journal changed')
            if prompt_limit_tokens is not None:
                if type(prompt_limit_tokens) is not int or prompt_limit_tokens < 1 or measured > prompt_limit_tokens:
                    raise PreparationError('predecessor FIT exceeds current prompt limit')
            reused_receipt = {**receipt, 'native_fit_reuse': {
                'schema': 'saea-native-fit-reuse-v1', 'source_path': str(path),
                'source_sha256': self._pins[path], 'new_tokenizer_contacts': 0,
                **({'runtime_repair': runtime_repair} if runtime_repair is not None else {})}}
            with self._lock:
                self.receipts.append(reused_receipt)
            return measured
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise PreparationError(f'cannot reuse predecessor native FIT: {error}') from error
