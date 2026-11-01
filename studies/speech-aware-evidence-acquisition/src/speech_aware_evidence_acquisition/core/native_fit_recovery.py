"""Durable FIT custody for native contact recovery, without tokenizer replay."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .atomic_file import publish_new
from .contact_journal import ContactJournal, canonical, comparable_request, digest, logical_contact
from .preparation import PreparationError, load_artifact


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
        with self._lock:
            logical = logical_contact(payload)
            key = (logical['sample_id'], logical['agent_contact_id'])
            if key in self._used:
                raise PreparationError('native FIT repeated agent_contact_id')
            if self.replay is not None:
                # Includes unknown-response rejection BEFORE any tokenizer work.
                observed_payload = self.replay.match(payload)
                source_path = self._source.path(payload)
                if source_path in self._pins:
                    result = self._reuse(source_path, payload, prompt_limit_tokens)
                    self._used.add(key)
                    return result
                if source_path.exists() or observed_payload is not None:
                    raise PreparationError('predecessor FIT missing or changed; no automatic tokenizer repeat')
            if self._gate is None:
                self._gate = self.factory()
            before = len(self._gate.receipts)
            measured = self._gate(request, **(
                {'prompt_limit_tokens': prompt_limit_tokens} if prompt_limit_tokens is not None else {}))
            receipts = self._gate.receipts[before:]
            if len(receipts) != 1:
                raise PreparationError('native FIT must produce exactly one bound receipt')
            self.journal.record(payload, self.runtime.request_fit_inputs((payload,)), receipts[0],
                measured, self.runtime.session_required_identity())
            self.receipts.append(dict(receipts[0]))
            self._used.add(key)
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
            if canonical(current) != canonical(document['inputs']):
                raise PreparationError('predecessor FIT configuration or source changed')
            receipt = document['receipt']
            artifact_path = Path(receipt['path'])
            artifact = json.loads(artifact_path.read_bytes())
            fit_plan = artifact['inputs']['fit_plan']
            expected_root = (self.runtime._data_root / 'prepared' / fit_plan['run_id'] / fit_plan['attempt_id']).resolve()
            if artifact_path.resolve().parent != expected_root or not expected_root.is_relative_to(self.runtime._data_root.resolve()):
                raise PreparationError('predecessor FIT artifact escaped preparation root')
            expected_inputs = {**current, 'native_contact': artifact['inputs']['native_contact'], 'fit_plan': fit_plan}
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
            self.receipts.append({**receipt, 'native_fit_reuse': {
                'schema': 'saea-native-fit-reuse-v1', 'source_path': str(path),
                'source_sha256': self._pins[path], 'new_tokenizer_contacts': 0}})
            return measured
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise PreparationError(f'cannot reuse predecessor native FIT: {error}') from error
