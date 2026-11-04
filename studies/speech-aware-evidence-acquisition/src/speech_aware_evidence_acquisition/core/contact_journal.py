"""Durable request/response custody for interrupted native generation.

An intent without a response is UNKNOWN, never permission to resend. Records
preserve raw transport bytes before parsing. Capture does not authorize recovery. Explicit replay binds an aborted
attempt and reuses only verified observed responses without new metering.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

from ..contracts import RUN_ID_PATTERN
from .atomic_file import publish_new


class ContactJournalError(ValueError):
    """The recorded contact cannot be safely written or reconciled."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def _identifier(value):
    if not isinstance(value, str) or not value or value in ('.', '..') or '/' in value or '\\' in value:
        raise ContactJournalError('nonempty confined identifier required')
    return value


def logical_contact(payload):
    """Read exact native contact custody; keep fallback requests distinct."""
    contacts = [item.get('contact_id') for item in payload.get('history', [])
                if isinstance(item, Mapping) and item.get('role') == 'control']
    if not contacts or not isinstance(contacts[-1], str) or not contacts[-1]:
        raise ContactJournalError('native contact identity required')
    return {'sample_id': payload['sample_id'], 'agent_contact_id': contacts[-1]}


class ContactJournal:
    """Create exclusive, fsynced records owned by one exact execution attempt."""

    def __init__(self, data_root, plan, arm, *, source_root=None):
        plan.validate(source_root=source_root)
        if not RUN_ID_PATTERN.fullmatch(plan.run_id):
            raise ContactJournalError('registered run identity required')
        attempt = _identifier(plan.attempt_id)
        self.root = Path(data_root).resolve() / 'contact-journals' / plan.run_id / attempt
        self.identity = {'run_id': plan.run_id, 'attempt_id': attempt, 'arm': arm,
            'protocol_hash': plan.protocol_hash, 'split_identity_hash': plan.split_identity_hash}

    def _ensure_directory(self):
        # Sync newly created directory entries as well as the record itself.
        missing = []
        current = self.root
        while not current.exists():
            missing.append(current)
            current = current.parent
        for path in reversed(missing):
            path.mkdir(exist_ok=True)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def _path(self, request_id, kind):
        key = digest(_identifier(request_id).encode())
        return self.root / (key + '.' + kind + '.json')

    def intent(self, payload):
        request_id = _identifier(payload['request_id'])
        blob = canonical(payload)
        record = {'schema': 'saea-contact-intent-v1', 'identity': self.identity,
            'request_id': request_id, **logical_contact(payload),
            'request_sha256': digest(blob), 'payload': payload}
        self._ensure_directory()
        try:
            publish_new(self._path(request_id, 'intent'), canonical(record))
        except FileExistsError as error:
            raise ContactJournalError('contact intent already exists; inspect, never resend') from error
        return digest(canonical(record))

    def response(self, request_id, request_sha256, response_bytes, latency_seconds):
        if (not isinstance(response_bytes, bytes) or type(latency_seconds) not in (int, float)
                or not math.isfinite(latency_seconds) or latency_seconds < 0):
            raise ContactJournalError('observed transport bytes and finite latency required')
        intent_path = self._path(request_id, 'intent')
        intent_blob = intent_path.read_bytes()
        intent = json.loads(intent_blob)
        self._validate_intent(intent, request_id)
        if intent['request_sha256'] != request_sha256:
            raise ContactJournalError('response does not match request intent')
        record = {'schema': 'saea-contact-response-v1', 'identity': self.identity,
            'request_id': request_id, 'request_sha256': request_sha256,
            'intent_sha256': digest(intent_blob), 'response_sha256': digest(response_bytes),
            'response_b64': base64.b64encode(response_bytes).decode('ascii'),
            'latency_seconds': latency_seconds}
        try:
            publish_new(self._path(request_id, 'response'), canonical(record))
        except FileExistsError as error:
            raise ContactJournalError('response already recorded; never overwrite') from error

    def _validate_intent(self, intent, request_id):
        if (intent.get('schema') != 'saea-contact-intent-v1' or intent.get('identity') != self.identity
                or intent.get('request_id') != request_id or intent['payload'].get('request_id') != request_id
                or intent.get('request_sha256') != digest(canonical(intent['payload']))
                or any(intent.get(k) != v for k, v in logical_contact(intent['payload']).items())):
            raise ContactJournalError('intent identity or bytes differ')

    def inspect(self, request_id):
        """Read a contact without mutating it; UNKNOWN can only be reconciled."""
        intent_path, response_path = self._path(request_id, 'intent'), self._path(request_id, 'response')
        if not intent_path.exists():
            if response_path.exists():
                raise ContactJournalError('orphan response has no intent')
            return {'state': 'absent', 'request_id': request_id}
        try:
            intent_blob = intent_path.read_bytes()
            intent = json.loads(intent_blob)
            self._validate_intent(intent, request_id)
            if not response_path.exists():
                return {'state': 'unknown', 'intent': intent}
            response = json.loads(response_path.read_bytes())
            blob = base64.b64decode(response['response_b64'], validate=True)
            latency = response['latency_seconds']
            if (response.get('schema') != 'saea-contact-response-v1' or response.get('identity') != self.identity
                    or response.get('request_id') != request_id
                    or response.get('intent_sha256') != digest(intent_blob)
                    or response.get('request_sha256') != intent['request_sha256']
                    or response.get('response_sha256') != digest(blob)
                    or type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
                raise ContactJournalError('response identity or bytes differ')
            return {'state': 'observed', 'intent': intent, 'response': response, 'response_bytes': blob}
        except (KeyError, TypeError, ValueError, UnicodeError) as error:
            raise ContactJournalError('contact journal integrity failed: ' + str(error)) from error


def comparable_request(payload):
    """Remove transport routing only; retain all instructions and evidence bytes.

    Only stable native supplied-evidence IDs are supported. Legacy raw evidence
    named after request IDs cannot be compared safely across attempts.
    """
    import copy
    value = copy.deepcopy(dict(payload))
    request_id = value.pop('request_id')
    value.get('decoding_params', {}).pop('id_slot', None)
    for index, source in enumerate(value.get('evidence_provenance', [])):
        if source.get('lock_key') == 'rails-supply':
            if source.get('relpath') != f'rails-message/{request_id}/{index}':
                raise ContactJournalError('unexpected request-derived evidence provenance')
            source['relpath'] = f'rails-message/CONTACT/{index}'
    if any(str(e.get('evidence_id', '')).startswith(request_id + '-supply-')
           for e in value.get('supplied_evidence', [])):
        raise ContactJournalError('legacy request-derived evidence IDs cannot be replayed')
    return value


class NativeContactReplay:
    """Index an aborted attempt's observed native contacts without new spend.

    The caller must separately register a recovery attempt. This object neither
    opens an attempt nor authorizes generation; it refuses changed or unknown
    contacts and requires the source's recorded spend to cover every response.
    """

    MAX_EMPTY_RECOVERY_ANCESTORS = 32

    def __init__(self, data_root, source_plan, recovery_plan, arm, *, _ancestry_seen=()):
        import threading
        if len(_ancestry_seen) > self.MAX_EMPTY_RECOVERY_ANCESTORS:
            raise ContactJournalError('native recovery ancestry exceeds supported depth')
        source_plan.validate(); recovery_plan.validate()
        if source_plan.attempt_id in _ancestry_seen or recovery_plan.attempt_id in _ancestry_seen:
            raise ContactJournalError('cyclic native recovery ancestry')
        if (source_plan.attempt_id == recovery_plan.attempt_id
                or source_plan.owner_allowance != recovery_plan.owner_allowance
                or any(getattr(source_plan, key) != getattr(recovery_plan, key) for key in (
                    'run_id', 'protocol_hash', 'split_identity_hash', 'carrier_lock_key',
                    'planned_model_calls', 'planned_speech_audio_seconds', 'planned_gpu_hours', 'execution_profile', 'split_role'))):
            raise ContactJournalError('recovery requires a fresh attempt of the unchanged plan')
        self.source_plan = source_plan
        self.source = ContactJournal(data_root, source_plan, arm)
        self.target_identity = ContactJournal(data_root, recovery_plan, arm).identity
        self._attempt_path = Path(data_root) / 'attempts' / source_plan.run_id / (source_plan.attempt_id + '.json')
        attempt_blob = self._attempt_path.read_bytes()
        attempt = json.loads(attempt_blob)
        if (attempt.get('state') != 'ABORTED' or attempt.get('run_id') != source_plan.run_id
                or attempt.get('attempt_id') != source_plan.attempt_id
                or attempt.get('carrier_lock_key') != source_plan.carrier_lock_key):
            raise ContactJournalError('source attempt must be explicitly aborted, never merely unobserved')
        from .attempts.constants import CONTACT_KIND_PREPARED_TEXT
        prepared_text = arm == "prepared-context-text"
        if prepared_text != (attempt.get("contact_kind") == CONTACT_KIND_PREPARED_TEXT):
            raise ContactJournalError("source contact kind differs; legacy text audio spend requires explicit reconciliation")
        self._attempt_sha256 = digest(attempt_blob)
        self.source_actual_usage = dict(attempt["actual"])
        self._by_contact, self._records, self._pins, self._used = {}, {}, {}, set()
        self._lock = threading.Lock()
        intents = list(self.source.root.glob('*.intent.json'))
        if not intents and not ((prepared_text or arm == 'raw-speech-agent-native-tools')
                                and self.source_actual_usage == {'calls':0,'audio_seconds':0.0}):
            raise ContactJournalError('source has no durable native contacts')
        audio = []
        for path in intents:
            record = json.loads(path.read_bytes())
            rid = record['request_id']
            if path != self.source._path(rid, 'intent'):
                raise ContactJournalError('journal filename does not match request')
            observed = self.source.inspect(rid)
            if observed['state'] not in ('observed', 'unknown'):
                raise ContactJournalError('unreadable predecessor contact')
            payload = observed['intent']['payload']
            logical = logical_contact(payload)
            key = (logical['sample_id'], logical['agent_contact_id'])
            comparison = canonical(comparable_request(payload))
            variants = self._by_contact.setdefault(key, {})
            if comparison in variants:
                raise ContactJournalError('duplicate logical request in source; reconcile before replay')
            variants[comparison] = rid
            self._records[rid] = observed
            for kind in ('intent', 'response'):
                p = self.source._path(rid, kind)
                self._pins[p] = digest(p.read_bytes()) if p.exists() else None
            if prepared_text:
                from ..rails.prepared_text import validate_prepared_text_payload
                validate_prepared_text_payload(payload)
            audio.append(0.0 if prepared_text else payload["audio_seconds"])
        # An unmatched response cannot silently disappear from recovery costs.
        expected = {self.source._path(rid, 'response') for rid, obs in self._records.items() if obs['state'] == 'observed'}
        if set(self.source.root.glob('*.response.json')) != expected:
            raise ContactJournalError('orphan journal response')
        actual = attempt['actual']
        if (type(actual['calls']) is not int or type(actual['audio_seconds']) not in (int, float)
                or actual['calls'] != len(intents) or not math.isclose(actual['audio_seconds'], math.fsum(audio), rel_tol=1e-12, abs_tol=1e-9)):
            raise ContactJournalError('source attempt usage and contact journal differ; reconcile before recovery')
        self.scientific_source_plan = source_plan
        self._empty_ancestry = None
        self._empty_ancestor = None
        self._ancestry_pins = {}
        resume_path = Path(data_root)/'runs'/source_plan.run_id/source_plan.attempt_id/'resume-declaration.json'
        if resume_path.exists():
            # The immediate source owns every nonzero contact. Skipping a prior
            # zero-contact startup is safe; merging spent ancestor journals is
            # a different accounting contract and remains explicitly refused.
            if not (prepared_text or arm == 'raw-speech-agent-native-tools'):
                raise ContactJournalError('nonempty or unsupported recovery ancestry')
            resume_blob = resume_path.read_bytes(); resume = json.loads(resume_blob)
            if (resume.get('schema') != 'saea-resume-declaration-v1'
                    or resume.get('run_id') != source_plan.run_id
                    or resume.get('attempt_id') != source_plan.attempt_id):
                raise ContactJournalError('source recovery declaration differs')
            declaration = resume['native_recovery_receipt']
            if not isinstance(declaration, dict) or set(declaration) != {'path','sha256'}:
                raise ContactJournalError('pinned ancestor recovery required')
            receipt_path = Path(declaration['path'])
            if not receipt_path.is_absolute() or not receipt_path.resolve().is_relative_to(Path(data_root).resolve()):
                raise ContactJournalError('ancestor receipt escaped data root')
            receipt_blob = receipt_path.read_bytes(); receipt = json.loads(receipt_blob)
            if (digest(receipt_blob) != declaration['sha256']
                    or set(receipt) != {'schema', 'source', 'target_plan'}
                    or receipt.get('schema') != 'saea-native-recovery-v1'):
                raise ContactJournalError('ancestor receipt changed')
            from ..contracts import ExecutionPlan
            ancestor_target = ExecutionPlan(**receipt['target_plan'])
            if (ancestor_target != source_plan
                    or ancestor_target.owner_allowance != source_plan.owner_allowance):
                raise ContactJournalError('ancestor target differs from immediate source')
            ancestor = NativeContactReplay(data_root,ExecutionPlan(**receipt['source']['plan']),source_plan,arm,
                _ancestry_seen=(*_ancestry_seen,recovery_plan.attempt_id))
            if (ancestor.source_binding() != receipt['source'] or ancestor.request_ids
                    or ancestor.source_actual_usage != {'calls':0,'audio_seconds':0.0}):
                raise ContactJournalError('spent ancestor requires explicit multi-predecessor recovery')
            expected = [{'attempt_id':ancestor.source_plan.attempt_id,'mode':'native-contact-journal',
                         'pin_sha256':ancestor.source_binding()['attempt_sha256']}]
            if resume.get('predecessors') != expected:
                raise ContactJournalError('ancestor declaration provenance differs')
            self._empty_ancestry = {'resume':{'path':str(resume_path.resolve()),'sha256':digest(resume_blob)},
                'receipt':dict(declaration),'source':ancestor.source_binding()}
            self._empty_ancestor = ancestor
            self.scientific_source_plan = ancestor.scientific_source_plan
            self._ancestry_pins = {resume_path: digest(resume_blob),
                receipt_path: digest(receipt_blob),
                ancestor._attempt_path: ancestor._attempt_sha256, **ancestor._pins}
            self._pins.update(self._ancestry_pins)

    def source_binding(self):
        """Pin complete predecessor spend, including contacts not reused."""
        from dataclasses import asdict
        plan = asdict(self.source_plan)
        if self.source_plan.owner_allowance is not None:
            plan['owner_allowance'] = asdict(self.source_plan.owner_allowance)
        return {'schema': 'saea-native-replay-source-v1',
            'plan': plan, 'arm': self.source.identity['arm'],
            'attempt_path': str(self._attempt_path.resolve()),
            'attempt_sha256': self._attempt_sha256,
            'actual_usage': dict(self.source_actual_usage),
            **({'empty_ancestry':self._empty_ancestry} if self._empty_ancestry else {})}

    def receipt(self, request_id):
        self._unchanged(request_id)
        record = self.source.inspect(request_id)
        if record['state'] != 'observed':
            raise ContactJournalError('replay receipt requires an observed response')
        return {'schema': 'saea-native-contact-replay-v1',
            'source_identity': self.source.identity,
            'source_intent_sha256': record['response']['intent_sha256'],
            'source_response_sha256': record['response']['response_sha256'],
            'new_model_calls': 0}

    @property
    def request_ids(self):
        return frozenset(self._records)

    def assert_all_contacts_observed(self):
        """Audit pinned predecessor custody before dispatching any fresh work.

        This does not consume responses or reconcile unknown outcomes. Changed
        custody requires a newly verified replay, never a cached assertion.
        """
        if digest(self._attempt_path.read_bytes()) != self._attempt_sha256:
            raise ContactJournalError('source attempt changed after recovery inspection')
        for path, expected in self._pins.items():
            current = digest(path.read_bytes()) if path.exists() else None
            if current != expected:
                raise ContactJournalError('source custody changed after recovery inspection')
        self._assert_contact_inventory()
        if any(record['state'] != 'observed' for record in self._records.values()):
            raise ContactJournalError('contact outcome unknown; reconcile before dispatching any work')

    def _assert_contact_inventory(self):
        # Hash pins alone cannot detect files added to a formerly empty journal.
        # Audit every zero-contact ancestor before dispatch or response reuse.
        intents = {self.source._path(rid, 'intent') for rid in self._records}
        responses = {self.source._path(rid, 'response') for rid, record in self._records.items()
                     if record['state'] == 'observed'}
        if (set(self.source.root.glob('*.intent.json')) != intents
                or set(self.source.root.glob('*.response.json')) != responses):
            raise ContactJournalError('source contact inventory changed after recovery inspection')
        if self._empty_ancestor is not None:
            self._empty_ancestor._assert_contact_inventory()

    def assert_target(self, data_root, plan, arm):
        expected = ContactJournal(data_root, plan, arm)
        if expected.identity != self.target_identity or expected.root.parent != self.source.root.parent:
            raise ContactJournalError('replay is bound to another recovery attempt')

    def _unchanged(self, rid):
        if digest(self._attempt_path.read_bytes()) != self._attempt_sha256:
            raise ContactJournalError('source attempt changed after recovery inspection')
        self._assert_empty_ancestry()
        for kind in ('intent', 'response'):
            p = self.source._path(rid, kind)
            current = digest(p.read_bytes()) if p.exists() else None
            if current != self._pins[p]:
                raise ContactJournalError('source contact changed after recovery inspection')

    def _assert_empty_ancestry(self):
        # Keep hot-path work proportional to ancestry, not predecessor contacts.
        for path, expected in self._ancestry_pins.items():
            current = digest(path.read_bytes()) if path.exists() else None
            if current != expected:
                raise ContactJournalError('empty ancestor changed after recovery inspection')
        if self._empty_ancestor is not None:
            self._empty_ancestor._assert_contact_inventory()

    def match(self, payload):
        """Bind a prepared request to exact original bytes, without consuming it."""
        import copy
        logical = logical_contact(payload)
        key = (logical['sample_id'], logical['agent_contact_id'])
        variants = self._by_contact.get(key)
        if variants is None:
            self._assert_empty_ancestry()
            return None
        rid = variants.get(canonical(comparable_request(payload)))
        if rid is None:
            # A response persisted before the one allowed answer-only re-ask
            # may be resumed by issuing that exact not-yet-sent fallback.
            from .reasoning_knobs import answer_only_decoding, fallback_reserve
            for original_id in variants.values():
                record = self._records[original_id]
                self._unchanged(original_id)
                if record['state'] != 'observed':
                    continue
                original = record['intent']['payload']
                decoding = original.get('decoding_params', {})
                if fallback_reserve(decoding) is None:
                    continue
                response = json.loads(record['response_bytes'])
                if response.get('text') != '' or response.get('tool_calls'):
                    continue
                finish = response.get('wire_finish_reason', response.get('finish_reason'))
                if finish not in ('length', None):
                    continue
                expected = copy.deepcopy(original)
                expected['decoding_params'] = answer_only_decoding(decoding)
                if comparable_request(payload) == comparable_request(expected):
                    return None
            raise ContactJournalError('known contact request changed; refusing replay or resend')
        self._unchanged(rid)
        record = self._records[rid]
        if record['state'] != 'observed':
            raise ContactJournalError('contact outcome unknown; reconcile before sending')
        return copy.deepcopy(record['intent']['payload'])

    def consume(self, payload):
        """Consume a verified response once; never fall through on a mismatch."""
        rid = payload['request_id']
        record = self._records.get(rid)
        if record is None:
            self._assert_empty_ancestry()
            return None
        self._unchanged(rid)
        if record['state'] != 'observed' or canonical(payload) != canonical(record['intent']['payload']):
            raise ContactJournalError('replay payload differs from original observed request')
        with self._lock:
            if rid in self._used:
                raise ContactJournalError('predecessor contact already replayed')
            self._used.add(rid)
        return self.source.inspect(rid)
