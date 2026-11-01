"""Durable request/response custody for interrupted native generation.

An intent without a response is UNKNOWN, never permission to resend. Records
preserve raw transport bytes before parsing. This journal does not authorize
attempt recovery, substitute responses, or change metering.
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

    def __init__(self, data_root, plan, arm):
        plan.validate()
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
