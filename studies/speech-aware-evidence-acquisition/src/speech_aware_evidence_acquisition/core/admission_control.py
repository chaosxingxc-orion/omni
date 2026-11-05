"""Durable, attempt-bound stop admission; never cancels or retries contacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .atomic_file import publish_new

ENVIRONMENT_KEY = 'SAEA_ADMISSION_CONTROL_CONFIG'


class AdmissionDrained(RuntimeError):
    """Requested admission stop settled; incomplete flight must not be sealed."""


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def read_regular(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('admission control symlinks forbidden')
    return json.loads(path.read_bytes())


class AdmissionControl:
    def __init__(self, path, *, run_id, attempt_id):
        self.path = Path(path)
        self.config = read_regular(self.path)
        expected = {'schema', 'run_id', 'attempt_id', 'request_path', 'receipt_path'}
        if (set(self.config) not in (expected, expected | {'stop_policy'}) or self.config['schema'] != 'saea-admission-control-v1'
                or self.config['run_id'] != run_id or self.config['attempt_id'] != attempt_id):
            raise ValueError('admission control must bind current run and attempt')
        self.stop_policy = self.config.get('stop_policy', 'drain_admitted')
        if self.stop_policy not in ('drain_admitted', 'cancel_unstarted'):
            raise ValueError('invalid admission stop policy')
        self.stop_summary = None
        self.cancelled_trace_proof = None
        self.digest = hashlib.sha256(canonical(self.config)).hexdigest()
        self.request_path = Path(self.config['request_path'])
        self.receipt_path = Path(self.config['receipt_path'])
        if (not self.path.is_absolute() or any(not p.is_absolute() or p.parent != self.path.parent
                for p in (self.request_path, self.receipt_path))
                or len({self.path, self.request_path, self.receipt_path}) != 3):
            raise ValueError('distinct admission files in one absolute directory required')
        self.observed_path = self.receipt_path.with_name(self.receipt_path.name + '.observed')
        self.cancelled_trace_path = self.receipt_path.with_name(self.receipt_path.name + '.cancelled-traces.json')
        if len({self.path, self.request_path, self.receipt_path, self.observed_path, self.cancelled_trace_path}) != 5:
            raise ValueError('admission evidence paths must be distinct')
        self.request = None
        self.request_digest = None

    def should_stop(self):
        if self.request is not None:
            return True
        if not self.request_path.exists() and not self.request_path.is_symlink():
            if self.receipt_path.exists():
                raise ValueError('drained receipt without stop request; use a new attempt')
            return False
        request = read_regular(self.request_path)
        if (set(request) != {'schema', 'run_id', 'attempt_id', 'config_sha256', 'request_id'}
                or request['schema'] != 'saea-stop-admission-v1'
                or not isinstance(request['request_id'], str) or not request['request_id'].strip()):
            raise ValueError('invalid stop request schema')
        if (request['run_id'] != self.config['run_id']
                or request['attempt_id'] != self.config['attempt_id']
                or request['config_sha256'] != self.digest):
            # A stale/foreign request has no authority over this attempt.
            return False
        self.request = request
        self.request_digest = hashlib.sha256(canonical(request)).hexdigest()
        return True

    def observe_stop(self, info):
        if not self.should_stop():
            raise ValueError('observation requires stop')
        event = {'schema': 'saea-admission-stop-observed-v1',
            'run_id': self.config['run_id'], 'attempt_id': self.config['attempt_id'],
            'config_sha256': self.digest, 'stop_request_sha256': self.request_digest,
            'request_id': self.request['request_id'], **info}
        publish_new(self.observed_path, canonical(event))

    def publish_cancelled_traces(self, traces):
        """Persist partial trace records, including tool cost, before a drain receipt."""
        if self.stop_summary is None or set(traces) != set(self.stop_summary['cancelled_sample_ids']):
            raise ValueError('cancelled trace identities differ from stop summary')
        payload = {'schema': 'saea-admission-cancelled-traces-v1',
            'run_id': self.config['run_id'], 'attempt_id': self.config['attempt_id'],
            'config_sha256': self.digest, 'stop_request_sha256': self.request_digest,
            'samples': traces}
        data = canonical(payload)
        publish_new(self.cancelled_trace_path, data)
        self.cancelled_trace_proof = {'path': str(self.cancelled_trace_path),
            'sha256': hashlib.sha256(data).hexdigest(), 'sample_count': len(traces),
            'trace_record_count': sum(len(records) for records in traces.values())}

    def record_stop_summary(self, summary):
        self.stop_summary = summary

    def finish(self, checkpoint, admitted_sample_ids):
        """Verify durable rows after workers settle; not a lease-release receipt."""
        from .row_checkpoint import read_checkpoint
        if not self.should_stop():
            raise ValueError('drain receipt requires an observed stop request')
        ids = list(admitted_sample_ids)
        if len(ids) != len(set(ids)):
            raise ValueError('unique admitted samples required')
        checkpoint = Path(checkpoint)
        proof = None
        if checkpoint.exists():
            doc = read_checkpoint(checkpoint)
            if (doc.header['run_id'], doc.header['attempt_id']) != (self.config['run_id'], self.config['attempt_id']):
                raise ValueError('checkpoint identity differs')
            by_id = doc.by_sample()
            if doc.torn_tail_bytes or any(s not in by_id or not by_id[s].keepable for s in ids):
                raise ValueError('admitted samples lack complete durable checkpoints')
            proof = doc.manifest_block()
        elif ids:
            raise ValueError('admitted samples require durable checkpoints')
        receipt = {'schema': 'saea-admission-drained-v1', 'run_id': self.config['run_id'],
            'attempt_id': self.config['attempt_id'], 'config_sha256': self.digest,
            'stop_request_sha256': self.request_digest, 'request_id': self.request['request_id'],
            'admitted_sample_ids': ids, 'checkpoint': proof, 'inflight_samples': 0,
            'generation_complete': False, 'generation_lease_released': False,
            'resume_requires_new_attempt_and_existing_recovery_validation': True}
        if self.stop_policy == 'cancel_unstarted':
            summary = self.stop_summary
            keys = {'admitted_sample_ids', 'cancelled_sample_ids', 'started_sample_ids', 'completed_sample_ids'}
            if not isinstance(summary, dict) or set(summary) != keys:
                raise ValueError('stop summary required')
            if any(not isinstance(summary[k], list) or any(not isinstance(i, str) for i in summary[k])
                   or len(summary[k]) != len(set(summary[k])) for k in keys):
                raise ValueError('invalid stop summary identifiers')
            admitted = set(summary['admitted_sample_ids'])
            cancelled = set(summary['cancelled_sample_ids'])
            completed = set(summary['completed_sample_ids'])
            started = set(summary['started_sample_ids'])
            if (cancelled & completed or cancelled | completed != admitted
                    or not started <= completed or completed != set(ids)):
                raise ValueError('stop summary accounting differs')
            if self.cancelled_trace_proof is None:
                raise ValueError('cancelled trace custody required')
            receipt['stop_summary'] = summary
            receipt['cancelled_trace_custody'] = self.cancelled_trace_proof
        if self.receipt_path.exists():
            old = read_regular(self.receipt_path)
            previous_ids = old.get('admitted_sample_ids')
            if not isinstance(previous_ids, list) or any(not isinstance(s, str) for s in previous_ids):
                raise ValueError('existing drain receipt differs')
            expected_ids = ids or previous_ids
            if len(expected_ids) != len(set(expected_ids)) or any(
                    not checkpoint.exists() or s not in by_id or not by_id[s].keepable for s in expected_ids):
                raise ValueError('existing drain receipt checkpoint differs')
            if old != {**receipt, 'admitted_sample_ids': expected_ids}:
                raise ValueError('existing drain receipt differs')
            return old
        publish_new(self.receipt_path, canonical(receipt))
        return receipt


def from_environment(*, run_id, attempt_id):
    path = os.environ.get(ENVIRONMENT_KEY)
    if not path:
        return None
    control = AdmissionControl(path, run_id=run_id, attempt_id=attempt_id)
    if control.stop_policy == 'cancel_unstarted' and os.path.lexists(control.observed_path):
        raise ValueError('existing stop observation; recovery requires a new attempt')
    return control


def request_stop(config_path, *, request_id):
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError('nonempty stop request id required')
    config = read_regular(config_path)
    control = AdmissionControl(config_path, run_id=config['run_id'], attempt_id=config['attempt_id'])
    request = {'schema': 'saea-stop-admission-v1', 'run_id': config['run_id'],
        'attempt_id': config['attempt_id'], 'config_sha256': control.digest, 'request_id': request_id}
    if control.request_path.exists():
        if read_regular(control.request_path) != request:
            raise ValueError('different stop request already exists')
    else:
        try:
            publish_new(control.request_path, canonical(request))
        except FileExistsError:
            if read_regular(control.request_path) != request:
                raise ValueError('different concurrent stop request already exists')
    control.should_stop()
    return request
