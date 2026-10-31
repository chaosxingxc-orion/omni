"""FIT must bind the executed members, including endpoint and process custody."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/analysis'))
import port_serving_evidence as audit
from speech_aware_evidence_acquisition.core.flight.receipts import FlightReceipts


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    model = {'backend': 'vllm', 'model_path': 'models/frozen'}
    dump(repo / 'configs/model/model.json', model)
    runtime = {'deployment': {'instances': [
        {'instance_id': m, 'host': '127.0.0.1', 'port': 8200 + i}
        for i, m in enumerate(('i0', 'i1'))]}}
    runtime_path = repo / 'docs/receipts/runtime.json'
    dump(runtime_path, runtime)
    identity = {'model_fragment': 'model',
                'model_fragment_sha256': audit.fragment_hash(audit.read_fragment(repo, 'model', 'model')),
                'runtime_receipt_relpath': 'receipts/runtime.json',
                'runtime_receipt_sha256': audit.sha(runtime_path),
                'binary_sha256': 'binary', 'model_files': {'weights': 'hash'}}
    dump(data / 'serving/parent/session.json', {
        'session_id': 'parent', 'identity': identity, 'deployment_instance_ids': ['i0', 'i1']})
    attached = []
    for i, member in enumerate(('i0', 'i1')):
        endpoint = {'host': '127.0.0.1', 'port': 8200 + i}
        process = {'pid': 100 + i, 'started_at': '2026-10-30T17:26:00Z'}
        dump(data / 'serving' / member / 'session.json', {
            'session_id': member, 'deployment_parent_session_id': 'parent',
            'identity': {**identity, **endpoint}, **process})
        dump(data / 'serving' / member / 'session-receipt.json', {
            'session_id': member, **endpoint, **process})
        attached.append({'session_id': member, 'instance': member, 'pid': 100 + i, **endpoint})
    serve = {'session_source': 'attached-multi-instance-serving-session',
             'deployment_record': {'model_fragment': 'model', 'instance_count': 2},
             'instance_sessions': attached}
    write_flight(data, serve)
    return repo, data, serve


def write_flight(data, serve):
    receipts = FlightReceipts('run', 'attempt')
    for stage, payload in [('preflight', {}), ('preprocess', {}), ('serve', serve)]:
        receipts.record(stage, payload)
    dump(data / 'runs/run/attempt/flight-receipts.json', receipts.as_json())


def check(repo, data):
    return audit.sessions(repo, data, coordinator='parent', model_fragment='model',
                          run_id='run', attempt_id='attempt')


def test_primary_is_executed_first_member(tmp_path):
    repo, data, serve = fixture(tmp_path)
    serve['instance_sessions'].reverse()
    write_flight(data, serve)
    result = check(repo, data)
    assert result['primary_fit_session'] == 'i1'
    assert set(result['members']) == {'i0', 'i1'}
    assert len(result['inputs']) == 8


@pytest.mark.parametrize('case', ['parent', 'pid', 'start', 'endpoint', 'runtime',
                                  'model', 'missing', 'duplicate', 'foreign'])
def test_foreign_or_incomplete_deployment_refused(tmp_path, case):
    repo, data, serve = fixture(tmp_path)
    path = data / 'serving/i0/session.json'
    member = copy.deepcopy(audit.read(path))
    if case == 'parent': member['deployment_parent_session_id'] = 'other'
    elif case == 'pid': member['pid'] = 999
    elif case == 'start': member['started_at'] = '2026-10-31T00:00:00Z'
    elif case == 'endpoint': member['identity']['port'] = 9999
    elif case == 'runtime': member['identity']['runtime_receipt_sha256'] = 'other'
    elif case == 'model': member['identity']['model_files'] = {'weights': 'other'}
    elif case == 'missing': serve['instance_sessions'].pop()
    elif case == 'duplicate': serve['instance_sessions'][1] = serve['instance_sessions'][0]
    elif case == 'foreign': serve['instance_sessions'][0]['session_id'] = 'other'
    dump(path, member)
    write_flight(data, serve)
    with pytest.raises(ValueError):
        check(repo, data)
