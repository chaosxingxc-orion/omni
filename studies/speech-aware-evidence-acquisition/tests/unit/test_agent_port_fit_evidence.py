"""Full native FIT joins real boundaries and accepts only receipted fallback pairs."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/analysis'))
import agent_port_fit_evidence as audit
from speech_aware_evidence_acquisition.core.preparation import _artifact_bytes


def payload(request='r1'):
    return {'request_id': request, 'sample_id': 'sample', 'media_relpath': 'speech.wav',
            'raw_speech_query': {'media_relpath': 'speech.wav', 'sha256': 'a' * 64},
            'history': [{'role': 'control', 'contact_id': 'contact'}],
            'decoding_params': {'max_tokens': 4096, 'enable_thinking': True}}


def trace(path, *, fallback=False, fault=None):
    first = payload()
    second = payload('r2')
    second['decoding_params'] = {'max_tokens': 1024, 'enable_thinking': False}
    rows = []
    for p in ([first, second] if fallback else [first]):
        key = p['request_id']
        rows += [{'channel': 'MODEL_REQUEST', 'record_id': key + '/request',
                  'payload': {'payload': p, 'payload_sha256': audit.digest(p)}},
                 {'channel': 'MODEL_WIRE_RESPONSE', 'record_id': key + '/wire-response',
                  'payload': {'request_id': key, 'sample_id': p['sample_id']}}]
    if fallback:
        rows.append({'channel': 'TOOL', 'record_id': 'contact/fallback', 'payload': {
            'schema': 'saea-thinking-answer-fallback-v1', 'agent_contact_id': 'contact',
            'attempt': 1, 'first_request_id': 'r1', 'reask_request_id': 'r2',
            'reask_enable_thinking': False, 'reask_max_tokens': 1024, 'first_finish_reason': 'length'}})
    if fault == 'missing-response': rows.pop(1)
    elif fault == 'duplicate-request': rows.append(copy.deepcopy(rows[0]))
    elif fault == 'wrong-sample': rows[1]['payload']['sample_id'] = 'other'
    elif fault == 'missing-fallback': rows.pop()
    elif fault == 'wrong-fallback': rows[-1]['payload']['reask_request_id'] = 'other'
    elif fault == 'hash': rows[0]['payload']['payload_sha256'] = 'other'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.mark.parametrize('fallback', [False, True])
def test_trace_joins_logical_contacts_and_fallback(tmp_path, fallback):
    path = tmp_path / 'trace.jsonl'
    trace(path, fallback=fallback)
    primary, count, retries = audit.trace_contacts(path, ['sample'])
    assert list(primary) == [('sample', 'contact')]
    assert count == 1 + int(fallback) and retries == int(fallback)
    assert primary[('sample', 'contact')]['boundary'] == audit.boundary_identity((payload(),))


@pytest.mark.parametrize('fault', ['missing-response', 'duplicate-request', 'wrong-sample',
                                  'missing-fallback', 'wrong-fallback', 'hash'])
def test_bad_contact_custody_refused(tmp_path, fault):
    path = tmp_path / 'trace.jsonl'
    trace(path, fallback=True, fault=fault)
    with pytest.raises(ValueError): audit.trace_contacts(path, ['sample'])


def fit_fixture(tmp_path):
    identity = {'model_fragment': 'model', 'model_fragment_sha256': 'model-hash',
                'runtime_receipt_sha256': 'runtime-hash', 'binary_sha256': 'binary', 'model_files': []}
    spec = {'run_id': 'generation', 'fit_run_id': 'fit', 'fit_attempt_id': 'attempt', 'fit_cap': 17, 'fit_session': 'i0'}
    contact = {'sample_id': 'sample', 'agent_contact_id': 'contact'}
    digest = audit.digest(contact)
    boundary = audit.boundary_identity((payload(),))
    inputs = {'native_contact': {**contact, 'contact_identity_sha256': digest, 'raw_speech_query': payload()['raw_speech_query']},
              'fit_plan': {'run_id': 'fit', 'attempt_id': 'attempt', 'planned_model_calls': 17},
              'sources': {'required_identity': identity, 'boundary_sha256': boundary}}
    output = {**contact, 'contact_identity_sha256': digest, 'session_id': 'i0',
              'session_identity_sha256': audit.digest(identity), 'generation_contacts': 0,
              'every_request_fits': True, 'contacts': 1, 'cumulative_contacts': 1, 'sample_count': 1,
              'contact_counts': {'POST /tokenize': 1}, 'boundary_sha256': boundary,
              'rows': [{'sample_id': 'sample', 'fits': True, 'margin_tokens': 100, 'measured_prompt_tokens': 20}]}
    journal = {**contact, 'contact_identity_sha256': digest, 'method': 'POST', 'route': '/tokenize',
               'run_id': 'fit', 'attempt_id': 'attempt', 'generation_run_id': 'generation',
               'session_id': 'i0', 'contact': 1}
    return tmp_path / (digest + '.request-fit.json'), spec, identity, inputs, output, journal


@pytest.mark.parametrize('fault', [None, 'runtime', 'session', 'unfit', 'boundary', 'endpoint', 'count'])
def test_native_fit_artifact_and_journal(tmp_path, fault):
    path, spec, identity, inputs, output, journal = fit_fixture(tmp_path)
    identity = copy.deepcopy(identity)
    if fault == 'runtime': identity['runtime_receipt_sha256'] = 'other'
    elif fault == 'session': output['session_id'] = 'parent'
    elif fault == 'unfit': output['rows'][0]['fits'] = False
    elif fault == 'boundary': output['boundary_sha256'] = 'other'
    elif fault == 'endpoint': journal['route'] = '/completion'
    elif fault == 'count': journal['contact'] = 2
    path.write_bytes(_artifact_bytes(kind='request-fit', inputs=inputs, output=output, contact_class='tokenizer'))
    path.with_name(path.name + '.contacts.jsonl').write_text(json.dumps(journal) + '\n')
    if fault:
        with pytest.raises(ValueError): audit.check_fit(path, spec, ['sample'], identity)
    else:
        pair, value, count, pins = audit.check_fit(path, spec, ['sample'], identity)
        assert pair == ('sample', 'contact') and count == 1 and len(pins) == 2
