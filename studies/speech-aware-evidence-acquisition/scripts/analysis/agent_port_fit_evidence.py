"""Audit full native FIT coverage against sealed requests, including receipted reasks.

Stream traces and retain only identities, hashes and counters. Never emit response
text. Logical contacts have one tokenizer audit; a registered answer-only reask
may add one generation request under the same logical identity.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
for directory in (REPO / 'src', REPO / 'scripts/analysis'):
    sys.path.insert(0, str(directory))
from speech_aware_evidence_acquisition.core.preparation import load_artifact
from speech_aware_evidence_acquisition.core.preparation_fit import boundary_identity, _native_raw_contact_identity
from speech_aware_evidence_acquisition.core.preparation import digest
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
import port_serving_evidence as serving
import direct_port_score_evidence as scored
import direct_port_read_manifest as registered


def trace_contacts(path, expected):
    requests, responses, fallback = {}, {}, {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            channel, value = row['channel'], row['payload']
            key = row['record_id'].split('/', 1)[0]
            if channel == 'MODEL_REQUEST':
                payload = value['payload']
                sample, contact, raw = _native_raw_contact_identity(payload)
                if sample not in expected or key in requests or payload['request_id'] != key:
                    raise ValueError('foreign or duplicate native request')
                if value['payload_sha256'] != digest(payload):
                    raise ValueError('native request hash differs')
                requests[key] = {'pair': (sample, contact), 'boundary': boundary_identity((payload,)),
                                 'raw': raw, 'decoding': payload.get('decoding_params', {})}
            elif channel == 'MODEL_WIRE_RESPONSE':
                if key in responses or value['request_id'] != key:
                    raise ValueError('duplicate or misbound wire response')
                responses[key] = value['sample_id']
            elif channel == 'TOOL' and value.get('schema') == 'saea-thinking-answer-fallback-v1':
                contact = value['agent_contact_id']
                if contact in fallback or value['attempt'] != 1 or value['reask_enable_thinking'] is not False:
                    raise ValueError('invalid duplicate fallback receipt')
                fallback[contact] = value
    if not requests or set(requests) != set(responses):
        raise ValueError('every actual request must have exactly one wire response')
    pairs = defaultdict(list)
    for request, value in requests.items():
        if responses[request] != value['pair'][0]:
            raise ValueError('response belongs to another sample')
        pairs[value['pair']].append(request)
    if {pair[0] for pair in pairs} != set(expected):
        raise ValueError('generation does not cover the complete roster')
    primary, used_fallback = {}, set()
    for pair, ids in pairs.items():
        if len(ids) == 1:
            if pair[1] in fallback:
                raise ValueError('fallback receipt lacks its second request')
            primary[pair] = requests[ids[0]]
        elif len(ids) == 2:
            receipt = fallback.get(pair[1])
            if not receipt or set(ids) != {receipt['first_request_id'], receipt['reask_request_id']}:
                raise ValueError('repeated contact lacks an exact fallback receipt')
            first, second = requests[receipt['first_request_id']], requests[receipt['reask_request_id']]
            if (first['raw'] != second['raw'] or second['decoding'].get('enable_thinking') is not False
                    or second['decoding'].get('max_tokens') != receipt['reask_max_tokens']
                    or receipt['first_finish_reason'] not in ('length', None)):
                raise ValueError('fallback request differs from its receipt')
            primary[pair] = first
            used_fallback.add(pair[1])
        else:
            raise ValueError('more than one fallback per logical contact')
    if used_fallback != set(fallback):
        raise ValueError('orphan fallback receipt')
    return primary, len(requests), len(fallback)


def check_fit(path, spec, expected, member_identity):
    document = serving.read(path)
    inputs = document['inputs']
    output = load_artifact(path, expected_sha256=serving.sha(path), kind='request-fit', inputs=inputs)
    native, plan = inputs['native_contact'], inputs['fit_plan']
    pair = (native['sample_id'], native['agent_contact_id'])
    identity = digest({'sample_id': pair[0], 'agent_contact_id': pair[1]})
    if (document['contact_class'] != 'tokenizer' or pair[0] not in expected
            or identity != native['contact_identity_sha256'] or path.name != identity + '.request-fit.json'
            or pair != (output['sample_id'], output['agent_contact_id'])
            or output['contact_identity_sha256'] != identity):
        raise ValueError('native FIT logical contact identity differs')
    if ((plan['run_id'], plan['attempt_id']) != (spec['fit_run_id'], spec['fit_attempt_id'])
            or plan['planned_model_calls'] != spec['fit_cap']
            or output['session_id'] != spec['fit_session']):
        raise ValueError('native FIT spend or serving identity differs')
    required = inputs['sources']['required_identity']
    if output['session_identity_sha256'] != digest(required):
        raise ValueError('native FIT required identity hash differs')
    for key in ('model_fragment', 'model_fragment_sha256', 'runtime_receipt_sha256', 'binary_sha256', 'model_files'):
        if required[key] != member_identity[key]:
            raise ValueError('native FIT model/runtime differs from executed deployment')
    if (output['generation_contacts'] != 0 or output['every_request_fits'] is not True
            or output['contacts'] != 1 or output['contact_counts'] != {'POST /tokenize': 1}
            or len(output['rows']) != 1 or output['sample_count'] != 1):
        raise ValueError('native raw FIT requires one successful tokenizer contact')
    row, = output['rows']
    if row['sample_id'] != pair[0] or row['fits'] is not True or row['margin_tokens'] < 0 or row['measured_prompt_tokens'] <= 0:
        raise ValueError('native FIT has an unfit or foreign row')
    journal_path = path.with_name(path.name + '.contacts.jsonl')
    journal = [json.loads(line) for line in journal_path.read_bytes().splitlines()]
    if len(journal) != 1:
        raise ValueError('native FIT journal must have exactly one contact')
    contact, = journal
    fields = {'method': 'POST', 'route': '/tokenize', 'run_id': spec['fit_run_id'],
              'attempt_id': spec['fit_attempt_id'], 'generation_run_id': spec['run_id'],
              'session_id': spec['fit_session'], 'sample_id': pair[0], 'agent_contact_id': pair[1],
              'contact_identity_sha256': identity}
    if any(contact.get(k) != v for k, v in fields.items()) or contact['contact'] != output['cumulative_contacts']:
        raise ValueError('native FIT journal custody differs')
    if output['boundary_sha256'] != inputs['sources']['boundary_sha256']:
        raise ValueError('native FIT request boundary differs')
    return pair, {'boundary': output['boundary_sha256'], 'raw': native['raw_speech_query']}, contact['contact'], {
        str(p): serving.sha(p) for p in (path, journal_path)}


def build(repo, data, experiment, read_id):
    registered.require_read(repo, read_id)
    if Path(experiment).name != experiment:
        raise ValueError('invalid experiment')
    experiment_path = repo / 'configs/experiment' / (experiment + '.json')
    doc = serving.read(experiment_path)
    cell, = doc['cells']
    base_path = repo / 'configs/baseline' / (cell['baseline_fragment'] + '.json')
    base = serving.read(base_path)
    if base.get('arm') != 'raw-speech-agent-native-tools' or not base.get('native_episode'):
        raise ValueError('requires the raw native Agent')
    fit = base['native_contact_fit']
    expected = doc['populations'][0]['ids']
    if not expected or len(expected) != len(set(expected)) or len(expected) != doc['population']['count']:
        raise ValueError('complete unique roster required')
    directory = data / 'runs' / cell['run_id'] / cell['attempt_id']
    manifest_path = directory / (cell['run_id'] + '.run-manifest.json')
    bundle = scored.verify_bundle(manifest_path)
    if set(bundle.split_ids) != set(expected) or bundle.manifest['status'] != 'COMPLETED':
        raise ValueError('sealed Agent roster differs')
    sessions = serving.sessions(repo, data, coordinator=doc['service_session'], model_fragment=cell['model_fragment'],
                                run_id=cell['run_id'], attempt_id=cell['attempt_id'])
    trace = Path(bundle.manifest['raw_trace']['path'])
    if not trace.is_absolute(): trace = directory / trace
    trace = scored.within(trace, directory)
    if serving.sha(trace) != bundle.manifest['raw_trace']['sha256']:
        raise ValueError('sealed trace hash differs')
    primary, generation_count, fallback_count = trace_contacts(trace, expected)
    if generation_count > doc['family_budget']['generation_calls_cap']:
        raise ValueError('generation exceeds registered cap')
    fit_cap = fit['contact_cap']
    if fit_cap % len(expected) or max(Counter(pair[0] for pair in primary).values()) > fit_cap // len(expected):
        raise ValueError('per-row logical contact cap exceeded')
    spec = {'run_id': cell['run_id'], 'fit_run_id': fit['run_id'], 'fit_attempt_id': fit['attempt_id'],
            'fit_cap': fit_cap, 'fit_session': sessions['primary_fit_session']}
    fit_dir = data / 'prepared' / fit['run_id'] / fit['attempt_id']
    paths = sorted(fit_dir.glob('*.request-fit.json'))
    if len(paths) != len(list(fit_dir.glob('*.contacts.jsonl'))) or not 0 < len(paths) <= fit_cap:
        raise ValueError('incomplete FIT envelopes or journals')
    observed, counts = {}, []
    pins = {**sessions['inputs'], **{str(p): serving.sha(p) for p in (experiment_path, base_path, manifest_path, trace)}}
    for path in paths:
        pair, value, contact, inputs = check_fit(path, spec, expected, sessions['members'][spec['fit_session']])
        if pair in observed:
            raise ValueError('duplicate logical FIT contact')
        observed[pair] = value
        counts.append(contact)
        pins.update(inputs)
    if set(observed) != set(primary) or sorted(counts) != list(range(1, len(paths) + 1)):
        raise ValueError('FIT and generation contact coverage differ')
    for pair, value in observed.items():
        if any(value[key] != primary[pair][key] for key in ('boundary', 'raw')):
            raise ValueError('generated request differs from the actual FIT boundary')
    return {'schema': 'saea-agent-complete-native-fit-evidence-v1', 'status': 'PASS',
            'read_run_id': read_id, 'generation_run_id': cell['run_id'], 'rows': len(expected),
            'generation_contacts': generation_count, 'logical_contacts': len(primary), 'fit_contacts': len(paths),
            'fallback_contacts': fallback_count, 'primary_fit_session': spec['fit_session'],
            'new_model_calls': 0, 'new_tokenizer_calls': 0, 'inputs': pins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    for key in ('data-root', 'out'): parser.add_argument('--' + key, type=Path, required=True)
    for key in ('experiment', 'read-id'): parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    result = build(args.repo_root, args.data_root, args.experiment, args.read_id)
    publish_new(args.out, (json.dumps(result, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps({k: v for k, v in result.items() if k != 'inputs'}))


if __name__ == '__main__': main()
