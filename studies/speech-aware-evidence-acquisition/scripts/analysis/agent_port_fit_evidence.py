"""Audit full native FIT coverage against sealed requests, including receipted reasks.

Stream traces and retain only identities, hashes and counters. Never emit response
text. Logical contacts have one tokenizer audit; a registered answer-only reask
may add one generation request under the same logical identity.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
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
import historical_fit_serving as historical


def require_read(repo, read_id):
    # Registration loads research-family artifacts. Defer it until an actual
    # report build; checkpoint/FIT readers do not need those unrelated files.
    import direct_port_read_manifest as registered
    registered.require_read(repo, read_id)



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
    if (not requests and expected) or set(requests) != set(responses):
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



def audit_source(data, fit, run_id, expected, full_roster, trace, sessions, *, predecessor=False):
    """Audit one attempt independently; retain even non-kept predecessor FIT costs."""
    primary, generation_count, fallback_count = trace_contacts(trace, expected) if trace is not None else ({}, 0, 0)
    if trace is None and (expected or not predecessor):
        raise ValueError('missing generation trace')
    fit_cap = fit['contact_cap']
    if fit_cap % len(full_roster) or max(Counter(pair[0] for pair in primary).values(), default=0) > fit_cap // len(full_roster):
        raise ValueError('per-row logical contact cap exceeded')
    spec = {'run_id': run_id, 'fit_run_id': fit['run_id'], 'fit_attempt_id': fit['attempt_id'],
            'fit_cap': fit_cap, 'fit_session': sessions['primary_fit_session']}
    fit_dir = data / 'prepared' / fit['run_id'] / fit['attempt_id']
    paths = sorted(fit_dir.glob('*.request-fit.json'))
    if len(paths) != len(list(fit_dir.glob('*.contacts.jsonl'))) or not 0 <= len(paths) <= fit_cap or (expected and not paths):
        raise ValueError('incomplete FIT envelopes or journals')
    observed, counts, selected = {}, [], {}
    pins = dict(sessions['inputs'])
    if trace is not None:
        pins[str(trace)] = serving.sha(trace)
    for path in paths:
        pair, value, contact, inputs = check_fit(path, spec, full_roster if predecessor else expected,
                                                sessions['members'][spec['fit_session']])
        if pair in observed:
            raise ValueError('duplicate logical FIT contact')
        observed[pair] = value
        counts.append(contact)
        pins.update(inputs)
        if pair[0] in expected:
            selected[pair] = value
    if set(selected) != set(primary) or sorted(counts) != list(range(1, len(paths) + 1)):
        raise ValueError('FIT and generation contact coverage differ')
    for pair, value in selected.items():
        if any(value[key] != primary[pair][key] for key in ('boundary', 'raw')):
            raise ValueError('generated request differs from the actual FIT boundary')
    return {'generation_contacts': generation_count, 'logical_contacts': len(primary),
            'fit_contacts': len(paths), 'kept_fit_contacts': len(selected),
            'non_kept_fit_contacts': len(paths) - len(selected), 'fallback_contacts': fallback_count,
            'primary_fit_session': spec['fit_session'], 'inputs': pins}


def predecessor_sources(repo, data, cell, doc, bundle):
    """Resolve all registered ancestors, including attempts with no retained rows.

    Request IDs are local to each attempt. Empty ancestors still retain their
    paid generation and FIT costs; a missing retained manifest is accepted only
    when the pinned checkpoint proves it has no keepable rows.
    """
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import (
        CheckpointCompatibility, MultiCheckpointCompatibility, validate_bridge_declarations)
    from speech_aware_evidence_acquisition.core.flight.resume import resume_declarations_from_document
    blocks = bundle.manifest.get('predecessor_attempts')
    single = bundle.manifest.get('predecessor_attempt')
    if blocks and single:
        raise ValueError('ambiguous predecessor custody')
    blocks = blocks or ([single] if single else [])
    declarations = resume_declarations_from_document(cell['resume_from']) if cell.get('resume_from') is not None else ()
    if not declarations:
        if blocks:
            raise ValueError('unregistered predecessor custody')
        return []
    bridge = CheckpointCompatibility.from_document(cell.get('checkpoint_compatibility'), attempt_id=cell['attempt_id'])
    validate_bridge_declarations(bridge, declarations)
    if isinstance(bridge, MultiCheckpointCompatibility):
        bridge.validate_ancestry(repo, declarations)
    by_id = {block['attempt_id']: block for block in blocks}
    if len(by_id) != len(blocks) or not set(by_id) <= {d.attempt_id for d in declarations}:
        raise ValueError('duplicate or undeclared predecessor custody')
    current = compose(repo, **bundle.manifest['config']['fragments'])
    if current.config_hash != bundle.manifest['config']['config_hash']:
        raise ValueError('FIT configuration differs from sealed execution')
    if current.values['cells'][0] != cell:
        raise ValueError('FIT cell differs from sealed configuration')
    receipt_path = bundle.manifest_path.parent / 'flight-receipts.json'
    receipts = scored.read_flight_receipts(receipt_path)
    proof = receipts['stages']['preflight'].get('checkpoint_compatibility', {})
    if isinstance(bridge, MultiCheckpointCompatibility):
        if (proof.get('schema') != 'saea-checkpoint-compatibility-proofs-v2'
                or set(proof.get('predecessors', {})) != {d.attempt_id for d in declarations}):
            raise ValueError('missing executed ancestor proof')
        proofs = proof['predecessors']
    else:
        proofs = {declarations[0].attempt_id: proof}
    sources, seen, fit_ids = [], set(), set()
    for declaration in declarations:
        item = bridge.bridge_for(declaration.attempt_id) if isinstance(bridge, MultiCheckpointCompatibility) else bridge
        source = _predecessor_source(repo, data, cell, doc, bundle, declaration,
                                     item, proofs[declaration.attempt_id], by_id.get(declaration.attempt_id))
        if seen.intersection(source['expected']):
            raise ValueError('overlapping retained predecessor rows')
        seen.update(source['expected'])
        fit_id = (source['fit']['run_id'], source['fit']['attempt_id'])
        if fit_id in fit_ids:
            raise ValueError('duplicate ancestor FIT cost source')
        fit_ids.add(fit_id)
        source['pins'][str(receipt_path)] = serving.sha(receipt_path)
        sources.append(source)
    return sources


def _predecessor_source(repo, data, cell, doc, bundle, declaration, bridge, proof, block):
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import git
    from speech_aware_evidence_acquisition.core.row_checkpoint import checkpoint_path, read_checkpoint, kept_trace_text
    if (proof.get('predecessor_commit') != bridge.predecessor_commit
            or proof.get('reviewed_source_changes') != bridge.reviewed_source_changes
            or proof.get('change') != bridge.change):
        raise ValueError('FIT predecessor lacks executed compatibility proof')
    def pinned(kind):
        name = bridge.predecessor_fragments[kind]
        return json.loads(git(repo, 'show', bridge.predecessor_commit + ':configs/' + kind + '/' + name + '.json'))
    old_doc, old_base = pinned('experiment'), pinned('baseline')
    old_cell, = old_doc['cells']
    if ((old_cell['run_id'], old_cell['attempt_id']) != (cell['run_id'], declaration.attempt_id)
            or old_doc['populations'] != doc['populations']
            or old_cell['model_fragment'] != cell['model_fragment']):
        raise ValueError('pinned predecessor roster, model or attempt differs')
    path = checkpoint_path(data / 'runs' / cell['run_id'] / declaration.attempt_id, cell['run_id'])
    checkpoint = read_checkpoint(path, expected_sha256=declaration.checkpoint_sha256)
    header = checkpoint.header
    if (header['run_id'] != cell['run_id'] or header['attempt_id'] != declaration.attempt_id
            or header['protocol_hash'] != proof.get('protocol_hash')
            or header['implementation'] != proof.get('implementation')):
        raise ValueError('checkpoint differs from executed compatibility proof')
    # Mirror the shared resume reader's additional single-output/trace custody
    # checks, so malformed complete flags never become retained FIT rows.
    kept_rows = []
    for row in checkpoint.rows:
        if row.keep_defect() is not None or len(row.output_rows) != 1:
            continue
        record = row.output_rows[0]
        if record.get('sample_id') != row.sample_id:
            continue
        wanted = {str(record.get('request_id')) + '/request', str(record.get('request_id')) + '/response'}
        if not wanted <= {json.loads(line)['record_id'] for line in row.trace_lines}:
            continue
        kept_rows.append(row)
    kept = [row.sample_id for row in kept_rows]
    if len(kept) != len(set(kept)) or not set(kept) <= set(doc['populations'][0]['ids']):
        raise ValueError('invalid kept predecessor roster')
    pins = {str(path): serving.sha(path)}
    trace = None
    if block is None:
        if kept:
            raise ValueError('missing retained predecessor manifest')
    else:
        if (block.get('mode') != 'checkpoint' or block.get('run_id') != cell['run_id']
                or block.get('checkpoint_sha256') != declaration.checkpoint_sha256
                or sorted(block['kept_sample_ids']) != sorted(kept)):
            raise ValueError('FIT predecessor differs from registered checkpoint')
        trace = scored.within(Path(block['trace_path']), bundle.manifest_path.parent)
        if (serving.sha(trace) != block['trace_sha256']
                or trace.read_text() != kept_trace_text(kept_rows)):
            raise ValueError('predecessor trace differs from pinned checkpoint')
        pins[str(trace)] = serving.sha(trace)
    record_path = data / 'attempts' / cell['run_id'] / (declaration.attempt_id + '.json')
    record = serving.read(record_path)
    if (record.get('state') != 'ABORTED' or record.get('run_id') != cell['run_id']
            or record.get('attempt_id') != declaration.attempt_id):
        raise ValueError('predecessor attempt must remain ABORTED')
    actual = record['actual']
    if (type(actual.get('calls')) is not int or actual['calls'] < 0
            or type(actual.get('audio_seconds')) not in (int, float)
            or not math.isfinite(actual['audio_seconds']) or actual['audio_seconds'] < 0):
        raise ValueError('invalid predecessor actual cost')
    pins[str(record_path)] = serving.sha(record_path)
    return {'fit': old_base['native_contact_fit'], 'doc': old_doc, 'cell': old_cell,
            'trace': trace, 'expected': kept, 'attempt_record': record_path, 'pins': pins,
            'predecessor_commit': bridge.predecessor_commit, 'actual': actual}


def predecessor_source(repo, data, cell, doc, bundle):
    """Compatibility entry point for callers explicitly expecting one ancestor."""
    sources = predecessor_sources(repo, data, cell, doc, bundle)
    if len(sources) > 1:
        raise ValueError('multiple predecessors; use predecessor_sources')
    return sources[0] if sources else None


def build(repo, data, experiment, read_id, *, historical_map=None, historical_map_sha256=None):
    require_read(repo, read_id)
    if (historical_map is None) != (historical_map_sha256 is None):
        raise ValueError('historical serving map requires an independent hash pin')
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
    predecessors = predecessor_sources(repo, data, cell, doc, bundle)
    historical_bindings, historical_pins = {}, {}
    if historical_map is not None:
        historical_bindings, historical_pins = historical.bindings(
            historical_map, historical_map_sha256, read_id=read_id,
            run_id=cell['run_id'], attempt_id=cell['attempt_id'], predecessors=predecessors)
    current_expected = set(expected) - {sample for p in predecessors for sample in p['expected']}
    reports = []
    if any((p['fit']['run_id'], p['fit']['attempt_id']) == (fit['run_id'], fit['attempt_id']) for p in predecessors):
        raise ValueError('current FIT identity reuses ancestor cost source')
    for predecessor in predecessors:
        old_doc, old_cell = predecessor['doc'], predecessor['cell']
        old_sessions = historical.resolve(data, predecessor, doc, sessions,
                                          historical_bindings.get(old_cell['attempt_id']))
        report = audit_source(data, predecessor['fit'], cell['run_id'], set(predecessor['expected']),
                              expected, predecessor['trace'], old_sessions, predecessor=True)
        report.update(attempt_id=old_cell['attempt_id'], rows=len(predecessor['expected']),
                      predecessor_commit=predecessor['predecessor_commit'], prior_actual=predecessor['actual'])
        report['serving_custody'] = old_sessions.get('custody', 'successor attachment; same service identity')
        report['inputs'].update(predecessor['pins'])
        if predecessor['actual']['calls'] and not report['fit_contacts']:
            raise ValueError('paid predecessor lacks FIT evidence')
        if predecessor['actual']['calls'] < report['generation_contacts']:
            raise ValueError('predecessor actual cost is below kept contacts')
        reports.append(report)
    report = audit_source(data, fit, cell['run_id'], current_expected, expected, trace, sessions)
    report.update(attempt_id=cell['attempt_id'], rows=len(current_expected))
    reports.append(report)
    totals = {key: sum(report[key] for report in reports) for key in
              ('generation_contacts', 'logical_contacts', 'fit_contacts', 'fallback_contacts',
               'kept_fit_contacts', 'non_kept_fit_contacts')}
    accounted_generation = report['generation_contacts'] + sum(p['actual']['calls'] for p in predecessors)
    if accounted_generation > doc['family_budget']['generation_calls_cap']:
        raise ValueError('generation exceeds registered cap')
    pins = {str(p): serving.sha(p) for p in (experiment_path, base_path, manifest_path)}
    pins.update(historical_pins)
    for report in reports:
        pins.update(report.pop('inputs'))
    return {'schema': 'saea-agent-complete-native-fit-evidence-v1', 'status': 'PASS',
            'read_run_id': read_id, 'generation_run_id': cell['run_id'], 'rows': len(expected),
            **totals, 'accounted_generation_contacts': accounted_generation,
            'primary_fit_session': sessions['primary_fit_session'],
            'attempts': reports, 'new_model_calls': 0, 'new_tokenizer_calls': 0, 'inputs': pins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    for key in ('data-root', 'out'): parser.add_argument('--' + key, type=Path, required=True)
    for key in ('experiment', 'read-id'): parser.add_argument('--' + key, required=True)
    parser.add_argument('--historical-map', type=Path)
    parser.add_argument('--historical-map-sha256', help='Hash from the independently reviewed READ record')
    args = parser.parse_args()
    result = build(args.repo_root, args.data_root, args.experiment, args.read_id,
                   historical_map=args.historical_map, historical_map_sha256=args.historical_map_sha256)
    publish_new(args.out, (json.dumps(result, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps({k: v for k, v in result.items() if k != 'inputs'}))


if __name__ == '__main__': main()
