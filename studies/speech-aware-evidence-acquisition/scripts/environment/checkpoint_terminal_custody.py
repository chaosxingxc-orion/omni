"""Read-only checkpoint terminal custody verification."""
from __future__ import annotations

import json
import hashlib
import re
import math
import os
from pathlib import Path

from speech_aware_evidence_acquisition.core.attempts import AttemptStore
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint


def inspect_generation(*, data_root, plan, arm, source_root=None):
    """Verify a single ABORTED attempt and its complete generation journal.

    This is only one part of the terminal audit. FIT, producer quiescence,
    ancestry, checkpoint provenance, and inventory pins are separate duties.
    No record is modified and an unknown response is never retried.
    """
    root = Path(data_root)
    journal = ContactJournal(root, plan, arm, source_root=source_root)
    attempt_path = AttemptStore(root / 'attempts').handle(plan.run_id, plan.attempt_id).path
    attempt = json.loads(attempt_path.read_bytes())
    if (attempt.get('schema') != 'saea-attempt-v1' or attempt.get('state') != 'ABORTED'
            or attempt.get('run_id') != plan.run_id or attempt.get('attempt_id') != plan.attempt_id
            or attempt.get('carrier_lock_key') != plan.carrier_lock_key):
        raise ValueError('attempt is not the registered ABORTED predecessor')
    if os.path.lexists(journal.root / 'retirement.json'):
        raise ValueError('retired predecessor cannot be resumed')
    if journal.root.is_symlink():
        raise ValueError('journal root cannot be a symlink')
    files = list(journal.root.iterdir()) if journal.root.exists() else []
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError('journal contains a non-regular file')
    intents = [p for p in files if p.name.endswith('.intent.json')]
    responses = {p for p in files if p.name.endswith('.response.json')}
    expected = set()
    audio = []
    seen = set()
    not_sent = []
    for path in intents:
        record = json.loads(path.read_bytes())
        rid = record['request_id']
        if rid in seen or path != journal._path(rid, 'intent'):
            raise ValueError('duplicate or misnamed generation intent')
        seen.add(rid)
        observed = journal.inspect(rid)
        if observed['state'] not in ('observed', 'not_sent'):
            raise ValueError('generation outcome unknown; reconciliation required')
        if observed['state'] == 'observed':
            expected.add(journal._path(rid, 'response'))
        else:
            not_sent.append(rid)
        seconds = observed['intent']['payload'].get('audio_seconds')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('generation audio usage is invalid')
        audio.append(seconds)
    delivery = {journal._path(rid, kind) for rid in seen for kind in ('dispatch', 'not_sent', 'reconciliation')
                if journal._path(rid, kind).exists()}
    if any(p not in intents and p not in responses and p not in delivery and not p.name.endswith('.fit.json') for p in files):
        raise ValueError('unexpected generation journal file')
    if responses != expected:
        raise ValueError('orphan generation response')
    actual = attempt.get('actual', {})
    calls, seconds = actual.get('calls'), actual.get('audio_seconds')
    if (type(calls) is not int or calls != len(intents)
            or type(seconds) not in (int, float) or not math.isfinite(seconds)
            or not math.isclose(seconds, math.fsum(audio), rel_tol=1e-12, abs_tol=1e-9)):
        raise ValueError('generation attempt and journal usage differ')
    if calls > plan.planned_model_calls or seconds > plan.planned_speech_audio_seconds:
        raise ValueError('generation actual exceeds authenticated plan budget')
    gpu_seconds = actual.get('gpu_seconds', 0.0)
    if (type(gpu_seconds) not in (int, float) or not math.isfinite(gpu_seconds)
            or gpu_seconds < 0 or gpu_seconds > plan.planned_gpu_hours * 3600):
        raise ValueError('generation GPU actual exceeds authenticated plan budget')
    return {'run_id': plan.run_id, 'attempt_id': plan.attempt_id,
            'observed_requests': sorted(seen - set(not_sent)), 'actual': actual,
            **({'not_sent_requests': sorted(not_sent)} if not_sent else {})}


def inspect_checkpoint(*, path, sha256, plan):
    """Read the native hash chain and bind its header to the predecessor plan."""
    document = read_checkpoint(Path(path), expected_sha256=sha256)
    for field in ('run_id', 'attempt_id', 'protocol_hash', 'split_identity_hash'):
        if document.header.get(field) != getattr(plan, field):
            raise ValueError('checkpoint header differs from predecessor ' + field)
    return document


def validate_ancestry(graph, root):
    if root not in graph:
        raise ValueError('candidate root missing from ancestor declarations')
    visited, visiting = set(), set()
    def visit(node):
        if node in visiting:
            raise ValueError('cyclic checkpoint ancestry')
        if node not in graph:
            raise ValueError('ancestor declaration omitted')
        if node not in visited:
            visiting.add(node)
            for parent in graph[node]:
                visit(parent)
            visiting.remove(node)
            visited.add(node)
    visit(root)
    if visited != set(graph):
        raise ValueError('ancestor is not reachable from candidate root')



def inspect_fit(*, journal, prepared_root, logical_prepared_root, fit_plan,
                session_identity, max_contacts, roster, expected_inputs=None, session_id=None):
    """Join a raw-speech FIT run in both directions, without contacting it.

    The caller must independently authenticate the plan, roster, limits, session,
    directory inventory, and mapping against the producer's pinned source.
    An artifact without wrapper custody is returned as known but unrecoverable;
    an endpoint without an artifact raises because its HTTP result is unknown.
    """
    from collections import Counter
    from dataclasses import asdict
    from speech_aware_evidence_acquisition.core.contact_journal import (
        canonical, comparable_request, digest, logical_contact)
    from speech_aware_evidence_acquisition.core.preparation import load_artifact, digest as object_digest
    from speech_aware_evidence_acquisition.core.preparation_fit import boundary_identity

    root = Path(prepared_root)
    logical_root = Path(logical_prepared_root)
    if not logical_root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError('explicit existing FIT mapping required')
    if type(max_contacts) is not int or max_contacts < 1 or not roster or len(set(roster)) != len(roster):
        raise ValueError('unique roster and positive contact limit required')
    fit_plan.validate()
    files = list(root.iterdir())
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError('invalid FIT directory entry')
    artifacts = {p for p in files if p.name.endswith('.request-fit.json')}
    endpoints = {p for p in files if p.name.endswith('.request-fit.json.contacts.jsonl')}
    unknown_files = {p for p in files if p not in artifacts and p not in endpoints
                     and not p.name.endswith('.lock')}
    if unknown_files:
        raise ValueError('unexpected FIT file or start marker')
    if {Path(str(p) + '.contacts.jsonl') for p in artifacts} != endpoints:
        raise ValueError('FIT endpoint without artifact or artifact without endpoint: unknown')
    custodies = list(journal.root.glob('*.fit.json')) if journal.root.exists() else []
    linked = {}
    for path in custodies:
        doc = json.loads(path.read_bytes())
        payload = doc['payload']
        if (path.is_symlink() or doc.get('schema') != 'saea-native-fit-custody-v1'
                or doc.get('identity') != journal.identity
                or path.name != digest(canonical(comparable_request(payload))) + '.fit.json'):
            raise ValueError('FIT custody identity differs')
        receipt = doc['receipt']
        logical_path = Path(receipt['path'])
        if logical_path.parent != logical_root:
            raise ValueError('FIT receipt outside explicit mapped directory')
        artifact = root / logical_path.name
        if artifact in linked or artifact not in artifacts:
            raise ValueError('duplicate or missing FIT artifact')
        linked[artifact] = doc
    sequences, counts, results = [], Counter(), []
    contact_ids = set()
    if fit_plan.planned_model_calls != len(roster) * max_contacts:
        raise ValueError('raw FIT cap differs from roster and program')
    for artifact in sorted(artifacts):
        doc = json.loads(artifact.read_bytes())
        inputs, native = doc['inputs'], doc['inputs']['native_contact']
        if expected_inputs is not None:
            for key in ('config_sha256', 'split_identity_hash'):
                if inputs.get(key) != expected_inputs.get(key):
                    raise ValueError('FIT scientific configuration differs')
            for key in ('implementation', 'runtime_receipt', 'required_identity'):
                if inputs['sources'].get(key) != expected_inputs['sources'].get(key):
                    raise ValueError('FIT source identity differs')
        output = load_artifact(artifact, expected_sha256=digest(artifact.read_bytes()),
                               kind='request-fit', inputs=inputs)
        if doc['contact_class'] != 'tokenizer' or inputs['fit_plan'] != asdict(fit_plan):
            raise ValueError('FIT artifact class or actual plan differs')
        sample, contact = native['sample_id'], native['agent_contact_id']
        identity = object_digest({'sample_id': sample, 'agent_contact_id': contact})
        if (sample not in roster or native.get('contact_identity_sha256') != identity
                or artifact.name != identity + '.request-fit.json'
                or output.get('sample_id') != sample or output.get('agent_contact_id') != contact
                or output.get('contact_identity_sha256') != identity
                or native.get('capacity_scope') != 'actual-variable-trajectory-contact'):
            raise ValueError('FIT sample/contact identity differs')
        if contact in contact_ids:
            raise ValueError('duplicate FIT contact identity')
        contact_ids.add(contact)
        counts[sample] += 1
        endpoint = Path(str(artifact) + '.contacts.jsonl')
        lines = endpoint.read_bytes().splitlines()
        events = [json.loads(line) for line in lines]
        if len(events) != 1:
            raise ValueError('raw FIT requires exactly one endpoint event')
        for event in events:
            if (event.get('run_id') != fit_plan.run_id or event.get('attempt_id') != fit_plan.attempt_id
                    or event.get('generation_run_id') != journal.identity['run_id']
                    or event.get('sample_id') != sample or event.get('agent_contact_id') != contact
                    or event.get('contact_identity_sha256') != identity
                    or event.get('method') != 'POST' or event.get('route') != '/tokenize'
                    or event.get('session_id') != output.get('session_id')
                    or (session_id is not None and event.get('session_id') != session_id)
                    or type(event.get('contact')) is not int or event['contact'] < 1):
                raise ValueError('FIT endpoint identity or route differs')
            sequences.append(event['contact'])
        routes = Counter(e['method'] + ' ' + e['route'] for e in events)
        if (output.get('contacts') != len(events) or output.get('contact_counts') != dict(routes)
                or output.get('cumulative_contacts') != events[-1]['contact']
                or output.get('session_identity_sha256') != object_digest(session_identity)
                or inputs['sources']['required_identity'] != session_identity):
            raise ValueError('FIT endpoint output or session differs')
        if (type(output.get('every_request_fits')) is not bool
                or not isinstance(output.get('rows'), list) or len(output['rows']) != 1
                or type(output['rows'][0].get('measured_prompt_tokens')) is not int
                or output['rows'][0]['measured_prompt_tokens'] < 0):
            raise ValueError('FIT artifact lacks a valid measured result')
        custody = linked.get(artifact)
        status = ('returned_missing_custody' if output['every_request_fits']
                  else 'capacity_rejected_missing_custody')
        if custody is not None:
            payload, receipt = custody['payload'], custody['receipt']
            custody_inputs = {**custody['inputs'], 'native_contact': native, 'fit_plan': asdict(fit_plan)}
            measured = custody['measured_prompt_tokens']
            if (receipt.get('sha256') != digest(artifact.read_bytes())
                    or receipt.get('contact_class') != 'tokenizer'
                    or receipt.get('sample_id') != sample or receipt.get('agent_contact_id') != contact
                    or receipt.get('contact_identity_sha256') != identity
                    or custody.get('endpoint_journal_sha256') != digest(endpoint.read_bytes())
                    or custody.get('session_identity') != session_identity
                    or inputs != custody_inputs or logical_contact(payload) != {'sample_id': sample, 'agent_contact_id': contact}
                    or inputs['sources']['boundary_sha256'] != boundary_identity((payload,))
                    or output.get('boundary_sha256') != boundary_identity((payload,))
                    or native.get('raw_speech_query') != payload.get('raw_speech_query')
                    or output.get('every_request_fits') is not True
                    or type(measured) is not int or measured < 0
                    or len(output['rows']) != 1 or output['rows'][0].get('measured_prompt_tokens') != measured):
                raise ValueError('FIT custody, input, boundary or output differs')
            status = 'success_recoverable'
        results.append({'sample_id': sample, 'agent_contact_id': contact, 'status': status,
                        'artifact_sha256': digest(artifact.read_bytes()), 'endpoint_contacts': len(events)})
    if any(n > max_contacts for n in counts.values()):
        raise ValueError('FIT sample contact limit exceeded')
    if sorted(sequences) != list(range(1, len(sequences) + 1)) or len(sequences) > fit_plan.planned_model_calls:
        raise ValueError('FIT sequence gap, duplicate or budget exceeded')
    return {'endpoint_calls': len(sequences), 'fit_run_id': fit_plan.run_id, 'fit_attempt_id': fit_plan.attempt_id,
            'contacts': results, 'missing_custody': sum(x['status'] != 'success_recoverable' for x in results)}



def _pinned(pin):
    if (not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}
            or not isinstance(pin['path'], str) or not Path(pin['path']).is_absolute()
            or not isinstance(pin['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', pin['sha256'])):
        raise ValueError('invalid exact evidence pin')
    path = Path(pin['path'])
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or redirected evidence')
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != pin['sha256']:
        raise ValueError('evidence pin changed')
    return blob


def inventory_snapshot(inventory):
    """Check exact bounded directory trees and all their bytes, read-only."""
    if not isinstance(inventory, list) or not inventory:
        raise ValueError('inventory roots required')
    seen, result = set(), {}
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {'root', 'files'} or not isinstance(item['files'], dict):
            raise ValueError('invalid inventory record')
        root = Path(item['root'])
        if not root.is_absolute() or root.is_symlink() or not root.is_dir() or root in seen:
            raise ValueError('invalid inventory root')
        seen.add(root)
        actual = {}
        for parent, dirs, files in os.walk(root, followlinks=False):
            for name in dirs:
                if (Path(parent) / name).is_symlink():
                    raise ValueError('inventory directory symlink')
            for name in files:
                p = Path(parent) / name
                if p.is_symlink() or not p.is_file():
                    raise ValueError('inventory non-regular file')
                actual[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != item['files']:
            raise ValueError('inventory set or hash changed')
        result[str(root)] = actual
    return result


def inspect_settlement(*, reconciliation, lease, attempt, observed, ports=None, not_sent=()):
    """Check the native stale-lease settlement receipt against preserved bytes."""
    rec = json.loads(_pinned(reconciliation))
    blob = _pinned(lease)
    lock = json.loads(blob)
    state = rec['state']
    quiet = rec['quiescence']
    if (not isinstance(rec.get('container'), str) or not rec['container']
            or state.get('Running') is not False or state.get('Status') != 'exited' or state.get('Pid') != 0
            or rec.get('attempt') != attempt or attempt.get('state') != 'ABORTED'
            or lock.get('run_id') != attempt.get('run_id') or lock.get('attempt_id') != attempt.get('attempt_id')
            or type(lock.get('pid')) is not int or lock['pid'] < 1
            or not isinstance(lock.get('session_id'), str) or not lock['session_id']
            or rec.get('lease_sha256') != hashlib.sha256(blob).hexdigest()
            or sorted(rec.get('observed_requests', [])) != sorted(observed)
            or len(set(rec.get('observed_requests', []))) != len(observed)
            or sorted(rec.get('not_sent_requests', [])) != sorted(not_sent)
            or len(set(rec.get('not_sent_requests', []))) != len(not_sent)
            or set(observed) & set(not_sent)
            or rec.get('no_model_calls') is not True
            or type(quiet.get('observed_at')) not in (int, float)
            or not math.isfinite(quiet.get('observed_at'))
            or quiet['observed_at'] <= 0
            or not isinstance(quiet, dict) or quiet.get('established_connections') != []
            or not isinstance(quiet.get('queues'), dict) or not quiet['queues']):
        raise ValueError('settlement receipt or preserved lease differs')
    if ports is not None and set(quiet['queues']) != {str(port) for port in ports}:
        raise ValueError('settlement quiet endpoints differ from deployment')
    for metrics in quiet['queues'].values():
        if (not isinstance(metrics, dict) or set(metrics) != {'vllm:num_requests_running','vllm:num_requests_waiting'}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v != 0 for v in metrics.values())):
            raise ValueError('settlement queues are not quiet')
    return lock



def inspect_natural_drain(*, observation, attempt, plan_sha256, attempt_sha256,
                          observed, runtime_receipt, session, session_sha256=None,
                          ports=None, runtime_receipt_sha256=None, not_sent=()):
    """Authenticate a producer's normal GenerationLease context exit.

    The absent lease is one fact observed while holding the stable guard; the
    receipt must also bind the exited producer, boot, attempt, session, quiet
    endpoints, and the immutable journal/attempt hashes.
    """
    doc = json.loads(_pinned(observation))
    if (doc.get('schema') != 'saea-natural-terminal-observation-v1'
            or doc.get('mode') != 'natural-drain'
            or doc.get('run_id') != attempt.get('run_id')
            or doc.get('attempt_id') != attempt.get('attempt_id')
            or attempt.get('state') != 'ABORTED'
            or doc.get('plan_sha256') != plan_sha256
            or doc.get('attempt_sha256') != attempt_sha256
            or doc.get('observed_requests') != sorted(observed)
            or doc.get('not_sent_requests', []) != sorted(not_sent)
            or set(observed) & set(not_sent)
            or doc.get('session_id') != session.get('session_id')
            or doc.get('lease_absent_under_guard') is not True
            or not isinstance(doc.get('boot_id'), str) or not doc['boot_id']
            or not isinstance(doc.get('producer_pid'), int) or doc['producer_pid'] < 1
            or not isinstance(doc.get('container'), dict)
            or not isinstance(doc['container'].get('id'), str)
            or doc['container'].get('state', {}).get('Running') is not False
            or doc['container'].get('state', {}).get('Status') != 'exited'
            or doc['container'].get('state', {}).get('Pid') != 0
            or not isinstance(doc.get('guard'), dict)
            or not isinstance(doc['guard'].get('device'), int)
            or not isinstance(doc['guard'].get('inode'), int)
            or type(doc.get('producer_start_ticks')) is not int
            or doc['producer_start_ticks'] < 1
            or not re.fullmatch('[0-9a-f]{64}', str(doc.get('producer_identity_sha256', '')))
            or not isinstance(doc.get('quiescence'), dict)):
        raise ValueError('natural drain evidence identity differs')
    quiet = doc['quiescence']
    if not isinstance(quiet.get('queues'), dict) or not quiet['queues']:
        raise ValueError('natural drain quiet queues missing')
    if ports is not None and set(quiet.get('queues', {})) != {str(p) for p in ports}:
        raise ValueError('natural drain quiet endpoints differ from deployment')
    if quiet.get('established_connections') != []:
        raise ValueError('natural drain has established inference connections')
    observed_at = quiet.get('observed_at')
    if type(observed_at) not in (int, float) or not math.isfinite(observed_at) or observed_at <= 0:
        raise ValueError('natural drain quiet observation is invalid')
    for metrics in quiet.get('queues', {}).values():
        if (not isinstance(metrics, dict) or set(metrics) != {'vllm:num_requests_running', 'vllm:num_requests_waiting'}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v != 0
                       for v in metrics.values())):
            raise ValueError('natural drain serving queue is active or unknown')
    if session_sha256 is not None and doc.get('session_sha256') != session_sha256:
        raise ValueError('natural drain session changed')
    if runtime_receipt_sha256 is None or doc.get('runtime_receipt_sha256') != runtime_receipt_sha256:
        raise ValueError('natural drain runtime receipt identity is unavailable')
    return {'mode': 'natural-drain', 'observed_requests': list(observed),
            'container_id': doc['container']['id'], 'boot_id': doc['boot_id'],
            'quiescence': quiet}

def validate_terminal_custody(*, cfg, candidate, plan, data):
    """Verify pinned, settled raw-speech predecessor custody without writes.

    This authenticates recorded evidence, not current remote process liveness;
    the launcher must recheck the real guard, lease, and deployment at startup.
    """
    from dataclasses import asdict, replace
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.core.session.vllm_resident import required_identity
    from speech_aware_evidence_acquisition.core.flight.resume import resume_declarations_from_document
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import (
        CheckpointCompatibility, ImageReleaseCompatibility, MultiCheckpointCompatibility,
        STUDY_PREFIX, authenticated_old_config, validate_bridge_declarations)
    from speech_aware_evidence_acquisition.core.roster import verify_roster
    from speech_aware_evidence_acquisition.core.preparation import artifact_inputs

    terminal = cfg['terminal']
    evidence = terminal['evidence']
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('terminal evidence required')
    blobs, documents = {}, []
    for pin in evidence:
        blob = _pinned(pin)
        if pin['path'] in blobs:
            raise ValueError('duplicate evidence path')
        blobs[pin['path']] = (pin['sha256'], blob)
        try:
            doc = json.loads(blob)
        except (ValueError, UnicodeError):
            continue
        if isinstance(doc, dict) and doc.get('schema') == 'saea-checkpoint-terminal-custody-v1':
            documents.append(doc)
    if len(documents) != 1:
        raise ValueError('one terminal custody index required')
    index = documents[0]
    if set(index) != {'schema', 'ancestors', 'inventory'} or not isinstance(index['ancestors'], list):
        raise ValueError('invalid custody index')
    def evidence_blob(pin):
        if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'} or pin['path'] not in blobs:
            raise ValueError('referenced pin missing from evidence')
        if blobs[pin['path']][0] != pin['sha256']:
            raise ValueError('referenced pin differs')
        return _pinned(pin)
    before = inventory_snapshot(index['inventory'])
    declarations = resume_declarations_from_document(terminal['resume_from'])
    bridge = CheckpointCompatibility.from_document(terminal['bridge'], attempt_id=cfg['target']['attempt_id'])
    validate_bridge_declarations(bridge, declarations)
    bridges = bridge.bridges if isinstance(bridge, MultiCheckpointCompatibility) else {declarations[0].attempt_id: bridge}
    expected = {d.attempt_id: d for d in declarations}
    if candidate['attempt_id'] not in expected:
        raise ValueError('candidate root missing from ancestor declarations')
    ancestors = index['ancestors']
    if (len(ancestors) != len(expected) or any(not isinstance(a, dict) for a in ancestors)
            or {a.get('attempt_id') for a in ancestors} != set(expected)):
        raise ValueError('ancestor set differs from resume declarations')
    roots = {str(Path(x['root'])) for x in index['inventory']}
    results, graph = [], {}
    for a in ancestors:
        allowed = {'attempt_id','plan','checkpoint','generation_root','prepared_root',
                   'logical_prepared_root','roster','bridge_key','reconciliation','lease','terminal_mode','producer_identity'}
        if set(a) - allowed or not {'attempt_id','plan','checkpoint','generation_root','prepared_root',
                      'logical_prepared_root','roster','bridge_key','reconciliation'} <= set(a):
            raise ValueError('invalid ancestor fields')
        mode = a.get('terminal_mode', 'stale-lease')
        if mode not in ('stale-lease', 'natural-drain'):
            raise ValueError('unknown terminal evidence mode')
        if mode == 'stale-lease' and ('lease' not in a):
            raise ValueError('stale lease evidence required')
        if mode == 'natural-drain' and a.get('lease') != {'absent': True}:
            raise ValueError('natural drain must explicitly declare absent lease')
        aid = a['attempt_id']
        if a['bridge_key'] != aid or not isinstance(bridges[aid], ImageReleaseCompatibility):
            raise ValueError('authenticated image release bridge required')
        old_bridge = bridges[aid]
        workspace = old_bridge.evidence.get('producer_workspace')
        if workspace is not None:
            for key in ('manifest', 'source_manifest', 'launch', 'container'):
                evidence_blob(workspace[key])
        files = old_bridge.old_source(run_id=candidate['run_id'], predecessor_attempt_id=aid)
        registered = json.loads(evidence_blob(a['plan']))
        raw_plan = dict(registered['execution_plan'])
        if 'owner_allowance' in registered:
            if 'owner_allowance' in raw_plan and raw_plan['owner_allowance'] != registered['owner_allowance']:
                raise ValueError('plan allowance differs')
            raw_plan['owner_allowance'] = registered['owner_allowance']
        source = ExecutionPlan(**raw_plan).validate(source_root=cfg['repo'])
        if source.run_id != candidate['run_id'] or source.attempt_id != aid:
            raise ValueError('ancestor plan identity differs')
        if aid == candidate['attempt_id'] and registered != plan:
            raise ValueError('registered source plan differs from compiler input')
        cp = a['checkpoint']
        evidence_blob(cp)
        if cp['sha256'] != expected[aid].checkpoint_sha256:
            raise ValueError('checkpoint declaration differs')
        doc = inspect_checkpoint(path=cp['path'], sha256=cp['sha256'], plan=source)
        old_docs, composed = authenticated_old_config(files, old_bridge.predecessor_fragments, doc.header,
                                                       old_bridge.evidence['startup_data_root'])
        if registered.get('composed_config_hash') != doc.header.get('config_hash'):
            raise ValueError('registered configuration and checkpoint differ')
        if aid == candidate['attempt_id'] and (candidate.get('protocol_hash', source.protocol_hash) != source.protocol_hash
                or candidate.get('split_identity_hash', source.split_identity_hash) != source.split_identity_hash
                or candidate.get('composed_config_hash', composed.config_hash) != composed.config_hash):
            raise ValueError('candidate science identity differs')
        values = composed.values
        budget = old_docs['experiment'].get('execution_plan', {})
        for field in ('execution_profile', 'split_role', 'planned_model_calls', 'planned_speech_audio_seconds', 'planned_gpu_hours'):
            if budget.get(field) != getattr(source, field):
                raise ValueError('registered plan budget differs from authenticated experiment: ' + field)
        cells = old_docs['experiment']['cells']
        if len(cells) != 1 or cells[0].get('run_id') != source.run_id or cells[0].get('attempt_id') != aid:
            raise ValueError('authenticated predecessor cell differs')
        inherited = cells[0].get('resume_from')
        previous = resume_declarations_from_document(inherited) if inherited is not None else ()
        if any(d.attempt_id not in expected or expected[d.attempt_id].checkpoint_sha256 != d.checkpoint_sha256 for d in previous):
            raise ValueError('ancestor declaration omitted or changed')
        graph[aid] = {d.attempt_id for d in previous}
        if values.get('arm') != 'raw-speech-agent-native-tools':
            raise ValueError('only raw-speech native custody is supported')
        if values.get('prepared_execution_units') is not None:
            raise ValueError('execution-unit FIT requires separate custody verification')
        spec = values['native_contact_fit']
        policy = values['policy']
        policy_path = STUDY_PREFIX + 'configs/policy/' + policy + '.json'
        program = json.loads(files[policy_path])
        if program.get('program_id') != policy or program.get('stop_rule') != 'native-role-episode':
            raise ValueError('authenticated phase program differs')
        roster = json.loads(evidence_blob(a['roster']))
        if (verify_roster(roster) != source.split_identity_hash
                or roster.get('carrier_lock_key') != source.carrier_lock_key
                or roster.get('split_role') != source.split_role):
            raise ValueError('roster and plan differ')
        ids = [x['sample_id'] for x in roster['items']]
        cap = spec['contact_cap']
        if type(program.get('max_contacts')) is not int or program['max_contacts'] < 2 or type(cap) is not int or cap != len(ids) * program['max_contacts']:
            raise ValueError('authenticated FIT cap differs')
        fit_plan = replace(source, run_id=spec['run_id'], attempt_id=spec['attempt_id'], planned_model_calls=cap)
        generation_root = Path(a['generation_root'])
        journal = ContactJournal(generation_root, source, values['arm'], source_root=cfg['repo'])
        attempt_path = AttemptStore(generation_root/'attempts').handle(source.run_id, aid).path
        cp_path = Path(cp['path'])
        if Path(a['logical_prepared_root']) != Path(old_bridge.evidence['startup_data_root'])/'prepared'/fit_plan.run_id/fit_plan.attempt_id:
            raise ValueError('logical FIT directory differs from startup data root')
        required_roots = {str(journal.root), str(attempt_path.parent), str(cp_path.parent), str(Path(a['prepared_root']))}
        if not required_roots <= roots:
            raise ValueError('inventory does not cover all custody roots')
        if any(str(root) != str(Path(root).resolve()) for root in required_roots):
            raise ValueError('custody roots must be absolute and contain no symlinks')
        if before[str(attempt_path.parent)].get(attempt_path.name) != hashlib.sha256(attempt_path.read_bytes()).hexdigest():
            raise ValueError('attempt missing from inventory')
        if before[str(cp_path.parent)].get(cp_path.name) != cp['sha256']:
            raise ValueError('checkpoint missing from inventory')
        if cp_path != generation_root/'runs'/source.run_id/aid/(source.run_id+'.rows.ckpt.jsonl'):
            raise ValueError('checkpoint is outside mapped attempt directory')
        observed = inspect_generation(data_root=generation_root, plan=source, arm=values['arm'], source_root=cfg['repo'])
        attempt = json.loads(attempt_path.read_bytes())
        evidence_blob(a['reconciliation'])
        if mode == 'stale-lease':
            evidence_blob(a['lease'])
            lease = inspect_settlement(reconciliation=a['reconciliation'], lease=a['lease'], attempt=attempt,
                                       observed=observed['observed_requests'], not_sent=observed.get('not_sent_requests', []))
        else:
            observation = json.loads(evidence_blob(a['reconciliation']))
            producer = json.loads(evidence_blob(a['producer_identity']))
            if (observation.get('producer_identity_sha256') != a['producer_identity']['sha256']
                    or producer.get('schema') != 'saea-producer-process-observation-v1'
                    or producer.get('run_id') != source.run_id or producer.get('attempt_id') != aid
                    or type(producer.get('observed_before_exit')) not in (int, float)
                    or not math.isfinite(producer['observed_before_exit']) or producer['observed_before_exit'] <= 0
                    or observation.get('captured_at', 0) < producer['observed_before_exit']
                    or producer.get('plan') != a['plan']
                    or producer.get('container_id') != observation.get('container', {}).get('id')
                    or producer.get('producer_pid') != observation.get('producer_pid')
                    or producer.get('producer_start_ticks') != observation.get('producer_start_ticks')
                    or producer.get('boot_id') != observation.get('boot_id')
                    or producer.get('guard') != observation.get('guard')
                    or producer.get('session_id') != observation.get('session_id')
                    or producer.get('session', {}).get('sha256') != observation.get('session_sha256')):
                raise ValueError('natural drain producer pin differs')
            live_launch = json.loads(evidence_blob(producer['launch_config']))
            live_intent = json.loads(evidence_blob(producer['launch_intent']))
            live_result = json.loads(evidence_blob(producer['launch_result']))
            evidence_blob(producer['runtime_receipt'])
            config = json.loads(evidence_blob(producer['config']))
            logical_locks = str(Path(live_launch['argv'][live_launch['argv'].index('--data') + 1])/'locks')
            if (config.get('schema') != 'saea-producer-observation-config-v1'
                    or any(config.get(k) != producer.get(k) for k in
                           ('plan','runtime_receipt','session','launch_config','launch_intent','launch_result','session_id'))
                    or live_intent.get('config_sha256') != hashlib.sha256(json.dumps(live_launch, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    or live_result.get('returncode') != 0
                    or live_result.get('stdout', '').strip() != producer['container_id']
                    or producer.get('process_cmdline') != live_launch.get('argv')
                    or producer.get('lease_identity') != {'run_id':source.run_id,'attempt_id':aid,
                        'pid':producer['producer_pid'],'session_id':producer['session_id']}
                    or not re.fullmatch('[0-9a-f]{64}', str(producer.get('lease_sha256', '')))
                    or live_launch.get('generation_lease', {}).get('lock_directory') != producer.get('lock_directory')
                    or not any(m.get('source') == producer['lock_directory'] and m.get('target') == logical_locks
                               and m.get('readonly') is False for m in live_launch['mounts'])
                    or live_launch.get('environment', {}).get('SAEA_OBSERVABILITY_RUN_ID') != source.run_id
                    or live_launch.get('environment', {}).get('SAEA_OBSERVABILITY_ATTEMPT_ID') != aid):
                raise ValueError('natural drain live producer evidence differs')
            session_doc = json.loads(evidence_blob(producer['session']))
            if (session_doc.get('session_id') != producer['session_id']
                    or session_doc.get('identity', {}).get('runtime_receipt_sha256') != observation.get('runtime_receipt_sha256')):
                raise ValueError('natural drain pinned session differs')
            lease = inspect_natural_drain(observation=a['reconciliation'], attempt=attempt,
                plan_sha256=hashlib.sha256(evidence_blob(a['plan'])).hexdigest(),
                attempt_sha256=hashlib.sha256(attempt_path.read_bytes()).hexdigest(),
                observed=observed['observed_requests'], not_sent=observed.get('not_sent_requests', []), runtime_receipt={},
                session={'session_id': observation.get('session_id')}, ports=None,
                session_sha256=observation.get('session_sha256'),
                runtime_receipt_sha256=observation.get('runtime_receipt_sha256'))
            lease['session_id'] = producer['session_id']
        custody_files = sorted(journal.root.glob('*.fit.json'))
        if not custody_files:
            raise ValueError('FIT session identity cannot be authenticated without custody')
        first = json.loads(custody_files[0].read_bytes())
        session = first['session_identity']
        inputs = first['inputs']
        if (inputs.get('split_identity_hash') != source.split_identity_hash
                or inputs.get('config_sha256') != artifact_inputs(config=values, split_identity_hash=source.split_identity_hash, sources={})['config_sha256']):
            raise ValueError('FIT session or composed config differs')
        runtime_path = session.get('runtime_receipt_relpath')
        if not isinstance(runtime_path, str) or runtime_path.startswith('/') or '..' in Path(runtime_path).parts:
            raise ValueError('invalid runtime receipt path')
        runtime_blob = files[STUDY_PREFIX + 'docs/' + runtime_path]
        runtime_pins = [p for p in evidence if p['sha256'] == hashlib.sha256(runtime_blob).hexdigest()]
        if len(runtime_pins) != 1 or evidence_blob(runtime_pins[0]) != runtime_blob:
            raise ValueError('authenticated runtime receipt evidence missing')
        runtime_receipt = json.loads(runtime_blob)
        if runtime_receipt != inputs['sources']['runtime_receipt']:
            raise ValueError('FIT runtime receipt differs from authenticated source')
        reconstructed = required_identity(fragment=old_docs['model'],
            model_fragment=old_bridge.predecessor_fragments['model'], runtime_receipt=runtime_receipt,
            receipt_path=runtime_pins[0]['path'], host=session['host'], port=session['port'])
        if reconstructed != session:
            raise ValueError('FIT serving identity differs from authenticated receipt')
        instances = runtime_receipt['deployment']['instances']
        ports = [i['port'] for i in instances]
        if len(set(ports)) != len(ports) or any(type(p) is not int or not 0 < p < 65536 for p in ports):
            raise ValueError('invalid deployment endpoints')
        if mode == 'stale-lease':
            inspect_settlement(reconciliation=a['reconciliation'], lease=a['lease'], attempt=attempt,
                               observed=observed['observed_requests'], ports=ports, not_sent=observed.get('not_sent_requests', []))
        else:
            ndoc = json.loads(_pinned(a['reconciliation']))
            if set(ndoc['quiescence'].get('queues', {})) != {str(p) for p in ports}:
                raise ValueError('natural drain quiet endpoints differ from deployment')
            if ndoc.get('runtime_receipt_sha256') != hashlib.sha256(runtime_blob).hexdigest():
                raise ValueError('natural drain runtime receipt differs')
            if producer['runtime_receipt']['sha256'] != hashlib.sha256(runtime_blob).hexdigest():
                raise ValueError('natural drain producer runtime pin differs')
            if session_doc.get('deployment_instance_ids') != [i['instance_id'] for i in instances]:
                raise ValueError('natural drain session deployment differs')
        implementation = {'runtime':'core/flight/runtime.py','transport':'core/model/transport.py',
            'preparation_fit':'core/preparation_fit.py','preparation':'core/preparation.py','config':'core/config.py',
            'runtime_view':'rails/adapters/runtime_view.py','core_client':'rails/core_client.py',
            'speech_features':'core/model/speech_features.py'}
        prefix = STUDY_PREFIX + 'src/speech_aware_evidence_acquisition/'
        wanted = {k:hashlib.sha256(files[prefix+v]).hexdigest() for k,v in implementation.items()}
        if inputs['sources']['implementation'] != wanted:
            raise ValueError('FIT implementation differs from authenticated source')
        expected_fit_inputs = artifact_inputs(config=values, split_identity_hash=source.split_identity_hash,
            sources={'boundary_sha256': inputs['sources']['boundary_sha256'],
                     'required_identity': session, 'runtime_receipt': runtime_receipt,
                     'implementation': inputs['sources']['implementation']})
        fit = inspect_fit(journal=journal, prepared_root=a['prepared_root'], logical_prepared_root=a['logical_prepared_root'],
                          fit_plan=fit_plan, session_identity=session, max_contacts=program['max_contacts'], roster=ids,
                          expected_inputs=expected_fit_inputs, session_id=lease.get('session_id'))
        from speech_aware_evidence_acquisition.core.contact_journal import comparable_request, canonical as contact_canonical, logical_contact
        fitted = {}
        for fit_path in custody_files:
            fit_doc = json.loads(fit_path.read_bytes())
            p = fit_doc['payload']
            identity = logical_contact(p)
            fitted.setdefault((identity['sample_id'], identity['agent_contact_id']), []).append(contact_canonical(comparable_request(p)))
        for intent_path in journal.root.glob('*.intent.json'):
            intent = json.loads(intent_path.read_bytes())
            payload = intent['payload']
            key = (intent['sample_id'], intent['agent_contact_id'])
            if contact_canonical(comparable_request(payload)) not in fitted.get(key, []):
                raise ValueError('observed generation lacks matching successful FIT')
        incomplete_rows, native_keepable = {}, 0
        for row in doc.rows:
            if row.sample_id not in ids:
                raise ValueError('checkpoint sample outside registered roster')
            defect = row.keep_defect()
            if defect is None:
                output = row.output_rows[0]
                trace_records = [json.loads(line) for line in row.trace_lines]
                present = {record['record_id'] for record in trace_records}
                rid = output.get('request_id')
                if len(row.output_rows) != 1 or output.get('sample_id') != row.sample_id:
                    defect = 'the stored record does not name its own sample'
                elif {f'{rid}/request', f'{rid}/response'} - present:
                    defect = 'the stored trace lacks the request/response the stored record names'
            if defect is not None:
                incomplete_rows[row.sample_id] = defect
                continue
            if defect is None:
                requests, responses = {}, {}
                for record in trace_records:
                    channel, record_id, payload = record['channel'], record['record_id'], record['payload']
                    if channel == 'MODEL_REQUEST' and record_id.endswith('/request'):
                        rid = record_id[:-len('/request')]
                        if rid in requests:
                            raise ValueError('checkpoint trace duplicate request')
                        requests[rid] = payload
                    elif channel == 'MODEL_RESPONSE' and record_id.endswith('/response'):
                        rid = record_id[:-len('/response')]
                        if rid in responses:
                            raise ValueError('checkpoint trace duplicate response')
                        responses[rid] = payload
                if set(requests) != set(responses):
                    raise ValueError('checkpoint trace request/response mismatch')
                for rid, request in requests.items():
                    contact = journal.inspect(rid)
                    if (contact['state'] != 'observed'
                            or contact['intent']['sample_id'] != row.sample_id
                            or request.get('payload') != contact['intent']['payload']
                            or request.get('payload_sha256') != contact['intent']['request_sha256']
                            or responses[rid].get('response_sha256') != contact['response']['response_sha256']):
                        raise ValueError('checkpoint trace differs from observed journal')
                    try:
                        parsed = json.loads(contact['response_bytes'].decode('utf-8'))
                        text = parsed['text']
                        usage = {str(k): int(v) for k, v in dict(parsed.get('usage', {})).items()}
                        if not isinstance(text, str):
                            raise ValueError('response text is not a string')
                    except (KeyError, TypeError, ValueError, UnicodeError) as error:
                        raise ValueError('checkpoint journal response is malformed') from error
                    if (responses[rid].get('text') != text
                            or responses[rid].get('response_text_sha256') != hashlib.sha256(text.encode('utf-8')).hexdigest()
                            or responses[rid].get('usage') != usage):
                        raise ValueError('checkpoint response text or usage differs from journal')
                rid = output.get('request_id')
                if rid in requests:
                    contact = journal.inspect(rid)
                    if (output.get('request_sha256') != contact['intent']['request_sha256']
                            or output.get('response_sha256') != contact['response']['response_sha256']):
                        raise ValueError('checkpoint canonical output hashes differ from observed journal')
                else:
                    from speech_aware_evidence_acquisition.rails.native_episode_receipt import validate_native_episode_delivery_receipt
                    from speech_aware_evidence_acquisition.core.delivery import (
                        native_abstention_records, native_context_abstention_records,
                        native_seal_refusal_abstention_records)
                    try:
                        delivery = validate_native_episode_delivery_receipt(output['native_episode_receipt'])
                        disposition = delivery.terminal_disposition
                        if delivery.kind == 'context-abstention':
                            derived = native_context_abstention_records(sample_id=row.sample_id, disposition=disposition)
                        elif delivery.kind == 'abstention':
                            source_contact = delivery.final_contact
                            source_id = source_contact['request_id']
                            if source_id not in requests or any(source_contact.get(k) != v for k, v in {
                                    'agent_contact_id':logical_contact(requests[source_id]['payload'])['agent_contact_id'],
                                    'request_sha256':requests[source_id]['payload_sha256'],
                                    'response_sha256':responses[source_id]['response_sha256'],
                                    'response_text_sha256':responses[source_id]['response_text_sha256']}.items()):
                                raise ValueError('abstention source differs from observed journal')
                            if disposition['reason'] == 'seal_refusal':
                                derived = native_seal_refusal_abstention_records(sample_id=row.sample_id,
                                    source_contact=source_contact, seal_refusal=disposition['seal_refusal'])
                            else:
                                derived = native_abstention_records(sample_id=row.sample_id,
                                    source_contact=source_contact, finish_reason=disposition['finish_reason'])
                        else:
                            raise ValueError('canonical request lacks observed model contact')
                        derived_id, request_doc, response_doc = derived
                        records = [r for r in trace_records if r['record_id'] in {derived_id+'/request', derived_id+'/response'}]
                        if (rid != derived_id or len(records) != 2
                                or {r['record_id']: (r['channel'], r['payload']) for r in records} != {
                                    derived_id+'/request':('TOOL', request_doc), derived_id+'/response':('TOOL', response_doc)}
                                or output.get('request_sha256') != request_doc['payload_sha256']
                                or output.get('response_sha256') != response_doc['response_sha256']):
                            raise ValueError('derived canonical identity differs')
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError('checkpoint derived output identity differs') from error
                native_keepable += 1
        results.append({'generation': observed, 'fit':fit, 'checkpoint_sha256':doc.sha256,
                        'checkpoint_complete_rows':doc.complete_rows, 'checkpoint_native_keepable_rows':native_keepable,
                        'checkpoint_incomplete_rows':incomplete_rows, 'lease_session_id':lease.get('session_id')})
    validate_ancestry(graph, candidate['attempt_id'])
    ledger_roots = {str(Path(a['generation_root']).resolve()) for a in ancestors}
    if len(ledger_roots) != 1:
        raise ValueError('same-run attempts require one shared actual ledger')
    ledger = AttemptStore(Path(next(iter(ledger_roots))) / 'attempts')
    total_calls, total_audio, total_gpu = ledger._sum_actual_for_run(candidate['run_id'])
    if inventory_snapshot(index['inventory']) != before:
        raise ValueError('inventory changed during audit')
    for pin in evidence: _pinned(pin)
    return {'schema':'saea-checkpoint-terminal-custody-result-v1', 'ancestors':results,
            'generation_calls':sum(x['generation']['actual']['calls'] for x in results),
            'generation_audio_seconds':math.fsum(x['generation']['actual']['audio_seconds'] for x in results),
            'fit_endpoint_calls':sum(x['fit']['endpoint_calls'] for x in results),
            'costs': [{'run_id': x['generation']['run_id'], 'attempt_id': x['generation']['attempt_id'],
                       'attempt': x['generation']['actual'],
                       'fit': {'run_id': x['fit'].get('fit_run_id'), 'endpoint_calls': x['fit']['endpoint_calls']}} for x in results],
            'same_run_accumulated': {'calls': total_calls, 'audio_seconds': total_audio, 'gpu_seconds': total_gpu},
            'activation_authorized':False, 'requires_live_guard_and_deployment_check':True}
