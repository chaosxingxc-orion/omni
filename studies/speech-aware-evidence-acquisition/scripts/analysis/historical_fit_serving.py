"""Verify archived FIT identities without borrowing a successor's deployment.

The snapshot is an explicit, hash-pinned set of historical documents. This does
not prove generation attachment or authorize a new deployment/flight.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

SCHEMA = 'saea-historical-fit-serving-v1'
IDENTITY_KEYS = ('model_fragment', 'model_fragment_sha256', 'runtime_receipt_sha256',
                 'binary_sha256', 'model_files')
MAP_SCHEMA = 'saea-historical-fit-serving-map-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value):
        raise ValueError('invalid historical identity')
    return value


def load_pin(root, pin):
    if not isinstance(pin, dict) or set(pin) != {'relpath', 'sha256'}:
        raise ValueError('invalid historical document pin')
    relative = Path(pin['relpath'])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('historical document escapes snapshot')
    if digest(path) != pin['sha256']:
        raise ValueError('historical document hash differs')
    return json.loads(path.read_bytes()), path


def verify(snapshot_path, snapshot_sha256, *, coordinator, model_fragment,
           fit_session, required_identity):
    """Bind an actual FIT artifact to preserved historical serving documents.

    Caller must obtain the snapshot hash from an independently reviewed record,
    and fit_session/required_identity from the verified historical FIT artifact.
    All parent/member/runtime files are required, even for a cost-only ancestor.
    """
    path = Path(snapshot_path)
    if digest(path) != snapshot_sha256:
        raise ValueError('historical snapshot hash differs')
    doc = json.loads(path.read_bytes())
    if set(doc) != {'schema', 'coordinator', 'model_fragment', 'parent', 'runtime', 'members'} or doc['schema'] != SCHEMA:
        raise ValueError('invalid historical snapshot schema')
    for value in (coordinator, model_fragment, fit_session):
        name(value)
    if (doc['coordinator'], doc['model_fragment']) != (coordinator, model_fragment):
        raise ValueError('historical snapshot names another deployment')
    root = path.parent
    parent, parent_path = load_pin(root, doc['parent'])
    runtime, runtime_path = load_pin(root, doc['runtime'])
    members = parent.get('deployment_instance_ids', [])
    if (parent.get('session_id') != coordinator or not members or len(members) != len(set(members))
            or set(doc['members']) != set(members) or fit_session not in members):
        raise ValueError('historical member inventory differs')
    identity = parent['identity']
    if (identity['model_fragment'] != model_fragment
            or identity['runtime_receipt_sha256'] != digest(runtime_path)):
        raise ValueError('historical model/runtime differs')
    # FIT required identity includes model/runtime pins; never substitute the
    # new deployment's runtime receipt, endpoints or process IDs.
    if any(required_identity.get(key) != identity.get(key) or key not in identity for key in IDENTITY_KEYS):
        raise ValueError('FIT artifact differs from historical model/runtime')
    instances = runtime['deployment']['instances']
    by_id = {item['instance_id']: item for item in instances}
    if len(by_id) != len(instances) or set(by_id) != set(members):
        raise ValueError('historical runtime member inventory differs')
    pins = {str(p): digest(p) for p in (path, parent_path, runtime_path)}
    identities = {}
    for member in members:
        name(member)
        pair = doc['members'][member]
        if not isinstance(pair, dict) or set(pair) != {'session', 'receipt'}:
            raise ValueError('historical member needs session and receipt')
        session, session_path = load_pin(root, pair['session'])
        receipt, receipt_path = load_pin(root, pair['receipt'])
        if (session.get('session_id') != member or receipt.get('session_id') != member
                or session.get('deployment_parent_session_id') != coordinator):
            raise ValueError('historical member parent differs')
        if any(session['identity'].get(key) != identity[key] for key in IDENTITY_KEYS):
            raise ValueError('historical member model/runtime differs')
        for key in ('host', 'port'):
            if session['identity'][key] != receipt[key] or receipt[key] != by_id[member][key]:
                raise ValueError('historical member endpoint differs')
        if (type(session.get('pid')) is not int or session['pid'] <= 0
                or not session.get('started_at') or session['pid'] != receipt.get('pid')
                or session['started_at'] != receipt.get('started_at')):
            raise ValueError('historical process receipt differs')
        identities[member] = session['identity']
        pins.update({str(p): digest(p) for p in (session_path, receipt_path)})
    return {'primary_fit_session': fit_session, 'members': identities, 'inputs': pins,
            'custody': 'hash-pinned historical FIT serving; not generation attachment',
            'new_model_calls': 0}


def bindings(path, expected_sha256, *, read_id, run_id, attempt_id, predecessors):
    """Load an independently pinned, exact-scope analysis mapping.

    The caller supplies the reviewed hash; computing it from the input file at
    invocation would only check self-consistency, not establish reviewed custody.
    Snapshot paths are relative to this map and confined to its directory.
    """
    path = Path(path)
    if digest(path) != expected_sha256:
        raise ValueError('historical serving map hash differs')
    doc = json.loads(path.read_bytes())
    if (set(doc) != {'schema', 'read_id', 'run_id', 'attempt_id', 'predecessors'}
            or doc['schema'] != MAP_SCHEMA
            or (doc['read_id'], doc['run_id'], doc['attempt_id']) != (read_id, run_id, attempt_id)
            or not isinstance(doc['predecessors'], dict)
            or set(doc['predecessors']) != {p['cell']['attempt_id'] for p in predecessors}):
        raise ValueError('historical serving map scope differs')
    result = {}
    for source in predecessors:
        attempt = source['cell']['attempt_id']
        entry = doc['predecessors'][attempt]
        if (not isinstance(entry, dict)
                or set(entry) != {'predecessor_commit', 'fit_run_id', 'fit_attempt_id', 'snapshot'}
                or entry['predecessor_commit'] != source['predecessor_commit']
                or (entry['fit_run_id'], entry['fit_attempt_id']) !=
                (source['fit']['run_id'], source['fit']['attempt_id'])):
            raise ValueError('historical serving map ancestor differs')
        _, snapshot = load_pin(path.parent, entry['snapshot'])
        result[attempt] = (snapshot, entry['snapshot']['sha256'])
    return result, {str(path): expected_sha256}


def resolve(data, source, current_doc, current_sessions, binding=None):
    """Select historical identities; the caller must still audit every FIT file.

    The first envelope selects a historical member only. Its complete artifact
    and journal, and all other envelopes, remain subject to audit_source checks.
    This function neither proves historical generation attachment nor changes
    the execution-time checkpoint compatibility gate.
    """
    if binding is None:
        if source['doc']['service_session'] != current_doc['service_session']:
            raise ValueError('changed service requires a pinned historical FIT snapshot')
        return current_sessions
    fit = source['fit']
    for value in (fit['run_id'], fit['attempt_id']):
        name(value)
    paths = sorted((Path(data) / 'prepared' / fit['run_id'] / fit['attempt_id']).glob('*.request-fit.json'))
    if not paths:
        raise ValueError('historical FIT snapshot requires an actual FIT artifact')
    envelope = json.loads(paths[0].read_bytes())
    selected = verify(*binding, coordinator=source['doc']['service_session'],
                      model_fragment=source['cell']['model_fragment'],
                      fit_session=envelope['output']['session_id'],
                      required_identity=envelope['inputs']['sources']['required_identity'])
    old = selected['members'][selected['primary_fit_session']]
    current = current_sessions['members'][current_sessions['primary_fit_session']]
    for key in IDENTITY_KEYS:
        if key != 'runtime_receipt_sha256' and old[key] != current[key]:
            raise ValueError('historical and successor model identities differ')
    return selected
