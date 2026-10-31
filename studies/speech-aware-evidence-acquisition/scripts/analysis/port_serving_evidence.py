"""Bind post-run FIT evidence to the exact attached resident deployment members."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from speech_aware_evidence_acquisition.core.config import fragment_hash, read_fragment
from speech_aware_evidence_acquisition.core.flight.receipts import read_flight_receipts


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def sessions(repo, data, *, coordinator, model_fragment, run_id, attempt_id):
    """Return the primary FIT session and all attached members with source pins.

    The runtime uses its first attached member for tokenizer calls, while
    generation can route across all members. A coordinator alias is not the
    member session_id written by the tokenizer gate. Verify both directions
    of membership and the actual flight receipt before selecting that member.
    """
    if any(not isinstance(v, str) or Path(v).name != v or v in {'', '.', '..'}
           for v in (coordinator, model_fragment, run_id, attempt_id)):
        raise ValueError('invalid serving or flight identity')
    parent_path = data / 'serving' / coordinator / 'session.json'
    parent = read(parent_path)
    members = parent.get('deployment_instance_ids', [])
    if (parent.get('session_id') != coordinator or not members
            or len(members) != len(set(members))):
        raise ValueError('coordinator must declare unique deployment members')
    model_path = repo / 'configs/model' / (model_fragment + '.json')
    if (parent['identity']['model_fragment'] != model_fragment
            or parent['identity']['model_fragment_sha256'] != fragment_hash(read_fragment(repo, 'model', model_fragment))):
        raise ValueError('registered model differs from coordinator identity')
    runtime_path = (repo / 'docs' / parent['identity']['runtime_receipt_relpath']).resolve()
    if not runtime_path.is_relative_to(repo.resolve()) or sha(runtime_path) != parent['identity']['runtime_receipt_sha256']:
        raise ValueError('runtime receipt differs from coordinator identity')
    runtime = read(runtime_path)
    instances = runtime['deployment']['instances']
    by_id = {i['instance_id']: i for i in instances}
    if len(by_id) != len(instances) or set(by_id) != set(members):
        raise ValueError('runtime and coordinator member inventories differ')
    flight_path = data / 'runs' / run_id / attempt_id / 'flight-receipts.json'
    flight = read_flight_receipts(flight_path)
    if (flight['run_id'], flight['attempt_id']) != (run_id, attempt_id):
        raise ValueError('flight receipt belongs to another run')
    serve = flight['stages']['serve']
    attached = serve['instance_sessions']
    if (serve['session_source'] != 'attached-multi-instance-serving-session'
            or serve['deployment_record']['model_fragment'] != model_fragment
            or serve['deployment_record']['instance_count'] != len(members)
            or len(attached) != len(members)
            or {m['session_id'] for m in attached} != set(members)):
        raise ValueError('flight did not attach to the complete registered deployment')
    paths = [parent_path, model_path, runtime_path, flight_path]
    identities = {}
    for observed in attached:
        member = observed['session_id']
        if Path(member).name != member or member in {'.', '..'}:
            raise ValueError('invalid member path')
        path = data / 'serving' / member / 'session.json'
        receipt_path = path.with_name('session-receipt.json')
        doc, receipt = read(path), read(receipt_path)
        if (doc['session_id'] != member or receipt['session_id'] != member
                or observed['instance'] != member or doc['deployment_parent_session_id'] != coordinator):
            raise ValueError('member does not belong to the expected coordinator')
        for key in ('model_fragment', 'model_fragment_sha256', 'runtime_receipt_sha256', 'binary_sha256', 'model_files'):
            if doc['identity'][key] != parent['identity'][key]:
                raise ValueError('member model/runtime identity differs')
        for key in ('host', 'port'):
            if not doc['identity'][key] == receipt[key] == by_id[member][key] == observed[key]:
                raise ValueError('member endpoint differs from executed attachment')
        if (doc['pid'] != receipt['pid'] or doc['pid'] != observed['pid']
                or doc['started_at'] != receipt['started_at']):
            raise ValueError('member process identity differs from executed attachment')
        identities[member] = doc['identity']
        paths.extend([path, receipt_path])
    return {'primary_fit_session': attached[0]['session_id'], 'members': identities,
            'inputs': {str(p): sha(p) for p in paths}}
