"""Source-bound candidate handoff primitives for the native SDK integration.

This module does not perform retrieval or dispatch. Callers must obtain each
knowledge value through the existing snapshot-bound provider, not a model.
"""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping

from .intent_slot_pipeline import digest

SCHEMA = 'saea-candidate-handoff-v1'
POLICY = 'saea-candidate-exact-projection-v1'


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_handoff(*, sample_id, source_role, target_role, source_response_sha256,
                  snapshot_sha256, candidate_ids, judgments, knowledge):
    """Bind typed sample judgments and explicitly supplied snapshot records.

    Knowledge dedup is intentionally conservative: content, scope, annotations
    and target role must all be equal. Every original ID is retained. Model
    judgments are only IDs here; free text and model-proposed rules are refused.
    """
    if not all(isinstance(v, str) and v for v in
               (sample_id, source_role, target_role, source_response_sha256, snapshot_sha256)):
        raise ValueError('handoff identity is incomplete')
    if len(source_response_sha256) != 64 or len(snapshot_sha256) != 64:
        raise ValueError('handoff requires SHA-256 identities')
    if not isinstance(candidate_ids, list) or not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('candidate identities must be unique and nonempty')
    if any(not isinstance(cid, str) or not cid for cid in candidate_ids):
        raise ValueError('invalid candidate identity')
    if not isinstance(judgments, list) or any(
        not isinstance(row, Mapping) or set(row) != {'candidate_id', 'status'}
        or row['candidate_id'] not in candidate_ids
        or row['status'] not in ('candidate', 'uncertain', 'conflict') for row in judgments
    ):
        raise ValueError('judgments must be typed sample observations')
    groups, keys, ids = [], {}, set()
    for record in knowledge:
        if not isinstance(record, Mapping) or set(record) != {
            'id', 'snapshot_sha256', 'text', 'scope', 'annotations', 'target_role'
        }:
            raise ValueError('knowledge must be a typed snapshot record')
        if (record['snapshot_sha256'] != snapshot_sha256 or record['target_role'] != target_role
            or not isinstance(record['id'], str) or not record['id']
            or not isinstance(record['text'], str) or not record['text']
            or record['id'] in ids):
            raise ValueError('knowledge source identity or scope is invalid')
        ids.add(record['id'])
        key = digest({'target_role': target_role, 'text': record['text'],
                      'scope': record['scope'], 'annotations': record['annotations']})
        if key not in keys:
            keys[key] = len(groups)
            groups.append({'ids': [], 'text': record['text'], 'text_sha256': _sha(record['text']),
                           'scope': copy.deepcopy(record['scope']),
                           'annotations': copy.deepcopy(record['annotations'])})
        groups[keys[key]]['ids'].append(record['id'])
    payload = {'schema': SCHEMA, 'policy': POLICY, 'sample_id': sample_id,
               'source_role': source_role, 'target_role': target_role,
               'source_response_sha256': source_response_sha256,
               'snapshot_sha256': snapshot_sha256, 'candidate_ids': list(candidate_ids),
               'judgments': copy.deepcopy(judgments), 'knowledge': groups}
    payload['handoff_sha256'] = digest(payload)
    return payload


def validate_handoff(handoff, *, sample_id, source_role, target_role,
                     source_response_sha256, snapshot_sha256, candidate_ids,
                     snapshot_values):
    """Refuse stale/cross-row packets and verify every supplied body against KB."""
    if not isinstance(handoff, Mapping):
        raise ValueError('handoff must be an object')
    body = dict(handoff)
    claim = body.pop('handoff_sha256', None)
    if claim != digest(body):
        raise ValueError('handoff digest differs')
    for key, expected in (('schema', SCHEMA), ('policy', POLICY), ('sample_id', sample_id),
                          ('source_role', source_role), ('target_role', target_role),
                          ('source_response_sha256', source_response_sha256),
                          ('snapshot_sha256', snapshot_sha256), ('candidate_ids', candidate_ids)):
        if body.get(key) != expected:
            raise ValueError('handoff identity mismatch: ' + key)
    if not isinstance(snapshot_values, Mapping):
        raise ValueError('snapshot values required')
    for group in body['knowledge']:
        if group['text_sha256'] != _sha(group['text']) or not group['ids']:
            raise ValueError('knowledge content identity differs')
        for source_id in group['ids']:
            if snapshot_values.get(source_id) != group['text']:
                raise ValueError('knowledge absent from pinned snapshot')
    return copy.deepcopy(body)


def normalized_response(kind, text, context, config):
    """A typed sample view; original text stays bound in the raw response."""
    from .intent_slot_pipeline import canonical, candidates, document, slot_results
    if kind == 'perception':
        rows = candidates(text, config['max_candidates'], duplicate_policy='group_v2')
        return canonical({'candidates': [
            {'intent_id': row['candidate_id'], 'distinction': 'sample hypothesis',
             'evidence_refs': list(dict.fromkeys(ref for source in row['sources'] for ref in source['evidence_refs'])),
             'unresolved_fields': list(dict.fromkeys(field for source in row['sources'] for field in source['unresolved_fields']))} for row in rows]})
    if kind == 'slot':
        ids = [row['candidate_id'] for row in context['candidates']]
        rows = slot_results(text, ids)
        projected = []
        for cid in ids:
            row = copy.deepcopy(rows[cid])
            # The schema has no mention identity or span: preserve multiplicity.
            row['evidence_refs'] = list(dict.fromkeys(row['evidence_refs']))
            row['unresolved_fields'] = list(dict.fromkeys(row['unresolved_fields']))
            projected.append(row)
        return canonical({'candidates': projected})
    if kind == 'solver':
        return canonical(document(text))
    raise ValueError('unsupported candidate role')


def project_snapshot_messages(messages, *, snapshot_sha256, values, facets):
    """Validate the assembled evidence against the pinned snapshot and deduplicate safely."""
    if len(messages) != 2 or not isinstance(messages[1].get('content'), list):
        raise ValueError('candidate input must use shared speech messages')
    result = copy.deepcopy(messages)
    parts, groups, seen = [], [], {}
    for part in result[1]['content']:
        if not isinstance(part, Mapping) or 'evidence_id' not in part:
            parts.append(part)
            continue
        eid, text, kind = part['evidence_id'], part.get('text'), part.get('knowledge_role')
        if kind == 'value':
            value = values.get(eid)
            if value is None or text != value.content:
                raise ValueError('rendered value differs from pinned snapshot')
            source = eid
            # Retain all distinct provenance/conditions: conservative exact dedup.
            scope = {'facets': list(value.associated_facet_ids), 'target': value.task_target,
                     'sources': list(value.source_sample_ids), 'provenance': list(value.provenance_refs)}
        elif kind == 'key' and eid.startswith('key:'):
            source = eid[4:]
            facet = facets.get(source)
            if facet is None or text != facet.representation:
                raise ValueError('rendered key differs from pinned snapshot')
            scope = {'facet': source}
        else:
            raise ValueError('unregistered evidence source in candidate handoff')
        if part.get('source_uri') != f'knowledge-snapshot://{snapshot_sha256}/{source}':
            raise ValueError('knowledge source URI differs from snapshot')
        identity = digest({'text': text, 'kind': kind, 'scope': scope})
        if identity in seen:
            groups[seen[identity]]['aliases'].append(eid)
            continue
        seen[identity] = len(groups)
        groups.append({'id': eid, 'aliases': [], 'kind': kind, 'text_sha256': _sha(text),
                       'scope_sha256': digest(scope)})
        parts.append(part)
    result[1]['content'] = parts
    return result, groups


def validate_trace_handoffs(receipt, records, snapshot_cache=None):
    """Recompute handoff identities from raw responses and the actual request payloads."""
    from .intent_slot_pipeline import candidates, digest
    block = receipt['intent_slot_pipeline']
    handoffs = block['node_handoffs']
    visits = receipt['role_visits']
    expected = ['perception'] + ['slot'] * len(block['slot_sources']) + ['solver']
    if [row['stage'] for row in visits] != expected or len(handoffs) != len(visits):
        raise ValueError('node handoff count differs from native visits')
    sample = None
    snapshot = None
    store_pins = None
    target_requests = []
    for index, (handoff, visit) in enumerate(zip(handoffs, visits)):
        body = dict(handoff)
        claimed = body.pop('handoff_sha256', None)
        if claimed != digest(body) or body.get('schema') != 'saea-sdk-node-handoff-v1' or body.get('policy') != POLICY:
            raise ValueError('node handoff digest/policy mismatch')
        contact = visit['execution_receipt']['final_contact']
        if body.get('target_request_id') != contact['request_id'] or body.get('target_request_sha256') != contact['request_sha256']:
            raise ValueError('node handoff target contact mismatch')
        request = records[contact['request_id'] + '/request']
        target_requests.append(contact['request_id'])
        payload = request['payload']
        if (request['payload_sha256'] != contact['request_sha256'] or digest(payload) != request['payload_sha256']
            or payload.get('sample_id') != body['sample_id']):
            raise ValueError('node handoff request/sample mismatch')
        sample = body['sample_id'] if sample is None else sample
        snapshot = body['snapshot_sha256'] if snapshot is None else snapshot
        store_pins = body['store_pins'] if store_pins is None else store_pins
        if body['store_pins'] != store_pins:
            raise ValueError('node handoff store pins differ')
        if any(pin.get('composite_snapshot_sha256') != snapshot for pin in store_pins.values()):
            raise ValueError('node handoff store pin composite differs')
        if body['sample_id'] != sample or body['snapshot_sha256'] != snapshot:
            raise ValueError('node handoff crossed sample or snapshot')
        if body['target_role'] != visit['role_id']:
            raise ValueError('node handoff target role mismatch')
        if index == 0:
            if body['source_role'] != 'bootstrap' or body['source_request_id'] is not None or body['candidate_ids']:
                raise ValueError('invalid bootstrap handoff')
        else:
            source = block['perception_source'] if visit['stage'] == 'slot' else block['slot_sources'][-1]['source']
            source_role = visits[0]['role_id'] if visit['stage'] == 'slot' else visits[-2]['role_id']
            source_text = records[source['request_id'] + '/response']['text']
            if (body['source_request_id'] != source['request_id'] or body['source_response_sha256'] != source['response_text_sha256']
                or body['source_role'] != source_role):
                raise ValueError('node handoff source mismatch')
            expected_ids = (block['slot_sources'][index-1]['candidate_ids'] if visit['stage'] == 'slot' else block['candidate_ids'])
            expected_branch = index-1 if visit['stage'] == 'slot' else 0
            if body['candidate_ids'] != expected_ids or body['branch_index'] != expected_branch:
                raise ValueError('node handoff candidate/branch mismatch')
            if visit['stage'] == 'slot':
                source_context = {}
                kind = 'perception'
            else:
                source_context = {'candidates': [{'candidate_id': cid} for cid in block['slot_sources'][-1]['candidate_ids']]}
                kind = 'slot'
            if block.get('coverage_recovery') and kind == 'slot':
                from .intent_slot_pipeline import canonical, document
                selected = block['slot_sources'][-1]['valid_ids']
                normalized = canonical({'candidates': [r for r in document(source_text)['candidates']
                                                       if r.get('candidate_id') in selected]})
            else:
                normalized = normalized_response(kind, source_text, source_context, {'max_candidates': 3})
            if body['normalized_sha256'] != _sha(normalized):
                raise ValueError('node handoff normalized source mismatch')
        evidence = payload.get('supplied_evidence', [])
        actual = [(v['evidence_id'], v.get('knowledge_role'), _sha(v['content']), v['source_uri']) for v in evidence]
        expected_evidence = [(g['id'], g['kind'], g['text_sha256'],
            'knowledge-snapshot://' + snapshot + '/' + (g['id'][4:] if g['kind'] == 'key' else g['id'])) for g in body['knowledge']]
        if actual != expected_evidence:
            raise ValueError('node handoff differs from supplied request evidence')
        projection = records.get(contact['request_id'] + '/native-evidence-projection', {}).get('org_native_evidence_projection')
        if projection is not None and projection.get('policy_id') == 'native-candidate-slot-view-v1':
            scope = projection.get('scope', {})
            if (projection.get('input_request_id') != contact['request_id']
                or projection.get('input_request_bytes_sha256') != contact['request_sha256']
                or scope.get('role_kind') != visit['stage'] and not (visit['stage'] == 'solver' and scope.get('role_kind') == 'solver')
                or (visit['stage'] in ('slot', 'solver') and (scope.get('candidate_ids') != body['candidate_ids'] or scope.get('fallback_reason') is not None))):
                raise ValueError('candidate projection scope differs from handoff')
    matching = [v for v in records.values() if isinstance(v, Mapping) and v.get('org_candidate_node_handoffs') == handoffs
                and v.get('org_sample_id') == sample and v.get('org_knowledge_snapshot_sha256') == snapshot
                and v.get('org_knowledge_store_pins') == store_pins]
    if len(matching) != 1:
        raise ValueError('node handoff trace record absent or differs')
    runtime_matches = [item for item in records.values() if isinstance(item, Mapping)
        and isinstance(item.get('use_contacts'), list)
        and {row.get('plan', {}).get('contact_id') for row in item['use_contacts']} >=
            {visit['execution_receipt']['final_contact']['agent_contact_id'] for visit in visits}]
    if len(runtime_matches) != 1 or runtime_matches[0].get('use_knowledge_snapshot_sha256') != snapshot:
        raise ValueError('handoff snapshot differs from runtime custody')
    runtime_by_contact = {row['plan']['contact_id']: row for row in runtime_matches[0]['use_contacts']}
    for handoff, visit in zip(handoffs, visits):
        source = runtime_by_contact[visit['execution_receipt']['final_contact']['agent_contact_id']]
        if source['query_state_sha256'] != handoff['query_state_sha256']:
            raise ValueError('handoff query state differs from runtime custody')
    from .knowledge import KnowledgeSnapshot
    document = matching[0].get('org_knowledge_snapshot')
    reference = matching[0].get('org_knowledge_snapshot_ref')
    if document is not None:
        verified = KnowledgeSnapshot.from_frozen_json(dict(document, snapshot_sha256=snapshot))
    else:
        if not isinstance(reference, Mapping) or reference.get('snapshot_sha256') != snapshot:
            raise ValueError('node handoff trace lacks pinned snapshot source')
        key = (reference.get('path'), snapshot)
        if snapshot_cache is None:
            verified = load_shared_snapshot(reference)
        else:
            if key not in snapshot_cache:
                verified = load_shared_snapshot(reference)
                snapshot_cache[key] = (verified, {value.value_id: value for value in verified.values},
                                       {facet.facet_id: facet for facet in verified.facets})
            verified = snapshot_cache[key][0]
    if store_pins:
        from .hashing import content_hash
        store_identity = content_hash({store_id: {'snapshot_sha256': pin['store_snapshot_sha256'],
            'release_sha256': pin['store_release_sha256']} for store_id, pin in store_pins.items()})
        if verified.snapshot_id != 'named-knowledge-stores:' + store_identity:
            raise ValueError('individual knowledge store pins differ from composite snapshot')
    if reference is not None and snapshot_cache is not None and document is None:
        _verified, values, facets = snapshot_cache[key]
    else:
        values = {value.value_id: value for value in verified.values}
        facets = {facet.facet_id: facet for facet in verified.facets}
    for handoff in handoffs:
        request = records[handoff['target_request_id'] + '/request']['payload']
        parts = [{'type':'input_audio'}] + [
            {'type':'text', 'text': row['content'], 'evidence_id': row['evidence_id'],
             'source_uri': row['source_uri'], 'knowledge_role': row.get('knowledge_role')}
            for row in request.get('supplied_evidence', [])]
        _messages, groups = project_snapshot_messages([{'role':'system','content':'snapshot'},
            {'role':'user','content': parts}], snapshot_sha256=snapshot, values=values, facets=facets)
        for source_group, audited in zip(handoff['knowledge'], groups):
            for alias in source_group.get('aliases', []):
                canonical_id = source_group['id']
                if source_group['kind'] == 'value':
                    left, right = values.get(canonical_id), values.get(alias)
                    if left is None or right is None or (left.content, left.associated_facet_ids, left.task_target, left.source_sample_ids, left.provenance_refs) != (right.content, right.associated_facet_ids, right.task_target, right.source_sample_ids, right.provenance_refs):
                        raise ValueError('node knowledge alias scope differs from snapshot')
                else:
                    raise ValueError('key aliases require identical facet identity')
            if {k:v for k,v in source_group.items() if k != 'aliases'} != {k:v for k,v in audited.items() if k != 'aliases'}:
                raise ValueError('node knowledge scope differs from pinned snapshot')
        if len(groups) != len(handoff['knowledge']):
            raise ValueError('node knowledge snapshot projection count differs')


def validate_inventory_against_request(inventory, request, candidate_ids):
    payload = request['payload']
    if inventory.get('sample_id') != payload.get('sample_id'):
        raise ValueError('evidence inventory sample mismatch')
    if inventory.get('candidate_ids') and inventory['candidate_ids'] != candidate_ids:
        raise ValueError('evidence inventory candidate scope mismatch')
    sources = inventory['sources']
    if not sources or sources[0]['source_id'] != payload.get('media_relpath'):
        raise ValueError('evidence audio source differs from request')
    supplied = payload.get('supplied_evidence', [])
    actual = [(row['evidence_id'], row['source_uri'], _sha(row['content'])) for row in supplied]
    expected = [(row['source_id'], row['source_uri'], row['source_sha256']) for row in sources[1:]]
    if actual != expected:
        raise ValueError('evidence inventory differs from actual request supply')


def persist_shared_snapshot(snapshot, root):
    """Durable immutable content-addressed source for all rows/checkpoints in a run."""
    import os
    import tempfile
    from pathlib import Path
    root = Path(root).resolve() / 'candidate-snapshot-sources-v1'
    root.mkdir(parents=True, exist_ok=True)
    identity = snapshot.content_hash()
    path = root / (identity + '.json')
    if not path.exists():
        data = (digestible_snapshot(snapshot)).encode('utf-8')
        fd, tmp = tempfile.mkstemp(prefix='.snapshot-', dir=root)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError:
                pass
            directory = os.open(root, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            os.unlink(tmp)
    # Existing objects are never overwritten; validate before any paid work.
    load_shared_snapshot({'schema':'saea-candidate-snapshot-source-v1', 'path':str(path), 'snapshot_sha256':identity})
    return {'schema':'saea-candidate-snapshot-source-v1', 'path':str(path), 'snapshot_sha256':identity}


def digestible_snapshot(snapshot):
    import json
    return json.dumps(snapshot.frozen_json(), ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def load_shared_snapshot(reference):
    import json
    from pathlib import Path
    from .knowledge import KnowledgeSnapshot
    if (not isinstance(reference, Mapping) or set(reference) != {'schema', 'path', 'snapshot_sha256'}
        or reference['schema'] != 'saea-candidate-snapshot-source-v1'):
        raise ValueError('invalid shared candidate snapshot reference')
    identity = reference['snapshot_sha256']
    path = Path(reference['path'])
    if (not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity)
        or not path.is_absolute() or path.name != identity + '.json' or path.parent.name != 'candidate-snapshot-sources-v1'):
        raise ValueError('invalid content-addressed snapshot path')
    try:
        snapshot = KnowledgeSnapshot.from_frozen_json(json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError('shared candidate snapshot unavailable or invalid') from error
    if snapshot.content_hash() != identity:
        raise ValueError('shared candidate snapshot hash mismatch')
    return snapshot
