#!/usr/bin/env python3
"""Read one sealed generation output for coverage, without model or trace access."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SCHEMA = 'saea-sealed-coverage-reader-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def read_pin(pin):
    if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
        raise ValueError('exact input pin required')
    path = Path(pin['path'])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('input must be absolute regular file')
    raw = path.read_bytes()
    if sha(raw) != pin['sha256']:
        raise ValueError('input SHA256 differs: ' + str(path))
    return raw


def classify(text, pairs, types):
    if not isinstance(text, str):
        return 'nonempty_invalid_structure'
    if not text.strip():
        return 'empty_answer'
    try:
        doc = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return 'nonempty_invalid_structure'
    shape = (isinstance(doc, dict) and set(doc) == {'scenario', 'action', 'entities'}
             and all(isinstance(doc.get(k), str) and doc[k].strip() for k in ('scenario', 'action'))
             and isinstance(doc['entities'], list)
             and all(isinstance(e, dict) and set(e) == {'type', 'filler'}
                     and all(isinstance(e.get(k), str) and e[k].strip() for k in ('type', 'filler'))
                     for e in doc['entities']))
    if not shape:
        return 'nonempty_invalid_structure'
    if (doc['scenario'], doc['action']) not in pairs or any(e['type'] not in types for e in doc['entities']):
        return 'invalid_label_or_entity_contract'
    return 'contract_legal_nonempty_entities' if doc['entities'] else 'contract_legal_empty_entities'


def terminal_category(ep):
    term = ep.get('terminal_disposition') or {}
    intr = ep.get('interrupted_role') or {}
    stage = term.get('stage') or intr.get('stage') or 'unknown'
    stage = {'perception': 'P', 'slot': 'Slot', 'solver': 'Decision', 'decision': 'Decision'}.get(stage, 'unknown')
    reason = term.get('reason')
    refusal = term.get('seal_refusal') or intr.get('seal_refusal') or {}
    message = refusal.get('message', '')
    if reason == 'finite_completion' and term.get('finish_reason') == 'length':
        kind = 'length'
    elif reason == 'seal_refusal' and message == 'intent-slot output contract refused: evidence reference was not supplied uniquely to the current sample and candidate':
        kind = 'reference'
    elif reason == 'seal_refusal' and message in {
        'intent-slot output contract refused: entity type and filler must be nonempty text',
        'intent-slot output contract refused: pipeline response does not satisfy its node schema',
    }:
        kind = 'field_or_schema'
    elif reason in ('coverage_budget_exhausted', 'recovery_budget_exhausted'):
        kind = 'budget'
    else:
        kind = 'unknown'
    return stage + ':' + kind


def recovery_contacts(ep):
    """Return only observable extra contacts, deduped across all archived visits."""
    visits = list(ep.get('role_visits') or [])
    if ep.get('interrupted_role'):
        visits.append(ep['interrupted_role'])
    seen = {}
    for visit in visits:
        if not isinstance(visit, dict):
            raise ValueError('malformed role visit')
        for contact in (visit.get('execution_receipt') or {}).get('contacts', []):
            rid = contact.get('request_id')
            if not isinstance(rid, str) or not rid:
                raise ValueError('contact request identity missing')
            old = seen.get(rid)
            if old and old['contact'] != contact:
                raise ValueError('conflicting duplicate contact identity')
            seen[rid] = {'contact': contact, 'stage': visit.get('stage')}
    found = {}
    suffixes = {':perception-length-repair': 'perception_length',
                ':slot-length-repair': 'slot_length', ':answer-only': 'answer_only'}
    agent_to_request = {item['contact'].get('agent_contact_id'): rid for rid, item in seen.items()}
    completed_final = {
        (visit.get('execution_receipt') or {}).get('final_contact', {}).get('request_id')
        for visit in ep.get('role_visits') or []
    }
    for rid, item in seen.items():
        agent = item['contact'].get('agent_contact_id', '')
        for suffix, kind in suffixes.items():
            if isinstance(agent, str) and agent.endswith(suffix):
                parent_id = agent[:-len(suffix)]
                parent = agent_to_request.get(parent_id)
                found[rid] = {'request_id': rid, 'kind': kind, 'stage': item['stage'],
                              'parent_request_id': parent,
                              'parent_binding': 'contact_id_suffix' if parent else 'unknown', 'cost': 'unknown',
                              'local_candidate_recovered': 'unknown',
                              'stage_recovered': True if rid in completed_final else 'unknown'}
    block = ep.get('intent_slot_pipeline') or {}
    slots = block.get('slot_sources') or []
    pending = ep.get('slot_recovery_pending') or {}
    pending_sources = pending.get('sources') or []
    sources = slots if block.get('coverage_recovery') else pending_sources
    if len(sources) > 1:
        for source in sources[1:]:
            contact = source.get('source') or {}
            rid = contact.get('request_id')
            if not isinstance(rid, str):
                raise ValueError('slot recovery source identity missing')
            # Pending custody can contain an earlier accepted contact that is
            # absent from interrupted_role; keep it once with its known ID.
            if rid not in seen:
                seen[rid] = {'contact': contact, 'stage': 'slot'}
            if rid in found:
                found[rid]['local_candidate_recovered'] = bool(source.get('valid_ids')) if isinstance(source.get('valid_ids'), list) else 'unknown'
            else:
                found[rid] = {'request_id': rid, 'kind': 'slot_candidate', 'stage': 'slot',
                              'parent_binding': 'unknown', 'cost': 'unknown',
                              'local_candidate_recovered': bool(source.get('valid_ids')) if isinstance(source.get('valid_ids'), list) else 'unknown',
                              'stage_recovered': 'unknown'}
    interrupted = ep.get('interrupted_role') or {}
    interrupted_contacts = (interrupted.get('execution_receipt') or {}).get('contacts', [])
    # Multiple contacts are evidence of additional activity, but without a
    # recovery relation their roles cannot be inferred. Keep every unresolved ID.
    uncertain = []
    known_parents = {item.get('parent_request_id') for item in found.values()}
    for contact in interrupted_contacts:
        rid = contact['request_id']
        if rid not in found and rid not in known_parents:
            uncertain.append({'request_id': rid, 'stage': interrupted.get('stage'),
                              'attribution': 'unknown', 'cost': 'unknown'})
    return list(found.values()), uncertain


def verify_finalized_successor(config, manifest, output, ids):
    # Full verifier authenticates attempt state, predecessor checkpoint and kept rows.
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    bundle = verify_bundle(Path(config['manifest']['path']))
    pred = manifest.get('predecessor_attempt') or {}
    if (bundle.attempt_state != 'FINALIZED' or bundle.manifest != manifest
            or Path(bundle.outputs_path).resolve() != Path(config['outputs']['path']).resolve()
            or pred.get('mode') != 'checkpoint' or pred.get('run_id') != manifest.get('run_id')
            or pred.get('counts') != {'kept': 1080, 'flown': 0, 'partial': 0}
            or set(pred.get('kept_sample_ids', [])) != set(ids)
            or len(pred.get('kept_sample_ids', [])) != 1080
            or len(bundle.outputs_rows) != 1080 or set(bundle.split_ids) != set(ids)):
        raise ValueError('finalized full-kept successor identity differs')
    return pred['attempt_id']


def evaluate(config):
    required = {'schema', 'manifest', 'outputs', 'roster', 'inventory', 'scores', 'expected_rows', 'out'}
    successor = config.get('schema') == 'saea-sealed-coverage-successor-v1'
    if successor:
        required = required | {'lock'}
    if set(config) != required or config['schema'] not in (SCHEMA, 'saea-sealed-coverage-successor-v1') or config['expected_rows'] != 1080:
        raise ValueError('exact 1080-row coverage configuration required')
    manifest = json.loads(read_pin(config['manifest']))
    output = read_pin(config['outputs'])
    roster = json.loads(read_pin(config['roster']))
    inventory = json.loads(read_pin(config['inventory']))
    if config['scores'] is not None:
        raise ValueError('scores not supported; use existing paired analysis')
    if (manifest.get('status') != 'COMPLETED' or manifest.get('outputs', {}).get('sha256') != sha(output)
            or (not successor and (manifest.get('row_checkpoint', {}).get('complete_rows') != 1080
            or manifest.get('row_checkpoint', {}).get('unrendered_rows') != 0))):
        raise ValueError('sealed generation output identity differs')
    split = manifest.get('split') or {}
    ids = split.get('ids')
    rid = [v['sample_id'] for v in roster['items']]
    if (not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids) or len(ids) != 1080 or len(set(ids)) != 1080
            or len(rid) != 1080 or len(set(rid)) != 1080 or ids != rid
            or split.get('count') != 1080 or split.get('identity_hash') != roster.get('split_identity_hash')):
        raise ValueError('manifest and locked roster differ')
    if successor:
        from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_load_bearing_surface
        generation = Path('/opt/saea-release/studies/speech-aware-evidence-acquisition')
        commit, _ = require_clean_load_bearing_surface(generation)
        if (commit != manifest.get('study_commit')
                or config['lock']['path'] != '/opt/saea-release/docs/datasets.lock.json'):
            raise ValueError('original generation release lock identity differs')
        lock_sha = sha(read_pin(config['lock']))
    else:
        lock_sha = manifest.get('cost', {}).get('umbrella_lock_sha256')
    if lock_sha != inventory.get('lock_sha256'):
        raise ValueError('inventory dataset lock differs')
    journal_attempt = verify_finalized_successor(config, manifest, output, ids) if successor else manifest['attempt']['attempt_id']
    pairs = {tuple(x) for x in inventory['pairs']}
    types = set(inventory['entity_types'])
    counts, failures, kinds = Counter(), Counter(), Counter()
    samples, contacts, uncertain_contacts = set(), [], []
    for line in output.splitlines():
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=unique)
        sid = row['sample_id']
        if sid in samples:
            raise ValueError('duplicate sample')
        samples.add(sid)
        category = classify(row.get('text'), pairs, types)
        counts[category] += 1
        ep = row.get('native_episode_receipt') or {}
        if category not in ('contract_legal_nonempty_entities', 'contract_legal_empty_entities'):
            failures[terminal_category(ep)] += 1
        recovered, uncertain = recovery_contacts(ep)
        for contact in uncertain:
            contact['sample_id'] = sid
            uncertain_contacts.append(contact)
        for contact in recovered:
            contact['sample_id'] = sid
            contact['terminal_contract_legal'] = category.startswith('contract_legal_')
            contact['final_correctness'] = 'unknown'
            kinds[contact['kind']] += 1
            contacts.append(contact)
    if len(samples) != 1080 or samples != set(ids):
        raise ValueError('outputs missing or extra samples')
    for contact in contacts + uncertain_contacts:
        contact['journal_locator'] = {
            'relative_directory': 'contact-journals/' + manifest['run_id'] + '/' + journal_attempt,
            'intent_filename': sha(contact['request_id'].encode()) + '.intent.json',
            'response_filename': sha(contact['request_id'].encode()) + '.response.json',
            'binding_status': 'candidate paths only; verify journal identity and hashes before use',
        }
    return {'schema': SCHEMA, 'run_id': manifest['run_id'], 'total_rows': 1080,
            'input_sha256': {k: config[k]['sha256'] for k in ('manifest', 'outputs', 'roster', 'inventory')},
            'journal_source_attempt': journal_attempt, 'partitions': dict(counts), 'terminal_failures': dict(failures),
            'observable_recovery_contacts': len(contacts), 'recovery_kinds': dict(kinds),
            'recovery_outcomes': {
                'local_candidate_recovered': sum(c['local_candidate_recovered'] is True for c in contacts),
                'local_candidate_not_recovered': sum(c['local_candidate_recovered'] is False for c in contacts),
                'local_candidate_unknown': sum(c['local_candidate_recovered'] == 'unknown' for c in contacts),
                'stage_recovered': sum(c['stage_recovered'] is True for c in contacts),
                'stage_recovery_unknown': sum(c['stage_recovered'] == 'unknown' for c in contacts),
                'stage_recovery_unit': 'recovery contacts; multiple contacts may belong to one stage',
                'terminal_contract_legal': sum(c['terminal_contract_legal'] for c in contacts),
                'final_correctness_unknown': len(contacts),
            },
            'recovery_costs': {'status': 'unknown', 'reason': 'no bound per-contact COST or journal input'},
            'recovery_contacts': contacts,
            'unknown_attribution_contact_count': len(uncertain_contacts),
            'unknown_attribution_contacts': uncertain_contacts,
            'limitations': ['Only explicitly observable recovery contacts; interrupted slot candidate retries may be unknown.',
                            'Recovery parent binding and per-contact costs require separately verified evidence.',
                            'No model accuracy or final correctness inferred.',
                            ('Successor bundle and predecessor were verified.' if successor else 'This reader validates pins and coverage, not a full bundle.')],
            'new_model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_bytes())
    out = Path(config['out'])
    if not out.is_absolute() or out.exists():
        raise ValueError('fresh absolute output required')
    result = evaluate(config)
    inputs = [config[k]['path'] for k in ('manifest', 'outputs', 'roster', 'inventory')]
    if 'lock' in config: inputs.append(config['lock']['path'])
    if any(out.resolve().parent == Path(path).resolve().parent for path in inputs):
        raise ValueError('output must be in an independent analysis directory')
    with out.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        import os
        os.fsync(handle.fileno())
    print(json.dumps({'output': str(out), 'sha256': sha(out.read_bytes()), 'rows': result['total_rows']}))


if __name__ == '__main__':
    main()
