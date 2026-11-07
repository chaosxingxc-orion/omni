"""Verify an external proposal against its wire receipt and trusted actor parent.

No native teacher identity or FIT is invented. Actor tokenization/FIT remains in
existing pair compilation and execution. These checks do not prove semantics.
"""
from pathlib import Path
import json
import copy
import re

import external_exploration as external
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha
from speech_aware_evidence_acquisition.rails.node_prompt_candidate import for_contract, QUERY_ID


def normalize_citations(candidate, groups):
    """Expand only uniquely resolved diagnosis IDs and complete cited metadata."""
    ids = [g['query_id'] for g in groups]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate source query IDs')
    changes = []
    def resolve(value):
        matches = [q for q in ids if q == value or q.startswith(value + ':')]
        if len(matches) != 1:
            raise ValueError('unknown or ambiguous citation')
        return matches[0]
    def replace(match):
        before = match.group(0)
        ref = before.rstrip('.')
        after = resolve(ref) + before[len(ref):]
        changes.append({'span': [match.start(), match.end()], 'before': before, 'after': after})
        return after
    result = copy.deepcopy(candidate)
    result['diagnosis'] = re.sub(QUERY_ID, replace, candidate['diagnosis'], flags=re.I)
    original = candidate['evidence_query_ids']
    if not isinstance(original, list) or any(q not in ids for q in original):
        raise ValueError('unknown evidence metadata')
    result['evidence_query_ids'] = sorted(set(original) | {resolve(c['before'].rstrip('.')) for c in changes})
    return result, {'schema': 'saea-gepa-citation-proof-v1', 'changes': changes,
        'original_candidate_sha256': sha(candidate), 'derived_candidate_sha256': sha(result),
        'original_evidence_query_ids': original, 'derived_evidence_query_ids': result['evidence_query_ids']}


def verify_derived(spec, *, contract, parent):
    fields = {'schema', 'prepared', 'candidate_index', 'request_sha256', 'response_sha256'}
    if set(spec) != fields or spec['schema'] != 'saea-gepa-derived-citation-v1':
        raise ValueError('exact derived citation source required')
    prepared = external.checked(spec['prepared'])
    if prepared.name != 'prepared.json':
        raise ValueError('original prepared manifest required')
    root, _, caller, request = external.load_prepared(prepared.parent)
    payload = external.request_payload(request)
    context, profile = external.read(root / 'context.json'), external.read(root / 'profile.json')
    if (context.get('parent') != parent or context.get('contract') != contract
            or payload['context'] != external.request_context(context, 'gepa')
            or payload['algorithm'] != 'gepa' or profile['algorithm'] != 'gepa'
            or sha(request) != spec['request_sha256']):
        raise ValueError('request, contract or parent differs')
    observed = caller.reconcile(request)
    if observed is None or observed['response']['wire_response_sha256'] != spec['response_sha256']:
        raise ValueError('unknown or altered response')
    response = observed['response']
    if response['finish_reason'] != 'stop':
        raise ValueError('external proposal generation incomplete')
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d: raise ValueError('duplicate proposal key')
            d[k] = v
        return d
    def invalid(_): raise ValueError('nonfinite JSON constant')
    text = response['text'].strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```json\r?\n(.*?)\r?\n```', text, re.DOTALL)
        if not match or '```' in match.group(1): raise ValueError('invalid JSON fence')
        text = match.group(1)
    doc = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    candidates = doc.get('candidates') if isinstance(doc, dict) else None
    if (not isinstance(doc, dict) or set(doc) != {'algorithm', 'candidates'} or doc['algorithm'] != 'gepa'
            or not isinstance(candidates, list) or not profile['min_candidates'] <= len(candidates) <= profile['max_candidates']
            or len({sha(c) for c in candidates}) != len(candidates)):
        raise ValueError('invalid proposal envelope')
    index = spec['candidate_index']
    if type(index) is not int or not 0 <= index < len(candidates):
        raise ValueError('valid explicit candidate index required')
    derived, proof = normalize_citations(candidates[index], payload['groups'])
    single = {'algorithm': 'gepa', 'candidates': [derived]}
    external.validate_proposals(canonical(single).decode(), dict(profile, min_candidates=1, max_candidates=1), context,
                                [g['query_id'] for g in payload['groups']])
    return {'mutation_text': canonical(derived).decode(), 'response_sha256': response['wire_response_sha256'],
            'contract_sha256': contract['contract_sha256'], 'parent_sha256': parent['candidate_sha256'],
            'usage': response['usage'], 'provider_usage': response['provider_usage'],
            'receipt_kind': 'external-chat-derived-citation-v1', 'teacher_model_id': response['model_id'],
            'request_sha256': sha(request), 'source_prepared_sha256': spec['prepared']['sha256'],
            'candidate_index': index, 'citation_proof': proof,
            'usage_scope': 'whole proposal request shared by sibling candidates; count once by request hash',
            'candidate_accepted': False, 'native_fit_completed': False}


def verify_external_mutation(spec, *, contract, parent):
    if isinstance(spec, dict) and spec.get('schema') == 'saea-gepa-derived-citation-v1':
        return verify_derived(spec, contract=contract, parent=parent)
    if not isinstance(spec, dict) or set(spec) != {'schema', 'prepared', 'collected', 'candidate_index'} or spec['schema'] != 'saea-gepa-external-mutation-v1':
        raise ValueError('exact external mutation source required')
    prepared = external.checked(spec['prepared'])
    collected = external.checked(spec['collected'])
    if prepared.name != 'prepared.json' or collected != prepared.parent / 'collected.json':
        raise ValueError('prepared and collected must share their original directory')
    root, manifest, caller, request = external.load_prepared(prepared.parent)
    record = external.read(collected)
    if record.get('schema') != 'saea-external-exploration-collected-v1' or record.get('prepared_sha256') != spec['prepared']['sha256']:
        raise ValueError('external collected preparation differs')
    payload = external.request_payload(request)
    context = external.read(root / 'context.json')
    if payload['context'] != external.request_context(context, 'gepa') or context.get('parent') != parent or context.get('contract') != contract:
        raise ValueError('external request belongs to another actor parent or contract')
    profile = external.read(root / 'profile.json')
    if profile['algorithm'] != 'gepa' or payload['algorithm'] != 'gepa':
        raise ValueError('GEPA external source required')
    observed = caller.reconcile(request)
    if observed is None or observed != record.get('response') or sha(request) != record.get('request_sha256'):
        raise ValueError('external response custody differs or is unknown')
    if observed['response']['finish_reason'] != 'stop':
        raise ValueError('external proposal generation incomplete')
    proposals = external.validate_proposals(observed['response']['text'], profile, context,
                                           [g['query_id'] for g in payload['groups']])
    if proposals != record.get('proposals') or record.get('candidate_accepted') is not False:
        raise ValueError('collected proposals differ from actual external response')
    index = spec['candidate_index']
    if type(index) is not int or not 0 <= index < len(proposals['candidates']):
        raise ValueError('valid explicit candidate index required')
    mutation_text = canonical(proposals['candidates'][index]).decode()
    for_contract(contract).validate_mutation(mutation_text, contract=contract,
                      expected_contract_sha256=contract['contract_sha256'],
                      expected_parent_sha256=parent['candidate_sha256'])
    response = observed['response']
    return {'mutation_text': mutation_text, 'response_sha256': response['wire_response_sha256'],
            'contract_sha256': contract['contract_sha256'], 'parent_sha256': parent['candidate_sha256'],
            'usage': response['usage'], 'provider_usage': response['provider_usage'],
            'receipt_kind': 'external-chat-v1', 'teacher_model_id': response['model_id'],
            'request_sha256': sha(request), 'source_collected_sha256': spec['collected']['sha256'],
            'candidate_index': index, 'usage_scope': 'whole proposal request shared by sibling candidates; count once by request hash',
            'candidate_accepted': False, 'native_fit_completed': False}
