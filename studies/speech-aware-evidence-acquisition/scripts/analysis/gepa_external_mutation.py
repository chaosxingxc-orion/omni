"""Verify an external proposal against its wire receipt and trusted actor parent.

No native teacher identity or FIT is invented. Actor tokenization/FIT remains in
existing pair compilation and execution. These checks do not prove semantics.
"""
from pathlib import Path
import json

import external_exploration as external
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha
from speech_aware_evidence_acquisition.rails.node_prompt_candidate import for_contract


def verify_external_mutation(spec, *, contract, parent):
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
