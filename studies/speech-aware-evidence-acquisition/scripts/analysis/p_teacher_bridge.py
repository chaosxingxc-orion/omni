"""Compile a pinned GEPA wire response into a P-only template and patch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import external_exploration as external
from gepa_external_mutation import verify_external_mutation
from speech_aware_evidence_acquisition.rails.node_prompt_candidate import for_contract, sha
from speech_aware_evidence_acquisition.rails.gepa_prompt_material import publish_template


def collected_proposal(spec, context, actor_context):
    import copy
    import p_teacher_proposals as proposals
    if set(spec) != {'schema', 'prepared', 'collected', 'candidate_index', 'request_sha256', 'response_sha256'}:
        raise ValueError('exact collected source required')
    prepared, collected = external.checked(spec['prepared']), external.checked(spec['collected'])
    if prepared.name != 'prepared.json' or collected != prepared.parent / 'collected.json':
        raise ValueError('original collected directory required')
    root, manifest, caller, request = external.load_prepared(prepared.parent)
    record, profile = external.read(collected), external.read(root / 'profile.json')
    observed = caller.reconcile(request)
    if (observed is None or observed != record.get('response') or sha(request) != spec['request_sha256']
            or record.get('request_sha256') != spec['request_sha256']
            or record.get('prepared_sha256') != spec['prepared']['sha256']
            or observed['response']['wire_response_sha256'] != spec['response_sha256']
            or observed['response']['finish_reason'] != 'stop' or record.get('candidate_accepted') is not False):
        raise ValueError('collected wire custody differs')
    payload = external.request_payload(request)
    if payload['context'] != context or external.read(root / 'context.json') != context:
        raise ValueError('request context differs')
    source = external.read(external.checked(manifest['source']))
    if payload['groups'] != source['document']['groups']:
        raise ValueError('source groups differ')
    doc = external.validate_proposals(observed['response']['text'], profile, context, [g['query_id'] for g in payload['groups']])
    if doc != record['proposals']:
        raise ValueError('collected proposals differ')
    import jsonschema
    jsonschema.Draft202012Validator(context['constraints']['teacher_output_schema']).validate(doc)
    index = spec['candidate_index']
    if type(index) is not int or not 0 <= index < len(doc['candidates']): raise ValueError('invalid candidate index')
    candidate = copy.deepcopy(doc['candidates'][index])
    if profile['algorithm'] != 'reflexion': raise ValueError('reflexion source required')
    mutation = candidate['memory']['prompt_mutation']
    original = copy.deepcopy(mutation)
    mutation['evidence_query_ids'] = sorted(set(mutation['evidence_query_ids']))
    proposals.compile_one(candidate, method='reflexion', axis='consumption', context=context, parent_context=actor_context, source=source)
    response = observed['response']
    return {'mutation_text': external.canonical(mutation).decode(), 'receipt_kind': 'external-reflexion-collected-v1',
            'request_sha256': sha(request), 'response_sha256': response['wire_response_sha256'], 'usage': response['usage'],
            'provider_usage': response['provider_usage'], 'candidate_index': index, 'source_collected_sha256': spec['collected']['sha256'],
            'normalization': {'original_sha256': sha(original), 'normalized_sha256': sha(mutation), 'sorted_evidence_only': True},
            'usage_scope': 'whole proposal request shared by sibling candidates; count once by request hash',
            'candidate_accepted': False, 'native_fit_completed': False}


def compile_ace(cfg):
    import p_teacher_proposals as proposals
    import jsonschema
    if set(cfg) != {'schema', 'prepared', 'collected', 'request_sha256', 'response_sha256', 'output_dir'}:
        raise ValueError('exact ACE config required')
    prepared, collected = external.checked(cfg['prepared']), external.checked(cfg['collected'])
    if prepared.name != 'prepared.json' or collected != prepared.parent / 'collected.json': raise ValueError('original directory required')
    root, manifest, caller, request = external.load_prepared(prepared.parent)
    record, context, profile = external.read(collected), external.read(root / 'context.json'), external.read(root / 'profile.json')
    observed = caller.reconcile(request)
    if (observed is None or observed != record.get('response') or sha(request) != cfg['request_sha256']
        or record.get('request_sha256') != cfg['request_sha256'] or record.get('prepared_sha256') != cfg['prepared']['sha256']
        or observed['response']['wire_response_sha256'] != cfg['response_sha256'] or observed['response']['finish_reason'] != 'stop'
        or record.get('candidate_accepted') is not False or profile['algorithm'] != 'ace'):
        raise ValueError('ACE wire custody differs')
    payload = external.request_payload(request)
    source = external.read(external.checked(manifest['source']))
    if payload['context'] != context or payload['groups'] != source['document']['groups']: raise ValueError('ACE source differs')
    doc = external.validate_proposals(observed['response']['text'], profile, context, [g['query_id'] for g in payload['groups']])
    if doc != record['proposals']: raise ValueError('ACE proposals differ')
    jsonschema.Draft202012Validator(context['constraints']['teacher_output_schema']).validate(doc)
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, candidate in enumerate(doc['candidates']):
        compiled = proposals.compile_one(candidate, method='ace', axis='consumption', context=context, parent_context=context, source=source)
        edits = [edit for operation in candidate['operations'] for edit in operation['edits']]
        patch = []
        for edit in edits:
            path = '/knowledge_stores/intent-boundaries/phase_supply/perception/' + edit['field']
            patch.extend([{'op': 'test', 'path': path, 'value': edit['old']}, {'op': 'replace', 'path': path, 'value': edit['new']}])
        result = {'schema': 'saea-p-ace-retrieval-patch-v1', 'candidate_id': candidate['id'], 'patch': patch,
                  'source_roles': compiled['source_roles'], 'source_candidate_sha256': sha(candidate),
                  'request_sha256': sha(request), 'response_sha256': observed['response']['wire_response_sha256'],
                  'usage': observed['response']['usage'], 'provider_usage': observed['response']['provider_usage'],
                  'usage_scope': 'count once by request hash', 'supply_ready': False, 'native_fit_completed': False}
        path = out / ('candidate-' + str(i) + '.json')
        data = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
        if path.exists() and path.read_bytes() != data: raise ValueError('output differs')
        if not path.exists():
            from speech_aware_evidence_acquisition.core.atomic_file import publish_new
            publish_new(path, data)
        results.append(result)
    return results


def compile_config(cfg):
    if set(cfg) != {'schema', 'source', 'tokenizer_identity', 'tokenizer_dir', 'output_dir', 'parent_template', 'actor_context'} or cfg['schema'] != 'saea-p-teacher-bridge-v1':
        raise ValueError('exact bridge configuration required')
    prepared = external.checked(cfg['source']['prepared'])
    context = external.read(prepared.parent / 'context.json')
    actor_context = external.read(external.checked(cfg['actor_context']))
    contract, parent = actor_context['contract'], actor_context['parent']
    if context['parent'] != parent or context['contract']['contract_sha256'] != contract['contract_sha256']:
        raise ValueError('actor context differs')
    if cfg['source'].get('schema') == 'saea-p-collected-proposal-v1':
        receipt = collected_proposal(cfg['source'], context, actor_context)
    else:
        receipt = verify_external_mutation(cfg['source'], contract=contract, parent=parent)
    identity = external.read(external.checked(cfg['tokenizer_identity']))
    tokenizer_dir = Path(cfg['tokenizer_dir'])
    for name, digest in identity['files'].items():
        if hashlib.sha256((tokenizer_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError('tokenizer bytes changed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    engine = for_contract(contract)
    child = engine.derive_candidate(receipt['mutation_text'], parent=parent, expected_parent_sha256=parent['candidate_sha256'], contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
    out = Path(cfg['output_dir']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    material = publish_template(out, candidate=child, contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count, tokenizer_identity=identity, expected_tokenizer_sha256=contract['frozen']['tokenizer_sha256'], parent=parent, expected_parent_sha256=parent['candidate_sha256'], mutation_text=receipt['mutation_text'])
    old = cfg['parent_template']
    parent_text = engine.executable_template(parent, contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
    if set(old) != {'relpath', 'sha256'} or old['sha256'] != hashlib.sha256(parent_text.encode()).hexdigest():
        raise ValueError('parent template pin differs')
    pin = dict(material['instruction_template'])
    pin['relpath'] = 'configs/' + pin['relpath']
    patch = [{'op': 'test', 'path': '/perception_policy/instruction_template', 'value': old}, {'op': 'replace', 'path': '/perception_policy/instruction_template', 'value': pin}]
    result = {'schema': 'saea-p-teacher-bridge-result-v1', 'source_receipt': receipt, 'material': material, 'patch': patch, 'candidate': child, 'new_api_calls': 0, 'new_npu_calls': 0}
    for name, value in [('result.json', result), ('patch.json', patch)]:
        path = out / name
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
        if path.exists() and path.read_bytes() != data:
            raise ValueError('output already exists with different content')
        if not path.exists():
            from speech_aware_evidence_acquisition.core.atomic_file import publish_new
            publish_new(path, data)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    cfg = external.read(args.config)
    if cfg.get('schema') == 'saea-p-ace-bridge-v1':
        result = compile_ace(cfg)
        print(json.dumps({'candidates': len(result), 'supply_ready': False, 'new_api_calls': 0}))
    else:
        result = compile_config(cfg)
        print(json.dumps({'candidate_sha256': result['candidate']['candidate_sha256'], 'material': result['material'], 'new_api_calls': 0}))

if __name__ == '__main__':
    main()
