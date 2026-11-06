"""Compile a source-bound GEPA v2 node into a NodeWorkload policy, offline.

Run in a separate writable configuration tree, never against a running tree.
External proposal custody is verified by the existing GEPA receipt verifier.
"""
import argparse
import copy
import json
from pathlib import Path

from gepa_external_mutation import verify_external_mutation
from intent_slot_workload import NodeWorkload
from speech_aware_evidence_acquisition.rails.gepa_prompt_material import publish_template
from speech_aware_evidence_acquisition.rails.node_prompt_candidate import PromptCandidateContract, sha
from speech_aware_evidence_acquisition.rails.tokenizer_runtime import load_text_token_counter
from tfrl_hardset_a0_workload import confined, sha as file_sha
from tfrl_hardset_a0_registration import pinned

PIPELINE = {'protocol_version': 'v2', 'node_rails': 'sdk-handoff-v1',
            'answer_only_fallback': 'budgeted-v1'}
HEADERS = {
    'slot': 'Extract separately for every supplied candidate; use only speech-supported entities.',
    'decision': 'Compare the evaluated candidates and select only an existing candidate ID.',
}


def prepare(repo, data, workload, node, tokenizer, feedback_ids, excluded_ids):
    if node not in HEADERS:
        raise ValueError('v2 target must be slot or decision')
    if workload.get('node_arm') != 'enhanced-p-slot-decision' or workload.get('pipeline') != PIPELINE:
        raise ValueError('GEPA v2 requires explicit v2 SDK pipeline')
    if node + '_policy' in workload:
        raise ValueError('target policy must be absent from the source workload')
    job = NodeWorkload(workload)
    baseline, _, _, _, binding = job.derive(repo, data, workload['material'])
    role = 'slot' if node == 'slot' else 'solver'
    path = confined(repo / 'configs', baseline['phase_material'][role]['instruction_template'])
    original = path.read_text(encoding='utf-8')
    if node == 'decision' and baseline['phase_material'][role]['response_format_id'] != 'slurp-solver-paired-label-json-v1':
        raise ValueError('unexpected Decision response format')
    # The uneditable original remains intact, including response format and handoff rules.
    engine = PromptCandidateContract(node)
    templates = {name: file_sha(confined(repo / 'configs', material['instruction_template']))
                 for name, material in baseline['phase_material'].items()}
    model_path = confined(repo, 'configs/model/' + binding['model_fragment'] + '.json')
    frozen = {'knowledge_sha256': sha(baseline['knowledge_stores']),
              'supply_sha256': sha({'knowledge_stores': baseline['knowledge_stores'],
                                    'projection': baseline.get('native_evidence_projection'),
                                    'projection_params': baseline.get('native_evidence_projection_params')}),
              'baseline_sha256': sha({'baseline': baseline, 'phase_template_sha256': templates}),
              'model_sha256': sha({'binding': binding, 'model_sha256': file_sha(model_path)}),
              'tokenizer_sha256': sha(tokenizer)}
    contract = engine.build_contract(base_template=engine.BLOCK_START + HEADERS[node] + engine.BLOCK_END + '\n\n' + original,
        frozen=frozen, feedback_query_ids=feedback_ids, excluded_query_ids=excluded_ids)
    return engine, contract, baseline


def compile_node(repo, data, workload, node, *, contract, expected_contract_sha256,
                 parent, expected_parent_sha256, external_mutation, count_tokens, tokenizer):
    engine, actual, baseline = prepare(repo, data, workload, node, tokenizer,
                                       contract['feedback_query_ids'], contract['excluded_query_ids'])
    if actual != contract or expected_contract_sha256 != actual['contract_sha256']:
        raise ValueError('pinned contract differs from derived workload')
    if parent.get('candidate_sha256') != expected_parent_sha256:
        raise ValueError('externally pinned parent differs')
    receipt = verify_external_mutation(external_mutation, contract=contract, parent=parent)
    child = engine.derive_candidate(receipt['mutation_text'], parent=parent,
        expected_parent_sha256=expected_parent_sha256, contract=contract,
        expected_contract_sha256=expected_contract_sha256, count_tokens=count_tokens)
    publication = publish_template(repo, candidate=child, parent=parent,
        expected_parent_sha256=expected_parent_sha256, mutation_text=receipt['mutation_text'],
        contract=contract, expected_contract_sha256=expected_contract_sha256,
        count_tokens=count_tokens, tokenizer_identity=tokenizer,
        expected_tokenizer_sha256=contract['frozen']['tokenizer_sha256'])
    result = copy.deepcopy(workload)
    role = 'slot' if node == 'slot' else 'solver'
    template = publication['instruction_template']
    result[node + '_policy'] = {'thinking': copy.deepcopy(baseline['thinking_policy'][role]),
        'instruction_template': {'relpath': 'configs/' + template['relpath'], 'sha256': template['sha256']}}
    # Derive again to exercise exactly the same path used by the registered workload.
    compiled = NodeWorkload(result).derive(repo, data, result['material'])[0]
    expected = copy.deepcopy(baseline)
    expected['phase_material'][role]['instruction_template'] = template['relpath']
    if compiled != expected:
        raise ValueError('candidate changed frozen workload beyond target template')
    return {'workload': result, 'candidate': child, 'publication': publication,
            'source_receipt': receipt, 'compiled_baseline_sha256': sha(compiled),
            'new_model_calls': 0, 'native_fit_completed': False, 'candidate_accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('seed', 'compile'))
    for name in ('repo', 'data', 'config', 'tokenizer', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    required = {'schema', 'node', 'workload', 'feedback_query_ids', 'excluded_query_ids'}
    if args.mode == 'compile':
        required |= {'contract', 'parent', 'external_mutation'}
    if set(spec) != required or spec['schema'] != 'saea-gepa-v2-node-adapter-v1':
        raise ValueError('exact adapter configuration required')
    repo, data = args.repo.resolve(strict=True), args.data.resolve(strict=True)
    workload = pinned(repo, spec['workload'])
    count, tokenizer = load_text_token_counter(args.tokenizer.resolve(strict=True))
    try:
        if args.mode == 'seed':
            engine, contract, _ = prepare(repo, data, workload, spec['node'], tokenizer,
                spec['feedback_query_ids'], spec['excluded_query_ids'])
            result = {'contract': contract, 'parent': engine.seed_candidate(contract=contract,
                expected_contract_sha256=contract['contract_sha256'], count_tokens=count),
                'tokenizer_identity': tokenizer, 'new_model_calls': 0}
        else:
            contract = pinned(repo, spec['contract'])
            parent = pinned(repo, spec['parent'])
            if (contract['feedback_query_ids'] != spec['feedback_query_ids'] or
                    contract['excluded_query_ids'] != spec['excluded_query_ids']):
                raise ValueError('configured partition differs')
            result = compile_node(repo, data, workload, spec['node'], contract=contract,
                expected_contract_sha256=contract['contract_sha256'], parent=parent,
                expected_parent_sha256=parent['candidate_sha256'], external_mutation=spec['external_mutation'],
                count_tokens=count, tokenizer=tokenizer)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write('\n')
    finally:
        if hasattr(count, 'close'):
            count.close()


if __name__ == '__main__':
    main()
