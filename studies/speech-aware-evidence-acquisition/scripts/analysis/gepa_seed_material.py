"""Build a source-pinned GEPA seed and native probe template on CPU only.

The seed reproduces the original Solver bytes. This prepares a mechanism probe,
not a learned candidate, native FIT, admission, or a completed GEPA search.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.rails import solver_prompt_candidate as prompt
from speech_aware_evidence_acquisition.rails.gepa_prompt_material import compile_template_probe
from speech_aware_evidence_acquisition.rails.tokenizer_runtime import load_text_token_counter
from tfgrpo_dense_h2_paired import cumulative_supply_bound
from tfrl_h1_observation import population, runtime_binding, H1, H2, DATASET, THINKING_ARMS
from tfrl_odd40_cells import raw, read, sha, confined

SCHEMA = 'saea-gepa-seed-material-v1'


def read_pin(repo, data, value):
    if not isinstance(value, dict) or set(value) != {'root', 'relpath', 'sha256'} or value['root'] not in ('repo', 'data'):
        raise ValueError('explicit rooted source pin required')
    root = repo if value['root'] == 'repo' else data
    path = confined(root, value['relpath'])
    if sha(path) != value['sha256']:
        raise ValueError('source pin changed: ' + value['relpath'])
    return path, read(path)


def pin(repo, path):
    return {'root': 'repo', 'relpath': str(path.relative_to(repo)), 'sha256': sha(path)}


def fixed_partition(repo, partition):
    h1, _ = population(repo)
    h2 = read(repo/H2)
    test = read(repo/'configs/dataset/slurp-speech-native-test-full.json')
    identity = {k: v for k, v in partition.items() if k not in ('partition_sha256', 'input_files_sha256')}
    if prompt.sha(identity) != partition.get('partition_sha256'):
        raise ValueError('partition content identity changed')
    expected_paths = [H1, H2, f'configs/dataset/{DATASET}.json', 'configs/dataset/slurp-speech-native-test-full.json']
    if partition.get('input_files_sha256') != {p: sha(repo/p) for p in expected_paths}:
        raise ValueError('partition fixed input identity changed')
    feedback, selection = partition['feedback_ids'], partition['selection_ids']
    h1_ids = {r['sample_id'] for r in h1['items']}
    excluded = {r['sample_id'] for r in h2['items']} | {r['sample_id'] for r in test['items']} | set(selection)
    if (len(feedback) != 233 or len(selection) != 80 or len(set(feedback)) != 233
            or len(set(selection)) != 80 or set(feedback) & set(selection)
            or set(feedback) | set(selection) != h1_ids or set(feedback) & excluded):
        raise ValueError('exact existing233/80 partition required')
    return feedback[:16], sorted(excluded)


def frozen_fields(repo, base, model, tokenizer):
    phases = base['phase_material']
    templates = {role: {'definition': material,
        'template_sha256': sha(confined(repo/'configs', material['instruction_template']))}
        for role, material in phases.items()}
    return {
        'knowledge_sha256': prompt.sha(base['knowledge_stores']),
        'perception_sha256': prompt.sha({'material': templates['perception'],
            'role': base['native_episode']['roles']['perception'],
            'thinking': base['thinking_policy']['perception']}),
        'supply_sha256': prompt.sha({'knowledge_stores': base['knowledge_stores'],
            'preparation_source_receipts': base.get('preparation_source_receipts', [])}),
        'model_sha256': prompt.sha(model),
        'decoding_sha256': prompt.sha({'model': {k:v for k,v in model.items() if k in (
            'decoding_params', 'reasoning_budget_tokens', 'answer_reserve_tokens', 'enable_thinking',
            'reasoning_context_policy', 'forced_close_message_sha256', 'prompt_order', 'reasoning_content_guard')},
            'thinking_policy': base['thinking_policy']}),
        'roles_sha256': prompt.sha({'native_episode': base['native_episode'], 'phase_material': templates}),
        'tools_sha256': prompt.sha(base.get('phase_tools', {})),
        'output_schema_sha256': prompt.sha({'solver_template_sha256': templates['solver']['template_sha256'],
            'response_format_id': phases['solver'].get('response_format_id'),
            'fill_params': phases['solver'].get('fill_params', {})}),
        'tokenizer_sha256': prompt.sha(tokenizer)}


def build(repo, data, spec_path, out):
    repo, data, out = repo.resolve(), data.resolve(), out.resolve()
    if not out.resolve().is_relative_to(repo/'configs') or out.exists():
        raise ValueError('fresh owned repository config output required')
    spec = read(spec_path)
    if set(spec) != {'schema', 'source_baseline', 'model', 'runtime_binding', 'runtime_receipt', 'partition', 'tokenizer_path', 'fit_estimate'} or spec['schema'] != 'saea-gepa-seed-build-v1':
        raise ValueError('exact seed build specification required')
    sources = {}
    for key in ('source_baseline', 'model', 'runtime_binding', 'runtime_receipt', 'partition'):
        _, sources[key] = read_pin(repo, data, spec[key])
    base, model, binding, partition = (sources[k] for k in ('source_baseline', 'model', 'runtime_binding', 'partition'))
    if runtime_binding(repo, binding) != model:
        raise ValueError('model differs from resident binding')
    if base.get('thinking_policy') != {'perception': THINKING_ARMS['baseline'], 'solver': THINKING_ARMS['baseline']}:
        raise ValueError('fixed P/S3072 source required')
    feedback_ids, excluded = fixed_partition(repo, partition)
    receipt_path = confined(repo/'docs', binding['runtime_receipt'])
    if read_pin(repo, data, spec['runtime_receipt'])[0] != receipt_path:
        raise ValueError('runtime receipt differs from binding')
    receipt = sources['runtime_receipt']
    tokenizer_path = Path(spec['tokenizer_path']).resolve(strict=True)
    if tokenizer_path != Path(receipt['model']['model_dir']).resolve(strict=True):
        raise ValueError('tokenizer differs from resident model material')
    estimate = spec['fit_estimate']
    if (set(estimate) != {'slot_tokens', 'completion_ceiling', 'base_prompt_tokens', 'tokens_per_supplied_item'}
            or any(type(v) is not int or v <= 0 for v in estimate.values())
            or estimate['slot_tokens'] != base['serving_slot_tokens']
            or estimate['completion_ceiling'] != model['decoding_params']['max_tokens']):
        raise ValueError('explicit bounded fit estimate differs from model geometry')
    template_path = confined(repo/'configs', base['phase_material']['solver']['instruction_template'])
    original = template_path.read_bytes().decode('utf-8')
    first, rest = original.split('\n\n', 1)
    count, tokenizer = load_text_token_counter(tokenizer_path)
    try:
        contract = prompt.build_contract(base_template=prompt.BLOCK_START+first+prompt.BLOCK_END+'\n\n'+rest,
            frozen=frozen_fields(repo, base, model, tokenizer), feedback_query_ids=sorted(partition['feedback_ids']),
            excluded_query_ids=excluded)
        candidate = prompt.seed_candidate(contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
        executable = prompt.executable_template(candidate, contract=contract,
            expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
        if executable.encode('utf-8') != template_path.read_bytes():
            raise ValueError('seed changed original Solver template bytes')
        supply = cumulative_supply_bound(base)
        compiled = compile_template_probe(repo, base, fit_estimate=estimate, max_supplied_items=supply,
            delivering_template_tokens=count(original), candidate=candidate, contract=contract,
            expected_contract_sha256=contract['contract_sha256'], count_tokens=count,
            tokenizer_identity=tokenizer, expected_tokenizer_sha256=prompt.sha(tokenizer))
    finally:
        if hasattr(count, 'close'):
            count.close()
    if compiled['program_ceiling'] != 3:
        raise ValueError('exact one-P two-S probe required')
    out.mkdir(parents=True)
    documents = {'contract': contract, 'candidate': candidate, 'tokenizer_identity': tokenizer,
        'compiled_baseline': compiled['baseline'], 'template_material': compiled['material']}
    for name, doc in documents.items():
        publish_new(out/(name+'.json'), raw(doc))
    manifest = {'schema': SCHEMA, 'source_pins': {k: spec[k] for k in sources},
        **{k: pin(repo, out/(k+'.json')) for k in documents},
        'runtime_receipt': pin(repo, receipt_path), 'source_solver_template': pin(repo, template_path),
        'feedback_probe_ids': feedback_ids, 'fit_estimate': estimate, 'max_supplied_items': supply,
        'program_ceiling': 3, 'normal_calls_per_row': 3, 'fallback_cap_per_row': 6,
        'native_fit_completed': False, 'new_model_calls': 0, 'seed_is_learned': False,
        'build_spec_sha256': sha(spec_path), 'purpose': 'Fixed16 seed mechanism probe; not GEPA search.'}
    publish_new(out/'complete.json', raw(manifest))
    return manifest


def verify(repo, data, material):
    """Reconstruct pinned seed provenance without publishing or model calls."""
    from speech_aware_evidence_acquisition.rails import consumption_module as cm
    from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
    if material.get('schema') != SCHEMA or material.get('seed_is_learned') is not False:
        raise ValueError('unlearned seed material required')
    sources = {k: read_pin(repo, data, v)[1] for k, v in material['source_pins'].items()}
    if set(sources) != {'source_baseline', 'model', 'runtime_binding', 'runtime_receipt', 'partition'}:
        raise ValueError('exact seed source pins required')
    base, model, binding = (sources[k] for k in ('source_baseline', 'model', 'runtime_binding'))
    if runtime_binding(repo, binding) != model or base.get('thinking_policy') != {
            r: THINKING_ARMS['baseline'] for r in ('perception', 'solver')}:
        raise ValueError('source runtime or thinking changed')
    receipt_path = confined(repo/'docs', binding['runtime_receipt'])
    receipt = read(receipt_path)
    if receipt != sources['runtime_receipt'] or pin(repo, receipt_path) != material['runtime_receipt']:
        raise ValueError('runtime receipt changed')
    ids, excluded = fixed_partition(repo, sources['partition'])
    if ids != material['feedback_probe_ids']:
        raise ValueError('probe feedback changed')
    docs = {k: read_pin(repo, data, material[k])[1] for k in (
        'contract', 'candidate', 'tokenizer_identity', 'compiled_baseline', 'template_material')}
    path = confined(repo/'configs', base['phase_material']['solver']['instruction_template'])
    if pin(repo, path) != material['source_solver_template']:
        raise ValueError('source Solver changed')
    original = path.read_text(); first, rest = original.split('\n\n', 1)
    estimate = material['fit_estimate']
    if (set(estimate) != {'slot_tokens','completion_ceiling','base_prompt_tokens','tokens_per_supplied_item'}
            or any(type(v) is not int or v <= 0 for v in estimate.values())
            or estimate['slot_tokens'] != base['serving_slot_tokens']
            or estimate['completion_ceiling'] != model['decoding_params']['max_tokens']):
        raise ValueError('seed geometry differs')
    count, tokenizer = load_text_token_counter(Path(receipt['model']['model_dir']))
    try:
        contract = prompt.build_contract(base_template=prompt.BLOCK_START+first+prompt.BLOCK_END+'\n\n'+rest,
            frozen=frozen_fields(repo, base, model, tokenizer),
            feedback_query_ids=sorted(sources['partition']['feedback_ids']), excluded_query_ids=excluded)
        candidate = prompt.seed_candidate(contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
        if docs['contract'] != contract or docs['candidate'] != candidate or docs['tokenizer_identity'] != tokenizer:
            raise ValueError('seed contract, candidate or tokenizer changed')
        published = docs['template_material']; template_pin = published['instruction_template']
        template = confined(repo/'configs', template_pin['relpath'])
        if sha(template) != template_pin['sha256'] or template.read_bytes() != path.read_bytes():
            raise ValueError('seed executable template changed')
        expected_material = {'instruction_template': template_pin, 'template_tokens': count(original),
            'candidate_sha256': candidate['candidate_sha256'], 'contract_sha256': contract['contract_sha256'],
            'tokenizer_sha256': prompt.sha(tokenizer), 'new_model_calls': 0, 'native_fit_completed': False}
        if published != expected_material:
            raise ValueError('seed template material changed')
        supply = cumulative_supply_bound(base)
        block = copy.deepcopy(base['native_episode']['consumption_module'])
        cm.verify_declared_module(block, base, repo_root=repo)
        if block['columns'] or block.get('review_loop') != 'off':
            raise ValueError('unvaried source required')
        block['columns'] = [{'column_id':'gepa','kind':'template','node':'solver',
            'template':template_pin,'template_tokens':count(original)}]
        block['template_frame'] = {'max_supplied_items':supply,'delivering_template_tokens':count(original),
            'twin':False,'fit_estimate':estimate}
        compiled = cm.compile_module(block, carrier_module_hooks(repo_root=repo), repo_root=repo)
        expected = compiled.apply_to(base); expected['native_episode']['consumption_module'] = block
        if expected != docs['compiled_baseline'] or material['max_supplied_items'] != supply:
            raise ValueError('compiled seed differs')
        if any(material.get(k) != v for k,v in {'program_ceiling':3,'normal_calls_per_row':3,
                'fallback_cap_per_row':6,'native_fit_completed':False,'new_model_calls':0}.items()):
            raise ValueError('seed budget or execution claim differs')
    finally:
        if hasattr(count, 'close'): count.close()
    return {**sources, **docs}


if __name__ == '__main__':
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'data', 'spec', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    assert_execution_environment()
    print(json.dumps(build(args.repo, args.data, args.spec, args.out), sort_keys=True))
