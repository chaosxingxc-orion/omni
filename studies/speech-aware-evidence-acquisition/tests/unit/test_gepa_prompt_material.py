import json
from pathlib import Path

import pytest
from speech_aware_evidence_acquisition.rails import gepa_prompt_material as m
from speech_aware_evidence_acquisition.rails import solver_prompt_candidate as c
from speech_aware_evidence_acquisition.rails.supply_variants import parse_template_variants


def inputs():
    original = (Path(__file__).parents[2]/'configs/probes/slurp_native_role_solver_v2.txt').read_text()
    first, rest = original.split('\n\n', 1)
    tokenizer = {'rendering': {'mode': 'text', 'add_generation_prompt': False,
        'add_special_tokens': False, 'truncation': False}, 'test_counter': 'words'}
    frozen = {k: c.sha(k) for k in c.FROZEN_FIELDS}
    frozen['tokenizer_sha256'] = c.sha(tokenizer)
    contract = c.build_contract(base_template=c.BLOCK_START+first+c.BLOCK_END+'\n\n'+rest,
        frozen=frozen, feedback_query_ids=['slurp:train:1'], excluded_query_ids=[])
    count = lambda text: len(text.split())  # fixture only, never production FIT
    seed = c.seed_candidate(contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
    return original, dict(candidate=seed, contract=contract,
        expected_contract_sha256=contract['contract_sha256'], count_tokens=count,
        tokenizer_identity=tokenizer, expected_tokenizer_sha256=c.sha(tokenizer))


def test_real_seed_bytes_reentry_and_native_parser(tmp_path):
    original, args = inputs()
    result = m.publish_template(tmp_path, **args)
    pin = result['instruction_template']
    assert (tmp_path/'configs'/pin['relpath']).read_bytes() == original.encode()
    assert m.publish_template(tmp_path, **args) == result
    group = parse_template_variants({'node': 'solver', 'max_supplied_items': 4,
        'fit_estimate': {'slot_tokens': 49152, 'completion_ceiling': 4096,
            'base_prompt_tokens': 9600, 'tokens_per_supplied_item': 120},
        'arms': [{'arm_id': 'seed', 'instruction_template': pin,
                  'template_tokens': result['template_tokens']},
                 {'arm_id': 'lock', 'deliver': True, 'template_tokens': result['template_tokens']}]},
        role_kind='solver', phase_id='solver', locked_template='probes/slurp_native_role_solver_v2.txt')
    assert group.arm('seed').instruction_template.as_json() == pin
    assert not result['native_fit_completed']


@pytest.mark.parametrize('fault', ['candidate', 'contract', 'tokenizer', 'count'])
def test_bad_inputs_leave_no_published_file(tmp_path, fault):
    _, args = inputs()
    if fault == 'candidate': args['candidate']['solver_instruction'] = 'changed'
    elif fault == 'contract': args['expected_contract_sha256'] = 'f'*64
    elif fault == 'tokenizer': args['expected_tokenizer_sha256'] = 'f'*64
    else: args['count_tokens'] = lambda _: 999999
    with pytest.raises(ValueError): m.publish_template(tmp_path, **args)
    assert list(tmp_path.iterdir()) == []


def test_conflicting_existing_bytes_are_not_overwritten(tmp_path):
    _, args = inputs(); result = m.publish_template(tmp_path, **args)
    target = tmp_path/'configs'/result['instruction_template']['relpath']
    target.write_text('conflict')
    with pytest.raises(ValueError, match='conflicts'): m.publish_template(tmp_path, **args)
    assert target.read_text() == 'conflict'


def test_symlink_parent_refused(tmp_path):
    _, args = inputs(); (tmp_path/'outside').mkdir(); (tmp_path/'configs').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError, match='redirected'): m.publish_template(tmp_path, **args)
    assert not list((tmp_path/'outside').iterdir())


def test_derived_template_requires_exact_parent_and_observed_mutation(tmp_path):
    original, args = inputs()
    parent = args['candidate']
    mutation = json.dumps({'schema': c.MUTATION_SCHEMA, 'module': c.MODULE,
        'parent_sha256': parent['candidate_sha256'], 'evidence_query_ids': ['slurp:train:1'],
        'diagnosis': 'Check evidence applicability.',
        'solver_instruction': 'Compare supplied candidates against the speech.'})
    args['candidate'] = c.derive_candidate(mutation, parent=parent,
        expected_parent_sha256=parent['candidate_sha256'], contract=args['contract'],
        expected_contract_sha256=args['expected_contract_sha256'], count_tokens=args['count_tokens'])
    args.update(parent=parent, expected_parent_sha256=parent['candidate_sha256'], mutation_text=mutation)
    result = m.publish_template(tmp_path, **args)
    text = (tmp_path/'configs'/result['instruction_template']['relpath']).read_text()
    assert text == 'Compare supplied candidates against the speech.\n\n' + original.split('\n\n', 1)[1]
    args['mutation_text'] = mutation.replace('applicability', 'unobserved')
    with pytest.raises(ValueError): m.publish_template(tmp_path, **args)
