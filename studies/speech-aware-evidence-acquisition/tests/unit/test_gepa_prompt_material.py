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


def test_real_seed_bytes_reentry_and_native_parser(tmp_path, monkeypatch):
    original, args = inputs()
    result = m.publish_template(tmp_path, **args)
    pin = result['instruction_template']
    assert (tmp_path/'configs'/pin['relpath']).read_bytes() == original.encode()
    def readonly(*args, **kwargs):
        raise OSError(30, 'Read-only file system')
    monkeypatch.setattr(m, 'publish_new', readonly)
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


def test_real_agent_probe_compiles_without_changing_knowledge_or_delivering_material(tmp_path):
    import shutil
    source = Path(__file__).parents[2]
    shutil.copytree(source/'configs', tmp_path/'configs')
    for relative in ['scripts/flights/nx/l338_port_pins.json',
        'docs/receipts/knowledge/slurp-kb-size-pool-annotated-exemplars-minus-r01.json']:
        (tmp_path/relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, tmp_path/relative)
    # Use the actual A0 component declaration and compiler, not a mocked policy.
    base = json.loads((source/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_text())
    for component in base['native_episode']['consumption_module']['components']:
        if component['state'] == 'on':
            path = Path(component['record'])
            (tmp_path/path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/path, tmp_path/path)
    original = json.loads(json.dumps(base))
    _, args = inputs()
    result = m.compile_template_probe(tmp_path, base, fit_estimate={
        'slot_tokens': 49152, 'completion_ceiling': 4096,
        'base_prompt_tokens': 9600, 'tokens_per_supplied_item': 120},
        max_supplied_items=48, delivering_template_tokens=args['count_tokens'](
            (source/'configs/probes/slurp_native_role_solver_v2.txt').read_text()), **args)
    assert result['program_ceiling'] == 3
    assert base == original
    built = result['baseline']
    for key in ('knowledge_stores', 'phase_material', 'thinking_policy'):
        assert built.get(key) == base.get(key)
    group = built['native_episode']['roles']['solver']['template_variants']
    assert [r['arm_id'] for r in group['arms']] == ['gepa', 'lock']
    assert group['arms'][-1]['deliver'] is True

    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config, prompt_remaining_contacts
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    built['carrier_lock_key'] = 'slurp'
    native = load_native_episode_config(built, default_phase_program(built['policy']))
    assert native.max_contacts == 3
    assert prompt_remaining_contacts(native, 'solver', 1) == 9
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    built['transport'] = 'vllm'
    before = json.loads(json.dumps(built))
    assert native_contact_custody_enabled(built)
    assert built == before
    for fault in ('column', 'extra_column', 'deliver', 'pin', 'extra_arm', 'twin',
                  'review', 'tool', 'perception', 'revision', 'branch', 'sampling',
                  'wrong_policy', 'extra_group_key', 'learned_component'):
        other = json.loads(json.dumps(built))
        episode = other['native_episode']
        module = episode['consumption_module']
        arms = episode['roles']['solver']['template_variants']['arms']
        if fault == 'column': module['columns'][0]['column_id'] = 'other'
        elif fault == 'extra_column': module['columns'].append(dict(module['columns'][0]))
        elif fault == 'deliver': arms[0]['deliver'] = True; arms[1]['deliver'] = False
        elif fault == 'pin': arms[0]['instruction_template']['sha256'] = '0' * 64
        elif fault == 'extra_arm': arms.append(dict(arms[0]))
        elif fault == 'twin': module['template_frame']['twin'] = True
        elif fault == 'review': episode['review_loop'] = 'on'
        elif fault == 'tool': other['phase_tools'] = {'solver': {'allowed_tools': ['lookup']}}
        elif fault == 'perception': episode['roles']['perception']['template_variants'] = {}
        elif fault == 'revision': episode['max_revisions'] = 1
        elif fault == 'branch': episode['roles']['solver']['branches'] = []
        elif fault == 'sampling': episode['roles']['solver']['samples'] = 2
        elif fault == 'wrong_policy': other['policy'] = 'phase-native-module-v1-c4'
        elif fault == 'extra_group_key': episode['roles']['solver']['template_variants']['arm_visits'] = 'all'
        else: module['components'].append({'component_id': 'tfgrpo-solver-experience', 'state': 'on'})
        assert not native_contact_custody_enabled(other), fault
    from speech_aware_evidence_acquisition.rails import consumption_module as cm
    from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
    for fault in ('other_name', 'two_candidates', 'twin', 'review'):
        block = json.loads(json.dumps(built['native_episode']['consumption_module']))
        if fault == 'other_name': block['columns'][0]['column_id'] = 'arbitrary'
        elif fault == 'two_candidates': block['columns'].append({**block['columns'][0], 'column_id': 'another'})
        elif fault == 'twin': block['template_frame']['twin'] = True
        else: block['review_loop'] = 'on'
        with pytest.raises(cm.ConsumptionModuleConfigError):
            cm.compile_module(block, carrier_module_hooks(repo_root=tmp_path), repo_root=tmp_path)


def pair_inputs(tmp_path):
    import shutil
    source = Path(__file__).parents[2]
    shutil.copytree(source/'configs', tmp_path/'configs')
    base = json.loads((source/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_text())
    paths = ['scripts/flights/nx/l338_port_pins.json',
             'docs/receipts/knowledge/slurp-kb-size-pool-annotated-exemplars-minus-r01.json']
    paths += [x['record'] for x in base['native_episode']['consumption_module']['components'] if x['state'] == 'on']
    for rel in paths:
        dest = tmp_path/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/rel, dest)
    original, seed = inputs()
    def child(parent_args, instruction):
        parent = parent_args['candidate']
        mutation = json.dumps({'schema': c.MUTATION_SCHEMA, 'module': c.MODULE,
            'parent_sha256': parent['candidate_sha256'], 'evidence_query_ids': ['slurp:train:1'],
            'diagnosis': 'Check applicability using supplied evidence.', 'solver_instruction': instruction})
        candidate = c.derive_candidate(mutation, parent=parent,
            expected_parent_sha256=parent['candidate_sha256'], contract=seed['contract'],
            expected_contract_sha256=seed['expected_contract_sha256'], count_tokens=seed['count_tokens'])
        return {**seed, 'candidate': candidate, 'parent': parent,
            'expected_parent_sha256': parent['candidate_sha256'], 'mutation_text': mutation}
    parent = child(seed, 'Compare candidates against speech and supplied evidence.')
    offspring = child(parent, 'Check applicability before selecting the complete supported answer.')
    kwargs = dict(parent_arguments=parent, child_arguments=offspring,
        fit_estimate={'slot_tokens':49152, 'completion_ceiling':4096,
            'base_prompt_tokens':9600, 'tokens_per_supplied_item':120},
        max_supplied_items=48, delivering_template_tokens=seed['count_tokens'](original))
    return base, kwargs


def test_current_nonseed_parent_child_native_four_contact_graph(tmp_path):
    import copy
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    base, args = pair_inputs(tmp_path); before = copy.deepcopy(base)
    result = m.compile_template_pair(tmp_path, base, **args)
    assert base == before and result['program_ceiling'] == 4
    built = result['baseline']; built.update(transport='vllm', carrier_lock_key='slurp')
    arms = built['native_episode']['roles']['solver']['template_variants']['arms']
    assert [a['arm_id'] for a in arms] == ['parent', 'child', 'lock']
    assert [a.get('deliver', False) for a in arms] == [False, False, True]
    assert result['declaration']['parent_candidate_sha256'] == args['child_arguments']['expected_parent_sha256']
    assert args['parent_arguments']['candidate']['parent_sha256'] is not None
    assert load_native_episode_config(built, default_phase_program(built['policy'])).max_contacts == 4
    assert native_contact_custody_enabled(built)
    for fault in ('order','extra','deliver','lineage','pin','knowledge','thinking','perception','tools','revision','declaration'):
        bad = copy.deepcopy(built); mod = bad['native_episode']['consumption_module']
        group = bad['native_episode']['roles']['solver']['template_variants']
        if fault == 'order': group['arms'][:2] = reversed(group['arms'][:2])
        elif fault == 'extra': group['arms'].append(copy.deepcopy(group['arms'][0]))
        elif fault == 'deliver': group['arms'][0]['deliver'] = True
        elif fault == 'lineage': mod['gepa_pair']['child_parent_sha256'] = '0'*64
        elif fault == 'pin': mod['columns'][0]['template']['sha256'] = '0'*64
        elif fault == 'knowledge': bad['knowledge_stores'] = {}
        elif fault == 'thinking': bad['thinking_policy'] = {'perception': {'enable_thinking': False}}
        elif fault == 'perception': bad['native_episode']['roles']['perception']['max_iterations'] = 2
        elif fault == 'tools': bad['phase_tools'] = {'solver': {'allowed_tools':['lookup']}}
        elif fault == 'revision': bad['native_episode']['max_revisions'] = 1
        else: mod.pop('gepa_pair')
        assert not native_contact_custody_enabled(bad), fault


@pytest.mark.parametrize('fault', ['wrong_parent', 'parent_mutation', 'child_mutation', 'contract', 'tokenizer'])
def test_pair_refuses_unverified_lineage(tmp_path, fault):
    import copy
    base, args = pair_inputs(tmp_path)
    folder = tmp_path/'configs/probes/gepa'
    before = {p.name: p.read_bytes() for p in folder.glob('*.txt')}
    if fault == 'wrong_parent': args['child_arguments']['parent'] = args['parent_arguments']['parent']
    elif fault == 'parent_mutation': args['parent_arguments']['mutation_text'] += ' altered'
    elif fault == 'child_mutation': args['child_arguments']['mutation_text'] += ' altered'
    elif fault == 'contract': args['child_arguments']['contract'] = copy.deepcopy(args['child_arguments']['contract']); args['child_arguments']['contract']['max_instruction_tokens'] = 1
    else: args['child_arguments']['tokenizer_identity'] = {}
    with pytest.raises(ValueError): m.compile_template_pair(tmp_path, base, **args)
    assert {p.name: p.read_bytes() for p in folder.glob('*.txt')} == before
