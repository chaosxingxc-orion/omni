"""Publish verified GEPA text for the existing native template-variant path.

This CPU operation is not Flight registration, native FIT, search completion or
candidate adoption. Use an independent writable configuration tree, never a live
pinned tree. The caller supplies independently verified tokenizer/lineage custody.
"""
from pathlib import Path
import re

from ..core.atomic_file import publish_new
from .node_prompt_candidate import for_contract, sha, text_sha
from .supply_variants import SupplyVariantTemplate


def publish_template(repo, *, candidate, contract, expected_contract_sha256,
                     count_tokens, tokenizer_identity, expected_tokenizer_sha256,
                     parent=None, expected_parent_sha256=None, mutation_text=None):
    """Verify first, then exclusively publish content-addressed UTF-8 bytes.

    Returns the exact ``instruction_template`` pin shape plus an independent
    text count for the arm's ``template_tokens``. Native FIT must still measure
    the complete speech/P/knowledge/framework request at execution time.
    """
    if (not isinstance(expected_tokenizer_sha256, str)
            or re.fullmatch('[a-f0-9]{64}', expected_tokenizer_sha256) is None
            or sha(tokenizer_identity) != expected_tokenizer_sha256
            or contract['frozen']['tokenizer_sha256'] != expected_tokenizer_sha256):
        raise ValueError('externally pinned tokenizer identity differs')
    if tokenizer_identity.get('rendering') != {
            'mode': 'text', 'add_generation_prompt': False,
            'add_special_tokens': False, 'truncation': False}:
        raise ValueError('untruncated plain-text tokenizer required')
    text = for_contract(contract).executable_template(candidate, contract=contract,
        expected_contract_sha256=expected_contract_sha256, count_tokens=count_tokens,
        parent=parent, expected_parent_sha256=expected_parent_sha256,
        mutation_text=mutation_text)
    count = count_tokens(text)
    if type(count) is not int or not 0 < count <= contract['max_template_tokens']:
        raise ValueError('executable template token bound exceeded')
    repo = Path(repo).resolve(strict=True)
    digest = text_sha(text)
    relative = 'probes/gepa/' + digest + '.txt'
    target = repo / 'configs' / relative
    # Refuse symlink redirection, including an existing same-content target.
    if target.resolve() != target:
        raise ValueError('template destination redirected')
    blob = text.encode('utf-8')
    if target.exists():
        # Validation re-compiles material in an immutable execution tree.
        # Existing content needs verification, not a temporary write attempt.
        if not target.is_file() or target.is_symlink() or target.read_bytes() != blob:
            raise ValueError('existing template conflicts with verified candidate')
    else:
        try:
            publish_new(target, blob)
        except FileExistsError:
            if not target.is_file() or target.is_symlink() or target.read_bytes() != blob:
                raise ValueError('existing template conflicts with verified candidate') from None
    return {'instruction_template': SupplyVariantTemplate(relative, digest).as_json(),
        'template_tokens': count, 'candidate_sha256': candidate['candidate_sha256'],
        'contract_sha256': expected_contract_sha256,
        'tokenizer_sha256': expected_tokenizer_sha256,
        'new_model_calls': 0, 'native_fit_completed': False}


def compile_template_probe(repo, baseline, *, fit_estimate, max_supplied_items,
                           delivering_template_tokens, **candidate_arguments):
    """Compile a single candidate beside the original delivering Solver.

    This is a same-P consumption mechanism probe, not a dynamic GEPA search.
    The native module compiler owns contact ceilings and the original policy;
    callers still register the exact roster, budget and model configuration.
    """
    import copy
    from . import consumption_module as cm
    from .adapters.slurp_module_hooks import carrier_module_hooks

    repo = Path(repo)
    block = baseline['native_episode']['consumption_module']
    cm.verify_declared_module(block, baseline, repo_root=repo)
    if block['columns'] or block.get('review_loop') != 'off':
        raise ValueError('template probe requires the unvaried review-off Agent')
    published = publish_template(repo, **candidate_arguments)
    block = copy.deepcopy(block)
    block['columns'] = [{'column_id': 'gepa', 'kind': 'template', 'node': 'solver',
        'template': published['instruction_template'],
        'template_tokens': published['template_tokens']}]
    block['template_frame'] = {'max_supplied_items': max_supplied_items,
        'delivering_template_tokens': delivering_template_tokens,
        'twin': False, 'fit_estimate': copy.deepcopy(fit_estimate)}
    compiled = cm.compile_module(block, carrier_module_hooks(repo_root=repo), repo_root=repo)
    result = compiled.apply_to(baseline)
    result['native_episode']['consumption_module'] = block
    cm.verify_declared_module(block, result, repo_root=repo)
    for key in ('knowledge_stores', 'thinking_policy', 'prepared_phase_supply'):
        if result.get(key) != baseline.get(key):
            raise ValueError('template probe changed frozen ' + key)
    if result['phase_material'] != baseline['phase_material']:
        raise ValueError('template probe changed delivering phase material')
    return {'baseline': result, 'material': published,
        'program_ceiling': compiled.program_ceiling, 'new_model_calls': 0,
        'purpose': 'same-P native template consumption probe; not GEPA search'}


PAIR_SCHEMA = 'saea-gepa-template-pair-v1'
PERCEPTION_PAIR_SCHEMA = 'saea-gepa-perception-template-pair-v1'
INDEPENDENT_PERCEPTION_PAIR_SCHEMA = 'saea-gepa-perception-template-pair-v2'

def pair_node(declaration):
    schemas = {PAIR_SCHEMA: 'solver', PERCEPTION_PAIR_SCHEMA: 'perception',
               INDEPENDENT_PERCEPTION_PAIR_SCHEMA: 'perception'}
    try:
        return schemas[declaration['schema']]
    except (KeyError, TypeError) as exc:
        raise ValueError('unsupported GEPA pair schema') from exc

def independent_perception_repeat(block):
    """Only the explicit v2 declaration makes the repeat consume its own P."""
    declaration = validate_pair_declaration(block)
    return declaration['schema'] == INDEPENDENT_PERCEPTION_PAIR_SCHEMA


FROZEN_PAIR_FIELDS = ('knowledge_stores', 'thinking_policy', 'phase_material',
                      'phase_tools', 'preparation_source_receipts')


def pair_frozen_identity(values):
    return sha({key: values.get(key) for key in FROZEN_PAIR_FIELDS})


def validate_pair_declaration(block):
    """Validate the exact graph declaration; external lineage remains caller-owned."""
    declaration = block.get('gepa_pair')
    fields = {'schema', 'contract_sha256', 'tokenizer_sha256',
              'parent_candidate_sha256', 'child_candidate_sha256',
              'child_parent_sha256', 'child_mutation_sha256', 'frozen_sha256',
              'parent_template', 'child_template'}
    if not isinstance(declaration, dict) or set(declaration) != fields:
        raise ValueError('exact GEPA pair declaration required')
    node = pair_node(declaration)
    for key in fields - {'schema', 'parent_template', 'child_template'}:
        if not isinstance(declaration[key], str) or re.fullmatch('[a-f0-9]{64}', declaration[key]) is None:
            raise ValueError('GEPA pair digest required')
    if (declaration['child_parent_sha256'] != declaration['parent_candidate_sha256']
            or declaration['child_candidate_sha256'] == declaration['parent_candidate_sha256']):
        raise ValueError('GEPA current parent/child lineage differs')
    columns = block.get('columns')
    if (not isinstance(columns, list) or len(columns) != 2
            or block.get('review_loop') != 'off'
            or block.get('prompt_contact_limits') != 'legacy-c17-lock-v1'
            or block.get('solver_tool_access') or block.get('answer_selection')
            or block.get('template_frame', {}).get('twin') is not (node == 'perception')
            or any(c.get('component_id') == 'tfgrpo-solver-experience'
                   and c.get('state') != 'off' for c in block.get('components', []))):
        raise ValueError('GEPA pair requires exact review-off two-column graph')
    for name, column in zip(('parent', 'child'), columns):
        if (set(column) != {'column_id', 'kind', 'node', 'template', 'template_tokens'}
                or column['column_id'] != name or column['kind'] != 'template'
                or column['node'] != node
                or column['template'] != declaration[name + '_template']):
            raise ValueError('GEPA pair column identity differs')
        pin = column['template']
        if (not isinstance(pin, dict) or set(pin) != {'relpath', 'sha256'}
                or not isinstance(pin['sha256'], str)
                or re.fullmatch('[a-f0-9]{64}', pin['sha256']) is None
                or pin['relpath'] != 'probes/gepa/' + pin['sha256'] + '.txt'):
            raise ValueError('GEPA pair requires content-addressed template pins')
    return declaration


def compile_template_pair(repo, baseline, *, parent_arguments, child_arguments,
                          fit_estimate, max_supplied_items, delivering_template_tokens,
                          perception_repeat_scope="shared-lock-perception"):
    """Compile explicit node interventions with frozen opposite-node templates.

    Solver uses shared P and parent/child/lock S. Explicit Perception v2 uses independent
    parent/child/AA/lock P-to-S branches (eight primary contacts).

    Each argument dictionary has publish_template's externally anchored inputs.
    A non-seed parent includes its own observed mutation and predecessor. This
    validates local lineage replay, not authenticity of the external journals.
    No evaluation/acceptance or cross-candidate recovery is performed here.
    """
    import copy
    from . import consumption_module as cm
    from .adapters.slurp_module_hooks import carrier_module_hooks
    repo = Path(repo)
    block = baseline['native_episode']['consumption_module']
    cm.verify_declared_module(block, baseline, repo_root=repo)
    if block.get('columns') or block.get('review_loop') != 'off' or block.get('gepa_pair'):
        raise ValueError('template pair requires the unvaried review-off Agent')
    if (parent_arguments['contract'] != child_arguments['contract']
            or parent_arguments['expected_contract_sha256'] != child_arguments['expected_contract_sha256']
            or parent_arguments['tokenizer_identity'] != child_arguments['tokenizer_identity']
            or child_arguments.get('parent') != parent_arguments['candidate']
            or child_arguments.get('expected_parent_sha256') != parent_arguments['candidate']['candidate_sha256']):
        raise ValueError('pair must bind child to the actual verified current parent')
    node = for_contract(parent_arguments['contract'])._contract.node
    if perception_repeat_scope not in ('shared-lock-perception', 'independent-perception'):
        raise ValueError('unknown perception repeat scope')
    if node != 'perception' and perception_repeat_scope != 'shared-lock-perception':
        raise ValueError('independent perception repeat requires Perception pair')
    # Verify both before publishing; rejected lineage cannot publish either arm.
    for args in (parent_arguments, child_arguments):
        for_contract(args['contract']).executable_template(args['candidate'], contract=args['contract'],
            expected_contract_sha256=args['expected_contract_sha256'], count_tokens=args['count_tokens'],
            parent=args.get('parent'), expected_parent_sha256=args.get('expected_parent_sha256'),
            mutation_text=args.get('mutation_text'))
    materials = {name: publish_template(repo, **args) for name, args in
                 (('parent', parent_arguments), ('child', child_arguments))}
    block = copy.deepcopy(block)
    block['columns'] = [{'column_id': name, 'kind': 'template', 'node': node,
        'template': value['instruction_template'], 'template_tokens': value['template_tokens']}
        for name, value in materials.items()]
    block['template_frame'] = {'max_supplied_items': max_supplied_items,
        'delivering_template_tokens': delivering_template_tokens, 'twin': node == 'perception',
        'fit_estimate': copy.deepcopy(fit_estimate)}
    child = child_arguments['candidate']
    perception_schema = (INDEPENDENT_PERCEPTION_PAIR_SCHEMA
        if perception_repeat_scope == 'independent-perception' else PERCEPTION_PAIR_SCHEMA)
    block['gepa_pair'] = {'schema': perception_schema if node == 'perception' else PAIR_SCHEMA,
        'contract_sha256': parent_arguments['expected_contract_sha256'],
        'tokenizer_sha256': parent_arguments['expected_tokenizer_sha256'],
        'parent_candidate_sha256': parent_arguments['candidate']['candidate_sha256'],
        'child_candidate_sha256': child['candidate_sha256'],
        'child_parent_sha256': child['parent_sha256'],
        'child_mutation_sha256': child['mutation_sha256'],
        'frozen_sha256': pair_frozen_identity(baseline),
        **{name + '_template': value['instruction_template'] for name, value in materials.items()}}
    validate_pair_declaration(block)
    compiled = cm.compile_module(block, carrier_module_hooks(repo_root=repo), repo_root=repo)
    result = compiled.apply_to(baseline)
    result['native_episode']['consumption_module'] = block
    cm.verify_declared_module(block, result, repo_root=repo)
    ceiling = 8 if node == 'perception' else 4
    if compiled.program_ceiling != ceiling or pair_frozen_identity(result) != pair_frozen_identity(baseline):
        raise ValueError('pair changed frozen system or contact graph')
    return {'baseline': result, 'materials': materials, 'declaration': block['gepa_pair'],
        'program_ceiling': ceiling, 'new_model_calls': 0,
        'purpose': (('independent P-to-S parent/child/AA/lock evaluation; external scoring required'
                     if perception_repeat_scope == 'independent-perception' else
                     'independent parent/child/lock P-to-S with shared-lock-P Solver repeat; external scoring required')
                    if node == 'perception' else 'same-P current-parent/child/lock evaluation; external scoring required')}
