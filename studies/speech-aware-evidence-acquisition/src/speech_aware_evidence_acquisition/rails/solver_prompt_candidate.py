"""Pure CPU GEPA prompt contract; no Flight, release, adoption or model calls.

Callers must obtain contract/parent pins and mutation bytes from independently
verified custody. Hashes supplied by a candidate are never trust anchors. Token
counts here do not replace native FIT of the actual wrapped execution request.
Semantic review must still reject memorized answers or malicious instructions;
syntactic boundaries cannot prove the meaning of arbitrary natural language.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

STRATEGY = 'gepa_solver_prompt_search_v1'
MODULE = 'solver_consumption_instruction'
KIND = 'solver_prompt'
CONTRACT_SCHEMA = 'saea-solver-prompt-contract-v1'
CANDIDATE_SCHEMA = 'saea-solver-prompt-candidate-v1'
MUTATION_SCHEMA = 'saea-solver-prompt-mutation-v1'
BLOCK_START = '<SAEA_SOLVER_CONSUMPTION>'
BLOCK_END = '</SAEA_SOLVER_CONSUMPTION>'
FROZEN_FIELDS = frozenset(name + '_sha256' for name in (
    'knowledge', 'perception', 'supply', 'model', 'decoding', 'roles', 'tools',
    'output_schema', 'tokenizer'))
CONTRACT_FIELDS = frozenset((
    'schema', 'strategy', 'candidate_kind', 'allowed_module', 'base_template',
    'base_template_sha256', 'original_instruction', 'frozen', 'feedback_query_ids',
    'excluded_query_ids', 'max_instruction_tokens', 'max_template_tokens', 'contract_sha256'))
MUTATION_FIELDS = frozenset(('schema', 'parent_sha256', 'module', 'evidence_query_ids',
                             'diagnosis', 'solver_instruction'))
CANDIDATE_FIELDS = frozenset((
    'schema', 'strategy', 'candidate_kind', 'module', 'contract_sha256',
    'parent_sha256', 'mutation_sha256', 'mutation', 'frozen', 'original_instruction',
    'solver_instruction', 'instruction_sha256', 'full_template', 'full_template_sha256',
    'instruction_tokens', 'template_tokens', 'candidate_sha256'))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(value):
    """Canonical JSON identity (text byte hashes use text_sha separately)."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _text(value, name):
    _require(isinstance(value, str) and value.strip(), name + ' must be nonempty text')
    _require(not any(ord(c) < 32 and c not in '\n\t' for c in value), name + ' contains control bytes')
    try:
        value.encode('utf-8')
    except UnicodeError as exc:
        raise ValueError(name + ' must be UTF-8') from exc
    return value


def _ids(value, name, *, train=False):
    _require(isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value),
             name + ' must be an ID list')
    _require(value == sorted(set(value)), name + ' must be sorted and unique')
    if train:
        _require(all(re.fullmatch(r'slurp:train:[A-Za-z0-9_.-]+', x) for x in value),
                 'feedback IDs must identify TRAIN queries')
    return value


def _split(template):
    _text(template, 'base template')
    _require(template.count(BLOCK_START) == template.count(BLOCK_END) == 1,
             'template requires exactly one consumption block')
    before, remaining = template.split(BLOCK_START)
    instruction, after = remaining.split(BLOCK_END)
    _require(BLOCK_END not in before and BLOCK_START not in after,
             'consumption block boundaries reversed')
    return before + BLOCK_START, instruction, BLOCK_END + after


def _instruction(value, contract):
    _text(value, 'solver instruction')
    # This deliberately narrow text-only block cannot create placeholders,
    # XML/chat roles, Jinja directives or nested consumption delimiters.
    _require(not any(c in value for c in '{}<>'), 'instruction contains template boundary syntax')
    _require(not re.search(r'slurp:(?:train|dev|test):', value, re.I),
             'query IDs must not be embedded in the solver instruction')
    _require(not any(q in value for q in contract['feedback_query_ids'] + contract['excluded_query_ids']),
             'query IDs must not be embedded in the solver instruction')
    return value


def build_contract(*, base_template, frozen, feedback_query_ids, excluded_query_ids,
                   max_instruction_tokens=2048, max_template_tokens=44032):
    """Freeze the trusted template, experiment identities and TRAIN partition.

    The returned digest must be pinned outside candidate-controlled artifacts.
    Identity values are content SHA256s, not declarations of native custody.
    """
    _, original, _ = _split(base_template)
    result = {'schema': CONTRACT_SCHEMA, 'strategy': STRATEGY, 'candidate_kind': KIND,
        'allowed_module': MODULE, 'base_template': base_template,
        'base_template_sha256': text_sha(base_template), 'original_instruction': original,
        'frozen': copy.deepcopy(frozen), 'feedback_query_ids': list(feedback_query_ids),
        'excluded_query_ids': list(excluded_query_ids),
        'max_instruction_tokens': max_instruction_tokens, 'max_template_tokens': max_template_tokens}
    result['contract_sha256'] = sha(result)
    validate_contract(result, expected_contract_sha256=result['contract_sha256'])
    return result


def validate_contract(contract, *, expected_contract_sha256):
    _require(isinstance(contract, dict) and set(contract) == CONTRACT_FIELDS, 'exact prompt contract required')
    _require(_digest(expected_contract_sha256) and contract['contract_sha256'] == expected_contract_sha256
        and sha({k: v for k, v in contract.items() if k != 'contract_sha256'}) == expected_contract_sha256,
        'externally pinned prompt contract differs')
    _require((contract['schema'], contract['strategy'], contract['candidate_kind'], contract['allowed_module'])
        == (CONTRACT_SCHEMA, STRATEGY, KIND, MODULE), 'unsupported prompt strategy, kind or module')
    frozen = contract['frozen']
    _require(isinstance(frozen, dict) and set(frozen) == FROZEN_FIELDS and all(_digest(x) for x in frozen.values()),
             'exact frozen knowledge/P/supply/model/decoding/interface/tokenizer identities required')
    for key, upper in (('max_instruction_tokens', 2048), ('max_template_tokens', 44032)):
        _require(type(contract[key]) is int and 0 < contract[key] <= upper, 'prompt token bound cannot widen')
    feedback = _ids(contract['feedback_query_ids'], 'feedback IDs', train=True)
    excluded = _ids(contract['excluded_query_ids'], 'excluded IDs')
    _require(feedback and not set(feedback).intersection(excluded), 'feedback and excluded IDs must be disjoint')
    _, original, _ = _split(contract['base_template'])
    _require(original == contract['original_instruction']
        and text_sha(contract['base_template']) == contract['base_template_sha256'], 'original template changed')
    _instruction(original, contract)
    return contract


def _decode_mutation(text):
    _text(text, 'mutation JSON')
    def pairs(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, 'duplicate mutation JSON key')
            result[key] = value
        return result
    def constant(_value):
        raise ValueError('nonfinite mutation JSON')
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError) as exc:
        raise ValueError('mutation must be exact JSON: ' + str(exc)) from exc
    _require(isinstance(value, dict) and set(value) == MUTATION_FIELDS, 'exact mutation fields required')
    return value


def validate_mutation(text, *, contract, expected_contract_sha256, expected_parent_sha256):
    validate_contract(contract, expected_contract_sha256=expected_contract_sha256)
    mutation = _decode_mutation(text)
    _require(mutation['schema'] == MUTATION_SCHEMA and mutation['module'] == MODULE,
             'unsupported mutation schema or module')
    _require(_digest(expected_parent_sha256) and mutation['parent_sha256'] == expected_parent_sha256,
             'mutation parent differs')
    ids = _ids(mutation['evidence_query_ids'], 'mutation evidence IDs', train=True)
    _require(ids and set(ids) <= set(contract['feedback_query_ids']), 'mutation references must be feedback-only')
    diagnosis = _text(mutation['diagnosis'], 'diagnosis')
    _require(not any(q in diagnosis for q in contract['excluded_query_ids']), 'diagnosis leaks excluded query IDs')
    mentioned = {value.rstrip('.') for value in
                 re.findall(r'slurp:(?:train|dev|test):[A-Za-z0-9_.-]+', diagnosis, re.I)}
    _require(mentioned <= set(ids), 'diagnosis references unlisted or non-feedback query IDs')
    _instruction(mutation['solver_instruction'], contract)
    return mutation


def _candidate(contract, instruction, *, parent_sha256, mutation_text, mutation, count_tokens):
    _instruction(instruction, contract)
    before, _, after = _split(contract['base_template'])
    template = before + instruction + after
    counts = []
    for text, bound in ((instruction, contract['max_instruction_tokens']), (template, contract['max_template_tokens'])):
        count = count_tokens(text)
        _require(type(count) is int and 0 < count <= bound, 'tokenizer count invalid or prompt exceeds bound; no truncation')
        counts.append(count)
    result = {'schema': CANDIDATE_SCHEMA, 'strategy': STRATEGY, 'candidate_kind': KIND,
        'module': MODULE, 'contract_sha256': contract['contract_sha256'], 'parent_sha256': parent_sha256,
        'mutation_sha256': None if mutation_text is None else text_sha(mutation_text),
        'mutation': copy.deepcopy(mutation), 'frozen': copy.deepcopy(contract['frozen']),
        'original_instruction': contract['original_instruction'], 'solver_instruction': instruction,
        'instruction_sha256': text_sha(instruction), 'full_template': template,
        'full_template_sha256': text_sha(template), 'instruction_tokens': counts[0], 'template_tokens': counts[1]}
    result['candidate_sha256'] = sha(result)
    return result


def seed_candidate(*, contract, expected_contract_sha256, count_tokens):
    validate_contract(contract, expected_contract_sha256=expected_contract_sha256)
    return _candidate(contract, contract['original_instruction'], parent_sha256=None,
                      mutation_text=None, mutation=None, count_tokens=count_tokens)


def _parent_pin(parent, contract, expected_parent_sha256, count_tokens):
    _require(isinstance(parent, dict) and set(parent) == CANDIDATE_FIELDS, 'exact parent candidate required')
    _require(_digest(expected_parent_sha256) and parent['candidate_sha256'] == expected_parent_sha256
        and sha({k: v for k, v in parent.items() if k != 'candidate_sha256'}) == expected_parent_sha256,
        'externally pinned parent differs')
    _require((parent['schema'], parent['strategy'], parent['candidate_kind'], parent['module'])
        == (CANDIDATE_SCHEMA, STRATEGY, KIND, MODULE), 'parent strategy or kind differs')
    _require(parent['contract_sha256'] == contract['contract_sha256'] and parent['frozen'] == contract['frozen']
        and parent['original_instruction'] == contract['original_instruction'], 'parent frozen contract differs')
    expected = _candidate(contract, parent['solver_instruction'], parent_sha256=parent['parent_sha256'],
                          mutation_text=None, mutation=parent['mutation'], count_tokens=count_tokens)
    for key in ('full_template', 'full_template_sha256', 'instruction_sha256', 'instruction_tokens', 'template_tokens'):
        _require(parent[key] == expected[key], 'parent template or tokenizer identity differs')


def derive_candidate(mutation_text, *, parent, expected_parent_sha256, contract,
                     expected_contract_sha256, count_tokens):
    """Derive only the allowed block; this is not candidate acceptance/release."""
    mutation = validate_mutation(mutation_text, contract=contract,
        expected_contract_sha256=expected_contract_sha256, expected_parent_sha256=expected_parent_sha256)
    _parent_pin(parent, contract, expected_parent_sha256, count_tokens)
    _require(mutation['solver_instruction'] != parent['solver_instruction'], 'rejected-noop: unchanged instruction')
    return _candidate(contract, mutation['solver_instruction'], parent_sha256=expected_parent_sha256,
                      mutation_text=mutation_text, mutation=mutation, count_tokens=count_tokens)


def verify_candidate(candidate, *, contract, expected_contract_sha256, count_tokens,
                     parent=None, expected_parent_sha256=None, mutation_text=None):
    """Replay from external anchors and exact observed mutation bytes.

    A derived parent's pin must already be established by prior verified search
    custody. This local verifier does not establish an entire native lineage.
    """
    _require(isinstance(candidate, dict) and set(candidate) == CANDIDATE_FIELDS, 'exact prompt candidate required')
    if parent is None:
        _require(expected_parent_sha256 is None and mutation_text is None, 'seed cannot carry mutation provenance')
        expected = seed_candidate(contract=contract, expected_contract_sha256=expected_contract_sha256,
                                  count_tokens=count_tokens)
    else:
        expected = derive_candidate(mutation_text, parent=parent, expected_parent_sha256=expected_parent_sha256,
            contract=contract, expected_contract_sha256=expected_contract_sha256, count_tokens=count_tokens)
    _require(candidate == expected, 'candidate differs from pinned template and observed mutation replay')
    return copy.deepcopy(expected)
