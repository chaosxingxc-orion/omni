"""Shared GEPA node prompt validation; each node has a distinct signed schema."""
import copy
import hashlib
import json
import re

QUERY_ID = r'slurp:(?:train|dev|test):[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)*'

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
        _require(all(x.startswith('slurp:train:') and re.fullmatch(QUERY_ID, x) for x in value),
                 'feedback IDs must identify TRAIN queries')
    return value


class PromptCandidateContract:
    """Node-specific immutable schema vocabulary over a single provenance engine."""

    def __init__(self, node):
        if node not in ('solver', 'perception', 'slot', 'decision'):
            raise ValueError('unsupported editable node')
        self.node = node
        self.instruction_field = node + '_instruction'
        self.STRATEGY = 'gepa_' + node + '_prompt_search_v1'
        self.MODULE = node + '_consumption_instruction'
        self.KIND = node + '_prompt'
        self.CONTRACT_SCHEMA = 'saea-' + node + '-prompt-contract-v1'
        self.CANDIDATE_SCHEMA = 'saea-' + node + '-prompt-candidate-v1'
        self.MUTATION_SCHEMA = 'saea-' + node + '-prompt-mutation-v1'
        self.BLOCK_START = '<SAEA_' + node.upper() + '_CONSUMPTION>'
        self.BLOCK_END = '</SAEA_' + node.upper() + '_CONSUMPTION>'
        if node in ('slot', 'decision'):
            self.FROZEN_FIELDS = frozenset(name + '_sha256' for name in (
                'knowledge', 'supply', 'baseline', 'model', 'tokenizer'))
        else:
            other = 'perception' if node == 'solver' else 'solver'
            self.FROZEN_FIELDS = frozenset(name + '_sha256' for name in (
                'knowledge', other, 'supply', 'model', 'decoding', 'roles', 'tools',
                'output_schema', 'tokenizer'))
        self.CONTRACT_FIELDS = frozenset((
            'schema', 'strategy', 'candidate_kind', 'allowed_module', 'base_template',
            'base_template_sha256', 'original_instruction', 'frozen', 'feedback_query_ids',
            'excluded_query_ids', 'max_instruction_tokens', 'max_template_tokens', 'contract_sha256'))
        self.MUTATION_FIELDS = frozenset(('schema', 'parent_sha256', 'module', 'evidence_query_ids',
            'diagnosis', self.instruction_field))
        self.CANDIDATE_FIELDS = frozenset((
            'schema', 'strategy', 'candidate_kind', 'module', 'contract_sha256',
            'parent_sha256', 'mutation_sha256', 'mutation', 'frozen', 'original_instruction',
            self.instruction_field, 'instruction_sha256', 'full_template', 'full_template_sha256',
            'instruction_tokens', 'template_tokens', 'candidate_sha256'))

    def _split(self, template):
        _text(template, 'base template')
        _require(template.count(self.BLOCK_START) == template.count(self.BLOCK_END) == 1,
                 'template requires exactly one consumption block')
        before, remaining = template.split(self.BLOCK_START)
        instruction, after = remaining.split(self.BLOCK_END)
        _require(self.BLOCK_END not in before and self.BLOCK_START not in after,
                 'consumption block boundaries reversed')
        return before + self.BLOCK_START, instruction, self.BLOCK_END + after


    def _instruction(self, value, contract):
        _text(value, 'node instruction')
        # This deliberately narrow text-only block cannot create placeholders,
        # XML/chat roles, Jinja directives or nested consumption delimiters.
        _require(not any(c in value for c in '{}<>'), 'instruction contains template boundary syntax')
        _require(not re.search(r'slurp:(?:train|dev|test):', value, re.I),
                 'query IDs must not be embedded in the node instruction')
        _require(not any(q in value for q in contract['feedback_query_ids'] + contract['excluded_query_ids']),
                 'query IDs must not be embedded in the node instruction')
        return value


    def build_contract(self, *, base_template, frozen, feedback_query_ids, excluded_query_ids,
                       max_instruction_tokens=2048, max_template_tokens=44032):
        """Freeze the trusted template, experiment identities and TRAIN partition.

        The returned digest must be pinned outside candidate-controlled artifacts.
        Identity values are content SHA256s, not declarations of native custody.
        """
        _, original, _ = self._split(base_template)
        result = {'schema': self.CONTRACT_SCHEMA, 'strategy': self.STRATEGY, 'candidate_kind': self.KIND,
            'allowed_module': self.MODULE, 'base_template': base_template,
            'base_template_sha256': text_sha(base_template), 'original_instruction': original,
            'frozen': copy.deepcopy(frozen), 'feedback_query_ids': list(feedback_query_ids),
            'excluded_query_ids': list(excluded_query_ids),
            'max_instruction_tokens': max_instruction_tokens, 'max_template_tokens': max_template_tokens}
        result['contract_sha256'] = sha(result)
        self.validate_contract(result, expected_contract_sha256=result['contract_sha256'])
        return result


    def validate_contract(self, contract, *, expected_contract_sha256):
        _require(isinstance(contract, dict) and set(contract) == self.CONTRACT_FIELDS, 'exact prompt contract required')
        _require(_digest(expected_contract_sha256) and contract['contract_sha256'] == expected_contract_sha256
            and sha({k: v for k, v in contract.items() if k != 'contract_sha256'}) == expected_contract_sha256,
            'externally pinned prompt contract differs')
        _require((contract['schema'], contract['strategy'], contract['candidate_kind'], contract['allowed_module'])
            == (self.CONTRACT_SCHEMA, self.STRATEGY, self.KIND, self.MODULE), 'unsupported prompt strategy, kind or module')
        frozen = contract['frozen']
        _require(isinstance(frozen, dict) and set(frozen) == self.FROZEN_FIELDS and all(_digest(x) for x in frozen.values()),
                 'exact frozen knowledge/P/supply/model/decoding/interface/tokenizer identities required')
        for key, upper in (('max_instruction_tokens', 2048), ('max_template_tokens', 44032)):
            _require(type(contract[key]) is int and 0 < contract[key] <= upper, 'prompt token bound cannot widen')
        feedback = _ids(contract['feedback_query_ids'], 'feedback IDs', train=True)
        excluded = _ids(contract['excluded_query_ids'], 'excluded IDs')
        _require(feedback and not set(feedback).intersection(excluded), 'feedback and excluded IDs must be disjoint')
        _, original, _ = self._split(contract['base_template'])
        _require(original == contract['original_instruction']
            and text_sha(contract['base_template']) == contract['base_template_sha256'], 'original template changed')
        self._instruction(original, contract)
        return contract


    def _decode_mutation(self, text):
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
        _require(isinstance(value, dict) and set(value) == self.MUTATION_FIELDS, 'exact mutation fields required')
        return value


    def validate_mutation(self, text, *, contract, expected_contract_sha256, expected_parent_sha256):
        self.validate_contract(contract, expected_contract_sha256=expected_contract_sha256)
        mutation = self._decode_mutation(text)
        _require(mutation['schema'] == self.MUTATION_SCHEMA and mutation['module'] == self.MODULE,
                 'unsupported mutation schema or module')
        _require(_digest(expected_parent_sha256) and mutation['parent_sha256'] == expected_parent_sha256,
                 'mutation parent differs')
        ids = _ids(mutation['evidence_query_ids'], 'mutation evidence IDs', train=True)
        _require(ids and set(ids) <= set(contract['feedback_query_ids']), 'mutation references must be feedback-only')
        diagnosis = _text(mutation['diagnosis'], 'diagnosis')
        _require(not any(q in diagnosis for q in contract['excluded_query_ids']), 'diagnosis leaks excluded query IDs')
        mentioned = {value.rstrip('.') for value in
                     re.findall(QUERY_ID, diagnosis, re.I)}
        _require(mentioned <= set(ids), 'diagnosis references unlisted or non-feedback query IDs')
        self._instruction(mutation[self.instruction_field], contract)
        return mutation


    def _candidate(self, contract, instruction, *, parent_sha256, mutation_text, mutation, count_tokens):
        self._instruction(instruction, contract)
        before, _, after = self._split(contract['base_template'])
        template = before + instruction + after
        counts = []
        for text, bound in ((instruction, contract['max_instruction_tokens']), (template, contract['max_template_tokens'])):
            count = count_tokens(text)
            _require(type(count) is int and 0 < count <= bound, 'tokenizer count invalid or prompt exceeds bound; no truncation')
            counts.append(count)
        result = {'schema': self.CANDIDATE_SCHEMA, 'strategy': self.STRATEGY, 'candidate_kind': self.KIND,
            'module': self.MODULE, 'contract_sha256': contract['contract_sha256'], 'parent_sha256': parent_sha256,
            'mutation_sha256': None if mutation_text is None else text_sha(mutation_text),
            'mutation': copy.deepcopy(mutation), 'frozen': copy.deepcopy(contract['frozen']),
            'original_instruction': contract['original_instruction'], self.instruction_field: instruction,
            'instruction_sha256': text_sha(instruction), 'full_template': template,
            'full_template_sha256': text_sha(template), 'instruction_tokens': counts[0], 'template_tokens': counts[1]}
        result['candidate_sha256'] = sha(result)
        return result


    def seed_candidate(self, *, contract, expected_contract_sha256, count_tokens):
        self.validate_contract(contract, expected_contract_sha256=expected_contract_sha256)
        return self._candidate(contract, contract['original_instruction'], parent_sha256=None,
                          mutation_text=None, mutation=None, count_tokens=count_tokens)


    def _parent_pin(self, parent, contract, expected_parent_sha256, count_tokens):
        _require(isinstance(parent, dict) and set(parent) == self.CANDIDATE_FIELDS, 'exact parent candidate required')
        _require(_digest(expected_parent_sha256) and parent['candidate_sha256'] == expected_parent_sha256
            and sha({k: v for k, v in parent.items() if k != 'candidate_sha256'}) == expected_parent_sha256,
            'externally pinned parent differs')
        _require((parent['schema'], parent['strategy'], parent['candidate_kind'], parent['module'])
            == (self.CANDIDATE_SCHEMA, self.STRATEGY, self.KIND, self.MODULE), 'parent strategy or kind differs')
        _require(parent['contract_sha256'] == contract['contract_sha256'] and parent['frozen'] == contract['frozen']
            and parent['original_instruction'] == contract['original_instruction'], 'parent frozen contract differs')
        expected = self._candidate(contract, parent[self.instruction_field], parent_sha256=parent['parent_sha256'],
                              mutation_text=None, mutation=parent['mutation'], count_tokens=count_tokens)
        for key in ('full_template', 'full_template_sha256', 'instruction_sha256', 'instruction_tokens', 'template_tokens'):
            _require(parent[key] == expected[key], 'parent template or tokenizer identity differs')


    def derive_candidate(self, mutation_text, *, parent, expected_parent_sha256, contract,
                         expected_contract_sha256, count_tokens):
        """Derive only the allowed block; this is not candidate acceptance/release."""
        mutation = self.validate_mutation(mutation_text, contract=contract,
            expected_contract_sha256=expected_contract_sha256, expected_parent_sha256=expected_parent_sha256)
        self._parent_pin(parent, contract, expected_parent_sha256, count_tokens)
        _require(mutation[self.instruction_field] != parent[self.instruction_field], 'rejected-noop: unchanged instruction')
        return self._candidate(contract, mutation[self.instruction_field], parent_sha256=expected_parent_sha256,
                          mutation_text=mutation_text, mutation=mutation, count_tokens=count_tokens)


    def verify_candidate(self, candidate, *, contract, expected_contract_sha256, count_tokens,
                         parent=None, expected_parent_sha256=None, mutation_text=None):
        """Replay from external anchors and exact observed mutation bytes.

        A derived parent's pin must already be established by prior verified search
        custody. This local verifier does not establish an entire native lineage.
        """
        _require(isinstance(candidate, dict) and set(candidate) == self.CANDIDATE_FIELDS, 'exact prompt candidate required')
        if parent is None:
            _require(expected_parent_sha256 is None and mutation_text is None, 'seed cannot carry mutation provenance')
            expected = self.seed_candidate(contract=contract, expected_contract_sha256=expected_contract_sha256,
                                      count_tokens=count_tokens)
        else:
            expected = self.derive_candidate(mutation_text, parent=parent, expected_parent_sha256=expected_parent_sha256,
                contract=contract, expected_contract_sha256=expected_contract_sha256, count_tokens=count_tokens)
        _require(candidate == expected, 'candidate differs from pinned template and observed mutation replay')
        return copy.deepcopy(expected)


    def executable_template(self, candidate, *, contract, expected_contract_sha256, count_tokens,
                            parent=None, expected_parent_sha256=None, mutation_text=None):
        """Compile verified editing metadata to the ordinary node template.

        The two delimiters are edit boundaries, not extra model instructions. The
        seed compiles to the original template bytes when the contract was formed
        by inserting delimiters around an existing instruction. Native FIT must
        still measure the complete runtime request, and release is caller-owned.
        """
        verified = self.verify_candidate(candidate, contract=contract,
            expected_contract_sha256=expected_contract_sha256, count_tokens=count_tokens,
            parent=parent, expected_parent_sha256=expected_parent_sha256, mutation_text=mutation_text)
        before, instruction, after = self._split(verified['full_template'])
        return before.removesuffix(self.BLOCK_START) + instruction + after.removeprefix(self.BLOCK_END)


def for_contract(contract):
    from . import solver_prompt_candidate, perception_prompt_candidate
    options = {solver_prompt_candidate.CONTRACT_SCHEMA: solver_prompt_candidate,
               perception_prompt_candidate.CONTRACT_SCHEMA: perception_prompt_candidate}
    for node in ('slot', 'decision'):
        instance = PromptCandidateContract(node)
        options[instance.CONTRACT_SCHEMA] = instance
    try:
        return options[contract['schema']]
    except (KeyError, TypeError) as exc:
        raise ValueError('unsupported prompt contract schema') from exc
