"""Explicit solver GEPA prompt contract; no model calls or automatic adoption."""
from .node_prompt_candidate import PromptCandidateContract, sha, text_sha, QUERY_ID

_contract = PromptCandidateContract('solver')
BLOCK_START = _contract.BLOCK_START
BLOCK_END = _contract.BLOCK_END
FROZEN_FIELDS = _contract.FROZEN_FIELDS
CONTRACT_FIELDS = _contract.CONTRACT_FIELDS
MUTATION_FIELDS = _contract.MUTATION_FIELDS
CANDIDATE_FIELDS = _contract.CANDIDATE_FIELDS
CONTRACT_SCHEMA = _contract.CONTRACT_SCHEMA
CANDIDATE_SCHEMA = _contract.CANDIDATE_SCHEMA
MUTATION_SCHEMA = _contract.MUTATION_SCHEMA
STRATEGY = _contract.STRATEGY
MODULE = _contract.MODULE
KIND = _contract.KIND
_split = _contract._split
_instruction = _contract._instruction
build_contract = _contract.build_contract
validate_contract = _contract.validate_contract
_decode_mutation = _contract._decode_mutation
validate_mutation = _contract.validate_mutation
_candidate = _contract._candidate
seed_candidate = _contract.seed_candidate
_parent_pin = _contract._parent_pin
derive_candidate = _contract.derive_candidate
verify_candidate = _contract.verify_candidate
executable_template = _contract.executable_template
