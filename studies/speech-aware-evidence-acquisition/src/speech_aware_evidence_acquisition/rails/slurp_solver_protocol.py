"""Label-only Solver JSON schema; never constrains fillers by observed answers."""
from collections.abc import Mapping, Sequence

FORMAT_ID = 'slurp-solver-paired-label-json-v1'


def response_format(actions_by_scenario, entity_types):
    """Preserve actual scenario/action pairs rather than taking a Cartesian product."""
    if not isinstance(actions_by_scenario, Mapping) or not actions_by_scenario:
        raise ValueError('nonempty paired label inventory required')
    pairs = {}
    for scenario, actions in actions_by_scenario.items():
        if (not isinstance(scenario, str) or not scenario or not isinstance(actions, Sequence)
                or isinstance(actions, (str, bytes)) or not actions
                or any(not isinstance(a, str) or not a for a in actions)
                or len(actions) != len(set(actions))):
            raise ValueError('unique nonempty paired labels required')
        pairs[scenario] = sorted(actions)
    if (not isinstance(entity_types, Sequence) or isinstance(entity_types, (str, bytes))
            or not entity_types or any(not isinstance(t, str) or not t for t in entity_types)
            or len(entity_types) != len(set(entity_types))):
        raise ValueError('unique entity types required')
    entity = {'type': 'object', 'additionalProperties': False,
              'required': ['type', 'filler'], 'properties': {
                  'type': {'type': 'string', 'enum': sorted(entity_types)},
                  'filler': {'type': 'string'}}}
    # Disjoint scenario constants make oneOf exact. No intent-specific slot
    # exclusions: observed associations are not an exhaustive legality schema.
    branches = [{'type': 'object', 'additionalProperties': False,
                 'required': ['scenario', 'action', 'entities'], 'properties': {
                     'scenario': {'type': 'string', 'const': scenario},
                     'action': {'type': 'string', 'enum': actions},
                     'entities': {'type': 'array', 'items': entity}}}
                for scenario, actions in sorted(pairs.items())]
    return {'type': 'json_schema', 'json_schema': {
        'name': 'slurp_solver_paired_labels', 'strict': True,
        'schema': {'oneOf': branches}}}


def registered_response_format(lock, root, **params):
    """Use the same governed TRAIN ontology as the already-supplied label list."""
    if params:
        raise ValueError('Solver protocol does not accept sample-specific parameters')
    from ..data.slurp import load_slurp_ontology
    ontology = load_slurp_ontology(lock, root)
    return response_format(ontology.actions_by_scenario, tuple(ontology.entity_types))
