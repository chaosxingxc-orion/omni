"""Audit actual same-P Solver inputs before accepting paired GEPA scores.

The caller must first verify the sealed bundle, declared template bytes and arm
contact response hashes (stage3_rollout_score already owns those checks). This
module establishes the additional model-facing input equality, not provenance
of candidate learning or authority to dispatch. No replay exception is added.
"""
import copy
from .tfrl_bank_request_audit import checked_control
from .solver_prompt_candidate import sha

ARMS = ('parent', 'child', 'lock')


def _project(request, instruction):
    value = copy.deepcopy(request)
    control = checked_control(value)
    messages = control.get('messages')
    if (not isinstance(messages, list) or len(messages) != 2
            or messages[1] != {'role': 'user', 'content': instruction}):
        raise ValueError('actual role user query differs from verified rendered template')
    history = value['history']
    if len([h for h in history if h.get('role') == 'control']) != 1:
        raise ValueError('first Solver request requires exactly one framework control')
    states = [h for h in history[:-1] if h.get('role') == 'agent_state']
    if not states:
        raise ValueError('actual shared perception state is absent')
    provenance = value.get('evidence_provenance')
    evidence = value.get('supplied_evidence')
    if not isinstance(provenance, list) or not isinstance(evidence, list) or len(provenance) != len(evidence):
        raise ValueError('complete evidence and provenance required')
    for index, source in enumerate(provenance):
        if source.get('lock_key') == 'rails-supply':
            if source.get('relpath') != f"rails-message/{value['request_id']}/{index}":
                raise ValueError('supply provenance request ownership differs')
            source['relpath'] = 'rails-message/<request>/' + str(index)
    # Only routing identities and independently verified instruction bytes are
    # normalized. Every other request/control/history field must stay equal.
    value.pop('request_id')
    control.pop('contact_id')
    # The role template is the user query. The global task/system instruction
    # remains in both payload and control and must match byte-for-byte.
    control['messages'][1]['content'] = '<verified-template>'
    envelope = copy.deepcopy(history[-1])
    for key in ('content', 'content_sha256', 'contact_id'):
        envelope.pop(key)
    value['history'] = history[:-1] + [{**envelope, 'verified_control': control}]
    return value, states


def verify_inputs(sample_id, arm_contacts, requests, *, rendered_instructions):
    """Verify exactly one primary request per arm, not reask/fallback context.

    ``arm_contacts`` maps each declared arm to its first native execution
    receipt contact. ``requests`` are the request payloads selected by those
    exact request IDs from a bundle-verified trace. Reasks remain separately
    counted and custody-checked by the native bundle; never silently compare
    their different prompts as if they were primary model inputs.
    """
    if (set(arm_contacts) != set(ARMS) or set(requests) != set(ARMS)
            or set(rendered_instructions) != set(ARMS)):
        raise ValueError('exact parent child lock inputs required')
    projections = {}; states = {}
    for arm in ARMS:
        contact, request = arm_contacts[arm], requests[arm]
        if (request.get('sample_id') != sample_id
                or request.get('request_id') != contact.get('request_id')
                or request.get('history', [{}])[-1].get('contact_id') != contact.get('agent_contact_id')):
            raise ValueError('request does not belong to exact row and arm contact')
        projections[arm], states[arm] = _project(request, rendered_instructions[arm])
    if len({requests[a]['request_id'] for a in ARMS}) != 3:
        raise ValueError('three distinct native Solver requests required')
    if any(projections[a] != projections['lock'] for a in ARMS):
        raise ValueError('paired Solver inputs differ outside verified instruction')
    common = projections['lock']
    return {'schema': 'saea-gepa-pair-input-audit-v1', 'sample_id': sample_id,
        'shared_context_sha256': sha(common),
        'shared_agent_state_sha256': sha(states['lock']),
        'shared_supply_sha256': sha({'evidence': common['supplied_evidence'],
                                    'provenance': common['evidence_provenance']}),
        'request_ids': {a: requests[a]['request_id'] for a in ARMS},
        'verified_primary_solver_requests': 3, 'new_model_calls': 0}
