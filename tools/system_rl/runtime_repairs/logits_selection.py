"""Bound diagnostic observation before generation; never change business wires.

Selections depend only on actual input content and configured node quotas.
Every admission, including exclusions, is durable and bound to one activation.
Missing controls/coverage remain a failed diagnostic, not a narrower PASS.
"""
from pathlib import Path
from dependencies import require
from run_package import pin, read, write
from policy_identity import digest

NODES = ('P', 'S1', 'S2', 'D')


def validate(selection):
    require(set(selection) == {'schema', 'inputs_per_node', 'calls_per_input'},
            'explicit bounded logits selection fields')
    require(selection['schema'] in ('omni-logits-input-selection-v1','omni-logits-input-selection-v2'), 'logits selection schema')
    require(set(selection['inputs_per_node']) == set(NODES), 'observe all four nodes')
    lower=1 if selection['schema'].endswith('v1') else 0
    require(any(selection['inputs_per_node'].values()) and
            all(type(v) is int and lower <= v <= 16 for v in selection['inputs_per_node'].values()),
            'bounded positive input quota per node')
    require(type(selection['calls_per_input']) is int and 2 <= selection['calls_per_input'] <= 16,
            'bounded repeated real draws per observed input')
    return sum(selection['inputs_per_node'].values()) * selection['calls_per_input']


def input_key(packet):
    wire = packet['wire']
    require(packet['stage'] in NODES, 'explicit actual node required for observation')
    # Same content definition as diagnostic_index, before prompt tokenization.
    # Seeds/request identifiers are excluded; all actual messages are retained.
    return digest(dict(node=packet['stage'], messages=wire['messages'],
                       model=wire.get('model'), chat_template_kwargs=wire.get('chat_template_kwargs', {})))


def decide(packet, capture, overlay_root, activation_ref):
    import fcntl
    selection = capture['selection']; validate(selection)
    key = packet['wire_sha256']
    require(key == digest(packet['wire']), 'selection actual wire mismatch')
    root = Path(overlay_root) / 'logits-selection'
    root.mkdir(parents=True, exist_ok=True)
    activation_ref = pin(activation_ref)
    policy_hash = digest(capture)
    identity = input_key(packet)
    target = root / (key + '.json')
    with (root / 'admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            previous = read(target)
            require(previous['wire_sha256'] == key and previous['input_key'] == identity and
                    previous['activation_ref'] == activation_ref and previous['capture_sha256'] == policy_hash,
                    'existing logits selection identity drift')
            return pin(target)
        rows = [read(p) for p in sorted(root.glob('*.json'))]
        require(all(r['activation_ref'] == activation_ref and r['capture_sha256'] == policy_hash for r in rows),
                'selection from another activation/configuration')
        node = packet['stage']
        admitted = [r for r in rows if r['node'] == node and r['selected']]
        identities = {r['input_key'] for r in admitted}
        chosen = identity in identities or len(identities) < selection['inputs_per_node'][node]
        if chosen:
            require(sum(r['input_key'] == identity for r in admitted) < selection['calls_per_input'],
                    'observed input draw bound exceeded; stop before model, never omit a group member')
        return write(target, dict(schema='omni-logits-selection-decision-v1',
            wire_sha256=key, input_key=identity, node=node, activation_ref=activation_ref,
            capture_sha256=policy_hash, selected=chosen, sequence=len(rows),
            reason='input_admitted_before_generation' if chosen else 'node_input_quota',
            outcome_used=False, business_wire_modified=False))


def decision(wire, capture, overlay_root, activation_ref, *, required=False):
    key = digest(wire); path = Path(overlay_root) / 'logits-selection' / (key + '.json')
    if not path.exists():
        require(not required, 'generation call lacks pre-generation observation decision')
        return None
    ref = pin(path); row = read(ref)
    require(row['schema'] == 'omni-logits-selection-decision-v1' and row['wire_sha256'] == key and
            row['activation_ref'] == pin(activation_ref) and row['capture_sha256'] == digest(capture),
            'observation selection custody mismatch')
    require(type(row['selected']) is bool and row['outcome_used'] is False and
            row['business_wire_modified'] is False, 'invalid pre-generation selection')
    return ref


def bind_lease(lease, capture, context):
    """Reuse the original journal, accounting and fence; add only a sidecar."""
    validate(capture['selection'])
    original = lease.transport
    def transport(inner):
        actual = original(inner)
        class Observed:
            def __getattr__(self, name): return getattr(actual, name)
            def submit(self, request_id, packet):
                context['owner_check']()
                decide(packet, capture, context['overlay_root'], context['activation_ref'])
                return actual.submit(request_id, packet)
        return Observed()
    lease.transport = transport
