"""Prove a runtime-only implementation repair preserves prepared supply bytes."""
from copy import deepcopy
import json
from .preparation import PreparationError, load_artifact, read_artifact_inputs, digest, canonical_bytes


def verified_runtime_repair(path, *, expected_sha256, inputs, produce):
    """Recompute actual supply; no identity alias or unchecked compatibility flag."""
    old = read_artifact_inputs(path)
    # Validate the old document and output against its full pinned file digest.
    output = load_artifact(path, expected_sha256=expected_sha256, kind='phase-supply', inputs=old)
    current = json.loads(canonical_bytes(inputs))
    comparable = deepcopy(current)
    try:
        before = old['sources']['implementation']['runtime']
        after = current['sources']['implementation']['runtime']
        if before == after or any(not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in (before,after)):
            raise ValueError('distinct implementation digests required')
        comparable['sources']['implementation']['runtime'] = before
    except (KeyError,TypeError,ValueError) as error:
        raise PreparationError('runtime-only recovery identity required') from error
    if comparable != old:
        raise PreparationError('prepared supply recovery changes non-runtime identity')
    rebuilt = produce()
    if canonical_bytes(rebuilt) != canonical_bytes(output):
        raise PreparationError('current runtime rebuilt different prepared supply')
    return output, {'schema':'saea-prepared-supply-runtime-recovery-v1',
        'artifact_sha256':expected_sha256, 'previous_inputs_sha256':digest(old),
        'current_inputs_sha256':digest(current), 'previous_runtime_sha256':before,
        'current_runtime_sha256':after,'output_sha256':digest(output),
        'recomputed_with_current_implementation':True, 'new_model_calls':0}
