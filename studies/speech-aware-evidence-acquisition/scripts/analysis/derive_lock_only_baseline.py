"""Derive a two-contact lock candidate without changing an existing experiment.

This emits preparation input, not a runnable/admitted cell. Request equivalence,
fresh preparation/FIT identities, registration and smoke are still required.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def derive(source, *, repo=ROOT):
    from speech_aware_evidence_acquisition.rails import consumption_module as cm
    from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
    if source['policy'] != 'phase-native-module-v1-c17':
        raise ValueError('requires the reviewed five-branch source')
    native = source['native_episode']
    arms = native['roles']['perception']['sampling_variants']['arms']
    if [a['arm_id'] for a in arms] != ['l1-support', 'l2-union', 'aa', 'lock-rep2', 'lock']:
        raise ValueError('unexpected diagnostic branch inventory')
    if arms[-1] != {'arm_id': 'lock', 'deliver': True}:
        raise ValueError('delivering lock has an unreviewed override')
    block = copy.deepcopy(native['consumption_module'])
    block['columns'] = []
    block.pop('sampling_frame')
    block['prompt_contact_limits'] = 'legacy-c17-lock-v1'
    composed = cm.compile_module(block, carrier_module_hooks(repo_root=repo), repo_root=repo)
    if composed.program_id != 'phase-native-module-v1-c2' or composed.program_ceiling != 2:
        raise ValueError('lock compiler did not produce two contacts')
    out = copy.deepcopy(source)
    out['policy'] = composed.program_id
    # Preserve the source's actual knowledge, prompt and preparation pins.
    # Replacing all policy-owned fields would silently restore lock defaults.
    del out['native_episode']['roles']['perception']['sampling_variants']
    out['native_episode']['consumption_module'] = block
    # c17 leaks 17/9 remaining contacts into P/S prompts; preserve those bytes
    # while the compiled c2 program independently enforces two actual calls.
    out['native_episode']['prompt_contact_limits'] = 'legacy-c17-lock-v1'
    for role in ('perception', 'solver'):
        if out['native_episode']['roles'][role]['max_iterations'] != 1:
            raise ValueError('lock role is not one visit')
    if native['max_revisions'] != 0 or native['review_loop'] != 'off':
        raise ValueError('unexpected iterative lock')
    return out


def budget_candidates(lock, model):
    """Prepare single-node budget contrasts; retain the other node's baseline."""
    from speech_aware_evidence_acquisition.core.reasoning_knobs import validate_thinking_policy
    if lock['policy'] != 'phase-native-module-v1-c2' or 'thinking_policy' in lock:
        raise ValueError('requires the unmodified two-contact budget control')
    if model.get('reasoning_budget_tokens') != 3072 or model.get('answer_reserve_tokens') != 1024:
        raise ValueError('unexpected control budget')
    result = {'control-3072': copy.deepcopy(lock)}
    for node in ('perception', 'solver'):
        for budget in (512, 1024, 2048):
            candidate = copy.deepcopy(lock)
            candidate['thinking_policy'] = {node: {'reasoning_budget_tokens': budget,
                                                   'answer_reserve_tokens': 1024}}
            validate_thinking_policy(candidate['thinking_policy'], {**model, **candidate})
            result[f'{node}-{budget}'] = candidate
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='Prepare single-node thinking budget contrasts too')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    source = json.loads(raw)
    candidate = derive(source)
    result = {'schema': 'saea-lock-only-candidate-v1', 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'candidate_baseline': candidate, 'status': 'PREPARATION_ONLY',
              'new_model_calls': 0, 'remaining': ['fresh preparation and FIT identity',
              'request equivalence', 'registered checkpoint projection', 'registration and smoke']}
    if args.model:
        result['model_sha256'] = hashlib.sha256(args.model.read_bytes()).hexdigest()
        result['budget_candidates'] = budget_candidates(candidate, json.loads(args.model.read_bytes()))
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')


if __name__ == '__main__':
    main()
