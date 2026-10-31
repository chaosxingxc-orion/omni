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
    composed = cm.compile_module(block, carrier_module_hooks(repo_root=repo), repo_root=repo)
    if composed.program_id != 'phase-native-module-v1-c2' or composed.program_ceiling != 2:
        raise ValueError('lock compiler did not produce two contacts')
    out = copy.deepcopy(source)
    out['policy'] = composed.program_id
    # Preserve the source's actual knowledge, prompt and preparation pins.
    # Replacing all policy-owned fields would silently restore lock defaults.
    del out['native_episode']['roles']['perception']['sampling_variants']
    out['native_episode']['consumption_module'] = block
    for role in ('perception', 'solver'):
        if out['native_episode']['roles'][role]['max_iterations'] != 1:
            raise ValueError('lock role is not one visit')
    if native['max_revisions'] != 0 or native['review_loop'] != 'off':
        raise ValueError('unexpected iterative lock')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    source = json.loads(raw)
    candidate = derive(source)
    result = {'schema': 'saea-lock-only-candidate-v1', 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'candidate_baseline': candidate, 'status': 'PREPARATION_ONLY',
              'new_model_calls': 0, 'remaining': ['fresh preparation and FIT identity',
              'request equivalence', 'registered checkpoint projection', 'registration and smoke']}
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')


if __name__ == '__main__':
    main()
