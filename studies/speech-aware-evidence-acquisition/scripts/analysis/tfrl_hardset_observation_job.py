"""Derive one finite H1 error recheck from a sealed full P/S observation.

CPU-only preparation: no model client, service changes, admission or registration.
The unchanged deterministic lock is a conditional execution recheck, not a new
independent stochastic draw and not a statistically confirmed hardset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

NAME = 'slurp-tfrl-h1-error-recheck-r2-v1'
RUN = 'SAEA-SLURP-TFRL-H1-ERROR-RECHECK-R2-V1'
SOURCE_RUN = 'SAEA-SLURP-TFRL-H1-FRESH-OBSERVATION-R1-V1'
SCORER = 'slurp-slu-official-train'
SCHEMA = 'saea-h1-conditional-error-recheck-v1'
SOURCE_KEYS = {'manifest', 'pool', 'queue', 'h1', 'h2', 'scores', 'run_manifest',
               'model', 'baseline', 'experiment', 'dataset', 'attempt'}


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(rows):
    result = {r['sample_id']: r for r in rows}
    require(len(result) == len(rows), 'duplicate sample identity')
    return result


def coverage(rows):
    intents = Counter(r['intent'] for r in rows)
    slots = Counter(s for r in rows for s in set(r['slot_types']))
    relations = {(r['intent'], s) for r in rows for s in set(r['slot_types'])}
    return {'rows': len(rows), 'intents': len(intents), 'slot_types': len(slots),
            'observed_relations': len(relations)}


def task_error(score):
    intent = score['scenario_action_exact']
    require(type(intent) in (int, float) and intent in (0, 1), 'exact intent required')
    counts = [score['slot_f1_exact_'+k] for k in ('tp', 'fp', 'fn')]
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                and v == int(v) for v in counts), 'valid exact slot counts required')
    return not (intent == 1 and counts[1:] == [0, 0])


def build(bundle, *, expected_manifest_sha256, expected_pool_sha256):
    """Verify pinned read-only inputs and emit a single bounded held job."""
    require(set(bundle) == SOURCE_KEYS, 'exact source bundle required')
    documents = {}
    for name, record in bundle.items():
        require(digest(record['text'].encode()) == record['sha256'], 'source bytes changed')
        documents[name] = ([json.loads(s) for s in record['text'].splitlines() if s]
                           if name in ('pool', 'queue') else json.loads(record['text']))
    require(bundle['run_manifest']['sha256'] == expected_manifest_sha256
            and bundle['pool']['sha256'] == expected_pool_sha256, 'trusted source pin differs')
    manifest, attempt = documents['run_manifest'], documents['attempt']
    require(manifest['run_id'] == SOURCE_RUN and manifest['status'] == 'COMPLETED'
            and manifest['attempt']['attempt_id'] == 'generation-v1'
            and attempt['run_id'] == SOURCE_RUN and attempt['attempt_id'] == 'generation-v1'
            and attempt['state'] == 'FINALIZED', 'finalized original observation required')
    require(manifest['scores']['sha256'] == bundle['scores']['sha256'],
            'sealed score hash differs')
    require(documents['manifest']['artifacts']['candidate-pool.jsonl'] == bundle['pool']['sha256']
            and documents['manifest']['artifacts']['difficulty-observation-queue.jsonl']
            == bundle['queue']['sha256'], 'fixed candidate manifest changed')
    h1 = indexed(documents['h1']['items']); h2 = indexed(documents['h2']['items'])
    pool = indexed(documents['pool']); queue = indexed(documents['queue'])
    media = indexed(documents['dataset']['items'])
    uid = lambda s: ':'.join(s.split(':')[:3])
    require(len(h1) == 313 and len(h2) == 312 and len(pool) == 625
            and set(h1) | set(h2) == set(pool) == set(media)
            and not ({uid(s) for s in h1} & {uid(s) for s in h2})
            and all(s.startswith('slurp:train:') for s in pool),
            'fixed disjoint H1/H2 within original625 required')
    require(not ({media[s]['audio_sha256'] for s in h1}
                 & {media[s]['audio_sha256'] for s in h2}), 'audio overlap')
    require(set(queue) <= set(pool) and all(r['model_calls_authorized'] is False
            for r in queue.values()), 'original metadata-only queue required')
    scores = documents['scores']
    require(set(scores) == {SCORER, 'cost'} and manifest['scorers'] == [SCORER, 'cost'],
            'official source scorers required')
    table = indexed(scores[SCORER]['per_sample'])
    require(set(table) == set(h1), 'full H1 scored coverage required')
    require(manifest['cost']['calls_used'] == attempt['actual']['calls'] == 626,
            'complete original P/S spend required')
    model, base = documents['model'], documents['baseline']
    require(model['decoding_params'] == {'temperature': 0, 'seed': 20260913, 'max_tokens': 4096},
            'exact deterministic lock decoding required')
    require(base['serving_slot_tokens'] == 49152, 'source context must be48k')
    roles = base['native_episode']['roles']
    require(roles['perception']['max_iterations'] == roles['solver']['max_iterations'] == 1
            and roles['planner']['max_iterations'] == roles['reviewer']['max_iterations'] == 0,
            'original two-role P/S graph required')
    block = base['native_episode']['consumption_module']
    require(not block['columns'] and block.get('review_loop') == 'off'
            and not any(c.get('state') == 'on' and c['component_id'] == 'tfgrpo-solver-experience'
                        for c in block['components']), 'unchanged experience-free lock required')
    selected = [sid for sid in h1 if task_error(table[sid])]
    require(selected, 'no measured H1 errors to recheck')
    audio = []
    for sid in selected:
        r, m = h1[sid], media[sid]
        require(type(r['audio_seconds']) in (int, float)
                and math.isfinite(r['audio_seconds']) and r['audio_seconds'] > 0
                and r['audio_seconds'] == m['audio_seconds']
                and '/slurp_real/' in m['media_relpath'], 'stable real media required')
        audio.append(r['audio_seconds'])
    # This script emits held descriptors; shared registration must validate them.
    roster = {'schema': 'saea-h1-error-recheck-roster-v1', 'roster_id': RUN,
              'source_roster_sha256': bundle['h1']['sha256'],
              'source_scores_sha256': bundle['scores']['sha256'],
              'items': [copy.deepcopy(h1[s]) for s in selected],
              'selection': 'All exact-task errors from full fixed H1 observation, original order.',
              'role': 'TRAIN H1 conditional recheck; never held-out evaluation'}
    prospective_base = copy.deepcopy(base)
    prospective_base.pop('prepared_phase_supply', None)
    prospective_base['native_contact_fit'].update(run_id=RUN+'-FIT', attempt_id='fit-v1',
                                                 contact_cap=2*len(selected))
    prospective_base['what_this_is'] = 'One finite H1 error recheck; fixed temperature0 lock; no stochastic hardness claim.'
    observations = [{'sample_id': s, 'full_ps_observations': 1,
                     'observed_task_errors': int(task_error(table[s])),
                     'eligible_for_next_pass': s in set(selected)} for s in h1]
    plan = {'schema': SCHEMA, 'job': NAME, 'run_id': RUN, 'rows': len(selected),
            'source_run_id': SOURCE_RUN, 'source_rows': 313,
            'source_pins': {k: v['sha256'] for k, v in bundle.items()},
            'original_queue_rows': len(queue), 'original_queue_h1_rows': len(set(queue) & set(h1)),
            'original_queue_h2_excluded': len(set(queue) & set(h2)),
            'old_error_candidates_now_correct': len((set(queue) & set(h1))-set(selected)),
            'new_h1_errors_outside_original_queue': len(set(selected)-set(queue)),
            'first_observation_missing_h1_rows': 0, 'existing_full_ps_observations_per_h1': 1,
            'planned_additional_passes': 1, 'normal_generation_calls': 2*len(selected),
            'fallback_inclusive_call_cap': 4*len(selected), 'fit_call_cap': 2*len(selected),
            'audio_seconds_unique': math.fsum(audio),
            'prior_source_cost': copy.deepcopy(manifest['cost']),
            'new_model_calls': 0, 'new_tokenizer_calls': 0,
            'model_decoding': copy.deepcopy(model['decoding_params']),
            'coverage': coverage([pool[s] for s in selected]),
            'sample_ids': selected, 'conditional_replication': True,
            'independent_stochastic_draws_claimed': False, 'hardness_confirmed': False,
            'statistical_posterior_updated': False, 'test_labels_read': False,
            'h2_scores_read': False, 'readiness': 'held-not-registered',
            'generation_authorized': False,
            'stop_rule': 'One additional full P/S pass only. Analyze agreement/differences before any further job.',
            'analysis': 'Report reproducible errors and changed outcomes conditional on fixed deployment; do not estimate iid error probabilities.',
            'remaining': ['shared workload validation for new finite job identity',
                          'prepare first-P supply using unchanged dataset view',
                          'register/admit existing shared runtime with held descriptor',
                          'restore task4096 serving identity under guard/quiet/no lease',
                          'retain native FIT/contact custody and detached runner']}
    return {'plan': plan, 'roster': roster, 'baseline': prospective_base,
            'source_model': copy.deepcopy(model), 'observations': observations}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--expected-manifest-sha256', required=True)
    p.add_argument('--expected-pool-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = build(json.loads(args.bundle.read_bytes()),
                   expected_manifest_sha256=args.expected_manifest_sha256,
                   expected_pool_sha256=args.expected_pool_sha256)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, doc in result.items():
        (args.out/(name+'.json')).write_bytes(raw(doc))
    print(json.dumps({k: v for k, v in result['plan'].items()
                      if k not in ('sample_ids', 'source_pins', 'prior_source_cost')}, sort_keys=True))


if __name__ == '__main__':
    main()
