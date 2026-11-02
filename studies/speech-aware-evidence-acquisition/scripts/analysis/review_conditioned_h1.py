"""Reproduce fixed-H1 counterexamples; never infer correctness from self-review.

The selected witnesses and decisions are explicit human/agent judgments.
This verifier only binds them to immutable candidate and original TRAIN bytes;
it does not automatically establish semantic entailment or grant admission.
"""
import hashlib
import json
from pathlib import Path

CANDIDATE_SHA = 'a8d10e9bd79f7cde4565405a72daa0387c6f63473f44e3646950f49607ea9f23'
ROSTER_SHA = '7f540a76dec8eed29cc2295f6090c4551e2910bfd61cc0d8e98f09ba044a51c0'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def review(candidate_path, material_path, roster_path):
    candidate_bytes = Path(candidate_path).read_bytes()
    roster_bytes = Path(roster_path).read_bytes()
    material_bytes = Path(material_path).read_bytes()
    if digest(candidate_bytes) != CANDIDATE_SHA or digest(roster_bytes) != ROSTER_SHA:
        raise ValueError('fixed candidate or H1 roster changed')
    candidate = json.loads(candidate_bytes)
    material = json.loads(material_bytes)
    ids = {r['sample_id'] for r in json.loads(roster_bytes)['items']}
    if (material['roster_sha256'] != ROSTER_SHA or len(ids) != 313
            or set(material['rows']) != ids or any(not s.startswith('slurp:train:') for s in ids)):
        raise ValueError('exact TRAIN H1 material required')
    rules = {r['rule_id']: r for r in candidate['rules']['rules']}
    if set(rules) != {f'conditioned-{i:02d}' for i in range(1, 6)}:
        raise ValueError('exact five-rule candidate required')
    rows = material['rows']

    def witness(sid, intent, present, absent):
        row = rows[sid]
        slots = {(e['type'], e['filler']) for e in row['entities']}
        if row['intent'] != intent or not set(present) <= slots or set(absent) & slots:
            raise ValueError('original H1 counterexample differs')
        return {'sample_id_sha256': digest(sid.encode()),
                'material_row_sha256': digest(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()),
                'intent': intent, 'gold_slot_types': sorted({e['type'] for e in row['entities']})}

    support1 = witness('slurp:train:007347:audio-1501764314', 'calendar_set',
                       [('person', "bola 's")], [])
    boundary1 = witness('slurp:train:006809:audio-1488633715', 'calendar_set',
                        [('event_name', "lily 's wedding")], [('person', "lily 's")])
    support2 = witness('slurp:train:015690:audio-1502100662', 'email_querycontact',
                       [('person', "jimmy johns 's")], [])
    counter4 = witness('slurp:train:010813:audio-1497265008-headset', 'lists_createoradd',
                       [('list_name', 'to do')], [('list_name', 'grocery shopping')])
    counter5a = witness('slurp:train:010000:audio-1501766725', 'cooking_recipe',
                        [('cooking_type', 'poach'), ('ingredient', 'egg')], [('food_type', 'egg')])
    counter5b = witness('slurp:train:010060:audio--1504190640', 'cooking_recipe',
                        [('food_type', 'brisket')], [('cooking_type', 'cook')])
    findings = [
        {'rule_id': 'conditioned-01', 'decision': 'revise-trigger-before-probe',
         'reason': 'Preserve possessive surface only after person evidence; a possessive token alone does not imply person.',
         'support': [support1], 'boundary': [boundary1]},
        {'rule_id': 'conditioned-02', 'decision': 'limited-support-not-validated',
         'reason': 'TRAIN supports business contact typed person in this context; two scoped H1 rows do not prove universal generalization.',
         'support': [support2]},
        {'rule_id': 'conditioned-03', 'decision': 'defer-insufficient-contrast',
         'reason': 'No reviewed positive/negative metaphor pair; physical-reference wording alone is not validated guidance.'},
        {'rule_id': 'conditioned-04', 'decision': 'reject-current-rule',
         'reason': 'Conflates the item added with the list name; H1 gold contradicts the proposal example.',
         'counterexamples': [counter4]},
        {'rule_id': 'conditioned-05', 'decision': 'reject-current-rule',
         'reason': 'Verb-food pattern does not establish cooking_type or distinguish food_type from ingredient.',
         'counterexamples': [counter5a, counter5b]},
    ]
    for finding in findings:
        rule = rules[finding['rule_id']]
        finding['rule_sha256'] = digest(json.dumps(rule, sort_keys=True, separators=(',', ':')).encode())
        finding['source_operation_sha256'] = rule['source']['sha256']
        finding['scope_rows'] = sum(row['intent'] in rule['scope']['task_ids'] for row in rows.values())
    return {'schema': 'saea-conditioned-h1-counterexample-review-v1',
            'status': 'candidate-rejected-before-npu-evaluation', 'eligible_for_h2': False,
            'candidate_sha256': digest(candidate_bytes), 'h1_roster_sha256': ROSTER_SHA,
            'private_material_sha256': digest(material_bytes), 'h1_rows': 313,
            'rules': 5, 'findings': findings, 'new_model_calls': 0, 'new_scoring_calls': 0,
            'h2_error_evidence_used': False, 'test_read': False,
            'limitations': ['Manual semantic judgments with mechanically checked TRAIN witnesses.',
                           'This audit is not a model performance evaluation or statistical coverage proof.',
                           'Source-operation custody remains a separate prerequisite.',
                           'No corrected candidate is created or admitted by this report.']}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('candidate', 'material', 'roster', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    result = review(args.candidate, args.material, args.roster)
    with args.output.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'rules': result['rules'], 'new_model_calls': 0}))
