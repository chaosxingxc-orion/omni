"""Audit released TRAIN exemplar/card associations without inference or new sampling."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def audit(knowledge, study):
    identities = {}

    def read(name, snapshot=False):
        raw = (knowledge / name).read_bytes()
        doc = json.loads(raw)
        identities[name] = {'file_sha256': hashlib.sha256(raw).hexdigest()}
        if snapshot:
            body = {k: v for k, v in doc.items() if k != 'snapshot_sha256'}
            assert digest(body) == doc['snapshot_sha256'], name
            identities[name]['snapshot_sha256'] = doc['snapshot_sha256']
        return doc

    cards = read('authored-slot-contracts-v2.json')
    assert identities['authored-slot-contracts-v2.json']['file_sha256'] == '8994f773c886f2c207cd9db6436507e53e7f7da79ff7bdf441ab3799f244d34c'
    slots = read('slot-contracts-snapshot.json', True)
    intents = read('intent-boundaries-snapshot.json', True)
    exemplars = read('train-exemplars-snapshot.json', True)
    bindings = {(b['intent_id'], row['slot_id']) for row in cards['rows']
                for b in row['intent_bindings']}
    assert len(bindings) == sum(len(r['intent_bindings']) for r in cards['rows'])
    observations = Counter()
    empty = 0
    assert len({v['value_id'] for v in exemplars['values']}) == len(exemplars['values'])
    for value in exemplars['values']:
        # This snapshot uses the locked annotated-exemplar serialization.
        fields = value['content'].split(' | ', 2)
        assert len(fields) == 3 and fields[1] == value['task_target']
        intent, entities = fields[1:]
        if entities == 'none':
            empty += 1
            continue
        types = []
        for item in entities.split('; '):
            match = re.fullmatch(r'([a-z][a-z0-9_]*)=(.+)', item)
            assert match, value['value_id']
            types.append(match[1])
        observations.update((intent, slot) for slot in set(types))
    slot_names = {r['slot_id'] for r in cards['rows']}
    assert len(slot_names) == len(cards['rows'])
    assert {v['value_id'] for v in slots['values']} == {'slot-contract:' + s for s in slot_names}
    sources = {
        'slot_cards': {s for r in cards['rows'] for s in r['source_support']['source_sample_ids']},
        'intent_cards': set(intents['source_sample_ids']),
        'train_exemplars': set(exemplars['source_sample_ids']),
    }
    for name, snapshot in [('slot_cards', slots), ('intent_cards', intents),
                           ('train_exemplars', exemplars)]:
        actual = {s for value in snapshot['values'] for s in value['source_sample_ids']}
        assert actual == set(snapshot['source_sample_ids']) == sources[name]
    utterance = lambda sample: ':'.join(sample.split(':', 3)[:3])
    overlaps, populations = {}, {}
    for name, filename in [('fold_a', 'slurp-speech-native-evo-fold-a-1080'),
                           ('fold_b', 'slurp-speech-native-evo-fold-b-1080'),
                           ('construction', 'slurp-speech-native-stage3-construction-r01c')]:
        raw = (study / 'configs/dataset' / (filename + '.json')).read_bytes()
        ids = {r['sample_id'] for r in json.loads(raw)['items']}
        populations[name] = {'count': len(ids), 'file_sha256': hashlib.sha256(raw).hexdigest()}
        overlaps[name] = {key: {'samples': len(ids & src),
            'utterances': len({utterance(s) for s in ids} & {utterance(s) for s in src})}
            for key, src in sources.items()}
    return {
        'schema': 'omni-knowledge-binding-audit-v1', 'inputs': identities,
        'populations': populations, 'slot_cards': len(cards['rows']),
        'intent_cards': len(intents['values']), 'exemplar_rows': len(exemplars['values']),
        'empty_slot_exemplars': empty, 'nonempty_slot_exemplars': len(exemplars['values']) - empty,
        'declared_pairs': len(bindings), 'observed_pairs': len(observations),
        'observed_pairs_without_binding': [list(x) for x in sorted(set(observations) - bindings)],
        'bindings_without_exemplar_observation': [list(x) for x in sorted(bindings - set(observations))],
        'pairs_by_observed_exemplar_support': dict(sorted(Counter(observations.values()).items())),
        'source_counts': {k: {'samples': len(v), 'utterances': len({utterance(s) for s in v})}
                          for k, v in sources.items()},
        'source_population_overlaps': overlaps,
        'limits': ['Association coverage only within the locked TRAIN exemplar snapshot; not the entire original dataset ontology.',
                   'No original annotation or audio reread, semantic correctness check, current model request or accuracy measurement.',
                   'Source-ID overlap audit is not proof of absence of all historical teacher or ancestor exposure.',
                   'Empty exemplar marker none is counted as an empty annotation, not an entity type or a valid runtime JSON answer.'],
        'model_calls': 0, 'new_samples': 0, 'runtime_changes': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--knowledge', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.knowledge, args.study)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('exemplar_rows', 'empty_slot_exemplars',
          'declared_pairs', 'observed_pairs', 'observed_pairs_without_binding',
          'bindings_without_exemplar_observation', 'source_population_overlaps')}))
