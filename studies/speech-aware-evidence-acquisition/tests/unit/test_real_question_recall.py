import copy

import pytest

from speech_aware_evidence_acquisition.rails.real_question_recall import (
    QuestionSourceError, aggregate_max, validate_members,
)


def fixture():
    sid = 'slurp:train:000001:audio-a'
    facet = dict(facet_id='q1', representation_kind='text',
                 representation='TRAIN utterance 1 (annotated-exemplar retrieval key): "set alarm"',
                 source_sample_ids=[sid])
    value = dict(value_id='value:1', associated_facet_ids=['q1'],
                 source_sample_ids=[sid], task_target='alarm_set',
                 provenance_refs=['governed-jsonl:datasets/slurp/metadata/train.jsonl#slurp_id=1'])
    return {'facets': [facet], 'values': [value]}, {'1': {'slurp_id': 1, 'sentence': 'set alarm', 'scenario': 'alarm', 'action': 'set'}}


def test_real_source_passes_and_exclusion_is_utterance_level():
    snapshot, metadata = fixture()
    assert validate_members(snapshot, metadata, [], {'alarm_set': 'intent:alarm_set'}) == [('q1', '1', 'intent:alarm_set')]
    assert validate_members(snapshot, metadata, ['slurp:train:000001:audio-other'], {'alarm_set': 'intent:alarm_set'}) == []


@pytest.mark.parametrize('change', ['synthetic', 'nontrain', 'text', 'intent', 'untraceable'])
def test_reject_bad_source(change):
    snapshot, metadata = fixture()
    if change == 'synthetic': snapshot['facets'][0]['representation'] = 'synthetic set alarm'
    if change == 'nontrain': snapshot['facets'][0]['source_sample_ids'] = ['slurp:test:000001:audio-a']
    if change == 'text': metadata['1']['sentence'] = 'another sentence'
    if change == 'intent': snapshot['values'][0]['task_target'] = 'alarm_remove'
    if change == 'untraceable': snapshot['values'][0]['provenance_refs'] = []
    with pytest.raises(QuestionSourceError):
        validate_members(snapshot, metadata, [], {'alarm_set': 'intent:alarm_set'})


def test_independent_vectors_deduplicate_intents_and_exclude_self():
    members = [('q1', '1', 'i1'), ('q2', '2', 'i1'), ('q3', '3', 'i2'), ('q4', '4', 'i3')]
    vectors = {'q1': [1, 0], 'q2': [0.8, 0.2], 'q3': [0.7, 0.3], 'q4': [-1, 0]}
    rows = aggregate_max({'slurp:train:000001:audio-x': [[1, 0]]}, vectors, members, k=5)
    result = rows['slurp:train:000001:audio-x']
    assert list(result) == ['i1', 'i2']
    assert 0 < result['i1']['intent-boundary'] < 1
    assert aggregate_max({'slurp:train:000002:audio-x': [[-1, 0]]}, vectors, members, k=5)['slurp:train:000002:audio-x'] == {'i3': {'intent-boundary': 1.0}}


def test_evaluation_query_identity_is_not_train_source_validation():
    members = [('q1', '1', 'i1'), ('q2', '2', 'i2')]
    vectors = {'q1': [1, 0], 'q2': [0, 1]}
    for sid in ('simple-id', 'slurp:test:000001:audio-x'):
        assert list(aggregate_max({sid: [[1, 0]]}, vectors, members, k=5)[sid]) == ['i1']
