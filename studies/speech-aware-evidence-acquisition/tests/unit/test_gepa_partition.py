import pytest
from speech_aware_evidence_acquisition.rails.gepa_partition import build_partition


def row(i, intent='a', utterance=None, audio=None):
    return {'sample_id':f'slurp:train:{i:06d}:audio-{i}',
            'utterance_id':f'slurp:train:{i if utterance is None else utterance:06d}',
            'audio_sha256':f'{i if audio is None else audio:064x}', 'intent':intent}


def test_partition_proportions_stable_order_and_complete_roster():
    rows = [row(i, 'a' if i < 12 else 'b') for i in range(20)]
    a = build_partition(rows, [], target_selection=5, seed=3)
    assert a == build_partition(list(reversed(rows)), [], target_selection=5, seed=3)
    assert a['coverage']['selection'] == {'a':3, 'b':2}
    assert set(a['feedback_ids']).isdisjoint(a['selection_ids'])
    assert set(a['feedback_ids'] + a['selection_ids']) == {r['sample_id'] for r in rows}
    assert not a['selection_is_unseen_validation']


def test_transitive_utterance_and_audio_component_never_split():
    rows = [row(0), row(1, utterance=0), row(2, audio=1), row(3), row(4)]
    d = build_partition(rows, [], target_selection=3, seed=4)
    connected = {r['sample_id'] for r in rows[:3]}
    side = set(d['selection_ids'])
    assert connected <= side or connected.isdisjoint(side)
    assert any(set(c['sample_ids']) == connected for c in d['components'])


def test_closest_reachable_below_target_does_not_split_components():
    rows = [row(i, utterance=i//2) for i in range(6)]
    d = build_partition(rows, [], target_selection=3, seed=1)
    assert d['actual_selection'] == 2
    assert len(d['feedback_ids']) == 4


@pytest.mark.parametrize('field', ['sample_id','utterance_id','audio_sha256'])
def test_h2_or_test_overlap_is_rejected(field):
    rows = [row(i) for i in range(4)]
    other = {k:v for k,v in row(99).items() if k != 'intent'}
    other[field] = rows[0][field]
    with pytest.raises(ValueError, match='exclusion overlap'):
        build_partition(rows, [other], target_selection=2, seed=1)


def test_mixed_intent_audio_duplicates_disclosed_and_grouped():
    rows = [row(0,'a'),row(1,'b',audio=0),row(2,'a'),row(3,'b')]
    d=build_partition(rows,[],target_selection=2,seed=1)
    assert any(c['intents']==['a','b'] and len(c['sample_ids'])==2 for c in d['components'])


def test_duplicate_query_and_bad_audio_rejected():
    with pytest.raises(ValueError, match='duplicate search'):
        build_partition([row(0),row(0)],[],target_selection=1,seed=1)
    bad=row(0);bad['audio_sha256']='unknown'
    with pytest.raises(ValueError, match='audio SHA'):
        build_partition([bad,row(1)],[],target_selection=1,seed=1)


def test_all_connected_cannot_fake_a_split():
    with pytest.raises(ValueError, match='no complete component'):
        build_partition([row(0),row(1,utterance=0)],[],target_selection=1,seed=1)
