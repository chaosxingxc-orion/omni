"""Versioned bounded ACE wire format; never repairs historical responses."""
import copy
import json

import pytest

from speech_aware_evidence_acquisition.rails import ace_incremental as m
from test_ace_incremental import fixture, runner, reflection, delta


def compact_reflection(index=0):
    value = reflection(index)
    value['schema'] = 'saea-ace-reflection-v3'
    for row in value['findings'] + value['signals']:
        row['refs'] = ['r0']
        del row['trajectory_ids']
    value['findings'][0]['finding_id'] = 'f0'
    return value


def compact_delta():
    value = delta()
    value['schema'] = 'saea-ace-delta-v3'
    value['deltas'][0]['finding_ids'] = ['f0']
    return value


def test_compact_identity_preserves_evidence_and_legacy_bytes():
    _, old = fixture()
    _, doc = fixture(reflector_contract=m.COMPACT_CONTRACT_V3)
    assert old['input_sha256'] != doc['input_sha256']
    assert old['groups'] == doc['groups'] and old['historical_usage'] == doc['historical_usage']
    prompt = m.reflect_messages(doc, 0, doc['parent']['items'], lambda messages: 10)
    payload = json.loads(prompt['messages'][1]['content'])
    assert payload['trajectory_references'] == {'r0': 'T0'}
    assert payload['original_group'] == doc['groups'][0]
    assert '"schema":"saea-ace-reflection-v3","findings":[],"signals":[]' in prompt['messages'][0]['content']
    assert 'trajectory_references' not in json.loads(m.reflect_messages(old, 0, old['parent']['items'], lambda messages: 10)['messages'][1]['content'])


def test_compact_refs_decode_exactly_and_do_not_mutate_response():
    _, doc = fixture(reflector_contract=m.COMPACT_CONTRACT_V3)
    response = compact_reflection(); saved = copy.deepcopy(response)
    checked = m.validate_reflection(json.dumps(response), doc['groups'][0], doc['parent']['items'], 4, m.COMPACT_CONTRACT_V3)
    assert checked['findings'][0]['trajectory_ids'] == ['T0']
    assert checked['signals'][0]['trajectory_ids'] == ['T0']
    assert response == saved
    with pytest.raises(ValueError):
        m.validate_reflection(json.dumps(response), doc['groups'][0], doc['parent']['items'], 4)
    with pytest.raises(ValueError):
        m.validate_reflection(json.dumps(reflection()), doc['groups'][0], doc['parent']['items'], 4, m.COMPACT_CONTRACT_V3)


@pytest.mark.parametrize('fault', ['long_lesson', 'long_reason', 'foreign_ref', 'duplicate_ref',
    'long_id', 'many', 'missing_schema', 'inner', 'unknown_item', 'truncated'])
def test_compact_reflection_rejects_bad_output(fault):
    _, doc = fixture(reflector_contract=m.COMPACT_CONTRACT_V3)
    value = compact_reflection()
    if fault == 'long_lesson': value['findings'][0]['lesson'] = '字' * 201
    elif fault == 'long_reason': value['signals'][0]['reason'] = 'x' * 121
    elif fault == 'foreign_ref': value['findings'][0]['refs'] = ['T0']
    elif fault == 'duplicate_ref': value['findings'][0]['refs'] = ['r0', 'r0']
    elif fault == 'long_id': value['findings'][0]['finding_id'] = 'F1'
    elif fault == 'many': value['findings'] *= 3
    elif fault == 'missing_schema': del value['schema']
    elif fault == 'inner': value = value['findings'][0]
    elif fault == 'unknown_item': value['signals'][0]['item_id'] = 'invented'
    text = json.dumps(value)
    if fault == 'truncated': text = text[:-1]
    with pytest.raises(ValueError):
        m.validate_reflection(text, doc['groups'][0], doc['parent']['items'], 4, m.COMPACT_CONTRACT_V3)


@pytest.mark.parametrize('fault', ['long_content', 'long_reason', 'many', 'old_schema', 'extra'])
def test_compact_delta_rejects_oversize_and_schema_drift(fault):
    _, doc = fixture(reflector_contract=m.COMPACT_CONTRACT_V3)
    value = compact_delta()
    if fault == 'long_content': value['deltas'][0]['content'] = 'x' * 201
    elif fault == 'long_reason': value['deltas'][0]['reason'] = 'x' * 121
    elif fault == 'many': value['deltas'] *= 3
    elif fault == 'old_schema': value['schema'] = 'saea-ace-delta-v1'
    else: value['extra'] = True
    checked = m.validate_reflection(json.dumps(compact_reflection()), doc['groups'][0], doc['parent']['items'], 4, m.COMPACT_CONTRACT_V3)
    with pytest.raises(ValueError): m.merge_delta(json.dumps(value), doc['parent']['items'], checked, doc['bounds'])


def test_compact_two_stage_known_response_replay_keeps_costs(tmp_path):
    answers = {(i, stage): compact_reflection(i) if stage == m.REFLECT else compact_delta()
               for i in range(2) for stage in (m.REFLECT, m.CURATE)}
    _, doc, run, calls, _ = runner(tmp_path, reflector_contract=m.COMPACT_CONTRACT_V3, answers=answers)
    first = run()
    assert run() == first and len(calls) == 4
    assert first['cumulative_known_usage']['calls'] == 316
    assert not first['semantic_support_verified'] and not first['deployment_accepted']
    curator = json.loads(calls[1]['messages'][1]['content'])
    assert curator['reflection']['findings'][0]['trajectory_ids'] == ['T0']


def test_compact_invalid_known_response_is_not_resampled(tmp_path):
    bad = compact_reflection(); del bad['schema']
    _, _, run, calls, _ = runner(tmp_path, reflector_contract=m.COMPACT_CONTRACT_V3,
        answers={(0, m.REFLECT): bad})
    for _ in range(2):
        with pytest.raises(ValueError): run()
    assert len(calls) == 1


def test_compact_empty_result_and_empty_library_signals():
    _, doc = fixture(reflector_contract=m.COMPACT_CONTRACT_V3, empty_parent=True)
    empty = {'schema': 'saea-ace-reflection-v3', 'findings': [], 'signals': []}
    checked = m.validate_reflection(json.dumps(empty), doc['groups'][0], {}, 4, m.COMPACT_CONTRACT_V3)
    merged = m.merge_delta(json.dumps({'schema': 'saea-ace-delta-v3', 'deltas': [], 'unresolved_conflicts': []}),
        {}, checked, doc['bounds'])
    assert merged['items'] == {} and not merged['semantic_support_verified']
    with pytest.raises(ValueError):
        m.validate_reflection(json.dumps(compact_reflection()), doc['groups'][0], {}, 4, m.COMPACT_CONTRACT_V3)


def test_configured_character_limits_bind_identity_prompt_and_parser():
    _, default = fixture(reflector_contract=m.COMPACT_CONTRACT_V3)
    _, doc = fixture(reflector_contract=m.COMPACT_CONTRACT_V3,
        compact_text_chars=512, compact_reason_chars=300)
    assert doc['input_sha256'] != default['input_sha256']
    assert 'compact_text_chars' not in default['bounds']
    response = compact_reflection()
    response['findings'][0]['lesson'] = '字' * 512
    response['signals'][0]['reason'] = '理' * 300
    encoded = json.dumps(response)
    checked = m.validate_reflection(encoded, doc['groups'][0], doc['parent']['items'], 4,
        m.COMPACT_CONTRACT_V3, bounds=doc['bounds'])
    assert checked['findings'][0]['lesson'] == '字' * 512
    with pytest.raises(ValueError, match='lesson'):
        m.validate_reflection(encoded, doc['groups'][0], doc['parent']['items'], 4, m.COMPACT_CONTRACT_V3)
    reflect_prompt = m.reflect_messages(doc, 0, doc['parent']['items'], lambda messages: 10)['messages'][0]['content']
    curate_prompt = m.curate_messages(doc, 0, doc['parent']['items'], checked, {}, lambda messages: 10)['messages'][0]['content']
    assert 'lesson at most 512 Unicode characters; reason at most 300' in reflect_prompt
    assert 'content at most 512 Unicode characters; every reason at most 300' in curate_prompt
    value = compact_delta()
    value['deltas'][0].update(content='字' * 512, reason='理' * 300)
    merged = m.merge_delta(json.dumps(value), doc['parent']['items'], checked, doc['bounds'])
    assert '字' * 512 in merged['items'].values()
    value['deltas'][0]['reason'] += '超'
    with pytest.raises(ValueError, match='reason'):
        m.merge_delta(json.dumps(value), doc['parent']['items'], checked, doc['bounds'])
    response['findings'][0]['lesson'] += '超'
    with pytest.raises(ValueError, match='lesson'):
        m.validate_reflection(json.dumps(response), doc['groups'][0], doc['parent']['items'], 4,
            m.COMPACT_CONTRACT_V3, bounds=doc['bounds'])


@pytest.mark.parametrize('key', ['compact_text_chars', 'compact_reason_chars'])
@pytest.mark.parametrize('value', [0, -1, 4097, True, 1.5, '512'])
def test_invalid_configured_character_limits_rejected_before_contact(key, value):
    with pytest.raises(ValueError, match='character bounds'):
        fixture(reflector_contract=m.COMPACT_CONTRACT_V3, **{key: value})


def test_character_limits_cannot_silently_change_legacy_contract():
    with pytest.raises(ValueError, match='compact contract'):
        fixture(compact_text_chars=512)
    with pytest.raises(ValueError, match='compact contract'):
        fixture(reflector_contract=m.REFLECTOR_CONTRACT_V2, compact_reason_chars=300)
