"""Offline preparation and real executor reconciliation; no paid API calls."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
import external_exploration as exploration
from test_external_teacher import network


def write(path, value):
    path.write_text(json.dumps(value))
    return exploration.pin(path)


def job(tmp_path):
    evidence = {'query': {'supplied_evidence': {'development_evidence_ref': 'e0'},
                          'task_instruction': 'Task'},
                'rollouts': [{'source_trajectory': {'request': {'history': [
                    {'content': {'development_canonical_json_string': {'v': {'development_evidence_ref': 'e0'}}}}]}},
                             'reward': 0.5}] * 2}
    table = {'e0': ['complete supplied content']}
    evidence.update(development_evidence_dictionary=table,
                    development_evidence_dictionary_sha256=exploration.sha(table))
    source = write(tmp_path / 'source.json', {'document': {'groups': [
        {'query_id': 'slurp:train:one', 'group_sha256': 'a' * 64, 'evidence': evidence}]}})
    context = write(tmp_path / 'context-input.json', {'parent_memory': []})
    config = {'schema': 'saea-external-exploration-job-v1',
              'profile': exploration.pin(ROOT / 'configs/learners/tfrl/reflexion.json'),
              'source': source, 'query_ids': ['slurp:train:one'], 'context': context,
              'teacher': {'schema': 'saea-external-learner-v1', 'endpoint': 'https://example.invalid/v1/chat/completions',
                          'model': 'external-model', 'api_key_env': 'TEST_EXTERNAL_KEY', 'timeout_seconds': 10, 'stream': False},
              'output_root': str(tmp_path / 'prepared')}
    write(tmp_path / 'job.json', config)
    return tmp_path / 'job.json', tmp_path / 'prepared'


def proposals():
    return {'algorithm': 'reflexion', 'candidates': [
        {'id': str(i), 'evidence_query_ids': ['slurp:train:one'], 'hypothesis': 'test',
         'memory': {'trigger': 't', 'action': 'a' + str(i), 'exception': 'e'},
         'retry_protocol': 'fixed inputs', 'transfer_scope': 'same_task_retry', 'validation': 'paired'}
        for i in range(2)]}


def test_expand_canonical_and_cycle():
    assert exploration.expand_evidence({'development_canonical_json_string': {'b': {'development_evidence_ref': 'a'}}}, {'a': [1, '文']}) == '{"b":[1,"文"]}'
    with pytest.raises(ValueError, match='cyclic'):
        exploration.expand_evidence({'development_evidence_ref': 'a'}, {'a': {'development_evidence_ref': 'a'}})
    with pytest.raises(ValueError, match='missing'):
        exploration.expand_evidence({'development_evidence_ref': 'missing'}, {})


def test_gepa_context_keeps_exclusion_roster_local():
    from test_solver_prompt_candidate import fixture
    contract, parent, _, _ = fixture()
    original = json.dumps(contract, sort_keys=True)
    projected = exploration.request_context({'contract': contract, 'parent': parent}, 'gepa')
    assert 'excluded_query_ids' not in projected['contract']
    assert projected['contract']['excluded_query_ids_commitment']['sha256'] == exploration.sha(contract['excluded_query_ids'])
    assert projected['contract']['feedback_query_ids'] == contract['feedback_query_ids']
    assert projected['parent'] == parent
    assert json.dumps(contract, sort_keys=True) == original


def test_prepare_preserves_duplicate_rollouts_and_complete_input(tmp_path):
    config, root = job(tmp_path)
    result = exploration.prepare(config)
    assert result['new_model_calls'] == 0 and result['rollouts'] == 2
    request = exploration.read(root / 'request.json')
    evidence = exploration.request_payload(request)['groups'][0]['evidence']
    assert evidence['query']['supplied_evidence'] == ['complete supplied content']
    assert len(evidence['rollouts']) == 2
    assert evidence['rollouts'][0]['source_trajectory']['request']['history'][0]['content'] == '{"v":["complete supplied content"]}'
    assert not list((root / 'contacts').glob('*/response/intent.json'))


def test_test_split_rejected_before_output(tmp_path):
    path, root = job(tmp_path)
    cfg = exploration.read(path); cfg['query_ids'] = ['slurp:test:one']; write(path, cfg)
    with pytest.raises(ValueError, match='TRAIN'):
        exploration.prepare(path)
    assert not root.exists()


@pytest.mark.parametrize('fault', [None, 'source', 'exposure', 'empty', 'duplicate',
                                 'blank', 'whitespace', 'not-string', 'unhashable', 'not-list'])
def test_development_feedback_is_pinned_and_exposure_bound(tmp_path, fault):
    path, root = job(tmp_path)
    cfg = exploration.read(path)
    feedback = {'schema': 'saea-exploration-development-feedback-v1',
                'source_sha256': cfg['source']['sha256'],
                'exposed_query_ids': cfg['query_ids'],
                'observations': [{'hypothesis': 'Non-exhaustive retrieval is not an authority.'}]}
    if fault == 'source': feedback['source_sha256'] = '0' * 64
    if fault == 'exposure': feedback['exposed_query_ids'] = ['slurp:train:unselected']
    invalid_exposures = {'empty': [], 'duplicate': cfg['query_ids'] * 2,
                         'blank': [''], 'whitespace': ['  '], 'not-string': [None],
                         'unhashable': [{}], 'not-list': cfg['query_ids'][0]}
    if fault in invalid_exposures: feedback['exposed_query_ids'] = invalid_exposures[fault]
    cfg['development_feedback'] = write(tmp_path/'feedback.json', feedback)
    write(path, cfg)
    if fault:
        with pytest.raises(ValueError, match='feedback must bind'):
            exploration.prepare(path)
        assert not root.exists()
    else:
        exploration.prepare(path)
        payload = exploration.request_payload(exploration.read(root/'request.json'))
        assert payload['development_feedback'] == feedback


def test_pinned_template_tampering_rejected(tmp_path):
    path, root = job(tmp_path); exploration.prepare(path)
    (root / 'template.md').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        exploration.load_prepared(root)


def test_collect_reconciles_wire_and_rejects_tampering(tmp_path, monkeypatch):
    path, root = job(tmp_path); exploration.prepare(path)
    _, _, caller, request = exploration.load_prepared(root)
    response = {'id': 'provider-test', 'model': 'external-model', 'choices': [
        {'finish_reason': 'stop', 'message': {'content': json.dumps(proposals())}}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}
    calls = network(caller, monkeypatch, [response])
    caller(request)
    assert len(calls) == 1
    result = exploration.collect(root)
    assert result['proposals'] == proposals()
    assert result['candidate_accepted'] is False and result['semantic_validation'] == 'pending'
    assert exploration.collect(root) == result
    directory, _ = caller._binding(request)
    (directory / 'response/response.bin').write_text('{}')
    with pytest.raises(Exception, match='wire changed'):
        exploration.collect(root)


def test_no_response_cannot_collect(tmp_path):
    path, root = job(tmp_path); exploration.prepare(path)
    with pytest.raises(ValueError, match='never resend'):
        exploration.collect(root)


def test_algorithm_mismatch_and_duplicate_candidates_rejected():
    profile = {'algorithm': 'reflexion', 'min_candidates': 2, 'max_candidates': 8}
    data = proposals(); data['algorithm'] = 'ace'
    with pytest.raises(ValueError, match='algorithm-specific'):
        exploration.validate_proposals(json.dumps(data), profile, {})
    data = proposals(); data['candidates'][1] = data['candidates'][0]
    with pytest.raises(ValueError, match='duplicate'):
        exploration.validate_proposals(json.dumps(data), profile, {})


def test_profiles_have_distinct_pinned_templates():
    texts = []
    for algorithm in ('gepa', 'tfgrpo', 'ace', 'reflexion'):
        p = ROOT / f'configs/learners/tfrl/{algorithm}.json'
        profile = exploration.read(p)
        texts.append(exploration.checked(profile['template'], p.parent).read_text())
        assert profile['algorithm'] == algorithm and profile['max_tokens'] >= 131072
    assert len(set(texts)) == 4


def test_gepa_parent_scope_and_unsupplied_evidence_rejected():
    from test_solver_prompt_candidate import fixture
    contract, parent, mutation, _ = fixture()
    second = dict(mutation, solver_instruction='Check counterevidence before selecting a slot.')
    data = {'algorithm': 'gepa', 'candidates': [mutation, second]}
    profile = {'algorithm': 'gepa', 'min_candidates': 2, 'max_candidates': 8}
    context = {'parent': parent, 'contract': contract}
    assert exploration.validate_proposals(json.dumps(data), profile, context, ['slurp:train:1']) == data
    with pytest.raises(ValueError, match='not supplied'):
        exploration.validate_proposals(json.dumps(data), profile, context, ['slurp:train:2'])
    data['candidates'][1]['parent_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='parent differs'):
        exploration.validate_proposals(json.dumps(data), profile, context)
