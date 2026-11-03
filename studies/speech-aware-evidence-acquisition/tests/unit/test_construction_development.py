"""Development material is verified data; native execution remains authoritative."""
import copy
import hashlib
import json

import pytest

from speech_aware_evidence_acquisition.rails import construction_development as dev
from speech_aware_evidence_acquisition.rails import ace_incremental as ace
from speech_aware_evidence_acquisition.rails.grounded_synthesis_fusion import project_group_evidence
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
from test_ace_incremental import fixture, reflection, delta, runner
from test_ace_runtime import setup


def material_fixture(tmp_path):
    source, old = fixture(groups=1)
    group = source['batch']['groups'][0]
    group['query_id'] = 'slurp:train:000001:audio-one'
    group['query'] = json.dumps({'sample_id': group['query_id'], 'history': [],
        'task_instruction': 'Retain the requested action.', 'supplied_evidence': ['a card']})
    trajectory = json.loads(group['rollouts'][0]['trajectory'])
    trajectory['request'] = {'state': 'observed solver input'}
    group['rollouts'][0]['trajectory'] = json.dumps(trajectory)
    train = {'slurp_id': 1, 'scenario': 'calendar', 'action': 'set', 'entities': [],
        'tokens': [], 'sentence': 'create an event', 'sentence_annotation': 'create an event',
        'recordings': [{'file': 'audio-one.flac'}]}
    def pin(name, blob):
        (tmp_path / name).write_bytes(blob)
        return {'root': 'data', 'relpath': name, 'sha256': hashlib.sha256(blob).hexdigest()}
    train_pin = pin('train.jsonl', json.dumps(train).encode())
    diagnostic = {'schema': 'construction-diagnostic-evidence-private-v1',
        'original_train_sha256': train_pin['sha256'], 'groups': [{
            'query_id': group['query_id'], 'original_group_sha256': sha(group),
            'original_evidence': project_group_evidence(group),
            'diagnostic_train_only': {'slurp_id': 1, 'reference_transcript': train['sentence'],
                'reference_annotation': train['sentence_annotation']}}]}
    spec = {'schema': dev.SCHEMA, 'original_train': train_pin,
        'diagnostic': pin('diagnostic.json', json.dumps(diagnostic).encode()),
        'task_template': pin('task.txt', b'Find the first failing node and a runtime-observable action.')}
    return source, old, spec, diagnostic


def test_full_projection_preserves_all_records_and_legacy_default(tmp_path):
    source, _, _, _ = material_fixture(tmp_path); group = source['batch']['groups'][0]
    old = project_group_evidence(group); full = project_group_evidence(group, evidence_view='full-query-v1')
    assert old == project_group_evidence(group, evidence_view='compact-v1')
    assert 'supplied_evidence' not in old['query']
    assert full['query'] == json.loads(group['query'])
    assert full['rollouts'][0]['source_trajectory']['request'] == {'state': 'observed solver input'}
    assert old['full_query_sha256'] == full['full_query_sha256']
    assert full['audio_inspected'] is False


def test_material_binds_real_source_and_keeps_gold_diagnostic_only(tmp_path):
    source, old, spec, _ = material_fixture(tmp_path)
    material = dev.load_material(spec, source, repo_root=tmp_path, data_root=tmp_path)
    doc = ace.build_input(source, old['parent'], learning_config_sha256='a' * 64,
        **{k: v for k, v in old['bounds'].items() if k != 'max_calls'},
        evidence_view='full-query-v1', development_material=material)
    prompt = ace.reflect_messages(doc, 0, doc['parent']['items'], lambda _: 100)
    payload = json.loads(prompt['messages'][1]['content'])
    assert payload['diagnostic_train_only']['reference_transcript'] == 'create an event'
    assert payload['development_material_sha256'] == material['sha256']
    assert 'never evaluation Solver input' in prompt['messages'][0]['content']
    assert material['solver_input_permitted'] is False and prompt['native_fit_completed'] is False
    assert doc['historical_usage'] == old['historical_usage']


@pytest.mark.parametrize('fault', ['bytes', 'roster', 'group', 'evidence', 'transcript', 'annotation', 'train_pin', 'escape'])
def test_diagnostic_tampering_fails_before_prompt_or_fit(tmp_path, fault):
    source, _, spec, diagnostic = material_fixture(tmp_path)
    if fault == 'bytes':
        (tmp_path / 'diagnostic.json').write_text('{}')
    elif fault == 'escape':
        spec['diagnostic']['relpath'] = '../diagnostic.json'
    else:
        row = diagnostic['groups'][0]
        if fault == 'roster': diagnostic['groups'].append(copy.deepcopy(row))
        if fault == 'group': row['original_group_sha256'] = '0' * 64
        if fault == 'evidence': row['original_evidence']['reference'] = {}
        if fault == 'transcript': row['diagnostic_train_only']['reference_transcript'] = 'wrong'
        if fault == 'annotation': row['diagnostic_train_only']['reference_annotation'] = 'wrong'
        if fault == 'train_pin': diagnostic['original_train_sha256'] = '0' * 64
        blob = json.dumps(diagnostic).encode(); (tmp_path / 'diagnostic.json').write_bytes(blob)
        spec['diagnostic']['sha256'] = hashlib.sha256(blob).hexdigest()
    with pytest.raises(ValueError): dev.load_material(spec, source, repo_root=tmp_path, data_root=tmp_path)


def test_configured_development_request_uses_native_fit_and_seals(tmp_path, monkeypatch):
    runtime, transport, captured, source, spec, fits = setup(tmp_path, monkeypatch,
        evidence_view='full-query-v1', development_instruction='Identify first divergence and one intervention.',
        invalid_delta_policy='reject-collision-keep-parent-v1')
    result = runtime.run_phases()
    assert result['contact_count'] == len(transport.calls) == len(fits) == 2
    assert runtime.seal()['cost']['calls_used'] == 2 and captured['finalized']
    assert 'first divergence' in spec['initial_measurement']['messages'][0]['content']


def test_collision_policy_retains_entire_parent_and_exact_rejected_response():
    _, doc = fixture(invalid_delta_policy='reject-collision-keep-parent-v1')
    answer = delta('ADD', None, 'A prior operation must also be rolled back.')
    answer['deltas'].append(delta(content=doc['parent']['items']['untouched'])['deltas'][0])
    text = json.dumps(answer)
    merged = ace.merge_delta(text, doc['parent']['items'], reflection(), doc['bounds'])
    assert merged['items'] == doc['parent']['items'] and merged['operations'] == []
    assert merged['rejected_delta']['response_text_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert not merged['semantic_support_verified']


@pytest.mark.parametrize('fault', ['foreign-finding', 'invalid-json', 'foreign-id', 'oversize'])
def test_collision_policy_does_not_hide_other_errors(fault):
    _, doc = fixture(invalid_delta_policy='reject-collision-keep-parent-v1')
    answer = delta()
    if fault == 'foreign-finding': answer['deltas'][0]['finding_ids'] = ['unknown']
    if fault == 'foreign-id': answer['deltas'][0]['id'] = 'unknown'
    if fault == 'oversize': answer['deltas'][0]['content'] = 'x' * 1001
    text = '{' if fault == 'invalid-json' else json.dumps(answer)
    with pytest.raises(ValueError): ace.merge_delta(text, doc['parent']['items'], reflection(), doc['bounds'])


def test_rejected_delta_known_replay_does_not_repeat_contacts(tmp_path):
    answers = {(i, stage): reflection(i) if stage == ace.REFLECT else
               delta(content='Use observed evidence only.') for i in range(2) for stage in (ace.REFLECT, ace.CURATE)}
    _, _, run, calls, _ = runner(tmp_path, answers=answers, invalid_delta_policy='reject-collision-keep-parent-v1')
    first = run(); second = run()
    assert first == second and len(calls) == 4
    assert first['learning_usage']['calls'] == 4 and first['cumulative_known_usage']['calls'] == 316
    assert all('rejected_delta' in event['merge'] for event in first['updates'])


def test_native_context_limit_still_rejects_larger_development_prompt(tmp_path):
    source, old, spec, _ = material_fixture(tmp_path)
    material = dev.load_material(spec, source, repo_root=tmp_path, data_root=tmp_path)
    bounds = {k: v for k, v in old['bounds'].items() if k != 'max_calls'}
    doc = ace.build_input(source, old['parent'], learning_config_sha256='a' * 64,
        **bounds, evidence_view='full-query-v1', development_material=material)
    with pytest.raises(ValueError, match='no truncation'):
        ace.reflect_messages(doc, 0, doc['parent']['items'], lambda _: 49152)
