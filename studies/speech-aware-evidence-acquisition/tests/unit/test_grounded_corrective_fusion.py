"""Deterministic CPU contract tests; fixtures never estimate real model cost."""
import base64
import copy
import json

import pytest

from speech_aware_evidence_acquisition.rails import grounded_corrective_fusion as m
from speech_aware_evidence_acquisition.rails import grounded_synthesis_fusion as fusion
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (
    ExperienceLearner, UnknownLearningContact, empty_experience,
)


def fixture():
    references = [
        {'scenario': 'news', 'action': 'query', 'entities': [{'type': 'person', 'filler': 'a person'}]},
        {'scenario': 'calendar', 'action': 'set', 'entities': [{'type': 'timeofday', 'filler': 'noon'}]},
        {'scenario': 'weather', 'action': 'query', 'entities': [{'type': 'weather_descriptor', 'filler': 'snow'}]},
    ]
    groups = []
    for i, answer in enumerate(references):
        prediction = {**answer, 'entities': []}
        groups.append({'group_ref': f'G{i}', 'group_index': i, 'group_sha256': m.sha(i),
            'query_id': f'slurp:train:{i}', 'content': {
                'query': {'history': []}, 'reference': answer, 'omitted_query_fields': [],
                'rollouts': [{'scored_prediction': prediction, 'reward': 0.5,
                              'feedback': {'strict_task_success': 0}, 'prediction_observed': True}]}})
    parent = empty_experience('model')
    cost = {'calls': 312, 'prompt_tokens': 1000, 'completion_tokens': 200}
    source = {'source_native_custody_verified': True, 'unknown_contacts': 0,
              'original_total_usage': cost,
              'native_source_usage': [{'run_id': 'source', 'attempt_id': 'a', 'usage': cost}]}
    doc = {'schema': fusion.INPUT_SCHEMA, 'model_id': 'model', 'source_sha256': m.sha(source),
        'parent': parent, 'groups': groups,
        'proposals': [{'source_ref': f'S{i}O0', 'operation': {'operation': 'ADD', 'id': None,
            'content': 'Observe task evidence'}, 'shard_index': i, 'available_group_refs': [f'G{i}']}
            for i in range(3)],
        'shard_sources': [{'shard_index': i, 'available_group_refs': [f'G{i}']} for i in range(3)],
        'duplicate_sets': [], 'structural_conflicts': [], 'original_total_usage': cost}
    doc['input_sha256'] = m.sha(doc)
    binding = {'schema': m.EXECUTION_SCHEMA, 'model_id': 'model',
        'learning_config_sha256': 'a'*64, 'input_sha256': doc['input_sha256'],
        'identity': doc['input_sha256'], 'source_custody_sha256': m.sha(source), 'parent': parent,
        'learned_query_ids': [g['query_id'] for g in groups],
        'rollout_kind': 'shared-perception-adaptation', 'reward_mode': 'intent-exact-slot-f1-v1',
        'messages': [{'role': 'system', 'content': 'original system'},
                     {'role': 'user', 'content': m.canonical(fusion.compact_model_view(doc)).decode()}],
        'bounds': {'max_items': 48, 'max_characters': 10000, 'max_operations': 48}}
    request = {'schema': 'saea-tfgrpo-learning-request-v1', 'batch_sha256': m.sha(binding),
        'model_id': 'model', 'learning_config_sha256': 'a'*64, 'stage': m.STAGE,
        'identity': binding['identity'], 'messages': binding['messages']}
    payload = {'request_id': 'flight-1', 'history': [{'role': 'user', 'content': json.dumps({
        'source_messages': binding['messages'], 'stage': m.STAGE})}]}
    identity = {'run_id': 'failed', 'attempt_id': 'a'}
    native_intent = {'identity': identity, 'payload': payload, 'request_id': payload['request_id'],
                     'request_sha256': m.sha(payload)}
    body = {'text': '{invalid original output', 'usage': {'prompt_tokens': 20, 'completion_tokens': 10},
            'finish_reason': 'stop', 'reasoning': 'complete original reasoning remains private'}
    native_response = {'identity': identity, 'request_id': payload['request_id'],
        'request_sha256': m.sha(payload), 'intent_sha256': m.sha(native_intent),
        'response_sha256': m.sha(body), 'response_b64': base64.b64encode(m.canonical(body)).decode()}
    receipt = {'model_id': 'model', 'receipt_id': 'failed/a/flight-1', 'text': body['text'],
        'request_sha256': m.sha(payload), 'response_sha256': m.sha(body), 'usage': {'calls': 1, **body['usage']}}
    response = {'request_sha256': m.sha(request), 'response': receipt, 'response_sha256': m.sha(receipt)}
    failure = m.bind_failed_response(binding, request, response, native_intent, native_response,
                                     historical_usage=cost, failure_reason='invalid original fusion')
    corrective = m.build_corrective_input(doc, failure)
    return corrective, source


def output(corrective, group=0):
    row = corrective['catalog'][group]
    return {'schema': m.OUTPUT_SCHEMA, 'rules': [{
        'rule_id': 'R1', 'operation': {'operation': 'ADD', 'id': None, 'content': 'Consult observed evidence'},
        'source_refs': [f'S{group}O0'], 'support_group_refs': [row['group_ref']],
        'scope': 'Observed task boundary', 'trigger': 'Explicit observed evidence',
        'exceptions': ['Unobserved evidence'], 'rationale': 'Compare actual reference with prediction',
        'applicable_intents': [row['intent']], 'focused_entity_types': sorted({e['type'] for e in row['entities']}),
        'support_claims': [m._claim(row)],
        'prediction_counterexamples': [m._counterexample(row, row['predictions'][0])]}],
        'dispositions': [{'source_ref': f'S{i}O0', 'decision': 'retain' if i == group else 'reject',
            'rule_ids': ['R1'] if i == group else [], 'reason': 'Exact observed evidence or insufficient support'}
            for i in range(3)]}


def test_bound_failure_keeps_full_native_response_and_cost():
    c, _ = fixture()
    f = c['failure']
    assert f['cumulative_known_usage'] == {'calls': 313, 'prompt_tokens': 1020, 'completion_tokens': 210}
    body = json.loads(base64.b64decode(f['native_response']['response_b64']))
    assert body['reasoning'] == 'complete original reasoning remains private'
    assert not f['generation_authorized'] and not f['terminal_custody_verified']
    assert c['observed_validation_error'] == 'fusion response must be exact JSON'


@pytest.mark.parametrize('group,bad', [(0, 'takeaway_order'), (1, 'calendar_query')])
def test_wrong_support_intent_rejected(group, bad):
    c, _ = fixture(); result = output(c, group)
    result['rules'][0]['support_claims'][0]['intent'] = bad
    with pytest.raises(ValueError, match='support intent/entities'):
        m.validate_corrective_output(json.dumps(result), c)


@pytest.mark.parametrize('fault', ['type', 'filler', 'omission', 'query', 'prediction', 'contrast',
                                  'scope_intent', 'scope_slot', 'outside_shard', 'missing_disposition',
                                  'no_counterexample', 'same_prediction'])
def test_false_grounding_and_existing_checks_fail_closed(fault):
    c, _ = fixture(); result = output(c, 2); rule = result['rules'][0]
    claim = rule['support_claims'][0]
    if fault == 'type': claim['entities'][0]['type'] = 'temperature'
    elif fault == 'filler': claim['entities'][0]['filler'] = 'hot'
    elif fault == 'omission': claim['entities'] = []
    elif fault == 'query': claim['query_id'] = 'slurp:train:wrong'
    elif fault == 'prediction': rule['prediction_counterexamples'][0]['prediction_entities'] = claim['entities']
    elif fault == 'contrast': rule['prediction_counterexamples'][0]['contrast']['missing_entities'] = []
    elif fault == 'scope_intent': rule['applicable_intents'] = ['takeaway_order']
    elif fault == 'scope_slot': rule['focused_entity_types'] = ['temperature']
    elif fault == 'outside_shard':
        rule['source_refs'] = ['S0O0']
        result['dispositions'][0].update(decision='retain', rule_ids=['R1'])
        result['dispositions'][2].update(decision='reject', rule_ids=[])
    elif fault == 'missing_disposition': result['dispositions'].pop()
    elif fault == 'no_counterexample': rule['prediction_counterexamples'] = []
    else: rule['prediction_counterexamples'].append(copy.deepcopy(rule['prediction_counterexamples'][0]))
    with pytest.raises(ValueError):
        m.validate_corrective_output(json.dumps(result), c)


def test_mechanical_grounding_explicitly_not_semantic_proof():
    c, _ = fixture()
    checked = m.validate_corrective_output(json.dumps(output(c)), c)
    assert checked['label_and_entity_grounding_verified']
    assert not checked['semantic_support_verified'] and not checked['candidate_publishable']
    assert checked['semantic_review_required']
    assert 'news_query' in checked['operations'][0]['content']


def test_all_rejected_is_valid():
    c, _ = fixture(); result = output(c); result['rules'] = []
    for row in result['dispositions']: row.update(decision='reject', rule_ids=[])
    checked = m.validate_corrective_output(json.dumps(result), c)
    assert checked['items'] == {} and not checked['candidate_publishable']


@pytest.mark.parametrize('qid', ['slurp:test:1', 'slurp:dev:1', 'H2:1'])
def test_non_train_catalog_rejected(qid):
    c, _ = fixture(); ids = c['failure']['binding']['learned_query_ids']; ids[0] = qid
    with pytest.raises(ValueError, match='TRAIN'):
        m.reference_catalog(fusion.compact_model_view(c['document']), ids)


def test_one_prompt_contains_complete_failure_and_no_truncation():
    c, _ = fixture()
    measured = m.corrective_messages(c, count_tokens=lambda messages: 300)
    payload = json.loads(measured['messages'][1]['content'])
    assert payload['failed_output'] == {'format': 'verbatim-text', 'value': c['failure']['response']['response']['text']}
    assert payload['original_evidence'] == fusion.compact_model_view(c['document'])
    assert measured['max_new_calls'] == 1 and not measured['native_fit_completed']
    with pytest.raises(ValueError, match='no truncation'):
        m.corrective_messages(c, count_tokens=lambda messages: 49152)
    with pytest.raises(ValueError, match='widen'):
        m.corrective_messages(c, count_tokens=lambda messages: 1, max_items=49)


def test_correction_cannot_be_corrected_again():
    c, _ = fixture(); f = c['failure']; binding = copy.deepcopy(f['binding']); binding['identity'] = 'corrected'
    with pytest.raises(ValueError, match='original final'):
        m.bind_failed_response(binding, f['request'], f['response'], f['native_intent'], f['native_response'],
            historical_usage=f['historical_usage'], failure_reason='another correction')


def test_native_tampering_rejected():
    c, _ = fixture(); f = c['failure']; native = copy.deepcopy(f['native_response'])
    native['response_b64'] = base64.b64encode(b'{}').decode()
    with pytest.raises(ValueError, match='response bytes'):
        m.bind_failed_response(f['binding'], f['request'], f['response'], f['native_intent'], native,
            historical_usage=f['historical_usage'], failure_reason=f['failure_reason'])


def execution_fixture(tmp_path, monkeypatch, bad=False, unknown=False):
    c, source = fixture()
    monkeypatch.setattr(m, 'build_fusion_input', lambda given: copy.deepcopy(c['document']))
    calls, fits = [], []
    def call(request):
        calls.append(request)
        if unknown: raise OSError('transport disconnected')
        result = output(c)
        if bad: result['rules'][0]['support_claims'][0]['intent'] = 'takeaway_order'
        return {'model_id': 'model', 'receipt_id': 'new/a/flight-1', 'text': json.dumps(result),
                'usage': {'calls': 1, 'prompt_tokens': 30, 'completion_tokens': 15}}
    def make():
        return ExperienceLearner(tmp_path, model_id='model', learning_config_sha256='a'*64,
            metered_call=call, max_calls=1, preflight=lambda request: fits.append(request))
    terminal = {'failure_sha256': c['failure']['failure_sha256'], 'attempt_state': 'ABORTED',
        'observed_contacts': 1, 'unknown_contacts': 0, 'full_inventory_verified': True, 'native_fit_verified': True}
    measured = m.corrective_messages(c, count_tokens=lambda messages: 300)
    def execute():
        return m.execute_corrective_fusion(make(), c, measured, verify_source=lambda: source,
            verify_failed_attempt=lambda: terminal, count_tokens=lambda messages: 300)
    return execute, calls, fits, terminal, measured


def test_shared_execution_reuses_known_response_and_preserves_313_history(tmp_path, monkeypatch):
    execute, calls, fits, _, _ = execution_fixture(tmp_path, monkeypatch)
    first, second = execute(), execute()
    assert first == second and len(calls) == len(fits) == 1
    assert first['historical_fusion_custody']['original_total_usage']['calls'] == 313
    assert [r['usage']['calls'] for r in first['historical_fusion_custody']['native_source_usage']] == [312, 1]
    assert first['learning_usage']['calls'] == 1 and not first['deployment_accepted']


def test_invalid_correction_retained_never_resampled(tmp_path, monkeypatch):
    execute, calls, fits, _, _ = execution_fixture(tmp_path, monkeypatch, bad=True)
    for _ in range(2):
        with pytest.raises(ValueError, match='support intent/entities'): execute()
    assert len(calls) == len(fits) == 1 and not (tmp_path/'candidate.json').exists()
    assert len(list((tmp_path/'contacts').glob('*.response.json'))) == 1


def test_unknown_correction_never_resends(tmp_path, monkeypatch):
    execute, calls, fits, _, _ = execution_fixture(tmp_path, monkeypatch, unknown=True)
    with pytest.raises(OSError): execute()
    with pytest.raises(UnknownLearningContact): execute()
    assert len(calls) == len(fits) == 1


@pytest.mark.parametrize('fault', ['terminal', 'inventory', 'unknown', 'fit', 'prompt'])
def test_runtime_source_rejection_happens_before_spend(tmp_path, monkeypatch, fault):
    execute, calls, fits, terminal, measured = execution_fixture(tmp_path, monkeypatch)
    if fault == 'terminal': terminal['attempt_state'] = 'RUNNING'
    elif fault == 'inventory': terminal['full_inventory_verified'] = False
    elif fault == 'unknown': terminal['unknown_contacts'] = 1
    elif fault == 'fit': terminal['native_fit_verified'] = False
    else:
        measured['messages'][0]['content'] = 'changed prompt'
        measured['messages_sha256'] = m.sha(measured['messages'])
        measured['measurement_sha256'] = m.sha({k: v for k, v in measured.items() if k != 'measurement_sha256'})
    with pytest.raises(ValueError): execute()
    assert not calls and not fits


def compact_output(corrective):
    result = output(corrective)
    result['schema'] = m.COMPACT_OUTPUT_SCHEMA
    for rule in result['rules']:
        rule['support_claims'] = [{k: c[k] for k in ('group_ref', 'intent', 'entities')}
                                  for c in rule['support_claims']]
        rule['prediction_counterexamples'] = [{k: c[k] for k in (
            'group_ref', 'prediction_ref', 'prediction_intent', 'prediction_entities')}
            for c in rule['prediction_counterexamples']]
    return result


def test_compact_claims_resolve_to_same_verified_rules():
    c, _ = fixture()
    compact = compact_output(c)
    checked = m.validate_corrective_output(json.dumps(compact), c)
    full = m.validate_corrective_output(json.dumps(output(c)), c)
    assert checked['structured_claims'] == full['structured_claims']
    assert checked['corrective_output_sha256'] == m.sha(compact)
    assert checked['semantic_support_verified'] is False
    assert not checked['candidate_publishable']


@pytest.mark.parametrize('fault', ['intent', 'entity', 'prediction', 'unknown', 'extra'])
def test_compact_claims_do_not_weaken_grounding(fault):
    c, _ = fixture(); result = compact_output(c); rule = result['rules'][0]
    if fault == 'intent': rule['support_claims'][0]['intent'] = 'takeaway_order'
    elif fault == 'entity': rule['support_claims'][0]['entities'] = []
    elif fault == 'prediction': rule['prediction_counterexamples'][0]['prediction_intent'] = 'wrong'
    elif fault == 'unknown': rule['prediction_counterexamples'][0]['prediction_ref'] = 'G99P0'
    else: rule['support_claims'][0]['query_id'] = 'untrusted'
    with pytest.raises(ValueError): m.validate_corrective_output(json.dumps(result), c)


def test_model_view_keeps_every_group_without_duplicate_catalog():
    c, _ = fixture(); before = copy.deepcopy(c)
    measured = m.corrective_messages(c, count_tokens=lambda messages: 300)
    payload = json.loads(measured['messages'][1]['content'])
    assert payload['original_evidence'] == fusion.compact_model_view(c['document'])
    assert 'reference_and_counterexample_catalog' not in payload
    assert c == before
    assert m.COMPACT_OUTPUT_SCHEMA in measured['messages'][0]['content']
