"""Offline ACE adaptation contracts, not simulated research outcomes."""
import copy
import json

import pytest

from speech_aware_evidence_acquisition.rails import ace_incremental as m
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (
    ExperienceLearner, UnknownLearningContact, empty_experience, sha, canonical,
)


def fixture(groups=2, parent=None, empty_parent=False, **limits):
    parent = copy.deepcopy(parent) if parent is not None else empty_experience('model')
    if not parent['items'] and not empty_parent:
        parent['items'] = {'old': 'Preserve intent boundaries.', 'untouched': 'Use observed evidence only.'}
        parent['sha256'] = sha({k: v for k, v in parent.items() if k != 'sha256'})
    rows = []
    for i in range(groups):
        prediction = {'scenario': 'calendar', 'action': 'query', 'entities': []}
        trajectory = json.dumps({'response_text': json.dumps(prediction),
            'scored_answer_after_fixed_normalization': json.dumps(prediction)})
        rows.append({'query_id': f'slurp:train:{i}', 'query': json.dumps({'sample_id': f'slurp:train:{i}', 'history': []}),
            'answer': json.dumps({'scenario': 'calendar', 'action': 'set', 'entities': []}),
            'rollouts': [{'trajectory_id': f'T{i}', 'trajectory': trajectory,
                'critique': json.dumps({'strict_task_success': 0}), 'reward': 0.5,
                'model_id': 'model', 'policy_sha256': 'b'*64, 'experience_sha256': parent['sha256']}]})
    batch = {'model_id': 'model', 'parent': parent, 'groups': rows, 'policy_sha256': 'b'*64,
             'rollout_kind': 'shared-perception-adaptation', 'common': {'reward_mode': m.DENSE_REWARD}}
    usage = {'calls': 312, 'prompt_tokens': 1000, 'completion_tokens': 200}
    source = {'model_id': 'model', 'batch': batch, 'source_native_custody_verified': True,
        'unknown_contacts': 0, 'native_source_usage': [{'run_id': 'original', 'attempt_id': 'a', 'usage': usage}],
        'original_total_usage': copy.deepcopy(usage)}
    bounds = dict(learning_config_sha256='a'*64, resident_context=49152, completion_cap=100,
        reserve_tokens=100, max_groups=4, max_delta=4, max_findings=4, max_items=10,
        max_characters=1000, max_prompt_tokens_total=10000, max_completion_tokens_total=1000)
    bounds.update(limits)
    return source, m.build_input(source, parent, **bounds)


def reflection(index=0, signal='harmful'):
    return {'schema': 'saea-ace-reflection-v1', 'findings': [{'finding_id': 'F1',
        'lesson': 'Distinguish creation requests from lookups when evidence states the action.',
        'trajectory_ids': [f'T{index}'], 'confidence': 'tentative'}], 'signals': [{
        'item_id': 'old', 'label': signal, 'trajectory_ids': [f'T{index}'], 'reason': 'Observed mismatch association only'}]}


def delta(operation='MODIFY', item_id='old', content='When evidence asks creation, preserve calendar_set.'):
    return {'schema': 'saea-ace-delta-v1', 'deltas': [{'operation': operation, 'id': item_id,
        'content': content, 'finding_ids': ['F1'], 'reason': 'Local refinement from observed error'}],
        'unresolved_conflicts': []}


def test_all_wrong_equal_reward_groups_still_feed_reflector():
    source, doc = fixture()
    assert len(doc['groups']) == 2 and doc['bounds']['max_calls'] == 4
    assert not doc['candidate_publishable'] and doc['observed_associations_not_causal']
    assert all(g['evidence']['rollouts'][0]['feedback']['strict_task_success'] == 0 for g in doc['groups'])


@pytest.mark.parametrize('fault', ['test', 'duplicate_query', 'parent', 'policy', 'unknown', 'cost', 'too_many_groups'])
def test_source_provenance_and_budget_refused(fault):
    source, doc = fixture()
    if fault == 'test': source['batch']['groups'][0]['query_id'] = 'slurp:test:0'
    elif fault == 'duplicate_query': source['batch']['groups'][1]['query_id'] = source['batch']['groups'][0]['query_id']
    elif fault == 'parent': source['batch']['groups'][0]['rollouts'][0]['experience_sha256'] = 'f'*64
    elif fault == 'policy': source['batch']['groups'][0]['rollouts'][0]['policy_sha256'] = 'f'*64
    elif fault == 'unknown': source['unknown_contacts'] = 1
    elif fault == 'cost': source['original_total_usage']['calls'] = 0
    bounds = {k: v for k, v in doc['bounds'].items() if k != 'max_calls'}
    if fault == 'too_many_groups': bounds['max_groups'] = 1
    with pytest.raises(ValueError): m.build_input(source, doc['parent'], learning_config_sha256='a'*64, **bounds)


def test_incremental_modification_keeps_id_and_untouched_bytes():
    _, doc = fixture(); parent = copy.deepcopy(doc['parent']['items'])
    checked = m.merge_delta(json.dumps(delta()), parent, reflection(), doc['bounds'])
    assert checked['items']['old'] == delta()['deltas'][0]['content']
    assert checked['items']['untouched'] == parent['untouched']
    assert parent == doc['parent']['items']


def test_normalized_duplicate_add_coalesced_with_provenance():
    _, doc = fixture()
    checked = m.merge_delta(json.dumps(delta('ADD', None, '  PRESERVE  intent boundaries.  ')),
        doc['parent']['items'], reflection(), doc['bounds'])
    assert checked['items'] == doc['parent']['items'] and checked['operations'] == []
    assert checked['duplicates'][0]['existing_id'] == 'old'
    assert checked['delta']['deltas'][0]['finding_ids'] == ['F1']


@pytest.mark.parametrize('fault', ['missing_finding', 'double_touch', 'foreign_id', 'modify_duplicate',
                                  'conflict_touch', 'over_characters', 'over_operations', 'delete_content'])
def test_invalid_or_conflicting_deltas_fail_without_mutation(fault):
    _, doc = fixture(); answer = delta(); parent = copy.deepcopy(doc['parent']['items']); bounds = doc['bounds']
    if fault == 'missing_finding': answer['deltas'][0]['finding_ids'] = ['foreign']
    elif fault == 'double_touch': answer['deltas'].append(copy.deepcopy(answer['deltas'][0]))
    elif fault == 'foreign_id': answer['deltas'][0]['id'] = 'not-present'
    elif fault == 'modify_duplicate': answer['deltas'][0]['content'] = parent['untouched']
    elif fault == 'conflict_touch': answer['unresolved_conflicts'] = [{'item_ids': ['old'], 'finding_ids': ['F1'], 'reason': 'unresolved'}]
    elif fault == 'over_characters': answer['deltas'][0]['content'] = 'x'*1001
    elif fault == 'over_operations': bounds['max_delta'] = 1; answer['deltas'].append(delta('DELETE', 'untouched', '')['deltas'][0])
    else: answer['deltas'][0]['operation'] = 'DELETE'
    with pytest.raises(ValueError): m.merge_delta(json.dumps(answer), parent, reflection(), bounds)
    assert parent == doc['parent']['items']


def test_empty_delta_preserves_unresolved_conflict_without_promoting():
    _, doc = fixture(); answer = delta(); answer['deltas'] = []
    answer['unresolved_conflicts'] = [{'item_ids': ['old'], 'finding_ids': ['F1'], 'reason': 'Cannot resolve from observed evidence'}]
    checked = m.merge_delta(json.dumps(answer), doc['parent']['items'], reflection(), doc['bounds'])
    assert checked['items'] == doc['parent']['items'] and len(checked['unresolved_conflicts']) == 1
    assert not checked['semantic_support_verified']


def test_add_delete_delta_retains_unrelated_parent_entries():
    _, doc = fixture(); answer = delta('DELETE', 'old', '')
    answer['deltas'].append(delta('ADD', None, 'A scoped new lesson.')['deltas'][0])
    checked = m.merge_delta(json.dumps(answer), doc['parent']['items'], reflection(), doc['bounds'])
    assert 'old' not in checked['items'] and len(checked['items']) == 2
    assert checked['items']['untouched'] == doc['parent']['items']['untouched']
    assert [o['operation'] for o in checked['operations']] == ['DELETE', 'ADD']


def test_duplicate_adds_in_same_delta_coalesce_deterministically():
    _, doc = fixture(); answer = delta('ADD', None, 'One scoped lesson.')
    answer['deltas'].append(delta('ADD', None, '  ONE scoped lesson.  ')['deltas'][0])
    checked = m.merge_delta(json.dumps(answer), doc['parent']['items'], reflection(), doc['bounds'])
    assert len(checked['operations']) == len(checked['duplicates']) == 1


def test_source_rollout_parent_can_differ_only_with_explicit_offline_disclosure():
    source, doc = fixture(); parent = copy.deepcopy(doc['parent'])
    parent['items']['new'] = 'New parent knowledge.'
    parent['sha256'] = sha({k: v for k, v in parent.items() if k != 'sha256'})
    changed = m.build_input(source, parent, learning_config_sha256='a'*64,
        **{k: v for k, v in doc['bounds'].items() if k != 'max_calls'})
    assert changed['offline_stale_policy'] and changed['rollout_parent_sha256'] == source['batch']['parent']['sha256']


@pytest.mark.parametrize('parent_kind', ['candidate', 'rollout'])
def test_parent_knowledge_cannot_carry_evaluation_labels(parent_kind):
    source, doc = fixture()
    parent = copy.deepcopy(doc['parent'])
    contaminated = parent if parent_kind == 'candidate' else source['batch']['parent']
    contaminated['learned_query_ids'] = ['slurp:test:1']
    contaminated['sha256'] = sha({k: v for k, v in contaminated.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='only TRAIN'):
        m.build_input(source, parent, learning_config_sha256='a'*64,
            **{k: v for k, v in doc['bounds'].items() if k != 'max_calls'})


def test_reflection_cannot_claim_unknown_item_or_trajectory():
    _, doc = fixture(); result = reflection(); result['signals'][0]['item_id'] = 'invented'
    with pytest.raises(ValueError): m.validate_reflection(json.dumps(result), doc['groups'][0], doc['parent']['items'], 4)
    result = reflection(); result['findings'][0]['trajectory_ids'] = ['foreign']
    with pytest.raises(ValueError): m.validate_reflection(json.dumps(result), doc['groups'][0], doc['parent']['items'], 4)


def runner(tmp_path, *, fail=None, answers=None, **limits):
    source, doc = fixture(**limits); calls, fits = [], []
    def call(request):
        index = int(request['identity'].rsplit('-', 1)[1]); calls.append(request)
        if fail == 'unknown' and len(calls) == 1: raise OSError('unknown transport outcome')
        answer = reflection(index, 'harmful' if index == 0 else 'helpful') if request['stage'] == m.REFLECT else (
            delta() if index == 0 else delta('ADD', None, 'Consult observed temporal evidence before selecting a slot.'))
        if answers is not None: answer = answers[(index, request['stage'])]
        if fail == 'invalid' and request['stage'] == m.CURATE: answer['deltas'][0]['id'] = 'foreign'
        return {'model_id': 'model', 'receipt_id': str(len(calls)), 'text': json.dumps(answer),
                'usage': {'calls': 1, 'prompt_tokens': 10, 'completion_tokens': 5}}
    def run():
        learner = ExperienceLearner(tmp_path, model_id='model', learning_config_sha256='a'*64,
            metered_call=call, preflight=lambda r: fits.append(r), max_calls=doc['bounds']['max_calls'])
        return m.execute(learner, doc, count_tokens=lambda messages: 10, verify_source=lambda: copy.deepcopy(source))
    return source, doc, run, calls, fits


def test_two_stage_sequential_updates_and_known_response_reuse(tmp_path):
    source, doc, run, calls, fits = runner(tmp_path)
    first, second = run(), run()
    assert first == second and len(calls) == len(fits) == 4
    assert [r['stage'] for r in calls] == [m.REFLECT, m.CURATE, m.REFLECT, m.CURATE]
    assert first['items']['untouched'] == doc['parent']['items']['untouched']
    assert len(first['items']) == 3 and first['item_associations']['old']['harmful'] == 1
    assert first['item_associations']['old']['helpful'] == 1
    assert first['cumulative_known_usage']['calls'] == 316
    assert not first['deployment_accepted'] and first['requires_real_evaluation']
    assert first['observed_associations_not_causal'] and not first['semantic_support_verified']
    assert first['offline_stale_policy']
    # The next local reflector sees the incrementally refined library.
    second_input = json.loads(calls[2]['messages'][1]['content'])
    assert second_input['current_items']['old'] == delta()['deltas'][0]['content']
    assert second_input['offline_stale_policy'] and not second_input['generator_item_exposure_verified']
    assert first['item_associations']['old']['events'][0]['item_content_sha256'] != first['item_associations']['old']['events'][1]['item_content_sha256']


def test_deleted_and_readded_item_retains_historical_association_events(tmp_path):
    content = 'A scoped reusable lesson.'
    item_id = 'E'+m.hashlib.sha256(content.encode()).hexdigest()
    answers = {}
    for index in range(4):
        reflected = reflection(index)
        reflected['signals'] = []
        answers[index, m.REFLECT] = reflected
        answers[index, m.CURATE] = delta('ADD', None, content)
    answers[1, m.REFLECT]['signals'] = [{'item_id': item_id, 'label': 'harmful',
        'trajectory_ids': ['T1'], 'reason': 'Retrospective association only.'}]
    answers[1, m.CURATE] = delta('DELETE', item_id, '')
    answers[3, m.CURATE]['deltas'] = []
    _, _, run, calls, _ = runner(tmp_path, groups=4, answers=answers)
    result = run()
    assert len(calls) == 8 and result['items'][item_id] == content
    assert result['item_associations'][item_id]['harmful'] == 1
    assert result['item_associations'][item_id]['events'][0]['query_id'] == 'slurp:train:1'


def test_invalid_known_curator_output_never_resampled(tmp_path):
    _, _, run, calls, _ = runner(tmp_path, fail='invalid')
    for _ in range(2):
        with pytest.raises(ValueError): run()
    assert len(calls) == 2 and not (tmp_path/'candidate.json').exists()
    assert len(list((tmp_path/'contacts').glob('*.response.json'))) == 2
    failure = json.loads(next((tmp_path/'failures').glob('*.json')).read_bytes())
    assert failure['cumulative_known_usage']['calls'] == 314


def test_unknown_reflection_blocks_curator_and_retries(tmp_path):
    _, _, run, calls, _ = runner(tmp_path, fail='unknown')
    with pytest.raises(OSError): run()
    with pytest.raises(UnknownLearningContact): run()
    assert len(calls) == 1 and not (tmp_path/'candidate.json').exists()
    failures = [json.loads(p.read_bytes()) for p in (tmp_path/'failures').glob('*.json')]
    assert all(f['accounting']['unknown_contacts'] == 1 for f in failures)


def test_context_overflow_never_truncates_or_contacts(tmp_path):
    _, _, run, calls, fits = runner(tmp_path, resident_context=150)
    with pytest.raises(ValueError, match='no truncation'): run()
    assert not calls and not fits


def test_total_token_bound_stops_before_next_contact_and_preserves_known_cost(tmp_path):
    _, _, run, calls, _ = runner(tmp_path, max_prompt_tokens_total=15)
    with pytest.raises(ValueError, match='total token budget'): run()
    assert len(calls) == 1 and not (tmp_path/'candidate.json').exists()
    failure = json.loads(next((tmp_path/'failures').glob('*.json')).read_bytes())
    assert failure['cumulative_known_usage']['calls'] == 313


def test_foreign_contact_refused_before_new_spend(tmp_path):
    _, _, run, calls, _ = runner(tmp_path)
    (tmp_path/'contacts').mkdir(); (tmp_path/'contacts/foreign.intent.json').write_text('{}')
    with pytest.raises(ValueError, match='foreign'): run()
    assert not calls


def test_absent_contract_keeps_legacy_document_and_prompt_bytes():
    _, document = fixture()
    explicit_none = fixture(reflector_contract=None)[1]
    assert document == explicit_none
    assert 'reflector_contract' not in document['bounds']
    assert sha(document) == '513ae8e8428b7ae99a40c1df6f240e43eee67b35762472ea48e105de946852a0'
    measured = m.reflect_messages(document, 0, document['parent']['items'], lambda _: 10)
    assert measured['messages_sha256'] == 'bb30cb2359f5cd894f11a2fca13a158af55b301568d128c1e578792b42af1054'


@pytest.mark.parametrize('bad', ['', 'ace-reflector-contract-v1', 'ace-reflector-contract-v3', 2])
def test_unsupported_explicit_contract_refused(bad):
    with pytest.raises(ValueError, match='unsupported ACE reflector contract'):
        fixture(reflector_contract=bad)


@pytest.mark.parametrize('empty', [False, True])
def test_v2_prompt_binds_id_namespaces_and_evidence_semantics(empty):
    _, document = fixture(reflector_contract=m.REFLECTOR_CONTRACT_V2, empty_parent=empty)
    items = {} if empty else {'Z-item': 'Lesson Z.', 'A-item': 'Lesson A.'}
    measured = m.reflect_messages(document, 0, items, lambda _: 10)
    payload = json.loads(measured['messages'][1]['content'])
    assert payload['reflector_contract'] == m.REFLECTOR_CONTRACT_V2
    assert payload['allowed_item_ids'] == sorted(items)
    system = measured['messages'][0]['content']
    for condition in ('signals MUST be []', 'at most one signal per item_id',
            'distinct namespaces', 'facts directly shown', 'hypotheses, or must be omitted',
            'scorer matching rules', 'TP/FP/FN', 'strict_task_success must explicitly equal 1',
            'Extra false-positive slots remain errors', 'retrospective hypotheses'):
        assert condition in system


def test_v2_version_changes_input_and_prompt_hashes_but_not_evidence_or_budget():
    source, legacy = fixture()
    _, versioned = fixture(reflector_contract=m.REFLECTOR_CONTRACT_V2)
    assert versioned['input_sha256'] != legacy['input_sha256']
    assert versioned['groups'] == legacy['groups'] and versioned['source_sha256'] == legacy['source_sha256']
    assert {k:v for k,v in versioned['bounds'].items() if k != 'reflector_contract'} == legacy['bounds']
    assert m.reflect_messages(versioned,0,{},lambda _:10)['messages_sha256'] != m.reflect_messages(legacy,0,{},lambda _:10)['messages_sha256']
    rebuilt = m.build_input(source,versioned['parent'],learning_config_sha256=versioned['learning_config_sha256'],
        **{k:v for k,v in versioned['bounds'].items() if k != 'max_calls'})
    assert rebuilt == versioned


@pytest.mark.parametrize('fault', ['intent-as-item', 'duplicate-item', 'foreign-trajectory'])
def test_v2_keeps_strict_validator_without_dropping_invalid_signals(fault):
    _, document = fixture(reflector_contract=m.REFLECTOR_CONTRACT_V2)
    answer = reflection()
    if fault == 'intent-as-item':
        answer['signals'][0]['item_id'] = 'calendar_query'
    elif fault == 'duplicate-item':
        answer['signals'].append(copy.deepcopy(answer['signals'][0]))
    else:
        answer['findings'][0]['trajectory_ids'] = ['foreign']
    before = copy.deepcopy(answer)
    with pytest.raises(ValueError):
        m.validate_reflection(json.dumps(answer),document['groups'][0],document['parent']['items'],4)
    assert answer == before


def test_v2_empty_parent_add_then_next_reflection_uses_generated_item_id(tmp_path):
    content = 'A conditional lesson derived from observed task errors.'
    item_id = 'E'+m.hashlib.sha256(content.encode()).hexdigest()
    first = reflection(); first['signals'] = []
    second = reflection(1); second['signals'][0]['item_id'] = item_id
    unchanged = delta(); unchanged['deltas'] = []
    answers = {(0,m.REFLECT):first,(0,m.CURATE):delta('ADD',None,content),
        (1,m.REFLECT):second,(1,m.CURATE):unchanged}
    _, document, run, calls, fits = runner(tmp_path,empty_parent=True,
        reflector_contract=m.REFLECTOR_CONTRACT_V2,answers=answers)
    assert document['parent']['items'] == {}
    result = run()
    assert len(calls) == len(fits) == 4 and result['items'] == {item_id:content}
    assert json.loads(calls[0]['messages'][1]['content'])['allowed_item_ids'] == []
    assert json.loads(calls[2]['messages'][1]['content'])['allowed_item_ids'] == [item_id]
    assert result['cumulative_known_usage']['calls'] == 316
    assert run() == result and len(calls) == 4


def test_v2_invalid_empty_parent_signal_stays_failed_with_original_cost(tmp_path):
    answers = {(0,m.REFLECT):reflection()}
    _, _, run, calls, _ = runner(tmp_path,empty_parent=True,
        reflector_contract=m.REFLECTOR_CONTRACT_V2,answers=answers)
    for _ in range(2):
        with pytest.raises(ValueError,match='unknown or duplicate item'):
            run()
    assert len(calls) == 1 and not (tmp_path/'candidate.json').exists()
    failure = json.loads(next((tmp_path/'failures').glob('*.json')).read_bytes())
    assert failure['cumulative_known_usage']['calls'] == 313


def test_v2_cannot_reuse_legacy_contact_tree(tmp_path):
    _, _, legacy, calls, _ = runner(tmp_path)
    legacy()
    _, _, versioned, new_calls, _ = runner(tmp_path,reflector_contract=m.REFLECTOR_CONTRACT_V2)
    with pytest.raises(ValueError,match='durable result changed'):
        versioned()
    assert len(calls) == 4 and not new_calls


def test_v2_unknown_contact_still_blocks_retry_and_curator(tmp_path):
    _, _, run, calls, _ = runner(tmp_path,fail='unknown',reflector_contract=m.REFLECTOR_CONTRACT_V2)
    with pytest.raises(OSError):run()
    with pytest.raises(UnknownLearningContact):run()
    assert len(calls) == 1 and not (tmp_path/'candidate.json').exists()


def test_explicit_decision_context_separates_evidence_alias_from_editable_library():
    _, doc = fixture(empty_parent=True, decision_context_policy='explicit-editable-knowledge-v1')
    doc['groups'][0]['evidence']['development_evidence_dictionary']={'e5':'Read-only background intent.'}
    doc['input_sha256']=sha({k:v for k,v in doc.items() if k!='input_sha256'})
    context=m.decision_context(doc,0,{})
    assert context['allowed_existing_item_ids']==[]
    assert context['readonly_evidence_aliases']==['e5']
    assert context['allowed_operations']==['ADD'] and context['empty_delta_allowed']
    measured=m.curate_messages(doc,0,{},reflection(),{},lambda messages:100)
    body=json.loads(measured['messages'][1]['content'])
    assert body['original_group']==doc['groups'][0]
    assert body['decision_context']==context
    assert 'not targets' in measured['messages'][0]['content']
    reflected=m.reflect_messages(doc,0,{},lambda messages:100)
    assert json.loads(reflected['messages'][1]['content'])['decision_context']==context
    assert m.decision_context(doc,0,{'Eknown':'rule'})['allowed_existing_item_ids']==['Eknown']


def test_decision_context_opt_in_preserves_legacy_prompt():
    source, old = fixture()
    limits={k:v for k,v in old['bounds'].items() if k!='max_calls'}
    same=m.build_input(source,old['parent'],learning_config_sha256='a'*64,
                       decision_context_policy=None,**limits)
    assert same==old
    assert m.curate_messages(same,0,old['parent']['items'],reflection(),{},lambda messages:100)==m.curate_messages(old,0,old['parent']['items'],reflection(),{},lambda messages:100)
    with pytest.raises(ValueError,match='decision context policy'):
        fixture(decision_context_policy='invented')
