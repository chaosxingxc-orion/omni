"""Functional Reflexion contracts, not synthetic research outcomes."""
import copy
import json
import pytest
from speech_aware_evidence_acquisition.rails import reflexion_offline as m
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (
    ExperienceLearner, UnknownLearningContact, empty_experience, sha, canonical)
from test_ace_incremental import fixture as ace_fixture


def fixture(groups=3, **limits):
    source, _ = ace_fixture(groups=groups, empty_parent=True, max_groups=max(groups, 1))
    for i, group in enumerate(source['batch']['groups']):
        original = group['rollouts'][0]
        group['rollouts'] = [{**copy.deepcopy(original), 'trajectory_id': f'T{i}-{j}'}
                             for j in range(4)]
    bounds = dict(resident_context=49152, completion_cap=100, reserve_tokens=100,
        max_groups=51, max_items=48, max_characters=10000, max_entry_characters=200,
        max_prompt_tokens_total=100000, max_completion_tokens_total=10000)
    bounds.update(limits)
    return source, m.build_input(source, source['batch']['parent'],
        learning_config_sha256='a'*64, **bounds)


def reflection(index=0, content=None):
    return {'schema': 'saea-reflexion-memory-v1', 'reflection': {
        'content': content or f'When explicit evidence supports action {index}, preserve its scope.',
        'trajectory_ids': [f'T{index}-0']}}


def runner(tmp_path, *, answers=None, failure=None, **limits):
    source, document = fixture(**limits); calls = []; fits = []
    def call(request):
        index = int(request['identity'].rsplit('-', 1)[1]); calls.append(request)
        if failure == ('unknown', index):
            raise OSError('unknown transport')
        answer = answers[index] if answers is not None else reflection(index)
        text = 'bad JSON' if failure == ('invalid', index) else json.dumps(answer)
        return {'model_id': 'model', 'receipt_id': str(len(calls)), 'text': text,
                'usage': {'calls': 1, 'prompt_tokens': 10, 'completion_tokens': 5}}
    def run():
        learner = ExperienceLearner(tmp_path, model_id='model', learning_config_sha256='a'*64,
            metered_call=call, preflight=lambda r: fits.append(r), max_calls=document['bounds']['max_calls'])
        return m.execute(learner, document, count_tokens=lambda messages: 10,
                         verify_source=lambda: copy.deepcopy(source))
    return source, document, run, calls, fits


def test_all_four_failed_trajectories_and_equal_rewards_are_retained():
    source, doc = fixture()
    assert doc['bounds']['max_calls'] == 3 and not doc['parent']['items']
    assert len(doc['groups']) == 3
    assert all(len(g['evidence']['rollouts']) == 4 for g in doc['groups'])
    assert all(r['feedback']['strict_task_success'] == 0
               for g in doc['groups'] for r in g['evidence']['rollouts'])
    assert not doc['historical_actor_memory_conditioned']


def test_real_shared_learner_single_stage_fifo_and_known_response_replay(tmp_path):
    _, doc, run, calls, fits = runner(tmp_path, max_items=2)
    first = run(); second = run()
    assert first == second and len(calls) == len(fits) == 3
    assert all(r['stage'] == m.REFLECT for r in calls)
    assert len(first['items']) == 2 and len(first['learned_query_ids']) == 3
    assert first['updates'][2]['evicted'][0]['item_id'] == first['updates'][0]['item_id']
    last_prompt = json.loads(calls[2]['messages'][1]['content'])
    assert [r['content'] for r in last_prompt['memory']] == [
        reflection(0)['reflection']['content'], reflection(1)['reflection']['content']]
    assert first['cumulative_known_usage']['calls'] == 315
    assert not first['deployment_accepted'] and not first['semantic_support_verified']
    assert first['requires_real_evaluation'] and first['offline_transfer']


def test_duplicate_does_not_refresh_fifo_and_null_is_paid(tmp_path):
    texts = [reflection(0, 'Scoped A.'), reflection(1, 'Scoped B.'),
             reflection(2, 'Scoped A.'), {'schema': 'saea-reflexion-memory-v1', 'reflection': None},
             reflection(4, 'Scoped C.')]
    _, _, run, calls, _ = runner(tmp_path, groups=5, max_items=2, answers=texts)
    candidate = run()
    assert [e['action'] for e in candidate['updates']] == ['append','append','duplicate','empty','append']
    assert candidate['updates'][-1]['evicted'][0]['item_id'] == candidate['updates'][0]['item_id']
    assert list(candidate['items'].values()) == ['Scoped B.', 'Scoped C.']
    assert len(candidate['item_provenance'][candidate['updates'][0]['item_id']]) == 2
    assert len(calls) == candidate['learning_usage']['calls'] == 5


@pytest.mark.parametrize('index', [0,1,2])
def test_invalid_observed_output_preserves_cost_stops_and_never_resends(tmp_path, index):
    _, _, run, calls, _ = runner(tmp_path, failure=('invalid',index))
    for _ in range(2):
        with pytest.raises(ValueError, match='exact JSON'):
            run()
    assert len(calls) == index+1 and not (tmp_path/'candidate.json').exists()
    failures = [json.loads(p.read_bytes()) for p in (tmp_path/'failures').glob('*.json')]
    assert failures[-1]['accounting']['known_usage']['calls'] == index+1


@pytest.mark.parametrize('index', [0,1,2])
def test_unknown_blocks_later_stages_no_retry(tmp_path,index):
    _, _, run, calls, _ = runner(tmp_path,failure=('unknown',index))
    with pytest.raises(OSError):run()
    with pytest.raises(UnknownLearningContact):run()
    assert len(calls) == index+1


@pytest.mark.parametrize('fault', ['schema','extra','empty','long','foreign','duplicate','signals','duplicate_json'])
def test_distinct_strict_response_parser(fault):
    _, doc = fixture(); value = reflection()
    if fault=='schema':value['schema']='saea-ace-reflection-v1'
    elif fault=='extra':value['success']=True
    elif fault=='empty':value['reflection']['content']=' '
    elif fault=='long':value['reflection']['content']='x'*201
    elif fault=='foreign':value['reflection']['trajectory_ids']=['T1-0']
    elif fault=='duplicate':value['reflection']['trajectory_ids']=['T0-0','T0-0']
    elif fault=='signals':value['reflection']['signals']=[]
    text=json.dumps(value)
    if fault=='duplicate_json':text='{"schema":"saea-reflexion-memory-v1","reflection":null,"reflection":null}'
    with pytest.raises(ValueError):m.validate_reflection(text,doc['groups'][0],200)


@pytest.mark.parametrize('fault', ['test','parent','trajectory_parent','policy','unknown','cost','four','source_hash'])
def test_source_and_ancestry_rejections(fault):
    source, doc = fixture();parent=source['batch']['parent']
    if fault=='test':source['batch']['groups'][0]['query_id']='slurp:test:0'
    elif fault=='parent':
        parent['items']={'x':'not empty'};parent['sha256']=sha({k:v for k,v in parent.items() if k!='sha256'})
    elif fault=='trajectory_parent':source['batch']['groups'][0]['rollouts'][0]['experience_sha256']='f'*64
    elif fault=='policy':source['batch']['groups'][0]['rollouts'][0]['policy_sha256']='f'*64
    elif fault=='unknown':source['unknown_contacts']=1
    elif fault=='cost':source['original_total_usage']['calls']=0
    elif fault=='four':source['batch']['groups'][0]['rollouts'].pop()
    else:source['batch']['groups'][0]['rollouts'][1]['trajectory_id']='T0-0'
    with pytest.raises(ValueError):
        m.build_input(source,parent,learning_config_sha256='a'*64,
            **{k:v for k,v in doc['bounds'].items() if k!='max_calls'})


def test_budget_and_context_refuse_before_contact(tmp_path):
    _, _, run, calls, fits = runner(tmp_path,max_prompt_tokens_total=9)
    with pytest.raises(ValueError,match='budget'):run()
    assert not calls and not fits
    _, doc = fixture(resident_context=201)
    with pytest.raises(ValueError,match='context'):
        m.reflect_messages(doc,0,m.empty_memory(),lambda messages:2)


def test_character_capacity_evicts_whole_oldest_entry_without_truncation(tmp_path):
    answers=[reflection(i,letter*6) for i,letter in enumerate('abc')]
    _,_,run,_,_=runner(tmp_path,answers=answers,max_characters=10)
    candidate=run()
    assert list(candidate['items'].values())==['cccccc']
    assert len(candidate['updates'][1]['evicted'])==len(candidate['updates'][2]['evicted'])==1


def test_foreign_or_nonprefix_history_refuses_before_new_call(tmp_path):
    _,doc,run,calls,_=runner(tmp_path)
    root=tmp_path/'contacts';root.mkdir()
    request={'stage':m.REFLECT,'identity':f'{doc["input_sha256"]}:group-2'}
    (root/(sha(request)+'.intent.json')).write_bytes(canonical(request))
    with pytest.raises(ValueError,match='nonsequential'):run()
    assert not calls


def compact_reflection(content='When a slot is uncertain, use only explicit evidence; do not add alternatives.'):
    return {'schema': m.COMPACT_CONTRACT, 'reflection': {'content': content, 'refs': ['r0']}}


def test_compact_contract_changes_identity_retains_evidence_and_replays(tmp_path):
    source, legacy = fixture()
    _, doc, run, calls, fits = runner(tmp_path, reflector_contract=m.COMPACT_CONTRACT,
        answers=[compact_reflection() for _ in range(3)])
    assert doc['input_sha256'] != legacy['input_sha256']
    assert 'reflector_contract' not in legacy['bounds']
    first = run()
    assert first == run() and len(calls) == len(fits) == 3
    payload = json.loads(calls[0]['messages'][1]['content'])
    assert payload['original_group'] == legacy['groups'][0]
    assert payload['trajectory_references'] == {f'r{i}': f'T0-{i}' for i in range(4)}
    assert 'Never add speculative' in calls[0]['messages'][0]['content']
    assert first['updates'][0]['reflection']['reflection']['trajectory_ids'] == ['T0-0']
    assert not first['semantic_support_verified']


@pytest.mark.parametrize('fault', ['long', 'foreign', 'duplicate', 'legacy', 'extra', 'duplicate_json'])
def test_compact_invalid_known_output_is_retained_without_retry(tmp_path, fault):
    answer = compact_reflection()
    if fault == 'long': answer['reflection']['content'] = '字' * 201
    elif fault == 'foreign': answer['reflection']['refs'] = ['r4']
    elif fault == 'duplicate': answer['reflection']['refs'] = ['r0', 'r0']
    elif fault == 'legacy': answer = reflection()
    elif fault == 'extra': answer['reflection']['rationale'] = 'not allowed'
    else:
        _, doc = fixture(reflector_contract=m.COMPACT_CONTRACT)
        text = '{"schema":"saea-reflexion-compact-v2","reflection":null,"reflection":null}'
        with pytest.raises(ValueError, match='exact JSON'):
            m.validate_reflection(text, doc['groups'][0], 200, m.COMPACT_CONTRACT)
        return
    _, _, run, calls, fits = runner(tmp_path, reflector_contract=m.COMPACT_CONTRACT,
        answers=[answer] * 3)
    for _ in range(2):
        with pytest.raises(ValueError): run()
    assert len(calls) == len(fits) == 1
    assert len(list((tmp_path/'contacts').glob('*.response.json'))) == 1
    assert not (tmp_path/'candidate.json').exists()


def test_compact_null_and_unicode_boundary():
    _, doc = fixture(reflector_contract=m.COMPACT_CONTRACT)
    group = doc['groups'][0]
    assert m.validate_reflection(json.dumps({'schema': m.COMPACT_CONTRACT, 'reflection': None}),
        group, 200, m.COMPACT_CONTRACT)['reflection'] is None
    result = m.validate_reflection(json.dumps(compact_reflection('字' * 200)),
        group, 200, m.COMPACT_CONTRACT)
    assert len(result['reflection']['content']) == 200
    with pytest.raises(ValueError):
        m.validate_reflection(json.dumps(compact_reflection()), group, 200)


def test_error_feedback_variant_binds_new_evidence_without_changing_compact():
    _, old = fixture(reflector_contract=m.COMPACT_CONTRACT)
    _, new = fixture(reflector_contract=m.ERROR_FEEDBACK_CONTRACT)
    prior = m.reflect_messages(old,0,m.empty_memory(),lambda _:10)
    revised = m.reflect_messages(new,0,m.empty_memory(),lambda _:10)
    payload = json.loads(revised['messages'][1]['content'])
    assert 'literal_slot_errors' not in json.loads(prior['messages'][1]['content'])
    assert payload['literal_slot_errors']['official_score_recomputed'] is False
    assert payload['original_group'] == old['groups'][0]
    assert 'missing reference entity remains an error' in revised['messages'][0]['content']
    assert old['input_sha256'] != new['input_sha256']
    answer=compact_reflection();answer['schema']=m.ERROR_FEEDBACK_CONTRACT
    assert m.validate_reflection(json.dumps(answer),new['groups'][0],200,m.ERROR_FEEDBACK_CONTRACT)['reflection']['trajectory_ids']==['T0-0']
    with pytest.raises(ValueError):m.validate_reflection(json.dumps(answer),old['groups'][0],200,m.COMPACT_CONTRACT)
