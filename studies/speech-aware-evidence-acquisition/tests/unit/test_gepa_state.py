"""Controller tests use explicit score doubles, not native custody or algorithm gains."""
import copy
import multiprocessing
import os
import pytest
from speech_aware_evidence_acquisition.rails import gepa_state as m


def cand(i):
    return {'candidate_id': f'c{i}', 'instruction_sha256': str(i)*64}


def make(root, **kwargs):
    config = dict(feedback_ids=[f'slurp:train:f{i}' for i in range(6)],
        selection_ids=['slurp:train:s0','slurp:train:s1'], excluded_ids=['slurp:train:h2'],
        partition_sha256='a'*64, frozen_system_sha256='b'*64, initial_candidate=cand(0),
        seed=10, batch_size=2, max_rounds=3,
        budget={'calls':100,'prompt_tokens':10000,'completion_tokens':10000},
        inherited_usage={'calls':312,'prompt_tokens':2000,'completion_tokens':500})
    config.update(kwargs)
    return m.GepaSearchState.create(root, **config)


def execute(store, result, *, prompt=10, completion=5):
    state=store.read()['state']; config=store.read()['config']; phase=state['phase']
    n=1 if phase=='mutate' else len(config['selection_ids'] if phase in ('seed_eval','selection_eval') else state['active']['sample_ids'])
    # A malformed score vector is still an observed response; use a stable
    # request identity so the controller can retain and fail it durably.
    request_sha = 'f'*64 if any(isinstance(x, float) and x != x for x in result.get('scores', [])) else m.sha(result)
    intent=store.begin_stage(request_sha256=request_sha,
        allocation={'calls':n,'prompt_tokens':100,'completion_tokens':100})
    return store.observe(intent['stage_id'], response_sha256=m.sha({'result_repr':repr(result)}),
        actual_usage={'calls':n,'prompt_tokens':prompt,'completion_tokens':completion}, result=result)


def evaluate(store, identity, scores):
    state=store.read()['state']; config=store.read()['config']
    samples=config['selection_ids'] if state['phase'] in ('seed_eval','selection_eval') else state['active']['sample_ids']
    return execute(store, {'candidate':identity, 'sample_ids':samples, 'scores':scores})


def seed(store):
    evaluate(store,cand(0),[.4,.4])


def test_two_round_search_rejects_tie_and_accepts_only_after_full_selection(tmp_path):
    store=make(tmp_path);seed(store)
    first=store.next_round(); assert store.next_round()==first
    evaluate(store,first['parent'],[.3,.5]);execute(store,{'candidate':cand(1)})
    evaluate(store,cand(1),[.4,.4])
    assert store.read()['state']['rounds'][0]['outcome']=='rejected-no-strict-improvement'
    second=store.next_round(); assert not set(first['sample_ids']) & set(second['sample_ids'])
    evaluate(store,second['parent'],[.3,.5]);execute(store,{'candidate':cand(2)})
    evaluate(store,cand(2),[.6,.8])
    assert len(store.read()['state']['pool'])==1
    evaluate(store,cand(2),[.7,.7])
    final=store.finish(reason="configured evaluation stop")
    assert final['state']['winner_id']=='c2'
    assert final['state']['cumulative_usage']['calls']==326
    assert final==store.finish()
    assert m.GepaSearchState(tmp_path).read()==final


def test_unknown_blocks_parent_selection_next_contact_and_finish(tmp_path):
    store=make(tmp_path)
    intent=store.begin_stage(request_sha256='f'*64,allocation={'calls':2,'prompt_tokens':100,'completion_tokens':100})
    reopened=m.GepaSearchState(tmp_path)
    with pytest.raises(ValueError,match='unknown'):reopened.next_round()
    with pytest.raises(ValueError,match='unknown'):reopened.begin_stage(request_sha256='f'*64,allocation=intent['allocation'])
    with pytest.raises(ValueError,match='unknown'):reopened.finish()
    assert reopened.read()['state']['pending']==intent


def test_known_response_replay_is_free_but_changed_cost_rejected(tmp_path):
    store=make(tmp_path)
    result={'candidate':cand(0),'sample_ids':store.read()['config']['selection_ids'],'scores':[.4,.6]}
    intent=store.begin_stage(request_sha256='f'*64,allocation={'calls':2,'prompt_tokens':100,'completion_tokens':100})
    args=dict(response_sha256='d'*64,actual_usage={'calls':2,'prompt_tokens':20,'completion_tokens':10},result=result)
    first=store.observe(intent['stage_id'],**args)
    assert m.GepaSearchState(tmp_path).observe(intent['stage_id'],**args)==first
    args['actual_usage']['calls']=1
    with pytest.raises(ValueError,match='cost changed'):store.observe(intent['stage_id'],**args)


@pytest.mark.parametrize('fault',['partial','foreign','wrong_candidate','nan','outside','usage'])
def test_bad_observed_result_preserves_cost_and_stops(tmp_path,fault):
    store=make(tmp_path); config=store.read()['config']
    result={'candidate':cand(0),'sample_ids':config['selection_ids'],'scores':[.5,.5]}
    if fault=='partial':result['scores']=[.5]
    if fault=='foreign':result['sample_ids']=['slurp:train:h2','slurp:train:s1']
    if fault=='wrong_candidate':result['candidate']=cand(1)
    if fault=='nan':result['scores']=[float('nan'),.5]
    if fault=='outside':result['scores']=[1.1,.5]
    # Nonfinite score diagnostics preserve known spend without publishing NaN JSON.
    doc=execute(store,result,completion=101 if fault=='usage' else 5)
    assert doc['state']['phase']=='failed' and doc['state']['known_usage']['calls']==2
    assert len(doc['state']['observed_stages'])==1
    with pytest.raises(ValueError):store.finish()


def test_budget_reserves_whole_stage_before_intent(tmp_path):
    store=make(tmp_path,budget={'calls':1,'prompt_tokens':100,'completion_tokens':100})
    before=store.read()
    with pytest.raises(ValueError,match='budget'):store.begin_stage(request_sha256='f'*64,allocation={'calls':2,'prompt_tokens':100,'completion_tokens':100})
    assert store.read()==before


@pytest.mark.parametrize('fault',['overlap','h2','test','bad_hash','bool'])
def test_partition_and_config_rejected(tmp_path,fault):
    kw={}
    if fault=='overlap':kw['selection_ids']=['slurp:train:f0']
    if fault=='h2':kw['feedback_ids']=['slurp:train:h2']
    if fault=='test':kw['feedback_ids']=['slurp:test:0']
    if fault=='bad_hash':kw['partition_sha256']='z'*64
    if fault=='bool':kw['seed']=True
    with pytest.raises(ValueError):make(tmp_path,**kw)


def test_parent_rng_and_batch_survive_fresh_process(tmp_path):
    store=make(tmp_path);seed(store)
    def child():
        m.GepaSearchState(tmp_path).next_round();os._exit(73)
    p=multiprocessing.get_context('fork').Process(target=child);p.start();p.join(10)
    assert p.exitcode==73
    first=store.read(); active=store.next_round()
    assert first==store.read() and active==first['state']['active']


def test_snapshot_publication_crash_retains_previous_and_linked_state(tmp_path,monkeypatch):
    store=make(tmp_path);seed(store);before=store.read();original=m.publish_new
    def crash_before(*a):raise OSError('before publish')
    monkeypatch.setattr(m,'publish_new',crash_before)
    with pytest.raises(OSError):store.next_round()
    assert store.read()==before
    def crash_after(*a):original(*a);raise OSError('after publish')
    monkeypatch.setattr(m,'publish_new',crash_after)
    with pytest.raises(OSError):store.next_round()
    after=store.read(); assert after['sequence']==before['sequence']+1
    assert store.next_round()==after['state']['active']


def test_perfect_and_noop_consume_rounds_and_finite_feedback(tmp_path):
    store=make(tmp_path,max_rounds=2);seed(store)
    active=store.next_round();evaluate(store,active['parent'],[1,1])
    assert store.read()['state']['rounds'][0]['outcome']=='skipped-perfect'
    active=store.next_round();evaluate(store,active['parent'],[0,0])
    execute(store,{'candidate':cand(0)})
    assert store.read()['state']['rounds'][1]['outcome']=='rejected-noop'
    with pytest.raises(ValueError,match='exhausted'):store.next_round()
    assert store.finish()['state']['winner_id']=='c0'


def test_tampered_or_missing_snapshot_refuses(tmp_path):
    store=make(tmp_path);seed(store)
    (tmp_path/'000001.json').unlink()
    with pytest.raises(ValueError,match='gap'):store.read()


def test_active_round_cannot_disappear_without_reason(tmp_path):
    store=make(tmp_path);seed(store);active=store.next_round();before=store.read()
    with pytest.raises(ValueError,match='explicit reason'):store.finish()
    assert store.read()==before
    final=store.finish(reason='insufficient remaining wall-time for next stage')
    assert final['state']['termination']['interrupted_round']==active
    assert final['state']['rounds'][-1]['outcome']=='stopped-before-next-stage'
    assert final['state']['active'] is None
    assert final['state']['winner_id']=='c0'
    assert final['state']['known_usage']==before['state']['known_usage']
    assert m.GepaSearchState(tmp_path).finish()==final
    with pytest.raises(ValueError,match='reason changed'):store.finish(reason='different reason')


def execute_v2(store, result, *, calls=None):
    """Explicit graph-cost test double, never native execution evidence."""
    doc = store.read(); state = doc['state']; phase = state['phase']
    samples = (doc['config']['selection_ids'] if phase in ('seed_eval', 'selection_eval')
               else state['active']['sample_ids'])
    normal = 1 if phase == 'mutate' else len(samples) * 3
    intent = store.begin_stage(request_sha256=m.sha(result),
        allocation={'calls':normal, 'prompt_tokens':100, 'completion_tokens':100})
    return store.observe(intent['stage_id'], response_sha256=m.sha(result),
        actual_usage={'calls':normal if calls is None else calls, 'prompt_tokens':10, 'completion_tokens':5},
        result=result)


def prepare_pair(root):
    store = make(root, schema=m.PAIRED_SCHEMA)
    execute_v2(store, {'candidate':cand(0), 'sample_ids':store.read()['config']['selection_ids'], 'scores':[.4,.4]})
    active = store.next_round()
    execute_v2(store, {'candidate':active['parent'], 'sample_ids':active['sample_ids'], 'scores':[0,.1]})
    execute_v2(store, {'candidate':cand(1)})
    assert store.read()['state']['phase'] == 'paired_eval'
    return store


def pair_intent_result(store, parent_scores, child_scores):
    doc = store.read(); active = doc['state']['active']; config = doc['config']
    intent = store.begin_stage(request_sha256='e'*64,
        allocation={'calls':4*len(active['sample_ids']), 'prompt_tokens':100, 'completion_tokens':100})
    parent_scores = [float(v) for v in parent_scores]
    child_scores = [float(v) for v in child_scores]
    evidence = {'schema':m.PAIR_EVIDENCE_SCHEMA, 'stage_id':intent['stage_id'],
        'request_sha256':intent['request_sha256'], 'partition_sha256':config['partition_sha256'],
        'frozen_system_sha256':config['frozen_system_sha256'],
        'parent':active['parent'], 'child':active['child'], 'sample_ids':active['sample_ids'],
        'shared_perception_sha256':'c'*64, 'shared_supply_sha256':'d'*64,
        'execution_sha256':'f'*64, 'score_receipt_sha256':'a'*64,
        'scores_sha256':m.sha({'parent_scores':parent_scores, 'child_scores':child_scores,
                              'sample_ids':active['sample_ids']})}
    result = {'parent':active['parent'], 'child':active['child'], 'sample_ids':active['sample_ids'],
              'parent_scores':parent_scores, 'child_scores':child_scores, 'evidence':evidence}
    return intent, result


def observe_pair(store, intent, result, *, calls=8, response_sha256=None):
    return store.observe(intent['stage_id'], response_sha256=response_sha256 or m.sha(result.get('evidence')),
        actual_usage={'calls':calls, 'prompt_tokens':20, 'completion_tokens':10}, result=result)


def test_v2_uses_current_paired_parent_not_old_diagnostic_scores(tmp_path):
    store = prepare_pair(tmp_path)
    active = store.read()['state']['active']
    assert active['diagnostic_parent_scores'] == [0,.1] and 'parent_scores' not in active
    intent, result = pair_intent_result(store, [.9,.8], [.6,.6])
    final = observe_pair(store, intent, result)
    assert final['state']['rounds'][-1]['outcome'] == 'rejected-no-strict-improvement'
    assert final['state']['rounds'][-1]['paired_parent_scores'] == [.9,.8]
    assert len(final['state']['pool']) == 1
    assert final['state']['known_usage']['calls'] == 21  # 3*2 + 3*2 + 1 + 4*2
    assert m.GepaSearchState(tmp_path).read() == final


def test_v2_pair_tie_rejects_and_success_waits_for_complete_selection(tmp_path):
    store = prepare_pair(tmp_path)
    intent, result = pair_intent_result(store, [.2,.8], [.5,.5])
    assert observe_pair(store, intent, result)['state']['rounds'][-1]['outcome'] == 'rejected-no-strict-improvement'
    active = store.next_round()
    execute_v2(store, {'candidate':active['parent'], 'sample_ids':active['sample_ids'], 'scores':[.1,.1]})
    execute_v2(store, {'candidate':cand(2)})
    intent, result = pair_intent_result(store, [.3,.3], [.6,.8])
    paired = observe_pair(store, intent, result)
    assert paired['state']['phase'] == 'selection_eval' and len(paired['state']['pool']) == 1
    selected = execute_v2(store, {'candidate':cand(2), 'sample_ids':store.read()['config']['selection_ids'], 'scores':[.8,.8]})
    assert selected['state']['rounds'][-1]['outcome'] == 'accepted'
    assert len(selected['state']['pool']) == 2
    final = store.finish(reason='registered small search bound')
    assert final['state']['winner_id'] == 'c2'
    assert final['state']['cumulative_usage']['calls'] == 354  # 42 new + 312 inherited


@pytest.mark.parametrize('fault', ['parent', 'child', 'roster', 'missing_evidence', 'evidence_parent',
    'evidence_roster', 'stage', 'request', 'partition', 'frozen', 'perception', 'supply',
    'execution', 'score_receipt', 'score_pin', 'scores', 'response_pin', 'partial_calls'])
def test_v2_pair_rejects_unbound_evidence_and_retains_all_known_spend(tmp_path, fault):
    store = prepare_pair(tmp_path)
    intent, result = pair_intent_result(store, [.2,.2], [.5,.5])
    result = copy.deepcopy(result)
    if fault in ('parent', 'child'): result[fault] = cand(3)
    elif fault == 'roster': result['sample_ids'].reverse()
    elif fault == 'missing_evidence': del result['evidence']
    elif fault == 'evidence_parent': result['evidence']['parent'] = cand(3)
    elif fault == 'evidence_roster': result['evidence']['sample_ids'] = ['slurp:train:h2']
    elif fault == 'stage': result['evidence']['stage_id'] = '0'*64
    elif fault == 'request': result['evidence']['request_sha256'] = '0'*64
    elif fault == 'partition': result['evidence']['partition_sha256'] = '0'*64
    elif fault == 'frozen': result['evidence']['frozen_system_sha256'] = '0'*64
    elif fault == 'perception': result['evidence']['shared_perception_sha256'] = 'invalid'
    elif fault == 'supply': result['evidence']['shared_supply_sha256'] = 'invalid'
    elif fault == 'execution': result['evidence']['execution_sha256'] = 'invalid'
    elif fault == 'score_receipt': result['evidence']['score_receipt_sha256'] = 'invalid'
    elif fault == 'score_pin': result['evidence']['scores_sha256'] = '0'*64
    elif fault == 'scores': result['child_scores'][0] = .9
    calls = 7 if fault == 'partial_calls' else 8
    final = observe_pair(store, intent, result, calls=calls,
        response_sha256='0'*64 if fault == 'response_pin' else None)
    assert final['state']['phase'] == 'failed'
    assert final['state']['known_usage']['calls'] == 13 + calls
    assert len(final['state']['observed_stages']) == 4
    assert len(final['state']['pool']) == 1
    with pytest.raises(ValueError, match='failed'): store.finish()


def test_v2_pair_unknown_recovery_and_idempotent_observation(tmp_path):
    store = prepare_pair(tmp_path); intent, result = pair_intent_result(store, [.2,.2], [.6,.6])
    reopened = m.GepaSearchState(tmp_path)
    assert reopened.read()['state']['pending'] == intent
    with pytest.raises(ValueError, match='unknown'):
        reopened.begin_stage(request_sha256='e'*64, allocation=intent['allocation'])
    with pytest.raises(ValueError, match='unknown'): reopened.next_round()
    first = observe_pair(reopened, intent, result)
    assert observe_pair(m.GepaSearchState(tmp_path), intent, result) == first
    with pytest.raises(ValueError, match='cost changed'):
        observe_pair(reopened, intent, result, calls=7)


def test_v2_reserves_full_graph_and_preserves_v1_schema(tmp_path):
    old = make(tmp_path/'old')
    assert old.read()['schema'] == m.SCHEMA and 'evaluation_mode' not in old.read()['config']
    paired = make(tmp_path/'new', schema=m.PAIRED_SCHEMA)
    assert paired.read()['schema'] == m.PAIRED_SCHEMA
    before = paired.read()
    with pytest.raises(ValueError, match='complete sample roster'):
        paired.begin_stage(request_sha256='f'*64, allocation={'calls':5,'prompt_tokens':100,'completion_tokens':100})
    assert paired.read() == before
    limited = make(tmp_path/'limited', schema=m.PAIRED_SCHEMA,
        budget={'calls':5,'prompt_tokens':1000,'completion_tokens':1000})
    with pytest.raises(ValueError, match='budget'):
        limited.begin_stage(request_sha256='f'*64, allocation={'calls':6,'prompt_tokens':100,'completion_tokens':100})
    pair = prepare_pair(tmp_path/'pair'); before = pair.read()
    with pytest.raises(ValueError, match='complete sample roster'):
        pair.begin_stage(request_sha256='f'*64, allocation={'calls':7,'prompt_tokens':100,'completion_tokens':100})
    assert pair.read() == before


def test_v2_schema_cannot_silently_relabel_v1_history(tmp_path):
    import json
    store = make(tmp_path); seed(store)
    path = tmp_path/'000002.json'
    doc = json.loads(path.read_text()); doc['schema'] = m.PAIRED_SCHEMA
    doc['sha256'] = m.sha({k:v for k,v in doc.items() if k != 'sha256'})
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='schema changed'): store.read()
