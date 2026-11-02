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
    final=store.finish()
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
