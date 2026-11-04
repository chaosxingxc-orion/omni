"""Real executor protocol and ACE flow with fake network, never research evidence."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.rails.external_teacher import ExternalTeacher
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import ExperienceLearner, UnknownLearningContact
from speech_aware_evidence_acquisition.rails import ace_incremental as ace
from test_ace_incremental import fixture, reflection, delta


def teacher(tmp_path, stream=False, analysis_budget=False):
    study = Path(__file__).parents[2]
    script = study/'scripts/environment/configured_external_learner.py'
    prompt = tmp_path/'policy.md'; prompt.write_text('Analyze mechanisms and counterexamples.')
    cfg = {'schema': 'saea-external-learner-v1', 'endpoint': 'https://example.invalid/v1/chat/completions',
        'model': 'external-model', 'api_key_env': 'TEST_EXTERNAL_KEY', 'max_tokens': 131072,
        'timeout_seconds': 10, 'stream': stream,
        'research_prompt': {'path': str(prompt), 'sha256': hashlib.sha256(prompt.read_bytes()).hexdigest()}}
    if analysis_budget:
        example = json.loads((study/'configs/learners/glm-flash-external.example.json').read_bytes())
        cfg['analysis_budget'] = example['analysis_budget']
    return ExternalTeacher(tmp_path/'api', config=cfg,
        executor={'path': str(script), 'sha256': hashlib.sha256(script.read_bytes()).hexdigest()})


def request(caller):
    return {'schema': 'saea-tfgrpo-learning-request-v1', 'batch_sha256': 'b'*64,
        'model_id': caller.model_id, 'learning_config_sha256': caller.learning_config_sha256,
        'stage': 'stage', 'identity': 'query', 'messages': [{'role': 'user', 'content': 'complete evidence'}]}


def network(caller, monkeypatch, responses):
    monkeypatch.setenv('TEST_EXTERNAL_KEY', 'private-test-secret')
    calls = []
    class Response:
        status = 200; headers = {}
        def __init__(self, value): self.value = value
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return json.dumps(self.value).encode()
        def __iter__(self):
            v = self.value
            events = [{'id': v['id'], 'model': v['model'], 'choices': [{'index': 0,
                'delta': v['choices'][0]['message'], 'finish_reason': v['choices'][0]['finish_reason']}]},
                {'id': v['id'], 'model': v['model'], 'choices': [], 'usage': v['usage']}]
            for event in events:
                yield ('data: '+json.dumps(event)+'\n').encode(); yield b'\n'
            yield b'data: [DONE]\n'; yield b'\n'
    class Opener:
        def open(self, req, **kw):
            calls.append(json.loads(req.data))
            result = responses[len(calls)-1]
            if isinstance(result, Exception): raise result
            return Response(result)
    monkeypatch.setattr(caller.executor.urllib.request, 'build_opener', lambda *a: Opener())
    return calls


def answer(text, identity='r1', model='external-model'):
    return {'id': identity, 'model': model,
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': text}}],
        'usage': {'prompt_tokens': 20, 'completion_tokens': 10,
                  'completion_tokens_details': {'reasoning_tokens': 7}}}


@pytest.mark.parametrize('stream', [False, True])
def test_metered_receipt_roundtrip_and_tamper(tmp_path, monkeypatch, stream):
    c = teacher(tmp_path, stream); req = request(c)
    calls = network(c, monkeypatch, [answer('result')])
    c.preflight(req)
    assert not calls
    observed = c(req)
    assert observed['model_id'] == 'external-model' and observed['receipt_kind'] == 'external-chat-v1'
    assert observed['provider_usage']['completion_tokens_details']['reasoning_tokens'] == 7
    assert not observed['native_fit_completed']
    assert c.reconcile(req)['response'] == observed
    assert len(calls) == 1 and calls[0]['messages'][1:] == req['messages']
    directory, _ = c._binding(req)
    target = directory/'response'/('response.sse' if stream else 'response.bin')
    target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(ValueError, match='wire changed'): c.reconcile(req)
    assert len(calls) == 1


def test_unknown_never_redispatched(tmp_path, monkeypatch):
    c = teacher(tmp_path); req = request(c)
    calls = network(c, monkeypatch, [TimeoutError()])
    for _ in range(2):
        with pytest.raises(UnknownLearningContact): c(req)
    assert c.reconcile(req) is None and len(calls) == 1


@pytest.mark.parametrize('fault', ['identity', 'usage'])
def test_provider_mismatch_retained_without_fake_receipt(tmp_path, monkeypatch, fault):
    c = teacher(tmp_path); req = request(c); response = answer('result')
    if fault == 'identity': response['model'] = 'foreign'
    else: response['usage'].pop('prompt_tokens')
    calls = network(c, monkeypatch, [response])
    with pytest.raises(ValueError): c(req)
    with pytest.raises(ValueError): c(req)
    assert len(calls) == 1


def test_complete_ace_external_teacher_keeps_actor_identity(tmp_path, monkeypatch):
    c = teacher(tmp_path)
    source, old = fixture(groups=1)
    bounds = {k: v for k,v in old['bounds'].items() if k != 'max_calls'}
    bounds.update(resident_context=262144, completion_cap=131072, max_completion_tokens_total=262144)
    doc = ace.build_input(source, old['parent'], learning_config_sha256=c.learning_config_sha256,
        learner_model_id=c.model_id, **bounds)
    calls = network(c, monkeypatch, [answer(json.dumps(reflection()), 'reflect'),
                                    answer(json.dumps(delta()), 'curate')])
    learner = ExperienceLearner(tmp_path/'learner', model_id=c.model_id, metered_call=c,
        preflight=c.preflight, reconcile=c.reconcile, learning_config_sha256=c.learning_config_sha256,
        max_calls=2)
    # Explicit test-only tokenizer; this is not a provider-capacity claim.
    run = lambda: ace.execute(learner, doc, count_tokens=lambda messages: 20,
                             verify_source=lambda: copy.deepcopy(source))
    result = run()
    assert result == run() and len(calls) == 2
    assert result['model_id'] == result['actor_model_id'] == 'model'
    assert result['learner_model_id'] == 'external-model'
    assert result['items']['old'] == delta()['deltas'][0]['content']
    assert result['learning_usage'] == {'calls': 2, 'prompt_tokens': 40, 'completion_tokens': 20}
    assert not result['deployment_accepted'] and result['requires_real_evaluation']
    assert doc['source_sha256'] == ace.sha(source)
    # No external result is ever represented as an actor-native receipt.
    receipts = list((tmp_path/'learner/contacts').glob('*.response.json'))
    assert all(json.loads(p.read_bytes())['response']['receipt_kind'] == 'external-chat-v1' for p in receipts)


def test_external_ace_does_not_weaken_actor_source_check():
    source, doc = fixture(groups=1)
    source['batch']['groups'][0]['rollouts'][0]['model_id'] = 'external-model'
    with pytest.raises(ValueError, match='provenance'):
        ace.build_input(source, doc['parent'], learner_model_id='external-model',
            learning_config_sha256='a'*64, **{k:v for k,v in doc['bounds'].items() if k != 'max_calls'})


def test_crash_after_api_response_reconciles_without_network(tmp_path, monkeypatch):
    c = teacher(tmp_path); req = request(c)
    calls = network(c, monkeypatch, [answer('observed-before-learner-crash')])
    c(req)
    # New adapter instance uses disk receipts, never its sender, for reconciliation.
    c2 = teacher(tmp_path)
    monkeypatch.setattr(c2.executor, 'execute', lambda *a: pytest.fail('reconcile dispatched'))
    envelope = c2.reconcile(req)
    assert envelope['response']['text'] == 'observed-before-learner-crash'
    assert len(calls) == 1


def test_stream_reconcile_has_no_write_or_fsync_side_effect(tmp_path, monkeypatch):
    c = teacher(tmp_path, stream=True); req = request(c)
    calls = network(c, monkeypatch, [answer('durable')])
    c(req)
    monkeypatch.setattr(c.executor, 'persist', lambda *a: pytest.fail('reconcile wrote files'))
    monkeypatch.setattr(c.executor.os, 'fsync', lambda *a: pytest.fail('reconcile fsync'))
    assert c.reconcile(req)['response']['text'] == 'durable'
    assert len(calls) == 1


def test_external_budget_mismatch_rejected_before_any_dispatch(tmp_path, monkeypatch):
    c = teacher(tmp_path)
    source, old = fixture(groups=1)
    doc = ace.build_input(source, old['parent'], learning_config_sha256=c.learning_config_sha256,
        learner_model_id=c.model_id, **{k:v for k,v in old['bounds'].items() if k != 'max_calls'})
    learner = ExperienceLearner(tmp_path/'learner', model_id=c.model_id, metered_call=c,
        preflight=c.preflight, reconcile=c.reconcile, learning_config_sha256=c.learning_config_sha256,
        max_calls=2)
    monkeypatch.setattr(c.executor, 'execute', lambda *a: pytest.fail('budget mismatch dispatched'))
    with pytest.raises(ValueError, match='completion cap must match'):
        ace.execute(learner, doc, count_tokens=lambda m: 20, verify_source=lambda: source)
    assert not list((tmp_path/'learner/contacts').glob('*.intent.json'))


@pytest.mark.parametrize('stream', [False, True])
def test_budget_policy_roundtrip_preserves_evidence(tmp_path, monkeypatch, stream):
    c = teacher(tmp_path, stream, analysis_budget=True); req = request(c)
    calls = network(c, monkeypatch, [answer('bounded report')])
    observed = c(req)
    assert c.reconcile(req)['response'] == observed
    assert len(calls) == 1
    assert calls[0]['messages'][1:] == req['messages']
    assert '49152' in calls[0]['messages'][0]['content']
    assert 'analysis_budget' not in calls[0]


def test_reference_tfgrpo_external_chain_preserves_actor_and_resume(tmp_path, monkeypatch):
    from test_tfgrpo_learning import group, MODEL, POLICY
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience
    c = teacher(tmp_path)
    parent = empty_experience(MODEL)
    operations = json.dumps([{'operation': 'ADD', 'id': None,
                             'content': 'Check the action cue before choosing an intent.'}])
    calls = network(c, monkeypatch, [answer('Lower trajectory summary', 's1'),
        answer('Higher trajectory summary', 's2'),
        answer('<Experiences>1. Use action evidence rather than a mentioned object.</Experiences>', 'a1'),
        answer(operations, 'g1'), answer(operations, 'b1')])
    def learner():
        return ExperienceLearner(tmp_path/'reference', model_id=c.model_id, metered_call=c,
            preflight=c.preflight, reconcile=c.reconcile,
            learning_config_sha256=c.learning_config_sha256, max_calls=5, concurrency=1)
    kwargs = dict(agent_objective='Intent and slots', learning_objective='Generalizable evidence use',
                  policy_sha256=POLICY, actor_model_id=MODEL)
    result = learner().learn([group(parent)], parent, **kwargs)
    assert result['model_id'] == result['actor_model_id'] == MODEL
    assert result['learner_model_id'] == c.model_id
    assert result['learning_usage'] == {'calls': 5, 'prompt_tokens': 100, 'completion_tokens': 50}
    assert not result['deployment_accepted']
    assert learner().learn([group(parent)], parent, **kwargs) == result
    assert len(calls) == 5
    binding = json.loads((tmp_path/'reference/batch.json').read_bytes())
    assert binding['actor_model_id'] == MODEL and binding['learner_model_id'] == c.model_id
    contacts = [json.loads(p.read_bytes()) for p in (tmp_path/'reference/contacts').glob('*.response.json')]
    assert all(r['response']['model_id'] == c.model_id
               and r['response']['receipt_kind'] == 'external-chat-v1' for r in contacts)


@pytest.mark.parametrize('fault', ['actor', 'rollout', 'teacher_response'])
def test_reference_tfgrpo_external_identity_mismatch_fails_closed(tmp_path, monkeypatch, fault):
    from test_tfgrpo_learning import group, MODEL, POLICY
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience
    c = teacher(tmp_path); parent = empty_experience(MODEL); groups = [group(parent)]
    if fault == 'rollout':
        groups[0]['rollouts'][0]['model_id'] = c.model_id
    calls = network(c, monkeypatch, [answer('Summary', model='foreign' if fault == 'teacher_response' else c.model_id)])
    learner = ExperienceLearner(tmp_path/'reference', model_id=c.model_id, metered_call=c,
        preflight=c.preflight, reconcile=c.reconcile,
        learning_config_sha256=c.learning_config_sha256, max_calls=5, concurrency=1)
    with pytest.raises(ValueError):
        learner.learn(groups, parent, agent_objective='Intent', learning_objective='Evidence',
                      policy_sha256=POLICY, actor_model_id='wrong' if fault == 'actor' else MODEL)
    assert len(calls) == (1 if fault == 'teacher_response' else 0)
