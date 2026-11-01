"""Final-only imports preserve native costs and reject incomplete stage custody."""
import copy
import json
from dataclasses import asdict
import pytest
from test_frozen_core_adapter import _plan
from test_tfgrpo_learning import group, MODEL
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (
    ExperienceLearner, LearningError, canonical, empty_experience, sha)
from speech_aware_evidence_acquisition.rails.tfgrpo_final_import import build_import, load_import


def fixture(root, *, summary_finish="stop", group_finish="stop", final_text=None):
    plan = _plan()
    base = root/'runs'/plan.run_id/plan.attempt_id/'tfgrpo'
    journal = ContactJournal(root, plan, 'prepared-context-text')
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(canonical(value))
    def caller(request):
        stage = request['stage']; key = sha({'stage': stage, 'identity': request['identity']})
        rid = 'native-'+key
        text = {'summary': 'Useful observed behavior', 'advantage': '<Experiences>1. Use evidence</Experiences>',
            'group_update': '[{"operation":"ADD","id":null,"content":"Use evidence"}]',
            'batch_update': final_text or '[{"operation":"ADD","id":null,"content":"Unfinished'}[stage]
        body = {'text': text, 'usage': {'prompt_tokens': 10, 'completion_tokens': 4},
                'finish_reason': {'batch_update':'length','summary':summary_finish,'group_update':group_finish}.get(stage,'stop')}
        payload = {'prepared_text': 'saea-prepared-text-v1', 'request_id': rid,
            'sample_id': 'q1', 'history': [{'role': 'agent_state', 'content': json.dumps(
                {'stage': stage, 'source_messages': request['messages']})},
                {'role': 'control', 'contact_id': key}]}
        journal.intent(payload); journal.response(rid, sha(payload), canonical(body), 1)
        receipt = {'text': text, 'model_id': MODEL, 'receipt_id': f'{plan.run_id}/{plan.attempt_id}/{rid}',
            'usage': {'calls': 1, **body['usage']}, 'request_sha256': sha(payload), 'response_sha256': sha(body)}
        write(base/'contacts'/(key+'.intent.json'), {'request': request, 'payload': payload})
        write(base/'contacts'/(key+'.response.json'), {'request_sha256': sha(request), 'response': receipt})
        return receipt
    learner = ExperienceLearner(base/'learner', model_id=MODEL, metered_call=caller,
        learning_config_sha256='a'*64, max_calls=5, concurrency=1)
    with pytest.raises(LearningError, match='must be JSON'):
        learner.learn([group()], empty_experience(MODEL), agent_objective='Intent',
            learning_objective='Evidence', policy_sha256=group()['rollouts'][0]['policy_sha256'])
    write(root/'attempts'/plan.run_id/(plan.attempt_id+'.json'), {
        'state': 'ABORTED', 'contact_kind': 'prepared-text', 'run_id': plan.run_id,
        'attempt_id': plan.attempt_id, 'carrier_lock_key': plan.carrier_lock_key,
        'reserved': {'calls': plan.planned_model_calls, 'audio_seconds': float(plan.planned_speech_audio_seconds)},
        'actual': {'calls': 5, 'audio_seconds': 0.0}})
    batch = json.loads((base/'learner/batch.json').read_bytes())
    args = dict(data_root=root, **{k: batch[k] for k in ('groups','common','model_id','parent','policy_sha256','rollout_kind')})
    pin, operations, custody = build_import(data_root=root, source_plan=plan,
        native_source_plans=[plan], output_relpath='imports/final.json')
    return pin, args, journal, base, operations, custody


def test_final_import_preserves_failed_final_cost_and_full_group_identity(tmp_path):
    pin, args, journal, base, operations, custody = fixture(tmp_path)
    assert operations == [{'operation':'ADD','id':None,'content':'Use evidence'}]
    assert custody['original_total_usage'] == {'calls':5,'prompt_tokens':50,'completion_tokens':20}
    assert custody['imported_stage_contacts'] == 4
    assert custody['learned_query_ids'] == ['q1']
    assert custody['excluded_final_response']['retained_in_historical_cost']
    assert load_import(pin, **args) == (operations, custody)


@pytest.mark.parametrize('fault', ['unknown','orphan','extra_native','changed_group','cost','unexplained_ancestry',
                                   'missing_stage','duplicate_stage','changed_messages','changed_parent'])
def test_final_import_rejects_untrusted_history(tmp_path, fault):
    pin, args, journal, base, operations, custody = fixture(tmp_path)
    path = tmp_path/pin['relpath']; doc = json.loads(path.read_bytes())
    def repin(pin, value):
        target = tmp_path/pin['relpath']; target.write_bytes(canonical(value)); pin['sha256'] = sha(value)
    if fault == 'unknown': next(journal.root.glob('*.response.json')).unlink()
    elif fault == 'orphan': next(journal.root.glob('*.intent.json')).unlink()
    elif fault == 'extra_native': (journal.root/'extra.intent.json').write_text('{}')
    elif fault == 'changed_group': args['groups'] = copy.deepcopy(args['groups']); args['groups'][0]['answer'] = 'changed'
    elif fault == 'cost':
        p = doc['native_sources'][0]['attempt']; value = json.loads((tmp_path/p['relpath']).read_bytes())
        value['actual']['calls'] = 4; repin(p, value)
    elif fault == 'unexplained_ancestry': doc['native_sources'].append(copy.deepcopy(doc['native_sources'][0]))
    elif fault == 'missing_stage': doc['records'].pop()
    elif fault == 'duplicate_stage': doc['records'].append(copy.deepcopy(doc['records'][0]))
    elif fault == 'changed_messages':
        p = doc['records'][0]['intent']; value = json.loads((tmp_path/p['relpath']).read_bytes())
        value['messages'][1]['content'] = 'changed'; repin(p, value)
    else: args['parent'] = empty_experience('different')
    repin(pin, doc)
    with pytest.raises((LearningError, ValueError, FileNotFoundError)): load_import(pin, **args)


def test_final_import_refuses_a_second_completion_for_a_parseable_final(tmp_path):
    pin, args, journal, base, operations, custody = fixture(tmp_path)
    # Repinning only a bridge envelope cannot change the immutable native result.
    manifest = json.loads((tmp_path/pin['relpath']).read_bytes())
    row = next(r for r in manifest['records'] if r['stage'] == 'batch_update')
    path = tmp_path/row['response']['relpath']
    value = json.loads(path.read_bytes()); value['response']['text'] = '[]'
    value['response_sha256'] = sha(value['response']); path.write_bytes(canonical(value))
    row['response']['sha256'] = sha(value)
    (tmp_path/pin['relpath']).write_bytes(canonical(manifest)); pin['sha256'] = sha(manifest)
    with pytest.raises(LearningError): load_import(pin, **args)


def test_group_length_is_not_silently_accepted(tmp_path):
    with pytest.raises(LearningError, match='historical stage response'):
        fixture(tmp_path, group_finish='length')


def test_historical_nonempty_summary_length_is_preserved_and_disclosed(tmp_path):
    _, _, _, _, operations, custody = fixture(tmp_path, summary_finish='length')
    assert len(custody['historical_truncated_summaries']) == 2
    assert {r['trajectory_id'] for r in custody['historical_truncated_summaries']} == {'q1-t0','q1-t1'}
    assert custody['original_total_usage']['calls'] == 5
    assert len(operations) == 1


def test_verified_import_drives_exactly_one_fresh_final_call(tmp_path):
    pin, args, _, _, operations, custody = fixture(tmp_path)
    requests = []
    def caller(request):
        requests.append(request)
        return {'model_id':MODEL,'receipt_id':'fresh-final','text':json.dumps(operations),
                'usage':{'calls':1,'prompt_tokens':11,'completion_tokens':9}}
    learner = ExperienceLearner(tmp_path/'fresh', model_id=MODEL, metered_call=caller,
        learning_config_sha256='b'*64, max_calls=1, concurrency=1)
    candidate = learner.learn(args['groups'], args['parent'], **args['common'],
        policy_sha256=args['policy_sha256'], rollout_kind=args['rollout_kind'],
        final_import=pin, imported_operations=operations, stage_custody=custody)
    assert [r['stage'] for r in requests] == ['batch_update']
    assert sha(requests[0]['messages']) == custody['final_messages_sha256']
    assert candidate['learned_query_ids'] == ['q1']
    assert candidate['learning_usage']['calls'] == 1
    assert candidate['historical_stage_custody']['original_total_usage']['calls'] == 5
