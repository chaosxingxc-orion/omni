"""Corrective source uses real pinned journals/FIT, source ancestry is a boundary double."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
from speech_aware_evidence_acquisition.rails import grounded_fusion_runtime as m
from speech_aware_evidence_acquisition.rails import grounded_corrective_fusion as correction
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha, ExperienceLearner
from test_grounded_corrective_fusion import fixture as packet_fixture, output


def fixture(tmp_path, monkeypatch):
    initial, source = packet_fixture()
    source = {**source, 'model_id': 'model', 'batch': {
        'parent': initial['document']['parent'],
        'rollout_kind': 'shared-perception-adaptation',
        'common': {'reward_mode': 'intent-exact-slot-f1-v1'},
        'groups': [{'query_id': g['query_id']} for g in initial['document']['groups']]}}
    document = copy.deepcopy(initial['document'])
    document['source_sha256'] = sha(source)
    document['input_sha256'] = sha({k: v for k, v in document.items() if k != 'input_sha256'})
    binding = copy.deepcopy(initial['failure']['binding'])
    binding.update(source_custody_sha256=sha(source), input_sha256=document['input_sha256'],
                   identity=document['input_sha256'])
    from speech_aware_evidence_acquisition.rails.grounded_synthesis_fusion import compact_model_view
    binding['messages'][1]['content'] = canonical(compact_model_view(document)).decode()
    plan = ExecutionPlan(run_id='FAILED-FUSION', attempt_id='learning-v1',
        execution_profile='bounded-discovery-probe', carrier_lock_key='slurp', split_role='discovery',
        split_identity_hash='b'*64, planned_model_calls=1, planned_gpu_hours=1,
        planned_speech_audio_seconds=1, protocol_hash='c'*64)
    base = tmp_path/'runs'/plan.run_id/plan.attempt_id
    reg = {'schema': m.FUSION_RUNTIME_SCHEMA,
        'plan': {k: v for k, v in asdict(plan).items() if k != 'protocol_hash'},
        'model_id': 'model', 'decoding_params': {'max_tokens': 16384},
        'max_context_bytes': 2000000, 'runtime_identity_sha256': 'a'*64,
        'construction_roster_sha256': 'd'*64, 'completion_cap': 16384}
    reg['learning_config_sha256'] = sha(m.learning_configuration(reg))
    binding['learning_config_sha256'] = reg['learning_config_sha256']
    reg.update(batch_sha256=sha(binding), execution_sha256=sha(binding), contacts=[{
        'stage': m.STAGE, 'identity': binding['identity'],
        'sample_id': binding['learned_query_ids'][0], 'messages_sha256': sha(binding['messages'])}])
    request = {**initial['failure']['request'], 'batch_sha256': sha(binding),
        'learning_config_sha256': binding['learning_config_sha256'],
        'identity': binding['identity'], 'messages': binding['messages']}
    key = sha({'stage': m.STAGE, 'identity': binding['identity']})
    payload = {'request_id': 'flight-1', 'sample_id': binding['learned_query_ids'][0],
        'prepared_text': 'saea-prepared-text-v1',
        'history': [{'role': 'user', 'content': json.dumps({'source_messages': binding['messages'],
                                                        'stage': m.STAGE})},
                    {'role': 'control', 'contact_id': 'tfgrpo-'+key}]}
    native = ContactJournal(tmp_path, plan, 'prepared-context-text')
    native.intent(payload)
    body = {'text': '{invalid failed output', 'finish_reason': 'stop',
            'usage': {'prompt_tokens': 20, 'completion_tokens': 10}}
    native.response('flight-1', sha(payload), canonical(body), 1)
    observed = native.inspect('flight-1')
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(canonical(value))
        return {'relpath': str(path.relative_to(tmp_path)), 'sha256': sha(value)}
    def pin(path):
        return {'relpath': str(path.relative_to(tmp_path)),
                'sha256': __import__('hashlib').sha256(path.read_bytes()).hexdigest()}
    fitdoc = {'schema': 'saea-prepared-artifact-v1', 'kind': 'request-fit',
        'inputs': {'payload_sha256': sha(payload)}, 'contact_class': 'tokenizer',
        'output': {'every_request_fits': True}}
    fitdoc['output_sha256'] = sha(fitdoc['output'])
    fitpin = write(tmp_path/'prepared/failed-fit.json', fitdoc)
    fit = {'payload_sha256': sha(payload), 'artifact_relpath': fitpin['relpath'],
           'artifact_sha256': fitpin['sha256'], 'every_request_fits': True}
    receipt = {'receipt_id': f'{plan.run_id}/{plan.attempt_id}/flight-1', 'model_id': 'model',
        'text': body['text'], 'usage': {'calls': 1, **body['usage']},
        'request_sha256': sha(payload), 'response_sha256': observed['response']['response_sha256'],
        'fit_receipt_sha256': sha(fit)}
    local, bridge = base/'tfgrpo/learner/contacts', base/'tfgrpo/contacts'
    spec = {'schema': m.FAILED_SOURCE_SCHEMA, 'source_plan': asdict(plan),
        'attempt': write(tmp_path/'attempts'/plan.run_id/(plan.attempt_id+'.json'), {
            'run_id': plan.run_id, 'attempt_id': plan.attempt_id, 'carrier_lock_key': 'slurp',
            'state': 'ABORTED', 'contact_kind': 'prepared-text',
            'actual': {'calls': 1, 'audio_seconds': 0.0},
            'reserved': {'calls': 1, 'audio_seconds': 1.0}}),
        'execution_batch': write(base/'tfgrpo/learner/batch.json', binding),
        'registration': write(bridge/'registration.json', reg),
        'learner_files': [write(local/(key+'.intent.json'), request),
            write(local/(key+'.response.json'), {'request_sha256': sha(request),
                'response': receipt, 'response_sha256': sha(receipt)})],
        'bridge_files': [write(bridge/(key+'.intent.json'), {'request': request, 'payload': payload}),
            write(bridge/(key+'.response.json'), {'request_sha256': sha(request), 'response': receipt}),
            write(bridge/(key+'.fit.json'), fit)],
        'journal_files': [pin(native._path('flight-1', kind)) for kind in ('intent', 'response')],
        'fit_artifact': fitpin, 'failure_reason': 'invalid original fusion'}
    monkeypatch.setattr(m, 'load_finalized_shards', lambda *a, **k: copy.deepcopy(source))
    monkeypatch.setattr(m, 'build_fusion_input', lambda given: copy.deepcopy(document))
    monkeypatch.setattr(correction, 'build_fusion_input', lambda given: copy.deepcopy(document))
    monkeypatch.setattr(m, 'load_token_counter', lambda path: (lambda messages: 100, {'fixture': True}))
    return source, document, spec, native, plan, write


def test_native_failed_custody_preserves_exact_historical_spend(tmp_path, monkeypatch):
    source, document, spec, *_ = fixture(tmp_path, monkeypatch)
    failed, terminal = m.load_failed_fusion(spec, data_root=tmp_path, source=source, document=document)
    assert terminal['full_inventory_verified'] and terminal['native_fit_verified']
    assert failed['cumulative_known_usage'] == {'calls': 313, 'prompt_tokens': 1020, 'completion_tokens': 210}


@pytest.mark.parametrize('fault', ['extra_native', 'unknown', 'orphan', 'changed_native', 'attempt',
                                  'fit', 'bridge', 'missing_pin', 'candidate', 'recovery'])
def test_incomplete_failed_source_is_rejected(tmp_path, monkeypatch, fault):
    source, document, spec, journal, plan, write = fixture(tmp_path, monkeypatch)
    if fault == 'extra_native': journal.intent({'request_id': 'extra', 'sample_id': 'slurp:train:0',
        'history': [{'role': 'control', 'contact_id': 'extra'}]})
    elif fault == 'unknown': journal._path('flight-1', 'response').unlink()
    elif fault == 'orphan': journal._path('flight-1', 'intent').unlink()
    elif fault == 'changed_native': journal._path('flight-1', 'response').write_text('{}')
    elif fault == 'attempt':
        path = tmp_path/spec['attempt']['relpath']; doc = json.loads(path.read_bytes()); doc['state'] = 'OPENED'
        spec['attempt'] = write(path, doc)
    elif fault == 'fit':
        path = tmp_path/spec['fit_artifact']['relpath']; doc = json.loads(path.read_bytes())
        doc['output']['every_request_fits'] = False; path.write_bytes(canonical(doc))
    elif fault == 'bridge': (tmp_path/spec['bridge_files'][1]['relpath']).write_text('{}')
    elif fault == 'missing_pin': spec['journal_files'].pop()
    else:
        base = tmp_path/'runs'/plan.run_id/plan.attempt_id
        write(base/('tfgrpo/learner/candidate.json' if fault == 'candidate' else 'resume-declaration.json'), {})
    with pytest.raises((ValueError, FileNotFoundError)): m.load_failed_fusion(spec,
        data_root=tmp_path, source=source, document=document)


def prepared(tmp_path, monkeypatch):
    source, document, failed_spec, native, old_plan, write = fixture(tmp_path, monkeypatch)
    failed, terminal = m.load_failed_fusion(failed_spec, data_root=tmp_path, source=source, document=document)
    corrective = correction.build_corrective_input(document, failed)
    measured = correction.corrective_messages(corrective, count_tokens=lambda messages: 100)
    plan = ExecutionPlan(**{**asdict(old_plan), 'run_id': 'CORRECTIVE-FUSION'})
    reg, binding, custody = m.build_corrective_registration(plan, source=source, corrective=corrective,
        terminal=terminal, measured=measured, model_id='model', decoding_params={'max_tokens': 16384},
        max_context_bytes=2000000, runtime_identity_sha256='a'*64, roster_sha256='d'*64)
    source_doc = {'schema': 'saea-grounded-fusion-runtime-source-v1', 'strategy': m.CORRECTIVE_STRATEGY,
        'source': {}, 'failed_fusion': failed_spec, 'measurement': measured,
        'input_sha256': document['input_sha256'], 'corrective_input_sha256': corrective['corrective_input_sha256'],
        'tokenizer_relpath': 'tokenizer'}
    source_pin = write(tmp_path/'corrective-source.json', source_doc)
    values = {'tfgrpo_learning_registration': reg, 'prepared_training_feedback': {
        'schema': m.SOURCE_SCHEMA, 'source': {'root': 'data', **source_pin}}}
    return plan, values, source_doc, binding, custody, corrective, write


def test_same_source_router_reloads_native_custody_and_registration(tmp_path, monkeypatch):
    plan, values, doc, binding, custody, _, _ = prepared(tmp_path, monkeypatch)
    loaded, receipt = m.load_runtime_source(values, plan=plan, repo_root=tmp_path, data_root=tmp_path,
        sample_ids=('slurp:train:0',), roster_sha256='d'*64)
    assert loaded['binding'] == binding and loaded['custody'] == custody
    assert receipt['contacts'] == 1 and loaded['custody']['original_total_usage']['calls'] == 313
    assert values['tfgrpo_learning_registration']['schema'] == m.FUSION_RUNTIME_SCHEMA
    assert values['tfgrpo_learning_registration']['contacts'][0]['stage'] == m.STAGE


def test_registration_matches_actual_shared_execution_request(tmp_path, monkeypatch):
    plan, values, _, binding, _, corrective, _ = prepared(tmp_path, monkeypatch)
    loaded, _ = m.load_runtime_source(values, plan=plan, repo_root=tmp_path, data_root=tmp_path,
        sample_ids=('slurp:train:0',), roster_sha256='d'*64)
    calls = []
    def call(request):
        assert request['batch_sha256'] == sha(binding)
        assert request['identity'] == binding['identity']
        calls.append(request)
        return {'model_id': 'model', 'receipt_id': 'corrective/one',
            'text': json.dumps(output(corrective)), 'usage': {'calls': 1, 'prompt_tokens': 2, 'completion_tokens': 3}}
    learner = ExperienceLearner(tmp_path/'new-learner', model_id='model', metered_call=call,
        learning_config_sha256=binding['learning_config_sha256'], preflight=lambda r: None, max_calls=1)
    candidate = correction.execute_corrective_fusion(learner, corrective, loaded['measurement'],
        count_tokens=loaded['count_tokens'], verify_source=lambda: loaded['source'],
        verify_failed_attempt=lambda: loaded['failed_attempt'])
    assert len(calls) == 1 and candidate['historical_fusion_custody']['original_total_usage']['calls'] == 313
    assert candidate['learning_usage']['calls'] == 1 and not candidate['deployment_accepted']


@pytest.mark.parametrize('fault', ['measurement', 'tokenizer', 'registration', 'failure', 'roster'])
def test_runtime_rejects_drift_before_contact(tmp_path, monkeypatch, fault):
    plan, values, doc, _, _, _, write = prepared(tmp_path, monkeypatch)
    if fault == 'measurement': doc['measurement']['input_tokens'] += 1
    elif fault == 'tokenizer': doc['tokenizer_relpath'] = '../outside'
    elif fault == 'registration': values['tfgrpo_learning_registration']['batch_sha256'] = 'f'*64
    elif fault == 'failure': doc['failed_fusion']['attempt']['sha256'] = 'f'*64
    else: values['tfgrpo_learning_registration']['contacts'][0]['sample_id'] = 'slurp:test:0'
    pin = write(tmp_path/'corrective-source.json', doc)
    values['prepared_training_feedback']['source'] = {'root': 'data', **pin}
    with pytest.raises(ValueError): m.load_runtime_source(values, plan=plan, repo_root=tmp_path,
        data_root=tmp_path, sample_ids=('slurp:train:0',), roster_sha256='d'*64)


def test_same_runtime_dispatch_uses_corrective_strategy_and_reuses_response(tmp_path, monkeypatch):
    plan, values, _, binding, _, corrective, _ = prepared(tmp_path, monkeypatch)
    calls, fits = [], []
    class Caller:
        def __init__(self, root):
            self.root = root; self.model_id = 'model'; self.plan = plan
            self.registration = values['tfgrpo_learning_registration']
            self.client = SimpleNamespace(_closed=False)
        def preflight(self, request): fits.append(request)
        def reconcile(self, request): return None
        def __call__(self, request):
            assert request['batch_sha256'] == self.registration['batch_sha256']
            assert request['identity'] == self.registration['contacts'][0]['identity']
            calls.append(request)
            return {'model_id': 'model', 'receipt_id': 'corrective/one',
                'text': json.dumps(output(corrective)),
                'usage': {'calls': 1, 'prompt_tokens': 2, 'completion_tokens': 3}}
    monkeypatch.setattr(m, 'bind_runtime', lambda runtime, receipt_root: Caller(receipt_root))
    runtime = SimpleNamespace(_composed=SimpleNamespace(values=values), plan=plan,
        _repo_root=tmp_path, _data_root=tmp_path,
        _samples=[SimpleNamespace(sample_id='slurp:train:0')],
        _run_dir=lambda: tmp_path/'new-run')
    first, second = m.run_registered_fusion(runtime), m.run_registered_fusion(runtime)
    assert first == second and len(calls) == len(fits) == 1
    assert first['historical_fusion_custody']['original_total_usage']['calls'] == 313
    assert first['learning_usage']['calls'] == 1 and not first['deployment_accepted']
