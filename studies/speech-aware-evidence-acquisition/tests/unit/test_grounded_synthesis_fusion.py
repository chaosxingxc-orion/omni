"""Fusion tests exercise native custody; only original ancestry and bundle boundary are doubles."""
import copy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
from speech_aware_evidence_acquisition.rails import grounded_synthesis_fusion as m
from speech_aware_evidence_acquisition.rails.grounded_synthesis_plan import plan_batches
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience
from test_grounded_synthesis_plan import source, count


def fixture(tmp_path, monkeypatch):
    batch, ops = source()
    batch['model_id'] = 'm'
    batch['parent'] = empty_experience('m')
    bb, ob = m.canonical(batch), m.canonical(ops)
    plan = plan_batches(bb, ob, source_batch_sha256=m.sha(batch),
        source_operations_sha256=m.sha(ops), count_tokens=count,
        tokenizer_identity={'fixture': 1}, expected_groups=4, expected_operations=8,
        resident_context=500, completion_cap=100, safety_margin=100)

    def write(path, doc):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(m.canonical(doc))
        return {'relpath': str(path.relative_to(tmp_path)), 'sha256': m.sha(doc)}

    source_spec = {'source_batch': write(tmp_path/'original-batch.json', batch),
                   'operations': write(tmp_path/'original-ops.json', ops)}
    historical = {'calls': 309, 'prompt_tokens': 1010, 'completion_tokens': 505}
    custody = {'source': source_spec, 'model_id': 'm', 'source_native_custody_verified': True,
        'unknown_contacts': 0, 'source_batch_sha256': m.sha(batch),
        'source_operations_sha256': m.sha(ops), 'original_total_usage': historical,
        'native_source_usage': [{'run_id': 'prior', 'attempt_id': 'a', 'usage': historical}]}
    monkeypatch.setattr(m, 'load_source', lambda *a, **k: copy.deepcopy(custody))
    registration = m.registration_binding(plan, model_id='m', learning_config_sha256='a'*64,
        custody_sha256=m.sha(custody), max_operations=128, max_calls=2)
    binding = {'schema': m.SHARD_SCHEMA, 'plan': plan,
               'registration': registration, 'source_custody': custody}
    execution = ExecutionPlan(run_id='FUSION-SOURCE', attempt_id='learning-v1',
        execution_profile='bounded-discovery-probe', carrier_lock_key='slurp',
        split_role='discovery', split_identity_hash='b'*64, planned_model_calls=2,
        planned_gpu_hours=1, planned_speech_audio_seconds=1, protocol_hash='c'*64)
    base = tmp_path/'runs'/execution.run_id/execution.attempt_id
    journal = ContactJournal(tmp_path, execution, 'prepared-context-text')
    results = []; sealed_records = {}
    for shard in plan['shards']:
        index = shard['shard_index']; identity = m.contact_identity(plan, shard)
        key = m.sha({'stage': m.STAGE, 'identity': identity})
        request = {'schema': 'saea-tfgrpo-learning-request-v1',
            'batch_sha256': m.sha(binding), 'model_id': 'm',
            'learning_config_sha256': 'a'*64, 'stage': m.STAGE,
            'identity': identity, 'messages': shard['messages']}
        payload = {'request_id': f'flight-{index}', 'sample_id': f'slurp:train:{index}',
            'history': [{'role': 'user', 'content': json.dumps({
                'source_messages': shard['messages'], 'stage': m.STAGE})},
                {'role': 'control', 'contact_id': 'tfgrpo-'+key}]}
        text = json.dumps([{'operation': 'ADD', 'id': None, 'content': 'conditional rule'}])
        body = {'text': text, 'finish_reason': 'stop',
                'usage': {'prompt_tokens': 30, 'completion_tokens': 20}}
        bridge_payload = copy.deepcopy(payload)
        bridge_payload['decoding_params'] = {'id_slot': 0}
        payload['decoding_params'] = {'id_slot': index+1}
        journal.intent(payload)
        journal.response(payload['request_id'], m.sha(payload), m.canonical(body), 1)
        native = journal.inspect(payload['request_id'])
        sealed_records[payload['request_id']+'/response'] = {
            'response_sha256': native['response']['response_sha256'], 'text': text, 'usage': body['usage']}
        sealed_records[payload['request_id']+'/cost'] = {'calls': 1, **body['usage']}
        fitdoc = {'inputs': {'payload_sha256': m.sha(bridge_payload)}, 'output': {'every_request_fits': True}}
        fitpin = write(tmp_path/'prepared'/f'{key}.json', fitdoc)
        fit = {'payload_sha256': m.sha(bridge_payload), 'artifact_relpath': fitpin['relpath'],
               'artifact_sha256': fitpin['sha256'], 'every_request_fits': True}
        receipt = {'receipt_id': f'{execution.run_id}/{execution.attempt_id}/{payload["request_id"]}',
            'model_id': 'm', 'text': text, 'usage': {'calls': 1, **body['usage']},
            'request_sha256': m.sha(payload), 'response_sha256': native['response']['response_sha256'],
            'fit_receipt_sha256': m.sha(fit)}
        for directory, intent, response in (
            ('learner/contacts', request, {'request_sha256': m.sha(request),
             'response': receipt, 'response_sha256': m.sha(receipt)}),
            ('contacts', {'request': request, 'payload': bridge_payload},
             {'request_sha256': m.sha(request), 'response': receipt})):
            write(base/'tfgrpo'/directory/(key+'.intent.json'), intent)
            write(base/'tfgrpo'/directory/(key+'.response.json'), response)
        write(base/'tfgrpo/contacts'/(key+'.fit.json'), fit)
        result = {'shard_index': index, 'messages_sha256': shard['messages_sha256'],
            'source_group_indices': shard['group_indices'],
            'source_operation_indices': shard['operation_indices'],
            'output_operations': json.loads(text), 'response_text_sha256': m.sha(text),
            'candidate_publishable': False}
        write(base/'tfgrpo/learner/shards'/f'{index:04d}.json', result); results.append(result)
    current = {'calls': 2, 'prompt_tokens': 60, 'completion_tokens': 40}
    result = {'schema': m.SHARD_SCHEMA, 'status': 'SHARDS_COMPLETE_NOT_CONSOLIDATED',
        'batch_sha256': m.sha(binding), 'completed_shards': [0, 1], 'total_shards': 2,
        'candidate_publishable': False, 'final_consolidation': 'NOT_IMPLEMENTED', 'error_type': None,
        'shard_results': results, 'current_accounting': {'attempted_contacts': 2,
        'observed_contacts': 2, 'unknown_contacts': 0, 'known_usage': current},
        'historical_usage': historical,
        'cumulative_known_usage': {k: current[k]+historical[k] for k in m.USAGE}}
    manifest = {'plan': asdict(execution), 'result_kind': 'artifact',
                'cost': {'calls_used': 2, 'prompt_tokens_total': 60, 'completion_tokens_total': 40}}
    spec = {'schema': m.SOURCE_SCHEMA,
        'source_manifest': write(base/(execution.run_id+'.run-manifest.json'), manifest),
        'shards': write(base/'tfgrpo/learner/shards-complete.json', result),
        'execution_batch': write(base/'tfgrpo/learner/batch.json', binding)}
    write(tmp_path/'attempts'/execution.run_id/(execution.attempt_id+'.json'),
          {'state': 'FINALIZED', 'contact_kind': 'prepared-text',
           'actual': {'calls': 2, 'audio_seconds': 0.0}})
    seal = {'schema': 'saea-grounded-shards-flight-artifact-v1',
        'artifact': {'root': 'data', **spec['shards'], 'candidate_publishable': False},
        'batch_sha256': m.sha(binding), 'usage': current, 'historical_usage': historical,
        'candidate_publishable': False}
    monkeypatch.setattr(m, 'verify_bundle', lambda p: SimpleNamespace(
        attempt_state='FINALIZED', manifest=manifest,
        trace_records={**sealed_records, 'x': {'org_grounded_shards': seal}}))
    # FIT envelope validation has separate integration coverage; byte pin remains real.
    monkeypatch.setattr(m, 'load_artifact', lambda *a, **k: None)
    return spec, seal, journal, result, binding


def load(tmp, spec):
    return m.load_finalized_shards(spec, data_root=tmp, count_tokens=count,
                                   tokenizer_identity={'fixture': 1})


def test_finalized_native_source_preserves_all_spend(tmp_path, monkeypatch):
    spec, *_ = fixture(tmp_path, monkeypatch)
    value = load(tmp_path, spec)
    assert value['original_total_usage'] == {'calls': 311, 'prompt_tokens': 1070, 'completion_tokens': 545}
    assert [x['usage']['calls'] for x in value['native_source_usage']] == [309, 2]
    assert value['candidate_publishable'] is False


@pytest.mark.parametrize('fault', ['seal', 'artifact', 'attempt', 'cost', 'native', 'extra_native', 'fit'])
def test_refuses_invalid_source_without_contacts(tmp_path, monkeypatch, fault):
    spec, seal, journal, result, binding = fixture(tmp_path, monkeypatch)
    if fault == 'seal':
        seal['historical_usage'] = {'calls': 0}
    elif fault == 'artifact':
        (tmp_path/spec['shards']['relpath']).write_text('{}')
    elif fault == 'attempt':
        p = tmp_path/'attempts/FUSION-SOURCE/learning-v1.json'
        d = json.loads(p.read_bytes()); d['state'] = 'ABORTED'; p.write_bytes(m.canonical(d))
    elif fault == 'cost':
        result['cumulative_known_usage']['calls'] -= 1
        p = tmp_path/spec['shards']['relpath']; p.write_bytes(m.canonical(result))
        spec['shards']['sha256'] = m.sha(result)
        seal['artifact']['sha256'] = m.sha(result)
    elif fault == 'native':
        journal._path('flight-0', 'response').unlink()
    elif fault == 'extra_native':
        journal.intent({'request_id': 'extra', 'sample_id': 'slurp:train:0',
                        'history': [{'role': 'control', 'contact_id': 'extra'}]})
    else:
        next((tmp_path/'prepared').glob('*.json')).write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        load(tmp_path, spec)


def document(tmp_path, monkeypatch):
    spec, *_ = fixture(tmp_path, monkeypatch)
    return m.build_fusion_input(load(tmp_path, spec))


def output(doc):
    refs = [p['source_ref'] for p in doc['proposals']]
    return {'schema': m.OUTPUT_SCHEMA, 'rules': [{
        'rule_id': 'R1', 'operation': {'operation': 'ADD', 'id': None, 'content': 'conditional rule'},
        'source_refs': refs, 'support_group_refs': ['G0', 'G2'],
        'scope': 'specific observed intent', 'trigger': 'explicit evidence', 'exceptions': [],
        'rationale': 'compare observed prediction and reference'}],
        'dispositions': [{'source_ref': ref, 'decision': 'merge', 'rule_ids': ['R1'],
                          'reason': 'same supported conditional rule'} for ref in refs]}


def test_duplicates_recorded_without_inventing_support(tmp_path, monkeypatch):
    doc = document(tmp_path, monkeypatch)
    assert len(doc['duplicate_sets']) == 1
    assert len(doc['shard_sources']) == 2
    assert sum(len(s['available_proposals']) for s in doc['shard_sources']) == 8
    assert all('available_proposals' not in p for p in doc['proposals'])
    assert all(not p['rule_support_provided_by_shard'] for p in doc['proposals'])
    checked = m.validate_fusion_output(json.dumps(output(doc)), doc)
    assert len(checked['items']) == 1
    assert checked['candidate_publishable'] is checked['semantic_support_verified'] is False


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'foreign_group', 'foreign_source',
                                  'outside_shard', 'unlinked', 'duplicate_add'])
def test_output_requires_complete_consistent_evidence_links(tmp_path, monkeypatch, fault):
    doc = document(tmp_path, monkeypatch); result = output(doc)
    if fault == 'missing': result['dispositions'].pop()
    elif fault == 'duplicate': result['dispositions'].append(copy.deepcopy(result['dispositions'][0]))
    elif fault == 'foreign_group': result['rules'][0]['support_group_refs'] = ['G999']
    elif fault == 'foreign_source': result['rules'][0]['source_refs'] = ['S99O0']
    elif fault == 'outside_shard':
        result['rules'][0]['source_refs'] = [doc['proposals'][0]['source_ref']]
        result['rules'][0]['support_group_refs'] = ['G2']
    elif fault == 'unlinked':
        result['dispositions'][0].update(decision='reject', rule_ids=[])
    else:
        other = copy.deepcopy(result['rules'][0]);other['rule_id'] = 'R2';result['rules'].append(other)
    with pytest.raises(ValueError):
        m.validate_fusion_output(json.dumps(result), doc)


def test_all_rejected_valid_and_no_fake_candidate(tmp_path, monkeypatch):
    doc = document(tmp_path, monkeypatch)
    result = {'schema': m.OUTPUT_SCHEMA, 'rules': [], 'dispositions': [
        {'source_ref': p['source_ref'], 'decision': 'reject', 'rule_ids': [],
         'reason': 'unsupported by available evidence'} for p in doc['proposals']]}
    checked = m.validate_fusion_output(json.dumps(result), doc)
    assert checked['items'] == {} and not checked['candidate_publishable']


def test_real_counter_bounds_whole_input_without_truncating(tmp_path, monkeypatch):
    doc = document(tmp_path, monkeypatch); saved = copy.deepcopy(doc)
    measured = m.fusion_messages(doc, count_tokens=lambda messages: 100,
                                resident_context=250, completion_cap=100, reserve_tokens=50)
    assert measured['input_tokens'] == 100 and not measured['native_fit_completed']
    with pytest.raises(ValueError, match='exceeds context'):
        m.fusion_messages(doc, count_tokens=lambda _: 101, resident_context=250,
                          completion_cap=100, reserve_tokens=50)
    assert doc == saved


def test_tampered_projection_rejected(tmp_path, monkeypatch):
    spec, *_ = fixture(tmp_path, monkeypatch); value = load(tmp_path, spec)
    doc = m.build_fusion_input(value)
    evidence = [{k: g[k] for k in ('group_index', 'group_sha256', 'content')} for g in doc['groups']]
    evidence[0]['content']['reference'] = 'invented evidence'
    with pytest.raises(ValueError, match='projection differs'):
        m.build_fusion_input(value, evidence=evidence)


def test_conflicting_parent_updates_cannot_be_applied_sequentially(tmp_path, monkeypatch):
    doc = document(tmp_path, monkeypatch)
    doc['parent']['items'] = {'E1': 'old rule'}
    doc['parent']['sha256'] = m.sha({k:v for k,v in doc['parent'].items() if k!='sha256'})
    doc['input_sha256'] = m.sha({k:v for k,v in doc.items() if k!='input_sha256'})
    result = output(doc);result['rules'][0]['operation'] = {'operation':'UPDATE','id':'E1','content':'new'}
    other = copy.deepcopy(result['rules'][0]);other['rule_id']='R2'
    other['operation']={'operation':'DELETE','id':'E1','content':''};result['rules'].append(other)
    with pytest.raises(ValueError, match='multiply modified'):
        m.validate_fusion_output(json.dumps(result),doc)


def test_compact_view_retains_predictions_and_explicit_omissions(tmp_path, monkeypatch):
    doc = document(tmp_path, monkeypatch)
    evidence = doc['groups'][0]['content']
    evidence['rollouts'] = [
        {'scored_prediction': {'scenario':'x','action':'y','entities':[]}, 'reward':0.5,
         'feedback': {'strict_task_success':0}, 'prediction_observed':True,
         'raw_prediction':'PRIVATE RAW THOUGHT'} for _ in range(2)]
    doc['input_sha256'] = m.sha({k:v for k,v in doc.items() if k != 'input_sha256'})
    view = m.compact_model_view(doc)
    assert len(view['groups'][0]['prediction_variants']) == 1
    assert view['groups'][0]['prediction_variants'][0]['rollout_indices'] == [0, 1]
    assert 'PRIVATE RAW THOUGHT' not in json.dumps(view)
    assert view['groups'][0]['prediction_variants'][0]['prediction']['scenario'] == 'x'
    assert 'raw_prediction_and_chain_of_thought' in view['omitted_from_model_view']
    measured = m.fusion_messages(doc, count_tokens=lambda _: 1)
    assert 'operation.content MUST itself include' in measured['messages'][0]['content']
