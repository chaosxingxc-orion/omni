import json
import re
from itertools import permutations
import hashlib
from dataclasses import replace

import pytest

from speech_aware_evidence_acquisition.analysis.perception_node_score import score
from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import canonical
from speech_aware_evidence_acquisition.rails.ranked_candidate_indices import parse
from test_intent_slot_runner import recovery_harness
from test_llama_server_transport import _spawn_live_matching_process  # noqa: F401
from test_native_episode_runner import _wire_response
from test_rails_runner import _knowledge_snapshot


@pytest.mark.parametrize('count,output_k', [(n, k) for n in range(1, 6) for k in (1, 3)])
def test_dynamic_native_window(tmp_path, count, output_k):
    h, policy = recovery_harness(tmp_path)
    snapshot = _knowledge_snapshot()
    names = ('play_music', 'play_radio', 'play_video', 'play_game', 'play_movie')[:count]
    values = tuple(replace(snapshot.values[0], value_id='intent-boundaries::intent-boundary:'+name,
                           content='Description for '+name, task_target=name) for name in names)
    snapshot = replace(snapshot, values=values).validate()
    h.bootstrap_value_ids = tuple(v.value_id for v in values)
    template = tmp_path/'configs/probes/rank.txt'
    template.parent.mkdir(parents=True); template.write_text('Rank the candidates.')
    h.repo_root = tmp_path
    policy.update(max_candidates=output_k, ranking_window={'mode':'ranked-candidate-indices',
        'input_k':5, 'output_k':output_k, 'dynamic':True, 'max_tokens':128, 'transport':'finite-choice-v1',
        'template':{'relpath':'configs/probes/rank.txt','sha256':hashlib.sha256(template.read_bytes()).hexdigest()}})
    policy['coverage_recovery']['max_extra_contacts'] = output_k
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy), perception_only=True)
    sent = []
    answer = json.dumps(list(range(1, min(count, output_k)+1)), separators=(',', ':'))
    h.post_override = lambda _url, body: (sent.append(json.loads(body)) or _wire_response(answer))
    h.request_gate = lambda request: None
    try:
        node = h.run(snapshot=snapshot).results[0].node_outcome
    finally:
        h.close()
    assert node['ranking_result']['candidate_ids'] == list(names[:output_k])
    assert len(sent) == 1 and re.fullmatch(sent[0]['structured_outputs']['regex'], answer)
    result = score([{'sample_id':'s1','gold_intent':names[0]}], [{**node, 'text':node['response']}],
                   maximum=output_k, legal_ids=list(names))
    assert result['gold_recall_count'] == 1


@pytest.mark.parametrize('transport', ['json-schema', 'finite-choice-v1'])
@pytest.mark.parametrize('input_k,output_k,answer', [(3, 1, '[2]'), (3, 2, '[2,1]'), (5, 3, '[2,1,3]'), (3, 1, '[4]'), (3, 1, '[1]'), (3, 2, '[1,1]')])
def test_native_ranking_before_after_and_node_score(tmp_path, input_k, output_k, answer, transport):
    h, policy = recovery_harness(tmp_path)
    snapshot = _knowledge_snapshot()
    facet = snapshot.facets[0]
    values = tuple(replace(snapshot.values[0], value_id='intent-boundaries::intent-boundary:'+name,
                           content='Description for '+name, task_target=name) for name in ('play_music','play_radio','play_video','play_game','play_movie')[:input_k])
    snapshot = replace(snapshot, values=values).validate()
    h.bootstrap_value_ids = tuple(v.value_id for v in values)
    template = tmp_path/'configs/probes/rank.txt'
    template.parent.mkdir(parents=True)
    template.write_text('Prefer action and object.' if answer == '[1]' else 'Order the candidates by relevance to the audio.')
    h.repo_root = tmp_path
    policy.update(max_candidates=output_k, ranking_window={'mode':'ranked-candidate-indices',
        'input_k':input_k,'output_k':output_k,'max_tokens':128,'transport':transport,
        'template':{'relpath':'configs/probes/rank.txt','sha256':hashlib.sha256(template.read_bytes()).hexdigest()}}, coverage_recovery={
        'strategy':'complete-original-candidate-set-v1','max_extra_contacts':1})
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy), perception_only=True)
    sent = []
    def post(_url, body):
        sent.append(json.loads(body))
        return _wire_response(answer)
    h.post_override = post
    h.request_gate = lambda request: None
    try:
        summary = h.run(snapshot=snapshot)
    finally:
        h.close()
    node = summary.results[0].node_outcome
    assert node['response'] == answer
    invalid = answer in ('[4]', '[1,1]')
    assert node['ranking_result']['candidate_ids'] == ([] if invalid else ['play_music'] if answer == '[1]' else ['play_radio','play_music','play_video'][:output_k])
    result = score([{'sample_id':'s1','gold_intent':'play_radio'}], [{**node, 'text':node['response']}], maximum=output_k,
                   legal_ids=['play_music','play_radio','play_video','play_game','play_movie'])
    assert result['gold_recall_count'] == (0 if invalid or answer == '[1]' else 1), result
    assert result['ranking_input_gold_count'] == 1
    assert result['ranking_input_retained_count'] == (0 if invalid or answer == '[1]' else 1)
    assert len(sent) == 1
    assert ('Prefer action and object.' if answer == '[1]' else 'Order the candidates by relevance') in json.dumps(sent[0])
    assert sent[0]['max_tokens'] == 128
    if transport == 'finite-choice-v1':
        regex = sent[0]['structured_outputs']['regex']
        choices = [json.dumps(list(p), separators=(',', ':')) for p in permutations(range(1, input_k+1), output_k) if re.fullmatch(regex, json.dumps(list(p), separators=(',', ':')))]
        assert not re.fullmatch(regex, '[1,1]')
        assert not re.fullmatch(regex, '[1, 2]')
        assert 'response_format' not in sent[0]
        assert len(choices) == {(3, 1): 3, (3, 2): 6, (5, 3): 60}[input_k, output_k]
        assert all(len(x) == output_k and len(set(x)) == output_k
                   and all(1 <= i <= input_k for i in x)
                   for x in map(json.loads, choices))
    else:
        assert 'uniqueItems' not in json.dumps(sent[0]['response_format'])
        assert sent[0]['response_format']['json_schema']['schema']['minItems'] == output_k
        assert sent[0]['response_format']['json_schema']['schema']['maxItems'] == output_k
        assert sent[0]['response_format']['json_schema']['schema']['items']['enum'] == list(range(1, input_k+1))



def test_native_ranking_to_slot_handoff(tmp_path):
    h, policy = recovery_harness(tmp_path)
    snapshot = _knowledge_snapshot()
    values = tuple(replace(snapshot.values[0], value_id='intent-boundaries::intent-boundary:'+name,
                           content='Description for '+name, task_target=name) for name in ('play_music','play_radio','play_video'))
    snapshot = replace(snapshot, values=values).validate()
    h.bootstrap_value_ids = tuple(v.value_id for v in values)
    template = tmp_path/'configs/probes/rank.txt'; template.parent.mkdir(parents=True)
    template.write_text('Rank by relevance.')
    h.repo_root = tmp_path
    policy.update(max_candidates=1, ranking_window={'mode':'ranked-candidate-indices', 'input_k':3,
        'output_k':1, 'max_tokens':128, 'template':{'relpath':'configs/probes/rank.txt',
        'sha256':hashlib.sha256(template.read_bytes()).hexdigest()}}, coverage_recovery={
        'strategy':'complete-original-candidate-set-v1','max_extra_contacts':1})
    h.config = replace(h.config, intent_slot_pipeline_json=canonical(policy))
    h.request_gate = lambda request: None
    def post(_url, body):
        request = json.loads(body)
        schema = request.get('response_format', {}).get('json_schema', {}).get('schema', {})
        if schema.get('type') == 'array': return _wire_response('[2]')
        if schema.get('properties', {}).get('candidates'):
            return _wire_response(canonical({'candidates':[{'candidate_id':'play_radio','entities':[], 'evidence_refs':[], 'unresolved_fields':[]}]}))
        return _wire_response(canonical({'candidate_id':'play_radio'}))
    h.post_override = post
    try:
        summary = h.run(snapshot=snapshot)
    finally:
        h.close()
    receipt = summary.results[0].native_episode_receipt
    assert 'intent_slot_pipeline' in receipt, json.dumps(receipt, indent=2)
    assert receipt['intent_slot_pipeline']['candidate_ids'] == ['play_radio']
    from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import validate_receipt
    records = {rid: payload for _, rid, payload in h.trace.records}
    validate_receipt(receipt, trace_records=records)


@pytest.mark.parametrize('raw', ['[0]', '[4]', '[true]', '[1,1]', '[]', '[1,2]'])
def test_reject_invalid_indices(raw):
    with pytest.raises(ValueError):
        parse(raw, [{'intent_id': str(i)} for i in range(3)], 1)


def test_reject_duplicate_indices_with_correct_output_length():
    with pytest.raises(ValueError):
        parse('[1,1]', [{'intent_id': str(i)} for i in range(3)], 2)
