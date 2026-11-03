import copy
import hashlib
import json
import pytest
from speech_aware_evidence_acquisition.rails.gepa_pair_request_audit import verify_inputs, ARMS


def fixture():
    contacts, requests, instructions = {}, {}, {}
    for arm in ARMS:
        rid = 'request-' + arm; cid = 'contact-' + arm; instruction = 'instruction ' + arm
        control = {'contact_id': cid, 'sample_id': 'slurp:train:1',
            'messages': [{'role': 'system', 'content': 'frozen global instruction'}, {'role': 'user', 'content': instruction}]}
        text = json.dumps(control, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        history = [{'role': 'agent_state', 'state_id': 'phase:perception', 'content': 'same perception'},
            {'role': 'control', 'contact_id': cid, 'content': text,
             'content_sha256': hashlib.sha256(text.encode()).hexdigest()}]
        requests[arm] = {'request_id': rid, 'sample_id': 'slurp:train:1', 'task_instruction': 'frozen global instruction',
            'history': history, 'supplied_evidence': [{'content': 'frozen knowledge'}],
            'evidence_provenance': [{'lock_key': 'rails-supply', 'relpath': f'rails-message/{rid}/0', 'sha256': 'a'*64}],
            'decoding_params': {'temperature': 0.3, 'seed': 3}, 'speech_packet_sha256': 'b'*64}
        contacts[arm] = {'request_id': rid, 'agent_contact_id': cid}
        instructions[arm] = instruction
    return contacts, requests, instructions


def test_actual_primary_inputs_match_except_instruction_and_routing():
    contacts, requests, instructions = fixture()
    before = copy.deepcopy(requests)
    result = verify_inputs('slurp:train:1', contacts, requests, rendered_instructions=instructions)
    assert result['verified_primary_solver_requests'] == 3 and result['new_model_calls'] == 0
    assert before == requests


@pytest.mark.parametrize('fault', ['state', 'speech', 'decoding', 'evidence', 'provenance',
    'instruction', 'control_instruction', 'control_hash', 'extra_field', 'request_id', 'contact', 'row', 'absent_state'])
def test_different_or_forged_context_cannot_be_accepted(fault):
    contacts, requests, instructions = fixture(); row = requests['child']
    if fault == 'state': row['history'][0]['content'] = 'different P'
    elif fault == 'speech': row['speech_packet_sha256'] = 'c'*64
    elif fault == 'decoding': row['decoding_params']['temperature'] = 1.0
    elif fault == 'evidence': row['supplied_evidence'][0]['content'] = 'changed knowledge'
    elif fault == 'provenance': row['evidence_provenance'][0]['relpath'] = 'rails-message/wrong/0'
    elif fault == 'instruction': row['task_instruction'] = 'different'
    elif fault == 'control_instruction':
        ctl = json.loads(row['history'][-1]['content']); ctl['messages'][1]['content'] = 'different'
        row['history'][-1]['content'] = json.dumps(ctl)
        row['history'][-1]['content_sha256'] = hashlib.sha256(row['history'][-1]['content'].encode()).hexdigest()
    elif fault == 'control_hash': row['history'][-1]['content_sha256'] = '0'*64
    elif fault == 'extra_field': row['unregistered_flag'] = True
    elif fault == 'request_id': row['request_id'] = 'other'
    elif fault == 'contact': contacts['child']['agent_contact_id'] = 'other'
    elif fault == 'row': row['sample_id'] = 'slurp:test:1'
    else: row['history'].pop(0)
    with pytest.raises(ValueError):
        verify_inputs('slurp:train:1', contacts, requests, rendered_instructions=instructions)


def test_global_task_instruction_must_not_be_normalized_as_role_template():
    contacts,requests,instructions=fixture()
    row=requests['child'];row['task_instruction']='changed global task'
    control=json.loads(row['history'][-1]['content']);control['messages'][0]['content']=row['task_instruction']
    content=json.dumps(control);row['history'][-1].update(content=content,content_sha256=hashlib.sha256(content.encode()).hexdigest())
    with pytest.raises(ValueError,match='outside verified instruction'):
        verify_inputs('slurp:train:1',contacts,requests,rendered_instructions=instructions)
