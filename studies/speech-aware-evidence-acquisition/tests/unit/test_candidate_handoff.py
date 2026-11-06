import copy
import pytest
from speech_aware_evidence_acquisition.rails.candidate_handoff import build_handoff, validate_handoff


def test_snapshot_handoff_exact_dedup_and_isolation():
    record = {'id': 'a', 'snapshot_sha256': '1'*64, 'text': 'known',
              'scope': {'intent': 'x_y'}, 'annotations': {'condition': 'one'}, 'target_role': 'slot'}
    other = dict(record, id='b')
    distinct = dict(record, id='c', annotations={'condition': 'two'})
    args = dict(sample_id='sample', source_role='perception', target_role='slot',
                source_response_sha256='2'*64, snapshot_sha256='1'*64,
                candidate_ids=['x_y'], judgments=[{'candidate_id': 'x_y', 'status': 'conflict'}])
    packet = build_handoff(**args, knowledge=[record, other, distinct])
    assert [g['ids'] for g in packet['knowledge']] == [['a', 'b'], ['c']]
    assert packet == build_handoff(**args, knowledge=[record, other, distinct])
    validate_handoff(packet, **{k: args[k] for k in ('sample_id', 'source_role', 'target_role', 'source_response_sha256', 'snapshot_sha256', 'candidate_ids')},
                     snapshot_values={'a':'known', 'b':'known', 'c':'known'})
    with pytest.raises(ValueError):
        validate_handoff(packet, **{**{k: args[k] for k in ('sample_id', 'source_role', 'target_role', 'source_response_sha256', 'snapshot_sha256', 'candidate_ids')}, 'sample_id':'other'}, snapshot_values={})
    changed = copy.deepcopy(packet); changed['knowledge'][0]['text'] = 'model rule'
    with pytest.raises(ValueError):
        validate_handoff(changed, **{k: args[k] for k in ('sample_id', 'source_role', 'target_role', 'source_response_sha256', 'snapshot_sha256', 'candidate_ids')}, snapshot_values={'a':'known'})
    with pytest.raises(ValueError):
        build_handoff(**{**args, 'judgments':[{'candidate_id':'x_y','status':'candidate','rule':'invented'}]}, knowledge=[])
