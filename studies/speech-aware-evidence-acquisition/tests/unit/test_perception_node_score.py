import json
import pytest
from speech_aware_evidence_acquisition.analysis.perception_node_score import score, paired


def test_failure_rows_remain_in_denominator_and_native_grouping():
    roster = [{'sample_id': str(i), 'gold_intent': 'a'} for i in range(6)]
    row = {'intent_id': 'a', 'distinction': 'x', 'evidence_refs': [], 'unresolved_fields': []}
    good = json.dumps({'candidates': [row]})
    duplicate = json.dumps({'candidates': [row, row]})
    schema = {'type': 'object', 'required': ['candidates'], 'additionalProperties': False,
              'properties': {'candidates': {'type': 'array', 'minItems': 1, 'maxItems': 3,
                  'items': {'type': 'object', 'required': list(row), 'additionalProperties': False,
                            'properties': {'intent_id': {'enum': ['a', 'b']},
                              'distinction': {'type': 'string'}, 'evidence_refs': {'type': 'array'},
                              'unresolved_fields': {'type': 'array'}}}}}}
    context = {'pipeline_config': {'schema': 'intent-slot-v2', 'max_candidates': 3,
                                  'duplicate_candidate_policy': 'group_v2'},
               'response_format': {'json_schema': {'schema': schema}},
               'evidence_inventory': {'sample_id': 'sample', 'candidate_ids': [], 'sources': []}}
    def output(sid, text=good, finish='stop'):
        return {'sample_id': sid, 'text': text, 'finish_reason': finish, **context,
                'evidence_inventory': {**context['evidence_inventory'], 'sample_id': sid}}
    outputs = [output('0'), output('1', finish='length'), output('2', duplicate),
               output('3'), output('3'), {'sample_id': '5', 'text': good, 'finish_reason': 'stop'}]
    result = score(roster, outputs, maximum=3, legal_ids={'a', 'b'})
    assert [r['status'] for r in result['rows']] == ['valid', 'incomplete', 'valid', 'duplicate_output', 'missing', 'missing_native_context']
    assert (result['denominator'], result['valid_output_count'], result['gold_recall_count']) == (6, 2, 2)
    assert result['raw_duplicate_candidate_rows'] == 1
    assert not result['node_complete'] and not result['system_complete']
    assert paired(result, result) == {'denominator': 6, 'repairs': 0, 'regressions': 0}
    node = {**output('0'), 'response': good, 'stage': 'perception'}
    persisted = {'sample_id': '0', 'result_kind': 'node', 'text': good, 'node_outcome': node}
    assert score(roster[:1], [persisted], maximum=3, legal_ids={'a', 'b'})['gold_recall_count'] == 1
    with pytest.raises(ValueError, match='bind'):
        score(roster[:1], [{**persisted, 'text': 'different'}], maximum=3, legal_ids={'a', 'b'})
    with pytest.raises(ValueError):
        score(roster, outputs + [output('other')], maximum=3, legal_ids={'a'})


def test_native_terminal_status_cannot_be_promoted_by_parseable_text():
    roster = [{'sample_id': 's', 'gold_intent': 'a'}]
    for native_status, expected in [('invalid_output', 'invalid'),
                                    ('finite_completion', 'incomplete'),
                                    ('unexpected', 'invalid')]:
        node = {'sample_id': 's', 'response': '{"candidates": []}',
                'finish_reason': 'stop', 'status': native_status}
        row = {'sample_id': 's', 'result_kind': 'node',
               'text': node['response'], 'node_outcome': node}
        result = score(roster, [row], maximum=3, legal_ids={'a'})
        assert result['rows'][0]['status'] == expected
        assert result['valid_output_count'] == result['gold_recall_count'] == 0
